"""Shared helper for tests that exercise a real, non-synthetic recording too
large to ship as a repo fixture (issue #123: the on008083 EDF regression test
needs an actual ~192 MiB file, not a hand-built stand-in, to prove the fix
against the exact allocation pattern that triggered the bug).

Real-data tests are opt-in and never run by accident: they need network
access, are slow, and are unrelated to biosigio's platform-independent unit
tests. Two independent gates, both a ``pytest.skip`` (never a hard failure):

* the ``BIOSIGIO_REAL_DATA`` environment variable must be set (any non-empty
  value) to opt in at all.
* the URL must actually be reachable -- a short-timeout request decides
  "offline" so a flaky network never fails the suite.

Usage::

    from biosigio.tests.real_data import fetch_real_recording

    def test_something():
        path = fetch_real_recording(
            "https://data.nemar.org/on008083/v1.0.0/sub-001/ses-01/eeg/"
            "sub-001_ses-01_task-HierPrior_eeg.edf",
            min_bytes=190_000_000,
        )
        ...

The downloaded file is cached under ``BIOSIGIO_REAL_DATA_CACHE`` (default
``~/.cache/biosigio/real_data``), OUTSIDE the repository -- never committed,
and shared across biosigio worktrees/branches on the same machine so a
192 MiB download only ever happens once per host, not once per worktree.

NOTE for other biosigio worktrees adding their own real-data test (e.g. the
view-chunks or units-rescale work): this module did not exist before issue
#123's fix and is intentionally minimal (one function, one cache policy).
Add your own ``fetch_real_recording(url, ...)`` call here rather than
hand-rolling a second download/cache/skip helper, so every real-data test in
the suite shares one cache directory and one opt-in policy.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ENV_VAR = "BIOSIGIO_REAL_DATA"
CACHE_ENV_VAR = "BIOSIGIO_REAL_DATA_CACHE"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "biosigio" / "real_data"
_TIMEOUT_S = 30.0
_CHUNK_BYTES = 4 * 1024 * 1024
# data.nemar.org (and presumably other hosts behind the same WAF/CDN) resets the
# connection outright for the default `Python-urllib/x.y` User-Agent urllib sends
# with none set -- confirmed interactively while wiring this up: identical
# request, only the header differs, `Python-urllib/3.14` -> ConnectionResetError,
# `python-requests/2.0` / a browser UA -> 200. Any non-default UA string clears
# it, so this is a generic anti-bot header check, not a package-specific block.
_USER_AGENT = "biosigio-tests/1.0 (+https://github.com/neuromechanist/biosigio)"


def cache_dir() -> Path:
    """The shared real-data cache directory (overridable via ``BIOSIGIO_REAL_DATA_CACHE``)."""
    return Path(os.environ.get(CACHE_ENV_VAR) or _DEFAULT_CACHE_DIR)


def _sidecar_path(dest: Path) -> Path:
    """Where a cached file's known-good sha256 digest lives, once computed."""
    return dest.with_name(dest.name + ".sha256")


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integrity_error(dest: Path, expected_sha256: str | None) -> str | None:
    """``None`` if ``dest`` passes the integrity check (or none was requested),
    otherwise a message describing the mismatch.

    The digest is computed at most once per file, not once per test run: a
    sidecar ``<file>.sha256`` next to the cached file remembers it (written the
    first time this file is verified, whether that is a fresh download or an
    older cached file from before this check existed), so a warm-cache rerun
    compares two short hex strings instead of re-hashing a ~192 MiB file.
    """
    if expected_sha256 is None:
        return None
    sidecar = _sidecar_path(dest)
    actual = sidecar.read_text().strip() if sidecar.exists() else None
    if actual is None:
        actual = _sha256_of(dest)
        sidecar.write_text(actual + "\n")
    if actual.lower() != expected_sha256.lower():
        return f"sha256 mismatch: cached file is {actual}, expected {expected_sha256}"
    return None


def _discard(dest: Path) -> None:
    dest.unlink(missing_ok=True)
    _sidecar_path(dest).unlink(missing_ok=True)


def fetch_real_recording(
    url: str,
    *,
    filename: str | None = None,
    min_bytes: int = 1,
    sha256: str | None = None,
) -> Path:
    """Return a local path to ``url``, downloading it into the shared cache on
    first use. Calls ``pytest.skip`` (never raises) when real-data tests are
    not opted into, the URL cannot be fetched, or the file fails its integrity
    check.

    Args:
        url: Direct HTTPS URL to the file.
        filename: Cache filename; defaults to the URL's last path segment.
        min_bytes: Sanity floor on the cached file's size, so a previous
            partial/failed download (or an HTML error page saved under the
            expected name) is not silently reused.
        sha256: Expected sha256 hex digest of the file, when known. Verified
            on every call (cheaply -- see :func:`_integrity_error`), so a
            corrupted or substituted cache entry is caught rather than fed
            silently into a test.

    Returns:
        Path: local path to the cached file. Guaranteed to exist, be at least
        ``min_bytes``, and match ``sha256`` (when given) when this returns
        (otherwise the test was skipped).
    """
    if not os.environ.get(ENV_VAR):
        pytest.skip(
            f"real-data test skipped: set {ENV_VAR}=1 to opt in "
            "(downloads a real recording from the network)"
        )

    dest_dir = cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (filename or url.rsplit("/", 1)[-1])

    if dest.exists() and dest.stat().st_size >= min_bytes:
        mismatch = _integrity_error(dest, sha256)
        if mismatch is None:
            return dest
        _discard(dest)
        pytest.skip(
            f"real-data test skipped: cached file for {url} failed integrity check ({mismatch})"
        )

    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as e:
            # http.client.HTTPException covers a mid-transfer IncompleteRead
            # (and other low-level protocol errors) that urllib does not wrap
            # into URLError -- those raise directly out of resp.read().
            pytest.skip(f"real-data test skipped: could not fetch {url} ({e})")
    finally:
        # Removes the partial download on every failure path, including one
        # not listed above (e.g. a disk-full OSError while writing tmp is
        # already covered, but this also catches anything future code adds
        # here without needing a matching except clause). A no-op on success:
        # tmp.replace(dest) above already moved the file out from under it.
        tmp.unlink(missing_ok=True)

    if dest.stat().st_size < min_bytes:
        _discard(dest)
        pytest.skip(f"real-data test skipped: download of {url} looked truncated or invalid")

    mismatch = _integrity_error(dest, sha256)
    if mismatch is not None:
        _discard(dest)
        pytest.skip(
            f"real-data test skipped: download of {url} failed integrity check ({mismatch})"
        )

    return dest


def fetch_real_byte_ranges(
    url: str,
    ranges: list[tuple[int, int]],
    *,
    filename: str,
    sha256: str | None = None,
) -> Path:
    """Return a local path to the concatenation of byte ``ranges`` of ``url``,
    fetched with HTTP Range requests into the shared cache on first use.

    For a recording too large to download whole (hundreds of MB) when a test only
    needs its header and a few records: the excerpt is real bytes of the real file.
    Same opt-in, cache, User-Agent and skip policy as :func:`fetch_real_recording`
    (``BIOSIGIO_REAL_DATA`` unset, offline, a server that ignores ``Range``, a short
    read or a digest mismatch all ``pytest.skip``).

    Args:
        url: Direct HTTPS URL to the file.
        ranges: ``(first_byte, last_byte)`` pairs, both inclusive, in the order they
            are laid end to end in the excerpt.
        filename: Cache filename of the excerpt (the URL's own name would collide
            with a whole-file download of the same URL).
        sha256: Expected digest of the whole excerpt, verified on every call.
    """
    if not os.environ.get(ENV_VAR):
        pytest.skip(
            f"real-data test skipped: set {ENV_VAR}=1 to opt in "
            "(downloads part of a real recording from the network)"
        )
    expected = sum(last - first + 1 for first, last in ranges)
    dest_dir = cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size == expected:
        mismatch = _integrity_error(dest, sha256)
        if mismatch is None:
            return dest
        _discard(dest)
        pytest.skip(
            f"real-data test skipped: cached excerpt of {url} failed integrity check ({mismatch})"
        )

    tmp = dest.with_name(dest.name + ".part")
    try:
        try:
            with open(tmp, "wb") as fh:
                for first, last in ranges:
                    request = urllib.request.Request(
                        url,
                        headers={"User-Agent": _USER_AGENT, "Range": f"bytes={first}-{last}"},
                    )
                    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:
                        if resp.status != 206:
                            pytest.skip(
                                f"real-data test skipped: {url} ignored the Range request "
                                f"(HTTP {resp.status})"
                            )
                        body = resp.read()
                    if len(body) != last - first + 1:
                        pytest.skip(
                            f"real-data test skipped: short Range read from {url} "
                            f"({len(body)} of {last - first + 1} bytes)"
                        )
                    fh.write(body)
            tmp.replace(dest)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as e:
            pytest.skip(f"real-data test skipped: could not fetch {url} ({e})")
    finally:
        tmp.unlink(missing_ok=True)

    mismatch = _integrity_error(dest, sha256)
    if mismatch is not None:
        _discard(dest)
        pytest.skip(f"real-data test skipped: excerpt of {url} failed integrity check ({mismatch})")
    return dest
