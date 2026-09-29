"""BrainVision importer backed by MNE-Python (.vhdr + .vmrk + .eeg).

Many OpenNeuro BIDS EEG/iEEG datasets ship the BrainVision triplet rather than
EDF or EEGLAB .set. MNE reads it natively (only writing needs pybv), and is an
optional dependency imported lazily with a clear install hint. Channel types and
units come from the shared :mod:`_mne_common` mapping; ``.vmrk`` markers (which
MNE exposes as annotations) are read into ``Recording.events``.
"""

import contextlib
import errno
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterator

import pandas as pd

from ..core.emg import Recording
from ..exceptions import (
    BiosigIOError,
    classify_read_error,
    is_host_condition,
    is_resource_exhaustion,
)
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter

logger = logging.getLogger(__name__)

# ``DataFile=`` / ``MarkerFile=`` lines of the header's [Common Infos] section.
_FILE_REF = re.compile(r"^(\s*)(DataFile|MarkerFile)(\s*=\s*)(.*?)(\s*)$", re.IGNORECASE)
# Split after each ``\n`` and after a lone ``\r`` (old Mac line endings), keeping
# the ending on its line. Unlike ``str.splitlines`` this never breaks on ``\x85``,
# ``\x0b``, ``\x0c`` or `` ``, which a Latin-1/cp1252 header can carry inside a
# channel name and which MNE's own line reading (``StringIO``) does not split on.
_LINE_BREAK = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")
# Same-stem siblings tried for each key, in order (``.dat`` is the legacy data name).
# MarkerFile is patched too: MNE 1.13 recovers a stale MarkerFile= itself, but the
# 1.12.x line NEMAR runs does not (on002158 fails at ``open(mrk_fname)`` with
# FileNotFoundError). On MNE >= 1.13 the marker-only test row therefore passes with
# or without this patch; it proves the patch only on MNE 1.12.
_SIBLING_EXTS = {"datafile": (".eeg", ".dat"), "markerfile": (".vmrk",)}


def _header_encoding(settings: bytes) -> str:
    """The codepage MNE decodes a header with (``Codepage=``, else UTF-8; Latin-1 fallback).

    This intentionally mirrors ``mne.io.brainvision._aux_hdr_info`` so the patched
    copy is decoded by MNE exactly as the original would be. Two known, harmless
    differences: MNE passes ``re.IGNORECASE & re.MULTILINE`` (which is ``0``), so its
    search is case-sensitive and its ``"ANSI"`` check exact, where this one ignores
    case; and MNE lets an unknown codepage raise ``LookupError`` where this falls
    back to Latin-1. Untouched lines round-trip byte-for-byte under either codec,
    and an unknown ``Codepage=`` survives into the copy, so MNE still raises on it.
    """
    match = re.search(r"Codepage=(.+)", settings.decode("ascii", "ignore"), re.IGNORECASE)
    codepage = match.group(1).strip() if match else "utf-8"
    if codepage.upper() == "ANSI":
        codepage = "cp1252"
    try:
        settings.decode(codepage)
    except (LookupError, UnicodeDecodeError):
        return "latin-1"
    return codepage


# Recording-metadata key recording a successful stale-header recovery: which
# header key named which missing file, and which same-stem sibling was read
# instead, as ``{"DataFile": {"referenced": ..., "used": ...}, ...}``.
HEADER_RECOVERED_KEY = "brainvision_header_recovered"
# Canonical spelling of the two keys the resolver rewrites.
_KEY_NAMES = {"datafile": "DataFile", "markerfile": "MarkerFile"}


def _stage(target: str, staged: str) -> None:
    """Make ``staged`` (in the temporary directory) refer to the file ``target``.

    A symbolic link where the host allows one, else a hard link (NTFS allows one
    without the symlink privilege), else a byte copy. Only the copy's failure is
    raised: it is a host condition (``ENOSPC``, quota, a read-only temp root) and
    propagates as the raw ``OSError``.
    """
    for link in (os.symlink, os.link):
        try:
            link(target, staged)
            return
        except (OSError, NotImplementedError):
            continue
    shutil.copyfile(target, staged)


def _sibling_finder(stem: str):
    """Return ``find(exts)``: the first existing ``<stem><ext>`` for ``exts``, or None.

    Each extension is tried exactly first, then case-insensitively (``X.EEG``
    or ``X.VMRK`` beside ``X.vhdr`` on a case-sensitive filesystem), where a
    case-insensitive candidate is accepted only when it is the ONLY one: two
    files differing only in case leave the reference unresolved rather than
    guessing. The directory is listed once, lazily.
    """
    directory = os.path.dirname(stem)
    base = os.path.basename(stem)
    entries: list[str] | None = None

    def listing() -> list[str]:
        nonlocal entries
        if entries is None:
            try:
                entries = os.listdir(directory)
            except OSError as err:
                if is_resource_exhaustion(err):
                    raise
                logger.warning(
                    "Could not list %s to look for BrainVision siblings "
                    "case-insensitively (%s); trying exact names only.",
                    directory,
                    err,
                )
                entries = []
        return entries

    def find(exts: tuple[str, ...]) -> str | None:
        for ext in exts:
            exact = stem + ext
            if os.path.isfile(exact):
                return exact
            wanted = (base + ext).lower()
            matches = [
                name
                for name in listing()
                if name.lower() == wanted and os.path.isfile(os.path.join(directory, name))
            ]
            if len(matches) == 1:
                return os.path.join(directory, matches[0])
        return None

    return find


@contextlib.contextmanager
def resolved_vhdr(
    vhdr_path: str,
    *,
    substitutions: dict | None = None,
    staged_files: dict | None = None,
) -> Iterator[str]:
    """Yield a ``.vhdr`` path whose ``DataFile=``/``MarkerFile=`` point at files that exist.

    BIDS renames the BrainVision triplet on disk but not the references inside the
    header, so the header can name a pre-rename ``.eeg``/``.vmrk`` that no longer
    exists while the renamed file sits beside it with the header's stem (the same
    defect :meth:`EEGLABImporter._read_fdt` resolves for ``.fdt``). When a referenced
    file is missing and that same-stem sibling exists (see :func:`_sibling_finder`),
    a patched copy of the header is written to a temporary directory and yielded
    instead. Every file the header references is staged beside that copy under a
    fixed ASCII name (``data<ext>``, ``marker.vmrk``; a symbolic link, else a hard
    link, else a byte copy), and the copy names them by those bare names, so no
    header codepage ever has to spell the dataset's own path. Everything else in the
    copy (its encoding, its ``Codepage=``, its line endings) is the original's.
    Otherwise the original path is yielded unchanged, so a good header and an
    unrecoverable one behave exactly as MNE reads them. The dataset files are never
    modified.

    Args:
        vhdr_path: Path to the BrainVision ``.vhdr`` header.
        substitutions: Optional dict filled, when a patched copy is yielded, with
            ``{"DataFile": {"referenced": <name in the header>, "used": <sibling
            file name>}, ...}`` for each substituted key; callers record it under
            :data:`HEADER_RECOVERED_KEY`. Left empty otherwise.
        staged_files: Optional dict filled, when a patched copy is yielded, with
            ``{<path in the temporary directory>: <dataset file it stands for>}``
            for every reference, so :func:`brainvision_read_error` can name the
            dataset's file in an error. Left empty otherwise.

    Yields:
        The path to hand to MNE. A patched copy and its staged files are removed on
        exit, so a lazily read ``Raw`` (``preload=False``) must be used inside the
        block: MNE keeps the staged data path and opens it on every read.

    Raises:
        OSError: Reading the header failed with a host condition (see
            :func:`~biosigio.exceptions.is_host_condition`), or writing the
            patched copy or staging a file failed; both are raised unchanged.
    """
    directory = os.path.dirname(os.path.abspath(vhdr_path))
    stem = os.path.splitext(os.path.abspath(vhdr_path))[0]
    content: bytes | None = None
    read_error: OSError | None = None
    try:
        with open(vhdr_path, "rb") as f:
            content = f.read()
    except OSError as err:
        # Handled below, not here: yielding inside ``except`` would run the caller's
        # block with this OSError as the implicit ``__context__`` of anything it raises.
        read_error = err
    if content is None:
        assert read_error is not None  # content stays None only when open/read raised
        if is_host_condition(read_error):
            # The host, not the header (resource exhaustion, EACCES, EIO, a stale
            # network handle): raised as-is so a caller can retry.
            raise read_error
        # Unreadable header: hand MNE the original path and let it raise the real
        # read error, typed by the caller exactly as before this resolver existed.
        logger.warning(
            "Could not read BrainVision header %s to check its file references "
            "(errno %s: %s); handing it to the reader unchanged.",
            vhdr_path,
            errno.errorcode.get(read_error.errno, read_error.errno)
            if read_error.errno is not None
            else None,
            read_error.strerror or read_error,
        )
        read_error = None
        yield vhdr_path
        return
    encoding = _header_encoding(content)
    lines = _LINE_BREAK.split(content.decode(encoding))
    find_sibling = _sibling_finder(stem)

    found: dict[str, dict[str, str]] = {}
    # Every reference line: (line index, key, the parts around the value, target).
    references: list[tuple[int, str, str, str, str, str, str]] = []
    section = ""
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if body.strip().startswith("["):
            section = body.strip().lower()
            continue
        match = _FILE_REF.match(body)
        if section != "[common infos]" or match is None or not match.group(4):
            continue
        indent, key, sep, ref, trail = match.groups()
        target = os.path.join(directory, ref)
        if not os.path.isfile(target):
            sibling = find_sibling(_SIBLING_EXTS[key.lower()])
            if sibling is not None:
                target = sibling
                found[_KEY_NAMES[key.lower()]] = {
                    "referenced": ref,
                    "used": os.path.basename(sibling),
                }
        references.append((i, key, indent, sep, trail, line[len(body) :], target))

    if not found:
        yield vhdr_path
        return
    # A failure to create, stage or write the copy propagates unchanged (see
    # BrainVisionImporter.load); a failure to remove it after a successful read is
    # ignored rather than turned into a spurious error for a recording that loaded.
    with tempfile.TemporaryDirectory(prefix="biosigio-vhdr-", ignore_cleanup_errors=True) as tmp:
        staged: dict[str, str] = {}
        for i, key, indent, sep, trail, eol, target in references:
            if key.lower() == "markerfile":
                # MNE picks the marker reader by the exact ``.vmrk`` suffix, so an
                # upper-case ``X.VMRK`` sibling is staged under a lower-case name.
                name = "marker.vmrk"
            else:
                ext = os.path.splitext(target)[1]
                name = "data" + (ext if ext.isascii() else ".eeg")
            staged_path = os.path.join(tmp, name)
            if staged_path not in staged and os.path.isfile(target):
                _stage(os.path.abspath(target), staged_path)
            # A reference that resolves to nothing stays unstaged: MNE then raises
            # on it exactly as it would on the original header.
            staged[staged_path] = os.path.abspath(target)
            lines[i] = f"{indent}{key}{sep}{name}{trail}{eol}"
        # Every reference is now an ASCII name and every other line decoded from
        # ``encoding``, so the copy always encodes in the header's own codepage.
        tmp_vhdr = os.path.join(tmp, os.path.basename(vhdr_path))
        with open(tmp_vhdr, "wb") as f:
            f.write("".join(lines).encode(encoding))
        logger.info(
            "BrainVision header %s names missing files; reading it through a patched "
            "copy that points at the same-stem siblings: %s",
            vhdr_path,
            ", ".join(f"{k}: {v['referenced']} -> {v['used']}" for k, v in found.items()),
        )
        if substitutions is not None:
            substitutions.update(found)
        if staged_files is not None:
            staged_files.update(staged)
        yield tmp_vhdr


def brainvision_read_error(
    exc: Exception, used_vhdr: str, vhdr_path: str, staged_files: dict | None = None
) -> BiosigIOError:
    """Type a BrainVision read failure, naming the dataset's files, not the temp copy.

    The error is classified by :func:`~biosigio.exceptions.classify_read_error`,
    which re-raises a host condition unchanged. When MNE read a patched copy from
    :func:`resolved_vhdr`, its message quotes paths in the temporary
    ``biosigio-vhdr-*`` directory, which is deleted on exit: the copy is replaced
    with ``vhdr_path``, and each staged file (``staged_files``, as
    :func:`resolved_vhdr` fills it) with the dataset file it stood for, so the
    error points at files that exist.
    """
    typed = classify_read_error(exc, vhdr_path)
    if os.path.abspath(used_vhdr) == os.path.abspath(vhdr_path):
        return typed
    message = str(typed)
    real_dir = os.path.dirname(os.path.abspath(vhdr_path))
    tmp_dir = os.path.dirname(used_vhdr)
    pairs = [(used_vhdr, vhdr_path), *(staged_files or {}).items(), (tmp_dir, real_dir)]
    replacements = {}
    for old, new in pairs:
        replacements.setdefault(old, new)
        replacements.setdefault(os.path.realpath(old), new)
    # Longest first, so a file path is replaced before the directory inside it.
    for old, new in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        message = message.replace(old, new)
    if message == str(typed):
        return typed
    return type(typed)(message)


class BrainVisionImporter(BaseImporter):
    """Importer for BrainVision recordings via MNE-Python (.vhdr)."""

    def _read_events(self, raw) -> pd.DataFrame:
        """Read .vmrk markers (MNE annotations) into an events DataFrame."""
        annotations = raw.annotations
        # strict=True: the three arrays come from one Annotations object and are
        # co-length by MNE's contract; a mismatch is a bug we want surfaced, not
        # silently truncated.
        rows = [
            (float(onset), float(duration), str(description))
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description, strict=True
            )
        ]
        evt = pd.DataFrame(rows, columns=["onset", "duration", "description"])
        if not evt.empty:
            evt = evt.sort_values("onset").reset_index(drop=True)
            evt["onset"] = evt["onset"].astype("float64")
            evt["duration"] = evt["duration"].astype("float64")
        return evt

    def load(self, filepath: str) -> Recording:
        """Load a BrainVision recording (pass the ``.vhdr`` header path).

        Args:
            filepath: Path to the BrainVision ``.vhdr`` header (its ``.vmrk`` and
                ``.eeg`` siblings are resolved by MNE).

        Returns:
            Recording: channels carry their type and physical unit; ``.vmrk`` markers
            are read into ``Recording.events``.
        """
        mne = require_mne()
        # The resolver sits OUTSIDE the classifying try on purpose: what it can
        # raise is a failure to write the patched temporary header or stage a
        # file beside it (ENOSPC, EROFS, EACCES, quota), or a host condition
        # while reading the header. Each is a property of the host, not of the
        # recording, and must propagate as-is rather than become a typed
        # (possibly permanent) read failure. Any other unreadable header is not
        # raised here; the resolver yields the original path and MNE raises
        # inside the try below.
        recovered: dict = {}
        staged: dict = {}
        with resolved_vhdr(filepath, substitutions=recovered, staged_files=staged) as vhdr:
            try:
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            except Exception as e:
                # Resource exhaustion (MemoryError, thread/allocation-exhaustion
                # OSError/RuntimeError) is a host condition, not a file problem --
                # propagate unchanged rather than reclassifying it as a permanent
                # read failure (see biosigio.exceptions.is_resource_exhaustion).
                if is_resource_exhaustion(e):
                    raise
                raise brainvision_read_error(e, vhdr, filepath, staged) from e

        rec = raw_to_recording(raw)
        rec.set_metadata("source_file", filepath)
        if recovered:
            # A stale header read through its siblings is recorded, not silent.
            rec.set_metadata(HEADER_RECOVERED_KEY, recovered)

        events = self._read_events(raw)
        if not events.empty:
            rec.events = events

        return rec
