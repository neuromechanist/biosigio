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
from ..exceptions import classify_read_error, is_resource_exhaustion
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter

logger = logging.getLogger(__name__)

# ``DataFile=`` / ``MarkerFile=`` lines of the header's [Common Infos] section.
_FILE_REF = re.compile(r"^(\s*)(DataFile|MarkerFile)(\s*=\s*)(.*?)(\s*)$", re.IGNORECASE)
# Split after each ``\n`` and after a lone ``\r`` (old Mac line endings), keeping
# the ending on its line. Unlike ``str.splitlines`` this never breaks on ``\x85``,
# ``\x0b``, ``\x0c`` or `` ``, which a Latin-1/cp1252 header can carry inside a
# channel name and which MNE's own line reading (``StringIO``) does not split on.
# The value of a ``Codepage=`` key, up to (not including) its line ending.
_CODEPAGE_VALUE = re.compile(r"(Codepage\s*=\s*)[^\r\n]*", re.IGNORECASE)
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


def _declare_utf8(text: str) -> str:
    """``text`` with its ``Codepage=`` value set to ``UTF-8`` (line ending kept).

    Only the first ``Codepage=`` after the version line is rewritten, the one MNE
    reads. With none present MNE already decodes UTF-8, so nothing is added.
    """
    first, sep, rest = text.partition("\n")
    if not sep:  # lone-CR header: the version line ends at the first CR
        first, sep, rest = text.partition("\r")
    rest = _CODEPAGE_VALUE.sub(r"\g<1>UTF-8", rest, count=1)
    return first + sep + rest


# Recording-metadata key recording a successful stale-header recovery: which
# header key named which missing file, and which same-stem sibling was read
# instead, as ``{"DataFile": {"referenced": ..., "used": ...}, ...}``.
HEADER_RECOVERED_KEY = "brainvision_header_recovered"
# Canonical spelling of the two keys the resolver rewrites.
_KEY_NAMES = {"datafile": "DataFile", "markerfile": "MarkerFile"}


class BrainVisionHeaderRecoveryError(RuntimeError):
    """A header names missing files and the recovery copy could not be written.

    Raised by :func:`resolved_vhdr` when the same-stem siblings exist but their
    absolute paths cannot be spelled in any header encoding, not even UTF-8
    (a path holding undecodable, surrogate-escaped bytes). It is deliberately
    NOT a :class:`~biosigio.exceptions.BiosigIOError`: the recording itself is
    readable, and what failed is spelling this host's path to it, the same
    class of condition as a temp copy that cannot be written (which propagates
    as a raw ``OSError``). A caller that treats typed errors as permanent file
    failures therefore does not record this one as such.
    """


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
def resolved_vhdr(vhdr_path: str, *, substitutions: dict | None = None) -> Iterator[str]:
    """Yield a ``.vhdr`` path whose ``DataFile=``/``MarkerFile=`` point at files that exist.

    BIDS renames the BrainVision triplet on disk but not the references inside the
    header, so the header can name a pre-rename ``.eeg``/``.vmrk`` that no longer
    exists while the renamed file sits beside it with the header's stem (the same
    defect :meth:`EEGLABImporter._read_fdt` resolves for ``.fdt``). When a referenced
    file is missing and that same-stem sibling exists (matched exactly, else
    case-insensitively when exactly one file matches), a patched copy of the header
    (original encoding and line endings; only those two keys rewritten, to absolute
    paths) is written to a temporary directory and yielded instead; when the header's
    codepage cannot spell such a path, the copy is written as UTF-8 with its
    ``Codepage=`` rewritten to match. Otherwise the
    original path is yielded unchanged, so a good header and an unrecoverable one
    behave exactly as MNE reads them. The dataset files are never modified.

    Args:
        vhdr_path: Path to the BrainVision ``.vhdr`` header.
        substitutions: Optional dict filled, when a patched copy is yielded, with
            ``{"DataFile": {"referenced": <name in the header>, "used": <sibling
            file name>}, ...}`` for each substituted key; callers record it under
            :data:`HEADER_RECOVERED_KEY`. Left empty otherwise.

    Yields:
        The path to hand to MNE; a patched copy is removed on exit (MNE reads the
        header and markers when the ``Raw`` is built and keeps only the data path).

    Raises:
        BrainVisionHeaderRecoveryError: The siblings exist but no header encoding
            can spell their paths (see the class).
        OSError: Reading the header failed with resource exhaustion, or writing
            the patched copy failed; both are host conditions, raised unchanged.
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
        if is_resource_exhaustion(read_error):
            # The host, not the header: raised as-is so a caller can retry.
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
    # A marker sibling found only case-insensitively (``X.VMRK``): MNE picks the
    # marker reader by the file's exact suffix, so it must be handed a ``.vmrk``
    # name. Staged as a byte copy in the temp directory: (line index, parts).
    staged_marker: tuple[int, str, str, str, str, str, str] | None = None
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
        eol = line[len(body) :]
        target = os.path.join(directory, ref)
        if not os.path.isfile(target):
            sibling = find_sibling(_SIBLING_EXTS[key.lower()])
            if sibling is not None:
                target = sibling
                found[_KEY_NAMES[key.lower()]] = {
                    "referenced": ref,
                    "used": os.path.basename(sibling),
                }
                if key.lower() == "markerfile" and not sibling.endswith(".vmrk"):
                    staged_marker = (i, indent, key, sep, trail, eol, sibling)
        # Every reference becomes absolute: the patched copy lives elsewhere.
        lines[i] = f"{indent}{key}{sep}{os.path.abspath(target)}{trail}{eol}"

    if not found:
        yield vhdr_path
        return
    # A failure to create or write the copy propagates unchanged (see
    # BrainVisionImporter.load); a failure to remove it after a successful read is
    # ignored rather than turned into a spurious error for a recording that loaded.
    with tempfile.TemporaryDirectory(prefix="biosigio-vhdr-", ignore_cleanup_errors=True) as tmp:
        if staged_marker is not None:
            i, indent, key, sep, trail, eol, sibling = staged_marker
            staged = os.path.join(tmp, "marker.vmrk")
            shutil.copyfile(sibling, staged)
            lines[i] = f"{indent}{key}{sep}{staged}{trail}{eol}"
        text = "".join(lines)
        patched: bytes | None = None
        try:
            patched = text.encode(encoding)
        except UnicodeEncodeError:
            # An absolute path the header's codepage cannot spell (e.g. a Greek or
            # CJK directory under a cp1252 header): write the copy as UTF-8 instead
            # and declare it, which every character of the decoded header can be
            # spelled in.
            with contextlib.suppress(UnicodeEncodeError):
                patched = _declare_utf8(text).encode("utf-8")
        if patched is None:
            # Not even UTF-8 can spell it (an undecodable, surrogate-escaped path).
            # Handing MNE the stale header would only fail on the missing file and
            # hide the real cause, so say what happened instead.
            missing = ", ".join(f"{k}={v['referenced']}" for k, v in found.items())
            raise BrainVisionHeaderRecoveryError(
                f"BrainVision header {vhdr_path} names missing files ({missing}); the "
                "same-stem siblings exist, but the recovery copy pointing at them "
                "could not be written because their paths cannot be encoded in a "
                "header, not even as UTF-8"
            )
        tmp_vhdr = os.path.join(tmp, os.path.basename(vhdr_path))
        with open(tmp_vhdr, "wb") as f:
            f.write(patched)
        logger.info(
            "BrainVision header %s names missing files; reading it through a patched "
            "copy that points at the same-stem siblings: %s",
            vhdr_path,
            ", ".join(f"{k}: {v['referenced']} -> {v['used']}" for k, v in found.items()),
        )
        if substitutions is not None:
            substitutions.update(found)
        yield tmp_vhdr


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
        # raise is a failure to write the patched temporary header (ENOSPC,
        # EROFS, EACCES, quota), resource exhaustion while reading the header, or
        # BrainVisionHeaderRecoveryError (a sibling path no header can spell).
        # Each is a host condition, not a property of the recording, and must
        # propagate as-is rather than become a typed (possibly permanent) read
        # failure. Any other unreadable header is not raised here; the resolver
        # yields the original path and MNE raises inside the try below.
        recovered: dict = {}
        with resolved_vhdr(filepath, substitutions=recovered) as vhdr:
            try:
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            except Exception as e:
                # Resource exhaustion (MemoryError, thread/allocation-exhaustion
                # OSError/RuntimeError) is a host condition, not a file problem --
                # propagate unchanged rather than reclassifying it as a permanent
                # read failure (see biosigio.exceptions.is_resource_exhaustion).
                if is_resource_exhaustion(e):
                    raise
                raise classify_read_error(e, filepath) from e

        rec = raw_to_recording(raw)
        rec.set_metadata("source_file", filepath)
        if recovered:
            # A stale header read through its siblings is recorded, not silent.
            rec.set_metadata(HEADER_RECOVERED_KEY, recovered)

        events = self._read_events(raw)
        if not events.empty:
            rec.events = events

        return rec
