"""BrainVision importer backed by MNE-Python (.vhdr + .vmrk + .eeg).

Many OpenNeuro BIDS EEG/iEEG datasets ship the BrainVision triplet rather than
EDF or EEGLAB .set. MNE reads it natively (only writing needs pybv), and is an
optional dependency imported lazily with a clear install hint. Channel types and
units come from the shared :mod:`_mne_common` mapping; ``.vmrk`` markers (which
MNE exposes as annotations) are read into ``Recording.events``.
"""

import contextlib
import os
import re
import tempfile
from collections.abc import Iterator

import pandas as pd

from ..core.emg import Recording
from ..exceptions import classify_read_error, is_resource_exhaustion
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter

# ``DataFile=`` / ``MarkerFile=`` lines of the header's [Common Infos] section.
_FILE_REF = re.compile(r"^(\s*)(DataFile|MarkerFile)(\s*=\s*)(.*?)(\s*)$", re.IGNORECASE)
# Same-stem siblings tried for each key, in order (``.dat`` is the legacy data name).
_SIBLING_EXTS = {"datafile": (".eeg", ".dat"), "markerfile": (".vmrk",)}


def _header_encoding(settings: bytes) -> str:
    """The codepage MNE decodes a header with (``Codepage=``, else UTF-8; Latin-1 fallback)."""
    match = re.search(r"Codepage=(.+)", settings.decode("ascii", "ignore"), re.IGNORECASE)
    codepage = match.group(1).strip() if match else "utf-8"
    if codepage.upper() == "ANSI":
        codepage = "cp1252"
    try:
        settings.decode(codepage)
    except (LookupError, UnicodeDecodeError):
        return "latin-1"
    return codepage


@contextlib.contextmanager
def resolved_vhdr(vhdr_path: str) -> Iterator[str]:
    """Yield a ``.vhdr`` path whose ``DataFile=``/``MarkerFile=`` point at files that exist.

    BIDS renames the BrainVision triplet on disk but not the references inside the
    header, so the header can name a pre-rename ``.eeg``/``.vmrk`` that no longer
    exists while the renamed file sits beside it with the header's stem (the same
    defect :meth:`EEGLABImporter._read_fdt` resolves for ``.fdt``). When a referenced
    file is missing and that same-stem sibling exists, a patched copy of the header
    (original encoding and line endings; only those two keys rewritten, to absolute
    paths) is written to a temporary directory and yielded instead. Otherwise the
    original path is yielded unchanged, so a good header and an unrecoverable one
    behave exactly as MNE reads them. The dataset files are never modified.

    Args:
        vhdr_path: Path to the BrainVision ``.vhdr`` header.

    Yields:
        The path to hand to MNE; a patched copy is removed on exit (MNE reads the
        header and markers when the ``Raw`` is built and keeps only the data path).
    """
    directory = os.path.dirname(os.path.abspath(vhdr_path))
    stem = os.path.splitext(os.path.abspath(vhdr_path))[0]
    try:
        with open(vhdr_path, "rb") as f:
            content = f.read()
    except OSError:
        yield vhdr_path
        return
    encoding = _header_encoding(content)
    lines = content.decode(encoding).splitlines(keepends=True)

    stale = False
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
            siblings = (stem + ext for ext in _SIBLING_EXTS[key.lower()])
            sibling = next((p for p in siblings if os.path.isfile(p)), None)
            if sibling is not None:
                target, stale = sibling, True
        # Every reference becomes absolute: the patched copy lives elsewhere.
        eol = line[len(body) :]
        lines[i] = f"{indent}{key}{sep}{os.path.abspath(target)}{trail}{eol}"

    if not stale:
        yield vhdr_path
        return
    try:
        patched = "".join(lines).encode(encoding)
    except UnicodeEncodeError:
        # An absolute path the header's codepage cannot spell: read it as today.
        yield vhdr_path
        return
    with tempfile.TemporaryDirectory(prefix="biosigio-vhdr-") as tmp:
        tmp_vhdr = os.path.join(tmp, os.path.basename(vhdr_path))
        with open(tmp_vhdr, "wb") as f:
            f.write(patched)
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
        try:
            with resolved_vhdr(filepath) as vhdr:
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

        events = self._read_events(raw)
        if not events.empty:
            rec.events = events

        return rec
