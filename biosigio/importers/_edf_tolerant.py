"""Tolerant fallback for EDF/BDF files pyedflib's compliance checker rejects.

biosigIO's normal EDF/BDF path reads via ``pyedflib``, whose C-level ``open_file``
validator is stricter than the format's real-world variation. Four conditions,
all confirmed byte-intact and readable, currently discard entire recordings that
are perfectly usable (issue #109, issue #145 and the NEMAR Zarr backfill). The
first three are recovered through MNE-Python; the fourth is recovered by
pyedflib itself, without MNE (see below):

1. **Degenerate physical range** -- a channel with ``physical_min == physical_max``
   (a reference electrode in a referential montage, identically constant by
   construction). pyedflib: ``"... compliant (Physical Maximum)"`` /
   ``"(Physical Minimum)"``.
2. **Malformed numeric header field** -- a numeric header field padded with NUL
   bytes instead of the spec-mandated ASCII spaces (a real writer bug, not
   corruption; seen on ``number of data records``, but the same failure mode
   can hit any fixed-width numeric field). pyedflib:
   ``"... compliant (<field name>)"`` for a numeric field.
3. **Discontinuous EDF+D** -- the file is correctly marked discontinuous.
   pyedflib: ``"The file is discontinuous and cannot be read"``. It is read here only
   when its data records are verified contiguous (:func:`verify_records_contiguous`);
   one with a gap raises :class:`~biosigio.exceptions.DiscontinuousRecordingError`
   instead, because MNE would put every sample after the gap at the wrong time
   (issue #180).
4. **Late first time-keeping onset** (issue #145): an EDF+C/BDF+C recording cut
   out of a longer session at a data-record boundary, so the first record's
   time-keeping onset is 1 s or more instead of 0. EDFlib refuses it with the
   generic ``"(it contains format errors)"``. Unlike the first three this needs
   no MNE: the refusal comes from the walk over the time-stamped annotation lists
   (TALs), which runs only when annotations are read, so the samples are read by
   pyedflib itself with that walk switched off (:func:`probe_timekeeping_onset`).

:func:`classify_pyedflib_error` recognizes exactly the first three conditions from
pyedflib's exception message and returns ``None`` for anything else (including
the ``"Filesize"`` truncation check and the generic format-error message), so a
genuinely corrupt/truncated file is never misrouted to MNE; it keeps raising
through the caller's ``classify_read_error`` as
:class:`~biosigio.exceptions.CorruptFileError`, unless
:func:`probe_timekeeping_onset` proves the fourth condition is the only defect.

:func:`read_edf_tolerant` does the actual recovery: it re-reads the file with
MNE (which tolerates all three conditions), then rescales MNE's SI-volt output
back to the SAME native physical units ``pyedflib`` would have produced --
MNE and pyedflib apply an *identical* digital-to-physical calibration
(``(physical_max - physical_min) / (digital_max - digital_min)``), MNE just
additionally multiplies by a unit-to-SI factor pyedflib does not apply. That
factor is read back off the just-opened ``Raw`` (:func:`_channel_gains`), which
is exactly what MNE actually multiplied in -- including any of MNE's own edge
cases in matching the dimension string -- rather than a best-effort duplicate
that could silently disagree. It is, however, cross-checked (still in
:func:`_channel_gains`) against an independent recomputation from the same
dimension string, so that a *future* MNE release changing what that private
attribute means -- not merely whether it exists -- is caught as a loud failure
rather than silently trusted. Dividing back out by the (verified) factor makes
a recovered read numerically identical to a normal pyedflib read for any
channel a normal read would have produced -- see
``test_edf_fallback.py::test_fallback_matches_pyedflib_reading``, which is the
load-bearing proof, not an assumption.

For a channel with a degenerate physical range, no calibration is defined (the
digital-to-physical slope is 0/0), so that channel's samples are set to the
constant ``physical_min`` (== ``physical_max``) throughout -- mathematically the
correct simplification of the scaling formula when the physical range collapses
to a point, and exactly zero for the common real-world case of a
grounded/referential channel with ``physical_min == physical_max == 0``.

MNE has its own, unrelated notion of a "stim" channel that would otherwise
break this for one real-world channel shape: by default (``stim_channel=
"auto"``) it auto-detects any channel literally named ``"status"`` or
``"trigger"`` and, for it alone, (a) skips the physical calibration entirely
(``cal=1``, ``offset=0``, ``units=1``) AND (b) additionally reinterprets the
result with a ``bitwise_and(x, 2**17 - 1)`` trigger-code mask (see
``mne.io.edf.edf._read_segment_file``) -- a transform with no relationship at
all to the header's declared physical range. ``pyedflib`` has no notion of
"stim"; it always applies the header's calibration uniformly to every
channel, including one named ``"Status"`` (issue #109's own probe file,
`on004444`, has exactly this: channel 129 is ``"Status"``, declared physical
dimension ``"uV"``). :func:`read_edf_tolerant` therefore opens with
``stim_channel=False``, which turns this special-casing off entirely so every
channel -- ``"Status"`` included -- takes the SAME calibration+unit-scaling
path as any other channel, matching pyedflib exactly rather than needing a
per-channel carve-out.

A truncation safety net (:func:`_check_not_truncated`) still runs before any
fallback read: the on-disk file size is compared against the size the tolerantly
-parsed header implies, and a short file is treated as genuinely corrupt (the
original pyedflib error is re-raised) rather than silently handed to MNE, which
would otherwise just shrink the record count with a warning and return a
truncated recording.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import accumulate
from typing import Any

import numpy as np
import pandas as pd
import pyedflib

from ._labels import unique_channel_labels

# Reasons `classify_pyedflib_error` can return. Stable strings: stored verbatim
# in `Recording` metadata (`edf_tolerant_read_reason`) as part of the recovered-
# read provenance, so treat them as a small public vocabulary, not free text.
DEGENERATE_PHYSICAL_RANGE = "degenerate_physical_range"
MALFORMED_NUMERIC_FIELD = "malformed_numeric_field"
DISCONTINUOUS_DATARECORDS = "discontinuous_datarecords"
# Not returned by `classify_pyedflib_error` (the refusal message is generic):
# `probe_timekeeping_onset` proves it from the file, and the readers record it.
LATE_TIMEKEEPING_ONSET = "late_timekeeping_onset"

# Metadata key set on a Recording (and a streamed store) read from an EDF+D/BDF+D
# file whose data records were checked and found contiguous (see
# `verify_records_contiguous`).
DISCONTINUOUS_VERIFIED_CONTIGUOUS_KEY = "edf_discontinuous_verified_contiguous"
# The least a record's time-keeping onset may differ from the previous record's end
# before the pair counts as a gap (see `_pair_tolerance` for what it is raised to).
RECORD_ONSET_TOLERANCE_S = Decimal("1e-6")

# Labels of the EDF+/BDF+ annotations (TAL) channel, which MNE (like pyedflib)
# does not list as a signal; matched exactly, as MNE's ``_find_tal_idx`` does.
_ANNOTATION_LABELS = ("EDF Annotations", "BDF Annotations")

# pyedflib "... compliant (<field>)" field names that are purely numeric (ASCII
# digits, sign, decimal point/comma) and therefore fail in the same way a NUL-
# padded "Number of Datarecords" does (#issue on006233): the C validator's
# ASCII-to-number parse chokes on a trailing NUL where the spec requires a
# trailing space. MNE's own header parser (`_edf_str`/`_edf_str_num`) truncates
# at the first NUL either way, so it parses all of these identically whether
# they are NUL- or space-padded. "Physical Maximum"/"Physical Minimum" are
# deliberately excluded here -- those are handled by the degenerate-range path,
# which needs a different remedy (constant output) than "just parse it".
# "Filesize" is deliberately excluded -- that is a genuine truncation check and
# must keep surfacing as CorruptFileError.
_NUMERIC_FIELD_NAMES = (
    "number of datarecords",
    "duration",
    "digital maximum",
    "digital minimum",
    "sample in datarecord",
    "number of signals",
)


def classify_pyedflib_error(exc: Exception) -> str | None:
    """Classify a ``pyedflib.EdfReader`` open failure into a recoverable reason.

    Returns one of :data:`DEGENERATE_PHYSICAL_RANGE`, :data:`MALFORMED_NUMERIC_FIELD`,
    :data:`DISCONTINUOUS_DATARECORDS`, or ``None`` when the failure is not one of
    the three known-recoverable conditions (e.g. a genuine ``Filesize`` truncation,
    or a free-text/label field pyedflib also happens to validate).
    """
    low = str(exc).lower()
    if "discontinuous" in low:
        return DISCONTINUOUS_DATARECORDS
    if "physical maximum" in low or "physical minimum" in low:
        return DEGENERATE_PHYSICAL_RANGE
    if any(f"compliant ({name})" in low for name in _NUMERIC_FIELD_NAMES):
        return MALFORMED_NUMERIC_FIELD
    return None


def raise_host_open_error(filepath: str, reader_exc: Exception) -> None:
    """Raise the OS's own error when the host, not the file, stopped pyedflib opening it.

    pyedflib reports every ``open()`` the operating system refuses on an existing
    file (``EACCES``, ``EIO``, a stale network handle, ...) as ``OSError(123,
    "... was found but can't be accessed ...")``, dropping the real ``errno``, so
    that failure could not be told apart from a problem with the file. Re-opening
    the file here recovers the real ``OSError``; when it is a host condition (see
    :func:`~biosigio.exceptions.is_host_condition`) it is raised, chained from
    ``reader_exc``, and a caller then propagates it unchanged instead of typing
    it. Returns normally when the file opens, or fails for another reason.
    """
    from ..exceptions import is_host_condition

    try:
        with open(filepath, "rb") as fh:
            fh.read(1)
    except OSError as os_err:
        if is_host_condition(os_err):
            raise os_err from reader_exc


# --- Minimal, tolerant EDF/BDF header probe -----------------------------------
# pyedflib refuses to open these files at all, so its own (correct, but strict)
# header parser is unavailable to us. This probe reads the fixed-width ASCII
# header directly, exactly per the EDF/BDF spec layout, tolerant of NUL-padded
# fields the same way MNE is. It is intentionally independent of both pyedflib
# and MNE's internals: it only needs to answer "what does this header say",
# not "did MNE accept it".

_MAIN_HEADER_BYTES = 256
# (field name, width in bytes), in on-disk order, repeated once per signal.
_SIGNAL_FIELDS = (
    ("label", 16),
    ("transducer", 80),
    ("dimension", 8),
    ("physical_min", 8),
    ("physical_max", 8),
    ("digital_min", 8),
    ("digital_max", 8),
    ("prefilter", 80),
    ("samples_per_record", 8),
    ("reserved", 32),
)


def _decode_field(raw: bytes) -> str:
    """Tolerantly decode a fixed-width EDF/BDF ASCII header field.

    Real-world files pad short values with NUL bytes instead of the spec's
    ASCII spaces (issue #109's ``on006233`` example: the "number of data
    records" field contains ``b'421\\x00'``). Truncate at the first NUL
    (matching MNE's ``_edf_str``) before stripping surrounding whitespace.
    """
    return raw.decode("latin-1").split("\x00", 1)[0].strip()


def _decode_number(raw: bytes) -> float:
    """Tolerantly decode a fixed-width EDF/BDF ASCII numeric field.

    EDF permits ``,`` as the decimal separator; normalize to ``.`` first
    (matching MNE's ``_edf_str_num``).
    """
    return float(_decode_field(raw).replace(",", "."))


@dataclass(frozen=True)
class EdfHeaderProbe:
    """Tolerantly-parsed EDF/BDF header fields, independent of pyedflib/MNE."""

    reserved: str
    number_of_datarecords: int
    duration_of_data_record: float
    number_of_signals: int
    header_bytes: int
    channels: list[dict[str, Any]] = field(default_factory=list)


def probe_edf_header(filepath: str) -> EdfHeaderProbe:
    """Parse the fixed-width EDF/BDF header directly, tolerant of NUL padding.

    Includes every on-disk signal (the hidden EDF+/BDF+ annotations channel
    too, if present) -- callers match against a reader's channel list by label.
    """
    with open(filepath, "rb") as f:
        main = f.read(_MAIN_HEADER_BYTES)
        if len(main) < _MAIN_HEADER_BYTES:
            raise OSError(f"{filepath}: EDF/BDF header is truncated (<256 bytes)")
        reserved = _decode_field(main[192:236])
        # "-1" is the spec's own placeholder for "unknown, still recording".
        n_records = int(_decode_number(main[236:244]))
        duration = _decode_number(main[244:252])
        ns = int(_decode_field(main[252:256]))
        if ns < 0:
            raise OSError(f"{filepath}: invalid number of signals in header ({ns})")

        blocks: dict[str, list[bytes]] = {}
        for name, width in _SIGNAL_FIELDS:
            chunk = f.read(width * ns)
            if len(chunk) < width * ns:
                raise OSError(f"{filepath}: signal header is truncated")
            blocks[name] = [chunk[i * width : (i + 1) * width] for i in range(ns)]

    channels: list[dict[str, Any]] = []
    for i in range(ns):
        channels.append(
            {
                "label": _decode_field(blocks["label"][i]),
                "transducer": _decode_field(blocks["transducer"][i]),
                "dimension": _decode_field(blocks["dimension"][i]),
                "physical_min": _decode_number(blocks["physical_min"][i]),
                "physical_max": _decode_number(blocks["physical_max"][i]),
                "digital_min": int(round(_decode_number(blocks["digital_min"][i]))),
                "digital_max": int(round(_decode_number(blocks["digital_max"][i]))),
                "prefilter": _decode_field(blocks["prefilter"][i]),
                "samples_per_record": int(round(_decode_number(blocks["samples_per_record"][i]))),
            }
        )
    header_bytes = _MAIN_HEADER_BYTES * (1 + ns)
    return EdfHeaderProbe(
        reserved=reserved,
        number_of_datarecords=n_records,
        duration_of_data_record=duration,
        number_of_signals=ns,
        header_bytes=header_bytes,
        channels=channels,
    )


def _bytes_per_sample(filepath: str) -> int:
    """3 for a BDF/BDF+ file, 2 for EDF/EDF+, decided by the header, not the file name.

    A BDF file's first header byte is 0xFF (then ``BIOSEMI``), and a BDF+ file's reserved
    field starts with ``BDF+``; either is enough, so a BDF file named ``.edf`` (or the
    reverse) is still laid out correctly.
    """
    with open(filepath, "rb") as f:
        head = f.read(_MAIN_HEADER_BYTES)
    return 3 if head[:1] == b"\xff" or head[_RESERVED_AT].upper().startswith(b"BDF+") else 2


def _check_not_truncated(filepath: str, probe: EdfHeaderProbe, n_records: int) -> None:
    """Raise ``OSError`` if the file is shorter than the header implies.

    Guards against a file that is genuinely truncated *and* happens to also
    trip one of the three recoverable conditions -- pyedflib's own error
    message in that case may still name a recoverable condition first (its
    checks are not guaranteed to run in a fixed priority order), so this
    cannot be skipped just because ``classify_pyedflib_error`` matched.
    ``n_records == -1`` ("unknown, still recording") cannot be checked this
    way and is skipped; MNE's own reconciliation of records-vs-filesize is the
    only guard in that rare case, same as it would be for a compliant file.
    """
    if n_records < 0:
        return
    record_bytes = sum(ch["samples_per_record"] for ch in probe.channels) * _bytes_per_sample(
        filepath
    )
    expected = probe.header_bytes + n_records * record_bytes
    actual = os.path.getsize(filepath)
    if actual < expected:
        raise OSError(
            f"{filepath}: file is truncated (expected at least {expected} bytes "
            f"for {n_records} data record(s), found {actual})"
        )


# --- Discontinuous EDF+D/BDF+D (issue #180) -------------------------------------
# An EDF+D/BDF+D file gives every data record its own start time in its time-keeping
# TAL, and the records need not be contiguous. MNE (the fallback reader below) reads
# the reserved field without acting on it and concatenates the records, so every
# sample after a gap lands earlier than it was recorded while the annotations keep
# their true times. The TALs are therefore read here, directly, and a file with a gap
# is refused before MNE is asked for anything.


def _annotation_span(probe: EdfHeaderProbe, bytes_per_sample: int) -> tuple[int, int, int]:
    """``(offset in a record, width in bytes, record size in bytes)`` of the first annotations signal.

    EDF+ puts the time-keeping TAL in the FIRST annotations signal of each record.
    Raises ``FileReadError`` when the header lists none.
    """
    from ..exceptions import FileReadError

    offset = 0
    span: tuple[int, int] | None = None
    for ch in probe.channels:
        width = ch["samples_per_record"] * bytes_per_sample
        if span is None and ch["label"] in _ANNOTATION_LABELS:
            span = (offset, width)
        offset += width
    if span is None:
        raise FileReadError(
            "this file is labeled EDF+D/BDF+D but has no annotations signal, so the start "
            "time of its data records cannot be read"
        )
    return span[0], span[1], offset


# A time-keeping onset as written by real acquisition software, read more leniently
# than EDFlib reads it: a run of signs is accepted and the LAST one wins, so the
# ``+-0.001`` that 342 NEMAR files (on006233, on006914) carry in record 0 is -0.001 s.
# Used only to check continuity; the late-onset code above keeps EDFlib's own rule.
_LENIENT_ONSET = re.compile(rb"([+-]+)([0-9]+)(?:\.([0-9]+))?")


def _lenient_timekeeping_onset(block: bytes) -> tuple[int, int] | None:
    """The time-keeping onset opening an annotation block, as ``(units, digits)``, or ``None``.

    ``units`` is the onset as an integer count of ``10**-digits`` seconds. The onset
    ends at the first 0x14, a second 0x14 must follow at once (a time-keeping TAL has
    no duration and no text), and the text in front must be signs, digits and at most
    one decimal point.
    """
    end = block.find(b"\x14")
    if end < 0 or block[end + 1 : end + 2] != b"\x14":
        return None
    match = _LENIENT_ONSET.fullmatch(block, 0, end)
    if match is None:
        return None
    signs, whole, fraction = match.groups()
    try:
        units = int(whole + (fraction or b""))
    except ValueError:  # more digits than Python converts to an int: not an onset
        return None
    return (-units if signs[-1:] == b"-" else units), len(fraction or b"")


def _pair_tolerance(
    digits_a: int, digits_b: int, half_sample: Decimal | None, duration: Decimal
) -> Decimal:
    """How far two consecutive record onsets may be from exactly one record apart.

    Each onset is decimal text rounded to its digits, so the difference of two is off
    by up to ``10**-min(digits)``: three-decimal onsets of non-round record durations
    are not exact. The floor is :data:`RECORD_ONSET_TOLERANCE_S`; the ceiling is half a
    sample period of the fastest signal (``half_sample``), because a gap shorter than
    that cannot move a sample to another sample. A file with no data signal has no
    sample period, and the ceiling is then half a record (``duration``).
    """
    tolerance = Decimal(10) ** -min(digits_a, digits_b)
    tolerance = min(tolerance, half_sample if half_sample is not None else duration / 2)
    return max(tolerance, RECORD_ONSET_TOLERANCE_S)


def verify_records_contiguous(filepath: str, probe: EdfHeaderProbe | None = None) -> bool:
    """Whether ``filepath`` is an EDF+D/BDF+D file whose records were verified contiguous.

    Returns ``False`` without reading any record when the reserved field does not
    say ``EDF+D`` or ``BDF+D`` (case-insensitive, as :func:`read_edf_tolerant` treats
    the ``EDF+`` prefix). For a +D file it reads the time-keeping onset of every data
    record from the first annotations signal itself and returns ``True`` when each
    record starts where the one before it ended, within :func:`_pair_tolerance`: at
    least :data:`RECORD_ONSET_TOLERANCE_S`, more when the onsets are written with few
    decimals (their rounding), never more than half a sample period. A file labeled
    discontinuous whose records are in fact contiguous is readable like any
    continuous one.

    Onsets are read leniently (:func:`_lenient_timekeeping_onset`): ``+-0.001`` is
    -0.001 s, as 342 real NEMAR files write their first record. Whether the bytes
    that follow are annotations is the reader's concern, not this check's.

    Raises :class:`~biosigio.exceptions.DiscontinuousRecordingError` naming every gap
    (start and length, seconds on the file's own time axis) when any record starts
    elsewhere, an overlap or a record out of order included, and
    :class:`~biosigio.exceptions.FileReadError` when a record's time-keeping onset
    cannot be read or the records hold no samples, since continuity cannot be shown
    then. A header count of ``-1`` ("still recording") is replaced by what the file
    size holds. The width of a sample (2 or 3 bytes) comes from the header, not the
    file name.
    """
    from ..exceptions import DiscontinuousRecordingError, FileReadError, TimeGap

    if probe is None:
        probe = probe_edf_header(filepath)
    if not probe.reserved.upper().startswith(("EDF+D", "BDF+D")):
        return False

    offset, width, record_bytes = _annotation_span(probe, _bytes_per_sample(filepath))
    if record_bytes <= 0:
        raise FileReadError(
            f"{filepath}: this EDF+D/BDF+D file's data records hold no samples, so "
            "it cannot be shown to be contiguous"
        )
    n_records = probe.number_of_datarecords
    if n_records < 0:
        n_records = max(0, (os.path.getsize(filepath) - probe.header_bytes) // record_bytes)
    duration = Decimal(repr(probe.duration_of_data_record))
    fastest = max(
        (
            ch["samples_per_record"]
            for ch in probe.channels
            if ch["label"] not in _ANNOTATION_LABELS
        ),
        default=0,
    )
    half_sample = duration / (2 * fastest) if fastest > 0 and duration > 0 else None

    onsets: list[Decimal] = []
    digits: list[int] = []
    with open(filepath, "rb") as f:
        for index in range(n_records):
            f.seek(probe.header_bytes + index * record_bytes + offset)
            keeping = _lenient_timekeeping_onset(f.read(width))
            if keeping is None:
                raise FileReadError(
                    f"{filepath}: data record {index} has no readable time-keeping "
                    "annotation, so this EDF+D/BDF+D file cannot be shown to be contiguous"
                )
            units, places = keeping
            onsets.append(Decimal(units).scaleb(-places))
            digits.append(places)

    gaps = []
    for i in range(len(onsets) - 1):
        length = onsets[i + 1] - onsets[i] - duration
        if abs(length) > _pair_tolerance(digits[i], digits[i + 1], half_sample, duration):
            gaps.append(TimeGap(float(onsets[i] + duration), float(length)))
    if gaps:
        raise DiscontinuousRecordingError(filepath, gaps, float(duration))
    return True


# --- Late first time-keeping onset (issue #145) -------------------------------
# EDF+ and BDF+ keep a time-stamped annotation list (TAL) in every data record,
# and its first TAL is the record's "time-keeping" onset, ``+<seconds>\x14\x14\x00``.
# EDFlib (the C library inside pyedflib) refuses a file whose FIRST time-keeping
# onset is 1 s or more, which is what a recording cut out of a longer session at a
# record boundary looks like. The refusal comes from the TAL walk, which runs only
# when annotations are read, and EDFlib words it like any other bad TAL. So a file
# is recovered only after checking that this is its sole defect: its TALs, brought
# back by whole seconds into a small TAL-only copy, must pass EDFlib's own walk.

_GENERIC_FORMAT_ERROR = "(it contains format errors)"
_RESERVED_AT = slice(192, 236)
# A TAL onset as EDFlib reads it (`edflib_is_onset_number`): a sign, digits and at
# most one interior decimal point, ended by the 0x14 or 0x15 that follows it.
_TAL_ONSET = re.compile(rb"([+-])([0-9]+)(?:\.([0-9]+))?(?=[\x14\x15])")
_NON_NUL_RUN = re.compile(rb"[^\x00]+")


@dataclass(frozen=True, eq=False)
class TimekeepingOnset:
    """A file recoverable by reading it without annotations (see :func:`probe_timekeeping_onset`).

    ``first_onset`` is the first record's time-keeping onset in seconds: how far
    into the session that record lies. ``startdate`` is the header's start date and
    time plus that onset, the time of the first record; it is worked out here, once,
    so that a date ``datetime`` cannot hold is a reason to decline the file.

    The three arrays are what ``pyedflib.EdfReader.readAnnotations`` reports once
    the onsets are brought back, so they are relative to the first record, as for a
    compliant file: onset (seconds), duration (seconds, ``-1`` when absent) and
    description.
    """

    first_onset: float
    startdate: datetime
    onsets: np.ndarray
    durations: np.ndarray
    descriptions: np.ndarray

    @property
    def annotations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(onsets, durations, descriptions)``, in ``readAnnotations`` order."""
        return self.onsets, self.durations, self.descriptions


def _read_raw_header(f) -> tuple[bytes, dict[str, list[bytes]], int]:
    """The 256-byte main header, every per-signal field as raw bytes, and the signal count.

    Only called on a file pyedflib has just opened with annotations off, so the
    header is whole; a field that still does not parse raises ``ValueError``, which
    is the file's, while an ``OSError`` from the read itself is the host's and
    propagates.
    """
    main = f.read(_MAIN_HEADER_BYTES)
    n_signals = int(_decode_field(main[252:256]))
    blocks: dict[str, list[bytes]] = {}
    for name, width in _SIGNAL_FIELDS:
        chunk = f.read(width * n_signals)
        blocks[name] = [chunk[i * width : (i + 1) * width] for i in range(n_signals)]
    return main, blocks, n_signals


def _onset_units(match: re.Match[bytes]) -> tuple[int, int]:
    """A matched TAL onset as ``(signed integer, decimal digits)``, exact and float-free."""
    sign, whole, fraction = match.groups()
    units = int(whole + (fraction or b""))
    return (-units if sign == b"-" else units), len(fraction or b"")


def _format_onset(units: int, digits: int) -> bytes:
    whole, fraction = divmod(abs(units), 10**digits)
    text = f"{'-' if units < 0 else '+'}{whole}"
    if digits:
        text += f".{fraction:0{digits}d}"
    return text.encode("ascii")


def _first_timekeeping_onset(block: bytes) -> tuple[int, int] | None:
    """The time-keeping onset opening an annotation block, as ``(units, digits)``.

    Follows EDFlib's search: the first 0x14 before the block's last two bytes ends
    the onset, a second 0x14 must follow at once, and the text in front must be an
    onset number (a duration field there is refused). ``None`` when it is not.
    """
    end = block.find(b"\x14", 0, max(len(block) - 2, 0))
    if end < 0 or block[end + 1 : end + 2] != b"\x14":
        return None
    onset = _TAL_ONSET.match(block, 0, end + 1)
    if onset is None or onset.end() != end:
        return None
    return _onset_units(onset)


def _shift_tal_block(block: bytes, seconds: int) -> bytes | None:
    """One annotation block with every TAL onset brought back by ``seconds``.

    Only the onsets change. Every other byte, a malformed one included, is kept so
    that EDFlib's walk over the result sees the original's defects; the block is
    zero-padded back to its width through the run of NULs that ended it. ``None``
    when that cannot be done faithfully: the block's last byte is not a NUL (EDFlib
    refuses that itself, and padding would hide it), a TAL does not open with an
    onset, or the shifted TALs no longer leave room for the final NUL.
    """
    if not block or block[-1] != 0:
        return None
    out = bytearray()
    position = 0
    for run in _NON_NUL_RUN.finditer(block):
        tal = run.group()
        onset = _TAL_ONSET.match(tal)
        if onset is None:
            return None
        units, digits = _onset_units(onset)
        out += block[position : run.start()]
        out += _format_onset(units - seconds * 10**digits, digits)
        out += tal[onset.end() :]
        position = run.end()
    if len(out) >= len(block):
        return None
    return bytes(out) + b"\x00" * (len(block) - len(out))


@dataclass(frozen=True)
class _CopyPlan:
    """What :func:`_write_tal_only_copy` needs, worked out from the source before any scratch exists."""

    # The copy's main header and the annotation signals' own headers, as they go to disk.
    header: bytes
    n_records: int
    # Per annotation signal: (offset inside a data record, width in bytes).
    blocks: tuple[tuple[int, int], ...]
    record_bytes: int
    header_bytes: int
    # ``floor(first onset)``: the whole seconds every onset is brought back by.
    shift: int
    first_onset: float


def _plan_tal_copy(src: str, bdf: bool) -> _CopyPlan | None:
    """Read ``src``'s header and record 0's first annotation block; plan the TAL-only copy.

    ``None`` when this is not the defect: no annotation signal, no record, a short
    first block, or a first time-keeping onset under 1 s (which EDFlib accepts, so
    a refusal then has another cause). Nothing is written, so a caller creates its
    scratch space only once this has answered.

    Raises:
        ValueError: The header is not one, or an onset has more digits than
            ``int`` will parse.
        ArithmeticError: The first onset is too large for a float.
    """
    with open(src, "rb", buffering=0) as f:
        main, blocks, n_signals = _read_raw_header(f)
        n_records = int(_decode_field(main[236:244]))
        sample_size = 3 if bdf else 2
        wanted = b"BDF Annotations " if bdf else b"EDF Annotations "
        annotation = [i for i, label in enumerate(blocks["label"]) if label == wanted]
        if n_records < 1 or not annotation:
            return None
        samples = [int(_decode_field(raw)) for raw in blocks["samples_per_record"]]
        starts = list(accumulate(samples, initial=0))
        header_bytes = _MAIN_HEADER_BYTES * (1 + n_signals)

        first_at = header_bytes + starts[annotation[0]] * sample_size
        first_width = samples[annotation[0]] * sample_size
        f.seek(first_at)
        block = f.read(first_width)
        # A POSIX regular file is read short only at end of file, so a short read
        # means the file is truncated, and declining it is correct.
        if len(block) != first_width:
            return None

    first = _first_timekeeping_onset(block)
    if first is None or first[0] < 10 ** first[1]:
        return None

    patched = bytearray(main)
    patched[184:192] = str(_MAIN_HEADER_BYTES * (1 + len(annotation))).ljust(8).encode("ascii")
    patched[252:256] = str(len(annotation)).ljust(4).encode("ascii")
    header = bytes(patched) + b"".join(
        b"".join(blocks[name][i] for i in annotation) for name, _ in _SIGNAL_FIELDS
    )
    return _CopyPlan(
        header=header,
        n_records=n_records,
        blocks=tuple((starts[i] * sample_size, samples[i] * sample_size) for i in annotation),
        record_bytes=starts[-1] * sample_size,
        header_bytes=header_bytes,
        shift=first[0] // 10 ** first[1],
        first_onset=first[0] / 10 ** first[1],
    )


def _write_tal_only_copy(src: str, dst: str, plan: _CopyPlan) -> bool:
    """Write ``src``'s annotation signals, with onsets brought back, to ``dst``.

    The copy keeps the main header (header size and signal count patched), the
    annotation signals' own headers and, record by record, their bytes, a small
    fraction of the source. The onsets move by ``plan.shift`` seconds, so the first
    record starts under 1 s and EDFlib checks every other TAL and the
    record-to-record continuity as it would for a compliant file. ``False`` when a
    block is short or cannot be shifted faithfully.

    An ``OSError`` from creating or writing ``dst`` is the host's and propagates.
    """
    with open(src, "rb", buffering=0) as f, open(dst, "wb") as out:
        out.write(plan.header)
        for record in range(plan.n_records):
            for offset, width in plan.blocks:
                f.seek(plan.header_bytes + record * plan.record_bytes + offset)
                block = f.read(width)
                # As in _plan_tal_copy: a regular file is read short only at end of
                # file, so a short read means truncation, and declining is correct.
                if len(block) != width:
                    return False
                shifted = _shift_tal_block(block, plan.shift)
                if shifted is None:
                    return False
                out.write(shifted)
    return True


def _timekeeping_onset(filepath: str) -> TimekeepingOnset | None:
    with open(filepath, "rb") as f:
        reserved = f.read(_MAIN_HEADER_BYTES)[_RESERVED_AT]
    # Case-sensitive, like EDFlib: anything else is not an EDF+/BDF+ continuous file.
    if not reserved.startswith((b"EDF+C", b"BDF+C")):
        return None
    bdf = reserved.startswith(b"BDF")

    # Without annotations pyedflib still checks the header and the file size, so a
    # file that opens this way was refused by the annotation walk alone.
    try:
        header = pyedflib.EdfReader(filepath, annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS)
    except OSError as open_err:
        raise_host_open_error(filepath, open_err)
        return None
    try:
        header_start = header.getStartdatetime()
    finally:
        header.close()

    # The cheap exit comes before any scratch exists: most refusals end here.
    plan = _plan_tal_copy(filepath, bdf)
    if plan is None:
        return None
    # The recording starts at its first record. An onset that carries the date out
    # of datetime's range cannot be represented, so the file is declined.
    startdate = header_start + timedelta(seconds=plan.first_onset)

    # No handler around the scratch work: whatever the OS says about creating or
    # writing the copy is about this host, not about the file, and propagates as is.
    # Removing the scratch directory is different: a leftover copy holds only the
    # annotation lists, so it stays small and bounded. A cleanup failure is deliberately
    # not an error and must not turn a recovered file into a refused one.
    with tempfile.TemporaryDirectory(prefix="biosigio-tal-", ignore_cleanup_errors=True) as scratch:
        copy = os.path.join(scratch, "tal-only.bdf" if bdf else "tal-only.edf")
        if not _write_tal_only_copy(filepath, copy, plan):
            return None
        try:
            reader = pyedflib.EdfReader(copy)
        except OSError:
            # EDFlib's refusal of the copy is the answer (the original has another
            # defect); the OS refusing a copy just written is not, and raises here.
            with open(copy, "rb") as readable:
                readable.read(1)
            return None
        try:
            onsets, durations, descriptions = reader.readAnnotations()
        finally:
            reader.close()
    return TimekeepingOnset(plan.first_onset, startdate, onsets, durations, descriptions)


def probe_timekeeping_onset(filepath: str, exc: Exception) -> TimekeepingOnset | None:
    """Whether a late first time-keeping onset is the ONLY reason pyedflib refused ``filepath``.

    Called with the exception ``pyedflib.EdfReader(filepath)`` raised, after
    :func:`classify_pyedflib_error` found no MNE-recoverable cause. Returns ``None``
    unless every step holds, and the caller then raises as it always did:

    1. the message is EDFlib's generic ``"(it contains format errors)"``;
    2. the header's reserved field starts with ``EDF+C`` or ``BDF+C`` (case-sensitive);
    3. the file opens with annotations switched off, which still checks the header
       and the file size, so the refusal came from the annotation walk;
    4. record 0's first TAL is a time-keeping onset of 1 s or more, and the header's
       start plus that onset is a date ``datetime`` can represent;
    5. a TAL-only copy, its onsets brought back by whole seconds, opens in
       pyedflib's default mode, so EDFlib has checked every TAL and the continuity.

    The caller then opens ``filepath`` with
    ``annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS`` and reads the samples
    with pyedflib exactly as for any compliant file; the events are
    :attr:`TimekeepingOnset.annotations`. No MNE is involved.

    Only the file's own bytes are answered with ``None``: a header that is not one,
    an onset ``int`` cannot parse or a float cannot hold, a first record whose date
    ``datetime`` cannot represent, EDFlib refusing the copy. Every ``OSError`` raised
    while reading the file or creating, writing or reading back the scratch copy
    propagates unchanged, whatever its ``errno`` (a missing or read-only temporary
    directory is ``ENOENT`` or ``EACCES``): it says something about this host, and
    a caller must not record it as a permanent failure of a valid file.
    """
    if not str(exc).rstrip().endswith(_GENERIC_FORMAT_ERROR):
        return None
    try:
        return _timekeeping_onset(filepath)
    except (ValueError, ArithmeticError):
        return None


@dataclass
class EdfFallbackChannel:
    """One channel read via the tolerant fallback, in pyedflib-equivalent units."""

    label: str
    data: np.ndarray
    sample_frequency: float
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefilter: str
    transducer: str
    degenerate_physical_range: bool


@dataclass
class EdfFallbackRecording:
    """Everything :meth:`EDFImporter.load` needs from a tolerant fallback read."""

    channels: list[EdfFallbackChannel]
    events: pd.DataFrame
    filetype: int
    file_duration: float
    datarecord_duration: float
    # ``{new_label: header_label}`` for every repeated header label that was
    # suffixed (see :func:`~biosigio.importers._labels.unique_channel_labels`).
    renamed_labels: dict[str, str] = field(default_factory=dict)
    # True when the file is labeled EDF+D/BDF+D and its data records were checked
    # and found contiguous (see :func:`verify_records_contiguous`).
    verified_contiguous: bool = False


def _expected_gain(dimension: str) -> float:
    """The native-unit -> SI(volts) multiplier MNE's EDF/BDF reader SHOULD apply.

    Copied verbatim from ``mne.io.edf.edf._get_info``'s own lookup table:
    ``"uV"`` plus three micro-sign codepoint variants -> 1e-6, ``"mV"`` -> 1e-3,
    anything else (including empty/unrecognized) -> 1 (already-volts). This is
    NOT the value :func:`_channel_gains` returns -- it is an independent
    cross-check against what MNE's ``Raw`` actually used, so that if a future
    MNE release changes what its internal gain array *means* (not merely
    whether it exists), the disagreement is caught as a loud, immediate error
    rather than a silent, wrong-but-plausible rescale. See :func:`_channel_gains`.
    """
    if dimension in ("μV", "µV", "\x83\xcaV", "uV"):
        return 1e-6
    if dimension == "mV":
        return 1e-3
    return 1.0


def _channel_gains(raw, dimensions: list[str]) -> np.ndarray:
    """The per-channel native-unit -> SI(volts) multiplier MNE applied.

    Read directly off MNE's internal ``_raw_extras[0]["units"]`` -- the exact
    array MNE's own segment reader multiplies into the digital-to-physical
    calibration -- rather than used blindly. Two things can go wrong with a
    private attribute like this, and each is guarded separately:

    - **Shape**: if a future MNE's ``"units"`` array does not align 1:1 with
      ``raw.ch_names`` (``dimensions``, in the same order), indexing it later
      would raise a bare, uninformative ``IndexError`` instead of a clear one.
    - **Meaning**: even with the right shape, a future MNE could change what
      the array's values represent without changing its shape or key name at
      all -- the single worst outcome available here, since it would produce
      wrong-but-plausible data with no error and no warning. Guarding against
      it means not trusting the private attribute on faith: ``dimensions`` is
      independently rederived (:func:`_expected_gain`) from each channel's own
      physical-dimension string (already parsed independently of MNE, by
      :func:`probe_edf_header`), and any disagreement is a loud failure. This
      is deliberately a value check rather than a version pin -- the ``meg``
      extra floors at ``mne>=1.6`` with no ceiling, so nothing else would
      catch a future MNE changing this. (No per-channel exemption is needed
      here: ``read_edf_tolerant`` opens with ``stim_channel=False``, so every
      channel -- including one MNE would otherwise auto-detect as ``stim`` --
      is calibrated the normal way and this check applies uniformly.)
    """
    from ..exceptions import FileReadError

    try:
        gains = np.asarray(raw._raw_extras[0]["units"], dtype=float)
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise FileReadError(
            "EDF/BDF fallback reader could not read MNE's per-channel unit "
            f"gains ({type(exc).__name__}: {exc}); this MNE version may have "
            "changed its internal EDF/BDF reader layout"
        ) from exc

    if len(gains) != len(dimensions):
        raise FileReadError(
            "EDF/BDF fallback reader: MNE reported "
            f"{len(gains)} per-channel unit gain(s) for {len(dimensions)} "
            "channel(s); this MNE version may have changed its internal "
            "EDF/BDF reader layout"
        )

    expected = np.array([_expected_gain(d) for d in dimensions], dtype=float)
    mismatched = np.flatnonzero(gains != expected)
    if mismatched.size:
        raise FileReadError(
            "EDF/BDF fallback reader: MNE's per-channel unit gain disagrees "
            "with the value biosigIO independently computed from the "
            f"physical-dimension string for channel index(es) {mismatched.tolist()} "
            f"(MNE: {gains[mismatched].tolist()}, expected: {expected[mismatched].tolist()}); "
            "this MNE version may have changed how it interprets EDF/BDF "
            "physical-dimension units, which would otherwise make a recovered "
            "read silently disagree with a normal pyedflib read"
        )

    return gains


def _events_from_mne_annotations(raw) -> pd.DataFrame:
    """Same shape/sort as ``EDFImporter._events_from_annotations`` (pyedflib path)."""
    rows = []
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description, strict=False
    ):
        text = str(description)
        if text == "":
            continue
        rows.append((float(onset), float(duration), text))
    events = pd.DataFrame(rows, columns=["onset", "duration", "description"])
    if not events.empty:
        events = events.sort_values(by="onset").reset_index(drop=True)
        events["onset"] = events["onset"].astype("float64")
        events["duration"] = events["duration"].astype("float64")
    return events


def _is_mne_name_for(name: str, label: str, repeated: bool) -> bool:
    """Whether ``name`` is what MNE calls a header channel labelled ``label``.

    The label itself, or, only when ``label`` is ``repeated`` in the header, MNE's
    exact rename of a repeated label, ``<label>-<n>`` or ``<label>-<a..z>``
    (``_unique_channel_names``). MNE renames only labels that repeat, so for a
    unique label any ``<label>-...`` name is a DIFFERENT channel that happens to
    share the prefix (``A`` and ``A-b``), and a prefix test would accept a
    reordered pair.
    """
    if name == label:
        return True
    return repeated and re.fullmatch(re.escape(label) + r"-(\d+|[a-z])", name) is not None


def _check_channel_pairing(filepath: str, mne_names: list[str], header_labels: list[str]) -> None:
    """Fail loud unless the n-th MNE channel is the n-th header channel.

    Raises:
        FileReadError: At the first position where the names do not pair (see
            :func:`_is_mne_name_for`); attaching one row's scaling to another
            row's data would be silently wrong.
    """
    from ..exceptions import FileReadError

    counts = Counter(header_labels)
    for name, label in zip(mne_names, header_labels, strict=True):
        if not _is_mne_name_for(name, label, repeated=counts[label] > 1):
            raise FileReadError(
                f"{filepath}: fallback reader could not match channel {name!r} "
                f"read by MNE back to header channel {label!r} at the same position"
            )


def read_edf_tolerant(filepath: str, reason: str) -> EdfFallbackRecording:
    """Recover an EDF/BDF file pyedflib refuses to open, in pyedflib-equivalent units.

    ``reason`` is informational (one of the module-level constants); every
    condition is handled the same way here (read via MNE, rescale, zero out any
    degenerate channel), since a real file can combine more than one of them.

    Opens with ``stim_channel=False`` -- see the module docstring -- so a
    channel MNE would otherwise auto-detect as a trigger/status channel (e.g.
    one literally named ``"Status"``) is calibrated the same way as any other
    channel instead of being given MNE's unrelated, non-calibrated trigger-code
    treatment, which would otherwise silently disagree with a normal pyedflib
    read of that one channel (issue #109's own probe file has exactly this
    shape: a degenerate reference channel AND a "Status" channel in the same
    recording).

    An EDF+D/BDF+D file is checked first (:func:`verify_records_contiguous`): one
    with a gap raises :class:`~biosigio.exceptions.DiscontinuousRecordingError`, one
    whose records are contiguous is read and flagged ``verified_contiguous``.

    Raises ``ImportError`` if MNE (the ``meg`` extra) is not installed, and
    ``OSError``/``FileReadError`` for anything that turns out not to be
    recoverable after all: a truncation the safety net catches, a per-channel
    unit gain that disagrees with what biosigIO independently expects (see
    :func:`_channel_gains`), or a channel MNE read that cannot be paired back
    up with its header row.

    Channels are paired with header rows by POSITION: MNE lists the header's
    signals in on-disk order minus the EDF+/BDF+ annotations channel, so the
    n-th name in ``raw.ch_names`` is the n-th non-annotation header row. Each
    pairing is still checked by name, so a reader that reorders or drops a
    channel fails loud rather than attaching one row's scaling to another
    row's data. The check accepts MNE's own renaming of a repeated label: two
    header rows both labelled ``"EEG"`` come back from MNE as ``"EEG-0"`` and
    ``"EEG-1"``. The labels returned are biosigIO's (see
    :func:`~biosigio.importers._labels.unique_channel_labels`), not MNE's, so
    a recovered read names its channels exactly as a normal read would.
    """
    from ..exceptions import FileReadError
    from ._mne_common import require_mne

    probe = probe_edf_header(filepath)
    _check_not_truncated(filepath, probe, probe.number_of_datarecords)
    # Before MNE is touched: it would concatenate the records of a gapped EDF+D file
    # onto one contiguous axis (issue #180), so a file with a gap is refused here,
    # whatever else about it sent it down this path, and without needing MNE at all.
    verified_contiguous = verify_records_contiguous(filepath, probe)

    mne = require_mne()

    # stim_channel=False: see the module docstring -- without it, MNE would
    # auto-detect a channel named e.g. "Status" as a trigger channel and give
    # it its own uncalibrated, bitmasked treatment instead of the header's
    # declared physical scaling, breaking unit parity with pyedflib for it.
    raw = mne.io.read_raw(filepath, preload=True, verbose="ERROR", stim_channel=False)
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])

    # Pair by position (see the docstring): MNE drops only the annotations
    # channel, and renames a repeated label "<label>-<suffix>".
    probed_channels = [ch for ch in probe.channels if ch["label"] not in _ANNOTATION_LABELS]
    if len(probed_channels) != len(raw.ch_names):
        raise FileReadError(
            f"{filepath}: fallback reader (MNE {mne.__version__}) read "
            f"{len(raw.ch_names)} channels but the file's own header declares "
            f"{len(probed_channels)} signal channels"
        )
    _check_channel_pairing(filepath, list(raw.ch_names), [p["label"] for p in probed_channels])
    labels, renamed = unique_channel_labels(
        [p["label"] for p in probed_channels], filepath=filepath
    )

    gains = _channel_gains(raw, [p["dimension"] for p in probed_channels])

    channels: list[EdfFallbackChannel] = []
    for i, (label, probed) in enumerate(zip(labels, probed_channels, strict=True)):
        degenerate = probed["physical_min"] == probed["physical_max"]
        if degenerate:
            # The digital-to-physical slope is 0/0; the only value the scaling
            # formula can consistently produce is the (equal) physical_min/max
            # constant itself -- 0 in the common referential-reference-channel
            # case, but not assumed to be 0 in general.
            values = np.full(data.shape[1], probed["physical_min"], dtype=np.float64)
        else:
            values = data[i] / gains[i]
        channels.append(
            EdfFallbackChannel(
                label=label,
                data=values,
                sample_frequency=sfreq,
                physical_dimension=probed["dimension"] or "n/a",
                physical_min=probed["physical_min"],
                physical_max=probed["physical_max"],
                digital_min=probed["digital_min"],
                digital_max=probed["digital_max"],
                prefilter=probed["prefilter"] or "n/a",
                transducer=probed["transducer"],
                degenerate_physical_range=degenerate,
            )
        )

    events = _events_from_mne_annotations(raw)

    is_bdf = filepath.lower().endswith(".bdf")
    is_plus = probe.reserved.upper().startswith(("EDF+", "BDF+"))
    if is_bdf:
        filetype = pyedflib.FILETYPE_BDFPLUS if is_plus else pyedflib.FILETYPE_BDF
    else:
        filetype = pyedflib.FILETYPE_EDFPLUS if is_plus else pyedflib.FILETYPE_EDF

    return EdfFallbackRecording(
        channels=channels,
        events=events,
        filetype=filetype,
        file_duration=(raw.n_times / sfreq) if sfreq else 0.0,
        datarecord_duration=probe.duration_of_data_record,
        renamed_labels=renamed,
        verified_contiguous=verified_contiguous,
    )
