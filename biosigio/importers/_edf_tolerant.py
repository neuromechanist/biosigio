"""Tolerant fallback for EDF/BDF files pyedflib's compliance checker rejects.

biosigIO's normal EDF/BDF path reads via ``pyedflib``, whose C-level ``open_file``
validator is stricter than the format's real-world variation. Three conditions,
all confirmed byte-intact and readable by MNE-Python, currently discard entire
recordings that are perfectly usable (issue #109 and the NEMAR Zarr backfill):

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
   pyedflib: ``"The file is discontinuous and cannot be read"``.

:func:`classify_pyedflib_error` recognizes exactly these three conditions from
pyedflib's exception message and returns ``None`` for anything else (including
the ``"Filesize"`` truncation check), so a genuinely corrupt/truncated file is
never misrouted here -- it keeps raising through the caller's
``classify_read_error`` as :class:`~biosigio.exceptions.CorruptFileError`.

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
from dataclasses import dataclass, field
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
    return 3 if filepath.lower().endswith(".bdf") else 2


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
    """Same shape/sort as ``EDFImporter._read_annotations`` (pyedflib path)."""
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

    mne = require_mne()

    probe = probe_edf_header(filepath)
    _check_not_truncated(filepath, probe, probe.number_of_datarecords)

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
            f"{filepath}: fallback reader read {len(raw.ch_names)} channels but the "
            f"file's own header declares {len(probed_channels)} signal channels"
        )
    for name, probed in zip(raw.ch_names, probed_channels, strict=True):
        if name != probed["label"] and not name.startswith(f"{probed['label']}-"):
            raise FileReadError(
                f"{filepath}: fallback reader could not match channel {name!r} "
                f"read by MNE back to header channel {probed['label']!r} at the "
                "same position"
            )
    labels = unique_channel_labels([p["label"] for p in probed_channels])

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
    )
