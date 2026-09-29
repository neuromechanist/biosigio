"""Tolerant EDF/BDF fallback for pyedflib's overly strict compliance checks (#109).

pyedflib refuses to open real-world, byte-intact EDF/BDF files for three
conditions MNE-Python reads without complaint:

1. a channel with ``physical_min == physical_max`` (issue #109),
2. a numeric header field padded with NUL bytes instead of ASCII spaces,
3. a file correctly marked EDF+D (discontinuous).

Every fixture here is a REAL file on disk: written with ``pyedflib.EdfWriter``
(which can write EDF/BDF, just not these three malformed conditions -- pyedflib's
own writer refuses degenerate ranges, and has no API for writing a NUL-padded
field or an EDF+D file at all), then hand-patched at the exact byte offset the
condition lives at. No mocks, no fake readers: every read in this file, on both
the pyedflib and biosigIO sides, touches the same bytes on disk.

The single most important test is `test_fallback_matches_pyedflib_reading`:
it is the actual proof that the tolerant fallback's rescaled output is
numerically identical to pyedflib's own physical values, not an assumption.
"""

import os
import re
import struct
import tempfile

import numpy as np
import pyedflib
import pytest

pytest.importorskip("mne", reason="the tolerant EDF/BDF fallback requires the 'meg' extra (mne)")

from biosigio import Recording  # noqa: E402
from biosigio.exceptions import CorruptFileError, FileReadError  # noqa: E402
from biosigio.importers._edf_tolerant import (  # noqa: E402
    DEGENERATE_PHYSICAL_RANGE,
    DISCONTINUOUS_DATARECORDS,
    MALFORMED_NUMERIC_FIELD,
    _check_not_truncated,
    classify_pyedflib_error,
    probe_edf_header,
    read_edf_tolerant,
)

# Per-signal header field widths, in on-disk order (EDF/BDF spec, independent of
# the production module's own copy of this layout -- a test that imported the
# same constant it exercises could not catch an ordering bug in it).
_SIGNAL_FIELD_WIDTHS = {
    "label": (16, 0),
    "transducer": (80, 1),
    "dimension": (8, 2),
    "physical_min": (8, 3),
    "physical_max": (8, 4),
    "digital_min": (8, 5),
    "digital_max": (8, 6),
    "prefilter": (80, 7),
    "samples_per_record": (8, 8),
    "reserved": (32, 9),
}
_FIELD_ORDER_WIDTHS = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]


def _read_number_of_signals(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(252)
        return int(f.read(4).decode("ascii").strip())


def _patch_signal_field(path: str, ch_index: int, field: str, new_text: bytes) -> None:
    """Overwrite one per-signal header field on disk, space-padded to width."""
    ns = _read_number_of_signals(path)
    width, field_index = _SIGNAL_FIELD_WIDTHS[field]
    offset = 256 + sum(_FIELD_ORDER_WIDTHS[:field_index]) * ns + ch_index * width
    assert len(new_text) <= width
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(new_text.ljust(width, b" "))


def _patch_main_header_field(
    path: str, offset: int, width: int, new_text: bytes, pad: bytes
) -> None:
    assert len(new_text) <= width
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(new_text.ljust(width, pad))


def _write_edf(
    path: str,
    channels: list[dict],
    data: list[np.ndarray],
    *,
    annotations: list[tuple[float, float, str]] | None = None,
    file_type: int = pyedflib.FILETYPE_EDFPLUS,
) -> None:
    """Write a real, fully pyedflib-compliant EDF/BDF file."""
    w = pyedflib.EdfWriter(path, len(channels), file_type=file_type)
    try:
        w.setSignalHeaders(channels)
        for onset, duration, description in annotations or []:
            w.writeAnnotation(onset, duration, description)
        w.writeSamples(list(data))
    finally:
        w.close()


def _channel(
    label: str, sf: float, pmin: float, pmax: float, dmin: int = -32768, dmax: int = 32767
) -> dict:
    return {
        "label": label,
        "dimension": "uV",
        "sample_frequency": sf,
        "physical_max": pmax,
        "physical_min": pmin,
        "digital_max": dmax,
        "digital_min": dmin,
        "transducer": "",
        "prefilter": "",
    }


def _pyedflib_ground_truth(path: str) -> dict[str, np.ndarray]:
    """Read every real (non-annotation) signal via pyedflib, keyed by label."""
    r = pyedflib.EdfReader(path)
    try:
        out = {}
        for i, h in enumerate(r.getSignalHeaders()):
            out[h["label"].strip()] = r.readSignal(i)
        return out
    finally:
        r.close()


# --- classify_pyedflib_error ---------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "x: the file is not EDF(+) or BDF(+) compliant (Physical Maximum)",
            DEGENERATE_PHYSICAL_RANGE,
        ),
        (
            "x: the file is not EDF(+) or BDF(+) compliant (Physical Minimum)",
            DEGENERATE_PHYSICAL_RANGE,
        ),
        (
            "x: the file is not EDF(+) or BDF(+) compliant (Number of Datarecords)",
            MALFORMED_NUMERIC_FIELD,
        ),
        (
            "x: the file is not EDF(+) or BDF(+) compliant (Digital Maximum)",
            MALFORMED_NUMERIC_FIELD,
        ),
        (
            "x: the file is not EDF(+) or BDF(+) compliant (Sample in Datarecord)",
            MALFORMED_NUMERIC_FIELD,
        ),
        ("x: The file is discontinuous and cannot be read", DISCONTINUOUS_DATARECORDS),
        ("x: the file is not EDF(+) or BDF(+) compliant (Filesize)", None),
        ("x: the file is not EDF(+) or BDF(+) compliant (EDF+ Patientname)", None),
        ("x: the file is not EDF(+) or BDF(+) compliant (Prefilter)", None),
        ("some unrelated OSError", None),
    ],
)
def test_classify_pyedflib_error(message, expected):
    assert classify_pyedflib_error(OSError(message)) == expected


# --- Problem 1: physical_min == physical_max (issue #109) ---------------------


def test_pyedflib_writer_refuses_degenerate_range_directly(tmp_path):
    """pyedflib's own writer asserts on this -- confirms byte-patching is the
    only way to construct this real-world condition, not a shortcut."""
    path = str(tmp_path / "writer_refuses.edf")
    with pytest.raises(AssertionError):
        _write_edf(
            path,
            [_channel("REF", 100.0, 0.0, 0.0)],
            [np.zeros(100)],
        )


def test_degenerate_physical_range_recovered(tmp_path):
    path = str(tmp_path / "degenerate.edf")
    rng = np.random.default_rng(109)
    n = 300
    channels = [
        _channel("C0", 100.0, -200.0, 200.0),
        _channel("REF", 100.0, -200.0, 200.0),
        _channel("C1", 100.0, -500.0, 500.0),
    ]
    data = [rng.uniform(-200, 200, n), rng.uniform(-200, 200, n), rng.uniform(-500, 500, n)]
    _write_edf(path, channels, data)

    # Ground truth for the two normal channels, captured while the file is
    # still fully compliant (opening a file with a degenerate channel fails
    # before any signal can be read at all, so this MUST happen first).
    ground_truth = _pyedflib_ground_truth(path)

    # Make REF degenerate: physical_max := physical_min (both -200).
    _patch_signal_field(path, 1, "physical_max", b"-200")

    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")

    assert rec.metadata["edf_tolerant_read"] is True
    assert rec.metadata["edf_tolerant_read_reason"] == DEGENERATE_PHYSICAL_RANGE
    assert set(rec.channels) == {"C0", "REF", "C1"}
    assert rec.channels["REF"]["degenerate_physical_range"] is True
    assert "degenerate_physical_range" not in rec.channels["C0"]
    assert "degenerate_physical_range" not in rec.channels["C1"]

    # The degenerate channel must be an exact constant at physical_min -- not
    # MNE's own (nonzero, meaningless) fabricated calibration for it.
    ref_values = rec.signals["REF"].to_numpy()
    assert len(ref_values) == n
    np.testing.assert_array_equal(ref_values, np.full(n, -200.0))

    # The two normal channels must round-trip exactly like a normal read.
    # rtol (not a fixed atol): a fixed atol would silently loosen for a
    # small-native-magnitude channel, hiding exactly the drift this is meant
    # to catch (see test_fallback_matches_pyedflib_reading's "VLF" channel).
    np.testing.assert_allclose(rec.signals["C0"].to_numpy(), ground_truth["C0"], rtol=1e-9)
    np.testing.assert_allclose(rec.signals["C1"].to_numpy(), ground_truth["C1"], rtol=1e-9)


def test_degenerate_physical_range_nonzero_constant(tmp_path):
    """physical_min == physical_max at a NONZERO value: the correct fallback
    output is that constant, not a hardcoded 0 (0 is only the common case)."""
    path = str(tmp_path / "degenerate_nonzero.edf")
    rng = np.random.default_rng(7)
    n = 200
    channels = [_channel("C0", 50.0, -100.0, 100.0), _channel("REF", 50.0, -100.0, 100.0)]
    data = [rng.uniform(-100, 100, n), rng.uniform(-100, 100, n)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_min", b"37.5")
    _patch_signal_field(path, 1, "physical_max", b"37.5")

    with pytest.raises(OSError, match=r"(?i)physical"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")
    np.testing.assert_array_equal(rec.signals["REF"].to_numpy(), np.full(n, 37.5))


def test_degenerate_range_with_stim_channel_recovered(tmp_path):
    """Regression test for the real on004444 failure shape: a degenerate
    reference channel AND a "Status" channel declared in a voltage dimension,
    in the SAME recording. Neither condition alone reproduces this: MNE
    auto-detects "Status" as a `stim` channel and gives it its own
    uncalibrated, bitmasked treatment (see the module docstring in
    `_edf_tolerant.py`) that has nothing to do with the header's declared
    physical range, and would silently disagree with a normal pyedflib read
    of that channel if not handled -- this is exactly what broke on the real
    file before `read_edf_tolerant` started opening with `stim_channel=False`."""
    path = str(tmp_path / "degenerate_with_stim.edf")
    rng = np.random.default_rng(4444)
    n = 300
    channels = [
        _channel("C0", 100.0, -200.0, 200.0),
        _channel("REF", 100.0, -200.0, 200.0),
        _channel("Status", 100.0, -200.0, 200.0),
    ]
    # A real status/trigger channel carries small integer codes, not a
    # smoothly varying signal -- distinct enough from C0's data to catch a
    # channel-alignment bug, not just a scaling one.
    data = [
        rng.uniform(-200, 200, n),
        rng.uniform(-200, 200, n),
        rng.integers(-100, 100, n).astype(float),
    ]
    _write_edf(path, channels, data)

    # Ground truth captured before the file is degenerated (see
    # test_degenerate_physical_range_recovered for why this must happen
    # first): pyedflib has no notion of "stim" and always applies the
    # header's calibration uniformly, so this is what a normal read of
    # "Status" -- scaled into its declared (if nonsensical) microvolts --
    # actually looks like.
    ground_truth = _pyedflib_ground_truth(path)

    _patch_signal_field(path, 1, "physical_max", b"-200")
    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")

    assert rec.metadata["edf_tolerant_read_reason"] == DEGENERATE_PHYSICAL_RANGE
    np.testing.assert_array_equal(rec.signals["REF"].to_numpy(), np.full(n, -200.0))
    np.testing.assert_allclose(rec.signals["C0"].to_numpy(), ground_truth["C0"], rtol=1e-9)
    # The channel this regression is actually about: must match pyedflib's
    # calibrated value, NOT MNE's raw/bitmasked stim-channel treatment (which
    # would produce values far outside the declared [-200, 200] uV range).
    np.testing.assert_allclose(rec.signals["Status"].to_numpy(), ground_truth["Status"], rtol=1e-9)


# --- Problem 2: NUL-padded numeric header field --------------------------------


def test_malformed_numeric_field_recovered(tmp_path):
    path = str(tmp_path / "nul_padded.edf")
    rng = np.random.default_rng(6233)
    n = 400
    channels = [_channel("C0", 100.0, -300.0, 300.0), _channel("C1", 100.0, -50.0, 50.0)]
    data = [rng.uniform(-300, 300, n), rng.uniform(-50, 50, n)]
    _write_edf(path, channels, data)

    ground_truth = _pyedflib_ground_truth(path)

    # Number-of-datarecords field: main header offset 236, width 8. Same
    # numeric value ("4" data records of the writer's default 1s duration),
    # just NUL-padded instead of space-padded -- the real on006233 bug.
    _patch_main_header_field(path, 236, 8, b"4", pad=b"\x00")

    with pytest.raises(OSError, match=r"(?i)number of datarecords"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")
    assert rec.metadata["edf_tolerant_read"] is True
    assert rec.metadata["edf_tolerant_read_reason"] == MALFORMED_NUMERIC_FIELD
    assert rec.get_n_channels() == 2
    for label in ("C0", "C1"):
        assert len(rec.signals[label]) == n
        np.testing.assert_allclose(rec.signals[label].to_numpy(), ground_truth[label], rtol=1e-9)


# --- Problem 3: discontinuous EDF+D --------------------------------------------


def test_discontinuous_edfd_recovered(tmp_path):
    path = str(tmp_path / "discontinuous.edf")
    rng = np.random.default_rng(6910)
    n = 500
    channels = [_channel("C0", 100.0, -400.0, 400.0), _channel("C1", 100.0, -400.0, 400.0)]
    data = [rng.uniform(-400, 400, n), rng.uniform(-400, 400, n)]
    _write_edf(path, channels, data, annotations=[(1.0, 0.0, "trial_start")])

    ground_truth = _pyedflib_ground_truth(path)

    # Reserved field of the main header: offset 192, width 44, holds "EDF+C".
    with open(path, "rb") as f:
        f.seek(192)
        reserved = f.read(44)
    assert reserved.startswith(b"EDF+C")
    _patch_main_header_field(path, 192, 44, b"EDF+D", pad=b" ")

    with pytest.raises(OSError, match=r"(?i)discontinuous"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")
    assert rec.metadata["edf_tolerant_read"] is True
    assert rec.metadata["edf_tolerant_read_reason"] == DISCONTINUOUS_DATARECORDS
    for label in ("C0", "C1"):
        assert len(rec.signals[label]) == n
        np.testing.assert_allclose(rec.signals[label].to_numpy(), ground_truth[label], rtol=1e-9)

    assert not rec.events.empty
    assert rec.events.iloc[0]["description"] == "trial_start"
    assert rec.events.iloc[0]["onset"] == pytest.approx(1.0)


# --- The load-bearing unit-parity proof ----------------------------------------


def test_fallback_matches_pyedflib_reading(tmp_path):
    """THE core claim of this PR: on a file BOTH readers can open, the tolerant
    fallback's rescaled physical values are numerically identical to a normal
    pyedflib read -- not merely close, not merely 'reasonable'. This is proven,
    not assumed, precisely because this fixture has none of the three
    conditions and so is readable by pyedflib directly, giving a real ground
    truth to compare against on the exact same bytes.

    Includes one deliberately small-magnitude, "V"-native channel (VLF, an
    unrecognized-as-uV/mV dimension string so MNE and biosigIO both default
    its gain to 1): the comparison below uses `rtol`, not a fixed `atol`, so
    this channel's tiny native values are checked at the SAME relative
    precision as the volt-scale ones, not silently looser. Also includes a
    "Status" channel: MNE auto-detects that name as a trigger/status channel
    and gives it its OWN uncalibrated treatment unrelated to the header's
    declared range (see `_edf_tolerant.py`'s module docstring) -- without
    `read_edf_tolerant` opening with `stim_channel=False`, this channel alone
    would fail this exact parity claim (this was a real, shipped bug)."""
    path = str(tmp_path / "compliant.edf")
    rng = np.random.default_rng(944)
    n = 1000
    channels = [
        _channel("EEG1", 250.0, -400.0, 400.0),
        _channel("EEG2", 250.0, -123.45, 678.9),
        _channel("EMG1", 250.0, -1000.0, 1000.0),
        {**_channel("VLF", 250.0, -1e-5, 1e-5), "dimension": "V"},
        _channel("Status", 250.0, -200.0, 200.0),
    ]
    data = [
        rng.uniform(-400, 400, n),
        rng.uniform(-123.45, 678.9, n),
        rng.uniform(-1000, 1000, n),
        rng.uniform(-1e-5, 1e-5, n),
        rng.integers(-100, 100, n).astype(float),
    ]
    _write_edf(path, channels, data)

    # Sanity: pyedflib itself accepts this file (no recoverable condition).
    ground_truth = _pyedflib_ground_truth(path)

    fallback = read_edf_tolerant(path, reason=MALFORMED_NUMERIC_FIELD)
    by_label = {ch.label: ch for ch in fallback.channels}
    assert set(by_label) == set(ground_truth)
    for label, expected in ground_truth.items():
        np.testing.assert_allclose(
            by_label[label].data,
            expected,
            rtol=1e-9,
            err_msg=f"unit-parity mismatch on channel {label!r}",
        )
        assert by_label[label].degenerate_physical_range is False


def test_fallback_matches_pyedflib_reading_bdf(tmp_path):
    """Same unit-parity proof, on a BDF (24-bit, 3 bytes/sample) file, again
    including a "Trigger"-named channel (MNE's other default auto-stim name,
    alongside "Status")."""
    path = str(tmp_path / "compliant.bdf")
    rng = np.random.default_rng(945)
    n = 800
    channels = [
        _channel("C0", 200.0, -1000.0, 1000.0, dmin=-8388608, dmax=8388607),
        _channel("C1", 200.0, -50.0, 50.0, dmin=-8388608, dmax=8388607),
        _channel("Trigger", 200.0, -100.0, 100.0, dmin=-8388608, dmax=8388607),
    ]
    data = [
        rng.uniform(-1000, 1000, n),
        rng.uniform(-50, 50, n),
        rng.integers(-100, 100, n).astype(float),
    ]
    _write_edf(path, channels, data, file_type=pyedflib.FILETYPE_BDFPLUS)

    ground_truth = _pyedflib_ground_truth(path)
    fallback = read_edf_tolerant(path, reason=MALFORMED_NUMERIC_FIELD)
    by_label = {ch.label: ch for ch in fallback.channels}
    for label, expected in ground_truth.items():
        np.testing.assert_allclose(by_label[label].data, expected, rtol=1e-9)


# --- Negative tests: genuine corruption must still be reported as such --------


def test_truncated_file_still_raises_corrupt(tmp_path):
    path = str(tmp_path / "truncated.edf")
    rng = np.random.default_rng(1)
    n = 500
    channels = [_channel("C0", 100.0, -100.0, 100.0)]
    data = [rng.uniform(-100, 100, n)]
    _write_edf(path, channels, data)

    full_size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(full_size - 200)  # chop off part of the last data record

    with pytest.raises(OSError, match=r"(?i)filesize"):
        pyedflib.EdfReader(path)

    with pytest.raises(CorruptFileError):
        Recording.from_file(path, importer="edf")


def test_truncated_and_degenerate_still_raises_corrupt(tmp_path):
    """A file that is BOTH truncated AND has a degenerate channel: pyedflib
    happens to report the degenerate-range condition (not "Filesize" -- its
    checks are not priority-ordered by which is "most true"), which on its own
    would look recoverable. The truncation safety net must still catch this
    and preserve CorruptFileError -- proving corrupt_or_truncated cannot be
    silently bypassed just because the file also matches a recoverable pattern."""
    path = str(tmp_path / "truncated_degenerate.edf")
    rng = np.random.default_rng(2)
    n = 400
    channels = [_channel("C0", 100.0, -100.0, 100.0), _channel("REF", 100.0, -100.0, 100.0)]
    data = [rng.uniform(-100, 100, n), rng.uniform(-100, 100, n)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_max", b"-100")

    full_size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(full_size - 200)

    # pyedflib reports the degenerate range, NOT the truncation -- confirming
    # the safety net is genuinely load-bearing here, not redundant with pyedflib.
    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)
    assert (
        classify_pyedflib_error(OSError("... compliant (Physical Maximum)"))
        == DEGENERATE_PHYSICAL_RANGE
    )

    with pytest.raises(CorruptFileError):
        Recording.from_file(path, importer="edf")


def test_check_not_truncated_direct(tmp_path):
    """Unit-level check of the safety net itself, independent of the full
    pyedflib-error-message path."""
    path = str(tmp_path / "probe_truncate.edf")
    channels = [_channel("C0", 100.0, -100.0, 100.0)]
    data = [np.zeros(300)]
    _write_edf(path, channels, data)
    probe = probe_edf_header(path)
    _check_not_truncated(path, probe, probe.number_of_datarecords)  # full file: OK

    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 50)
    with pytest.raises(OSError, match=r"(?i)truncated"):
        _check_not_truncated(path, probe, probe.number_of_datarecords)


def test_mne_missing_degrades_to_original_error(tmp_path, monkeypatch, caplog):
    """If MNE (the `meg` extra) is not installed, loading a recoverable file
    must degrade to the SAME error pyedflib itself raised -- not crash with a
    raw ImportError, and not claim success it cannot deliver."""
    import biosigio.importers._mne_common as mne_common

    def _no_mne():
        raise ImportError("mne is not installed")

    monkeypatch.setattr(mne_common, "require_mne", _no_mne)

    path = str(tmp_path / "degenerate_no_mne.edf")
    channels = [_channel("C0", 100.0, -100.0, 100.0), _channel("REF", 100.0, -100.0, 100.0)]
    data = [np.zeros(100), np.zeros(100)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_max", b"-100")

    with caplog.at_level("WARNING", logger="biosigio.importers.edf"):
        with pytest.raises(CorruptFileError) as info:
            Recording.from_file(path, importer="edf")

    # The install hint is not lost: the pyedflib error is chained to the
    # ImportError, and the log names the missing extra.
    original = info.value.__cause__
    assert isinstance(original, OSError)
    assert isinstance(original.__cause__, ImportError)
    assert "'meg' extra" in caplog.text and path in caplog.text


# --- probe_edf_header -----------------------------------------------------------


def test_probe_edf_header_tolerant_of_nul_padding(tmp_path):
    path = str(tmp_path / "probe.edf")
    channels = [_channel("C0", 100.0, -10.0, 10.0), _channel("REF", 100.0, -10.0, 10.0)]
    data = [np.linspace(-10, 10, 200), np.full(200, 5.0)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_min", b"5")
    _patch_signal_field(path, 1, "physical_max", b"5")
    _patch_main_header_field(path, 236, 8, b"2", pad=b"\x00")

    probe = probe_edf_header(path)
    assert probe.number_of_datarecords == 2
    labels = [c["label"] for c in probe.channels]
    assert "C0" in labels and "REF" in labels
    ref = next(c for c in probe.channels if c["label"] == "REF")
    assert ref["physical_min"] == ref["physical_max"] == 5.0


def test_probe_edf_header_rejects_short_file(tmp_path):
    path = tmp_path / "short.edf"
    path.write_bytes(struct.pack("B", 0) * 100)  # far short of the 256-byte main header
    with pytest.raises(OSError):
        probe_edf_header(str(path))


# --- Streaming path (zarr_stream._EdfSource) must agree with the in-memory path -

# #944 kept EDF/BDF off the MNE streaming path specifically because a streamed
# read and an in-memory read disagreeing on units would be a silent
# correctness bug -- exactly the hazard a fallback that *also* uses MNE could
# reintroduce if the two call sites drifted. These tests exercise
# `_EdfSource`'s fallback branch (zarr_stream.py) directly, not just
# `EDFImporter.load`.


def _dequant(store_path: str, gname: str) -> np.ndarray:
    import zarr

    g = zarr.open_group(store_path, mode="r")[gname]
    base = np.asarray(g["0"][:])
    scale = np.asarray(g["0"].attrs["scale"])[:, None]
    offset = np.asarray(g["0"].attrs["offset"])[:, None]
    return base * scale + offset


def test_edf_source_fallback_matches_edf_importer_degenerate(tmp_path):
    """Direct, full-precision check of `_EdfSource`'s fallback branch
    (zarr_stream.py:179-217, the read at :352-353) against `EDFImporter.load`
    for the degenerate-physical-range condition. No int16 quantization is
    involved here, unlike the store-level test below, so this pins the
    streaming path's raw values exactly, not just "close enough"."""
    from biosigio.exporters.zarr_stream import _EdfSource

    path = str(tmp_path / "source_degenerate.edf")
    rng = np.random.default_rng(217)
    n = 400
    channels = [_channel("C0", 100.0, -100.0, 100.0), _channel("REF", 100.0, -100.0, 100.0)]
    data = [rng.uniform(-100, 100, n), rng.uniform(-100, 100, n)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_max", b"-100")
    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")
    assert rec.metadata["edf_tolerant_read"] is True

    source = _EdfSource(path, force_modality="EEG")
    try:
        assert source.extra_metadata["edf_tolerant_read"] is True
        assert source.extra_metadata["edf_tolerant_read_reason"] == DEGENERATE_PHYSICAL_RANGE
        assert source.n_samples == n
        labels = [ch["label"] for ch in source.channels]
        streamed = source.read(list(range(len(labels))), 0, n)
    finally:
        source.close()

    for i, label in enumerate(labels):
        np.testing.assert_array_equal(streamed[i], rec.signals[label].to_numpy())
        if label == "REF":
            np.testing.assert_array_equal(streamed[i], np.full(n, -100.0))


def test_edf_source_fallback_matches_edf_importer_discontinuous(tmp_path):
    """Same direct check, for the discontinuous-EDF+D condition."""
    from biosigio.exporters.zarr_stream import _EdfSource

    path = str(tmp_path / "source_discontinuous.edf")
    rng = np.random.default_rng(6910)
    n = 500
    channels = [_channel("C0", 100.0, -400.0, 400.0), _channel("C1", 100.0, -400.0, 400.0)]
    data = [rng.uniform(-400, 400, n), rng.uniform(-400, 400, n)]
    _write_edf(path, channels, data)
    _patch_main_header_field(path, 192, 44, b"EDF+D", pad=b" ")
    with pytest.raises(OSError, match=r"(?i)discontinuous"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")

    source = _EdfSource(path, force_modality="EEG")
    try:
        assert source.extra_metadata["edf_tolerant_read_reason"] == DISCONTINUOUS_DATARECORDS
        labels = [ch["label"] for ch in source.channels]
        streamed = source.read(list(range(len(labels))), 0, n)
    finally:
        source.close()

    for i, label in enumerate(labels):
        np.testing.assert_array_equal(streamed[i], rec.signals[label].to_numpy())


def test_stream_edf_degenerate_range_matches_in_memory_export(tmp_path):
    """End-to-end: `stream_to_zarr` on a physical_min == physical_max file must
    produce the SAME Zarr store (up to int16 quantization) as the in-memory
    `Recording.from_file` -> `to_zarr` path -- the reviewer-requested
    store-level regression test for the streaming fallback."""
    pytest.importorskip("zarr", reason="requires the 'zarr' extra")
    import biosigio

    path = str(tmp_path / "stream_degenerate.edf")
    rng = np.random.default_rng(944109)
    n = 1000  # 100 Hz x 10 s
    channels = [
        _channel("C0", 100.0, -200.0, 200.0),
        _channel("REF", 100.0, -200.0, 200.0),
        _channel("C1", 100.0, -400.0, 400.0),
    ]
    data = [rng.uniform(-200, 200, n), rng.uniform(-200, 200, n), rng.uniform(-400, 400, n)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_max", b"-200")
    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)

    with tempfile.TemporaryDirectory() as d:
        s_stream = os.path.join(d, "stream.zarr")
        s_inmem = os.path.join(d, "inmem.zarr")
        biosigio.stream_to_zarr(path, s_stream, force_modality="EEG", dtype="int16")
        rec = biosigio.Recording.from_file(path, importer="edf")
        for label in rec.channels:
            rec.channels[label]["modality"] = "EEG"
        rec.to_zarr(s_inmem, dtype="int16")

        a = _dequant(s_stream, "eeg_100hz")
        b = _dequant(s_inmem, "eeg_100hz")
        assert a.shape == b.shape
        for i in range(a.shape[0]):
            span = float(b[i].max() - b[i].min()) or 1.0
            assert np.max(np.abs(a[i] - b[i])) <= 6 * span / 65535


# --- _channel_gains: shape and semantic-drift guards ---------------------------

# These construct a REAL MNE `Raw` (by reading a real, fully-compliant file)
# and then perturb its own `_raw_extras[0]["units"]` in place -- not a mock
# class standing in for MNE, but MNE's real object with one field pushed into
# a state a future MNE release could plausibly produce on its own.


def test_channel_gains_shape_mismatch_raises(tmp_path):
    """A future MNE whose `units` array does not align 1:1 with `ch_names`
    must fail here, with a clear message, instead of a bare IndexError later
    at `gains[i]`."""
    import mne

    from biosigio.exceptions import FileReadError
    from biosigio.importers._edf_tolerant import _channel_gains

    path = str(tmp_path / "gains_shape.edf")
    channels = [_channel("C0", 100.0, -100.0, 100.0), _channel("C1", 100.0, -50.0, 50.0)]
    data = [np.zeros(100), np.zeros(100)]
    _write_edf(path, channels, data)

    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    raw._raw_extras[0]["units"] = raw._raw_extras[0]["units"][:1]  # drop one entry

    with pytest.raises(FileReadError, match=r"(?i)per-channel unit gain"):
        _channel_gains(raw, ["uV", "uV"])


def test_channel_gains_semantic_drift_raises(tmp_path):
    """The critical guard: same key, same shape, but the VALUE disagrees with
    what biosigIO independently computes from the dimension string -- exactly
    what a future MNE release changing what `_raw_extras[0]["units"]` means
    (not just whether it exists) would look like, and which nothing else in
    this module could otherwise catch (it is correct against mne 1.12.1, and
    the `meg` extra has no upper version bound)."""
    import mne

    from biosigio.exceptions import FileReadError
    from biosigio.importers._edf_tolerant import _channel_gains

    path = str(tmp_path / "gains_drift.edf")
    channels = [_channel("C0", 100.0, -100.0, 100.0)]
    data = [np.zeros(100)]
    _write_edf(path, channels, data)

    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    assert raw._raw_extras[0]["units"][0] == pytest.approx(1e-6)  # sanity: uV, as expected
    raw._raw_extras[0]["units"] = np.array([2e-6])  # simulate a future MNE meaning something else

    with pytest.raises(FileReadError, match=r"(?i)disagrees"):
        _channel_gains(raw, ["uV"])


def test_channel_gains_matches_expected_for_known_units(tmp_path):
    """Sanity check for the guard itself: uV, mV, and an unrecognized/absent
    unit (treated as already-volts, same as MNE) do not raise."""
    import mne

    from biosigio.importers._edf_tolerant import _channel_gains

    path = str(tmp_path / "gains_normal.edf")
    channels = [
        _channel("C_uV", 100.0, -100.0, 100.0),
        {**_channel("C_mV", 100.0, -1.0, 1.0), "dimension": "mV"},
        {**_channel("C_none", 100.0, -1.0, 1.0), "dimension": ""},
    ]
    data = [np.zeros(100), np.zeros(100), np.zeros(100)]
    _write_edf(path, channels, data)

    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    gains = _channel_gains(raw, ["uV", "mV", ""])
    np.testing.assert_allclose(gains, [1e-6, 1e-3, 1.0])


# --- pairing MNE's channels back to header rows --------------------------------


def test_pairing_accepts_exactly_mnes_rename_forms():
    from biosigio.importers._edf_tolerant import _check_channel_pairing

    _check_channel_pairing(
        "f.edf", ["A-0", "A-B", "A-1", "A-a", "C-12"], ["A", "A-B", "A", "A", "C"]
    )


@pytest.mark.parametrize(
    ("mne_names", "header_labels", "bad"),
    [
        # A reordered pair: MNE's "A-B" sits where the header has "A". A prefix
        # test accepted it and would have swapped the two rows' scaling.
        (["A-B", "A"], ["A", "A-B"], "A-B"),
        (["A-0x"], ["A"], "A-0x"),  # not a running number or a single letter
        (["A-ab"], ["A"], "A-ab"),
        (["A0"], ["A"], "A0"),
        (["a.b-0"], ["a+b"], "a.b-0"),  # the label is matched literally, not as a regex
    ],
)
def test_pairing_rejects_anything_else(mne_names, header_labels, bad):
    from biosigio.importers._edf_tolerant import _check_channel_pairing

    with pytest.raises(FileReadError, match=re.escape(f"could not match channel {bad!r}")):
        _check_channel_pairing("f.edf", mne_names, header_labels)
