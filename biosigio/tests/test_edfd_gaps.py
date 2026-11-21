"""An EDF+D/BDF+D file with a gap is refused, not read onto a wrong time axis (issue #180).

An EDF+D file gives each data record its own start time. MNE (the tolerant fallback
reader) ignores that and concatenates the records, so every sample after a gap lands
earlier than it was recorded while the annotations keep their true times. biosigIO now
reads the time-keeping annotation of every record itself and raises
``DiscontinuousRecordingError`` naming each gap, on both paths that read EDF: the
in-memory importer and the streaming Zarr exporter. A +D file whose records are in fact
contiguous is still read, and flagged.

The fixtures are real files on disk (``examples/edfd/``, see its NOTICE), written byte
for byte by ``edfd_builder``; the ramp inside each is its own true time in seconds, so a
sample placed on the wrong time shows as a value that disagrees with its position.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from biosigio import BiosigIOError, Recording, stream_to_zarr
from biosigio.exceptions import FileReadError

from .edfd_builder import FS, write_edf_plus

pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")

import zarr  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "edfd"
VERIFIED_KEY = "edf_discontinuous_verified_contiguous"


def _verify(path: str):
    """``verify_records_contiguous``, imported here so a test fails on its own when the
    function does not exist, rather than the whole module failing to import."""
    from biosigio.importers._edf_tolerant import verify_records_contiguous

    return verify_records_contiguous(path)


def _copy(name: str, tmp_path: Path) -> str:
    """A private copy of a committed fixture, so a test can patch it freely."""
    dest = tmp_path / name
    shutil.copy(EXAMPLES / name, dest)
    return str(dest)


def _refusal(path: str) -> BiosigIOError:
    with pytest.raises(BiosigIOError) as info:
        Recording.from_file(path, importer="edf")
    return info.value


# --- Gapped files are refused ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "gaps"),
    [
        ("edfd_gap_8s.edf", [(2.0, 8.0)]),
        ("edfd_gap_1s.edf", [(2.0, 1.0)]),
        ("bdfd_gap_8s.bdf", [(2.0, 8.0)]),
    ],
)
def test_importer_refuses_gapped_file_and_names_the_gap(name, gaps):
    path = str(EXAMPLES / name)
    err = _refusal(path)
    assert err.code == "discontinuous_recording"
    assert [tuple(g) for g in err.gaps] == gaps
    assert err.filepath == path
    message = str(err)
    assert path in message
    for start, length in gaps:
        assert f"{length:g} s at {start:g} s" in message


@pytest.mark.parametrize("name", ["edfd_gap_8s.edf", "edfd_gap_1s.edf", "bdfd_gap_8s.bdf"])
def test_streaming_export_refuses_gapped_file_and_writes_no_store(name, tmp_path):
    store = tmp_path / "store.zarr"
    with pytest.raises(BiosigIOError) as info:
        stream_to_zarr(str(EXAMPLES / name), str(store), dtype="float32", min_view_samples=8)
    assert info.value.code == "discontinuous_recording"
    assert str(EXAMPLES / name) in str(info.value)
    assert not store.exists()


def test_every_gap_is_named_in_order(tmp_path):
    # Records at 0, 1, 5, 6, 9: gaps of 3 s after t=2 and 2 s after t=7.
    path = write_edf_plus(tmp_path / "many.edf", [0, 1, 5, 6, 9], label="D")
    err = _refusal(str(path))
    assert [tuple(g) for g in err.gaps] == [(2.0, 3.0), (7.0, 2.0)]
    assert "3 s at 2 s" in str(err)
    assert "2 s at 7 s" in str(err)


def test_long_gap_list_is_kept_whole_but_summarized_in_the_message(tmp_path):
    onsets = [2 * i for i in range(31)]  # 30 one-second gaps
    path = write_edf_plus(tmp_path / "lots.edf", onsets, label="D")
    err = _refusal(str(path))
    assert len(err.gaps) == 30
    assert "and 20 more" in str(err)


def test_gap_is_measured_against_the_header_record_duration(tmp_path):
    # 0.5 s records at 0, 0.5, 1.5: the third starts 0.5 s after the second ends.
    path = write_edf_plus(tmp_path / "half.edf", [0, 0.5, 1.5], label="D", record_duration=0.5)
    err = _refusal(str(path))
    assert [tuple(g) for g in err.gaps] == [(1.0, 0.5)]
    assert err.record_duration == 0.5


def test_overlapping_or_backward_records_are_refused_too(tmp_path):
    path = write_edf_plus(tmp_path / "back.edf", [0, 1, 1.5, 3], label="D")
    err = _refusal(str(path))
    assert [tuple(g) for g in err.gaps] == [(2.0, -0.5), (2.5, 0.5)]


def test_gap_is_found_when_another_defect_sends_the_file_to_the_fallback_first(tmp_path):
    """pyedflib reports the first defect it meets. A NUL-padded numeric header field
    is another, unrelated, reason to take the fallback; the gap must still be found."""
    path = _copy("edfd_gap_8s.edf", tmp_path)
    with open(path, "r+b") as f:
        f.seek(236)  # the "number of data records" field
        f.write(b"4\x00\x00\x00\x00\x00\x00\x00")
    err = _refusal(path)
    assert [tuple(g) for g in err.gaps] == [(2.0, 8.0)]


def test_gap_is_found_when_the_header_record_count_is_unknown(tmp_path):
    path = _copy("edfd_gap_8s.edf", tmp_path)
    with open(path, "r+b") as f:
        f.seek(236)
        f.write(b"-1      ")
    with pytest.raises(BiosigIOError) as info:
        _verify(path)
    assert info.value.code == "discontinuous_recording"
    assert [tuple(g) for g in info.value.gaps] == [(2.0, 8.0)]


def test_unreadable_time_keeping_annotation_cannot_be_shown_contiguous(tmp_path):
    path = _copy("edfd_contiguous.edf", tmp_path)
    layout_record = 2 * (10 + 40)  # bytes per record in the fixture
    first_annotation = 768 + layout_record + 2 * 10  # record 1, after its 10 samples
    with open(path, "r+b") as f:
        f.seek(first_annotation)
        f.write(b"\x00" * 80)
    with pytest.raises(FileReadError) as info:
        Recording.from_file(path, importer="edf")
    assert info.value.code != "discontinuous_recording"
    assert "record 1" in str(info.value)


def test_edfd_file_without_an_annotations_signal_cannot_be_shown_contiguous(tmp_path):
    path = _copy("edfd_contiguous.edf", tmp_path)
    with open(path, "r+b") as f:
        f.seek(256 + 16)  # the second signal's label
        f.write(b"Other".ljust(16, b" "))
    # pyedflib itself refuses such a file as corrupt before the fallback is reached,
    # so the check is called directly: it must not guess a start time for the records.
    with pytest.raises(FileReadError, match="no annotations signal"):
        _verify(path)


# --- Tolerance ----------------------------------------------------------------------


def test_onset_jitter_below_the_tolerance_is_not_a_gap(tmp_path):
    path = write_edf_plus(
        tmp_path / "jitter.edf", [0, 1, 2.0000001, 3], label="D", onset_format=".7f"
    )  # 0.1 us off, written to 7 decimals
    assert b"+2.0000001" in Path(path).read_bytes()
    assert _verify(str(path)) is True


def test_onset_error_above_the_tolerance_is_a_gap(tmp_path):
    path = write_edf_plus(
        tmp_path / "off.edf", [0, 1, 2.00003, 3], label="D", onset_format=".5f"
    )  # 30 us off, written to 5 decimals: beyond the 10 us those decimals can round away
    err = _refusal(str(path))
    assert len(err.gaps) == 2  # the record after it starts 30 us early, then 30 us late
    assert err.gaps[0].length == pytest.approx(3e-5)


def test_coarsely_rounded_onsets_of_a_non_round_record_duration_are_not_gaps(tmp_path):
    """One third of a second per record, onsets rounded to milliseconds: every pair is
    off the exact 1/3 s by up to a millisecond, which is rounding, not a gap."""
    path = write_edf_plus(
        tmp_path / "thirds.edf",
        [0.0, 0.333, 0.667, 1.0],
        label="D",
        record_duration=0.333333,
        onset_format=".3f",
    )
    assert _verify(str(path)) is True


def test_a_gap_larger_than_the_rounding_of_coarse_onsets_is_still_found(tmp_path):
    path = write_edf_plus(
        tmp_path / "coarse_gap.edf",
        [0.0, 0.333, 0.667, 1.5],
        label="D",
        record_duration=0.333333,
        onset_format=".3f",
    )
    err = _refusal(str(path))
    assert [round(g.length, 3) for g in err.gaps] == [0.5]


def test_the_tolerance_never_reaches_half_a_sample_period(tmp_path):
    """Onsets written as whole numbers could round away up to a second; the tolerance
    is held below half a sample period (0.05 s at 10 Hz), so a 0.2 s gap is found."""
    path = write_edf_plus(tmp_path / "capped.edf", [0, 1, 2.2, 3.2], label="D")
    err = _refusal(str(path))
    assert [round(g.length, 3) for g in err.gaps] == [0.2]


# --- Onsets as real acquisition software writes them --------------------------------


def test_plus_minus_zero_first_onset_contiguous_file_reads_and_is_flagged(tmp_path):
    path = write_edf_plus(tmp_path / "pm.edf", ["+-0.001", 0.999, 1.999, 2.999], label="D")
    assert b"+-0.001" in Path(path).read_bytes()
    rec = Recording.from_file(str(path), importer="edf")
    assert rec.metadata[VERIFIED_KEY] is True
    assert rec.metadata["edf_tolerant_read"] is True


def test_plus_minus_zero_first_onset_gapped_file_is_refused(tmp_path):
    path = write_edf_plus(tmp_path / "pm_gap.edf", ["+-0.001", 0.999, 3.999, 4.999], label="D")
    err = _refusal(str(path))
    assert [(round(g.start, 3), round(g.length, 3)) for g in err.gaps] == [(1.999, 2.0)]


def test_run_of_signs_is_read_with_the_last_sign_winning(tmp_path):
    # "-+1" is +1 s, "+-1" is -1 s: records at -1, 0, 1 are contiguous only if read so.
    path = write_edf_plus(tmp_path / "signs.edf", ["+-1", 0, "+-+1", 2], label="D")
    # Pairs: (-1, 0) ok; (0, +1) ok; (+1, 2) ok.
    assert _verify(str(path)) is True


def test_unparseable_onset_in_a_later_record_is_a_refusal(tmp_path):
    path = write_edf_plus(tmp_path / "bad.edf", [0, 1, "+x", 3], label="D")
    with pytest.raises(FileReadError, match="record 2"):
        _verify(str(path))


# --- Which annotations signal, which end of the file --------------------------------


def test_only_the_first_annotations_signal_keeps_time(tmp_path):
    """EDF+ puts the time-keeping TAL in the FIRST annotations signal. Here the second
    one looks contiguous (0, 1, 2, 3) while the first has the 8 s gap."""
    path = write_edf_plus(
        tmp_path / "two_ann.edf", [0, 1, 10, 11], label="D", second_annotation_onsets=[0, 1, 2, 3]
    )
    err = _refusal(str(path))
    assert [tuple(g) for g in err.gaps] == [(2.0, 8.0)]


def test_a_second_annotations_signal_does_not_hide_contiguity_either(tmp_path):
    path = write_edf_plus(
        tmp_path / "two_ann_ok.edf",
        [0, 1, 2, 3],
        label="D",
        second_annotation_onsets=[0, 1, 10, 11],  # would be a gap if it were read
    )
    assert _verify(str(path)) is True


def test_a_gap_before_the_last_record_is_found(tmp_path):
    path = write_edf_plus(tmp_path / "late_gap.edf", [0, 1, 2, 5], label="D")
    err = _refusal(str(path))
    assert [tuple(g) for g in err.gaps] == [(3.0, 2.0)]


def test_gapped_bdfd_is_found_when_another_defect_sends_it_to_the_fallback_first(tmp_path):
    path = _copy("bdfd_gap_8s.bdf", tmp_path)
    with open(path, "r+b") as f:
        f.seek(236)
        f.write(b"4\x00\x00\x00\x00\x00\x00\x00")
    err = _refusal(path)
    assert [tuple(g) for g in err.gaps] == [(2.0, 8.0)]


def test_sample_width_comes_from_the_header_not_the_file_name(tmp_path):
    """A BDF+D recording named .edf is still laid out with 3-byte samples."""
    gapped = tmp_path / "bdf_named_edf.edf"
    shutil.copy(EXAMPLES / "bdfd_gap_8s.bdf", gapped)
    with pytest.raises(BiosigIOError) as info:
        _verify(str(gapped))
    assert [tuple(g) for g in info.value.gaps] == [(2.0, 8.0)]

    contiguous = write_edf_plus(tmp_path / "bdf_contiguous.edf", [0, 1, 2, 3], label="D", bdf=True)
    assert _verify(str(contiguous)) is True


def test_reserved_field_is_matched_without_regard_to_case(tmp_path):
    path = _copy("edfd_gap_8s.edf", tmp_path)
    with open(path, "r+b") as f:
        f.seek(192)
        f.write(b"edf+d".ljust(44, b" "))
    with pytest.raises(BiosigIOError) as info:
        _verify(path)
    assert info.value.code == "discontinuous_recording"


def test_records_without_samples_cannot_be_shown_contiguous(tmp_path):
    path = _copy("edfd_contiguous.edf", tmp_path)
    with open(path, "r+b") as f:
        f.seek(256 + 2 * (16 + 80 + 8 * 5 + 80))  # samples per record, both signals
        f.write(b"0       0       ")
        f.seek(236)
        f.write(b"-1      ")
    with pytest.raises(FileReadError, match="no samples"):
        _verify(path)


# --- A +D file whose records are contiguous stays readable -------------------------


def test_importer_reads_contiguous_edfd_and_records_that_it_was_verified():
    rec = Recording.from_file(str(EXAMPLES / "edfd_contiguous.edf"), importer="edf")
    assert rec.metadata["edf_tolerant_read"] is True
    assert rec.metadata["edf_tolerant_read_reason"] == "discontinuous_datarecords"
    assert rec.metadata[VERIFIED_KEY] is True
    # Every sample sits on its true time: the ramp equals sample_index / FS.
    values = np.asarray(rec.signals["EEG Ramp"].to_numpy(), dtype=float)
    np.testing.assert_allclose(values, np.arange(len(values)) / FS, atol=1e-6)
    assert len(values) == 40
    assert rec.events["description"].tolist() == ["E1", "E2"]
    np.testing.assert_allclose(rec.events["onset"].to_numpy(), [0.5, 2.5])


def test_streaming_export_reads_contiguous_edfd_and_records_that_it_was_verified(tmp_path):
    store = stream_to_zarr(
        str(EXAMPLES / "edfd_contiguous.edf"),
        str(tmp_path / "store.zarr"),
        dtype="float32",
        min_view_samples=8,
    )
    root = zarr.open(store, mode="r")
    metadata = root.attrs["recording_metadata"]
    assert metadata["edf_tolerant_read"] is True
    assert metadata[VERIFIED_KEY] is True


def test_file_not_labeled_discontinuous_is_read_as_before():
    rec = Recording.from_file(str(EXAMPLES / "edfc_control.edf"), importer="edf")
    assert "edf_tolerant_read" not in rec.metadata
    assert VERIFIED_KEY not in rec.metadata
    values = np.asarray(rec.signals["EEG Ramp"].to_numpy(), dtype=float)
    np.testing.assert_allclose(values, np.arange(len(values)) / FS, atol=1e-6)
    assert _verify(str(EXAMPLES / "edfc_control.edf")) is False


# --- The error itself ---------------------------------------------------------------


def test_the_error_survives_pickling():
    """Callers that convert in worker processes get the error back across a pipe."""
    import pickle

    from biosigio.exceptions import DiscontinuousRecordingError, TimeGap

    err = DiscontinuousRecordingError("/data/x.edf", [TimeGap(2.0, 8.0), TimeGap(12.0, 1.5)], 1.0)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is DiscontinuousRecordingError
    assert back.filepath == "/data/x.edf"
    assert back.gaps == err.gaps
    assert back.record_duration == 1.0
    assert str(back) == str(err)
    assert back.code == "discontinuous_recording"


def test_the_message_counts_and_names_gaps_or_overlaps():
    from biosigio.exceptions import DiscontinuousRecordingError, TimeGap

    one = str(DiscontinuousRecordingError("f.edf", [TimeGap(2.0, 8.0)], 1.0))
    two = str(DiscontinuousRecordingError("f.edf", [TimeGap(2.0, 8.0), TimeGap(5.0, -1.0)], 1.0))
    assert "has 1 gap or overlap between" in one
    assert "has 2 gaps or overlaps between" in two
    assert "-1 s at 5 s" in two
    assert "file's own time axis" in one


def _all_error_classes():
    from biosigio import exceptions

    seen, todo = [], [exceptions.BiosigIOError]
    while todo:
        cls = todo.pop()
        seen.append(cls)
        todo.extend(cls.__subclasses__())
    return seen


@pytest.mark.parametrize("cls", _all_error_classes(), ids=lambda c: c.__name__)
def test_every_error_code_has_a_default_reason(cls):
    from biosigio.exceptions import REASONS

    assert cls.code in REASONS


def test_onset_with_an_absurd_number_of_digits_is_unreadable_not_a_crash():
    """Python refuses to turn a digit string of several thousand characters into an
    int; that is an onset that cannot be read, not a ValueError out of the verifier."""
    from biosigio.importers._edf_tolerant import _lenient_timekeeping_onset

    assert _lenient_timekeeping_onset(b"+" + b"1" * 5000 + b"\x14\x14\x00") is None
    assert _lenient_timekeeping_onset(b"+12.5\x14\x14\x00") == (125, 1)


def test_a_file_with_no_data_signal_has_its_tolerance_held_to_half_a_record():
    from decimal import Decimal

    from biosigio.importers._edf_tolerant import _pair_tolerance

    # Whole-number onsets round by up to a second; with no sample period to cap it,
    # half of a 2 s record is the ceiling.
    assert _pair_tolerance(0, 0, None, Decimal("2")) == Decimal("1")
    # With a sample period the smaller of the two caps wins.
    assert _pair_tolerance(0, 0, Decimal("0.05"), Decimal("2")) == Decimal("0.05")
