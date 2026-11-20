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
    path = write_edf_plus(tmp_path / "jitter.edf", [0, 1, 2.0000001, 3], label="D")  # 0.1 us
    assert b"+2.0000001" in Path(path).read_bytes()
    assert _verify(str(path)) is True


def test_onset_error_above_the_tolerance_is_a_gap(tmp_path):
    path = write_edf_plus(tmp_path / "off.edf", [0, 1, 2.00001, 3], label="D")
    err = _refusal(str(path))
    assert len(err.gaps) == 2  # the record after it starts 10 us early, then 10 us late
    assert err.gaps[0].length == pytest.approx(1e-5)


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
