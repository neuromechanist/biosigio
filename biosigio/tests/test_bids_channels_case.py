"""A ``channels.tsv`` that differs from the data file only in case still applies (#136).

CHB-MIT's EDF headers say ``FP1-F7`` while a BIDS sidecar written for them says
``Fp1-F7``. Matching was exact, so such a row was skipped without a word and the
channel kept its header-guessed type and unit. A row with no exact match now
falls back to ignoring case when exactly one channel fits and nothing else
claims it, on the in-memory and the streaming path alike; an ambiguous row is
still left unapplied. Real EDF files and real sidecars throughout.
"""

import logging
import os

import numpy as np
import pytest

from biosigio import Recording
from biosigio.tests.test_bids_channels_units import write_channels_tsv
from biosigio.tests.test_edf_duplicate_labels import _write_edf

zarr = pytest.importorskip("zarr")
from biosigio.exporters.zarr_stream import stream_to_zarr  # noqa: E402

STEM = "sub-01_task-rest"


def _recording(tmp_path, labels, rows):
    path = os.path.join(tmp_path, f"{STEM}_eeg.edf")
    data = _write_edf(path, labels)
    write_channels_tsv(tmp_path, STEM, rows)
    return path, data


def _streamed(path, tmp_path) -> Recording:
    # One modality at the file's own rate, so the store is one group on one grid.
    store = stream_to_zarr(
        path,
        str(tmp_path / "streamed.zarr"),
        dtype="float32",
        force_modality="EEG",
        modality_rates={"EEG": 256},
    )
    return Recording.from_file(store)


def _report(store_or_rec) -> dict:
    return store_or_rec.metadata["channels_tsv_units"]


def test_case_only_difference_applies_in_memory(tmp_path, caplog):
    path, data = _recording(
        tmp_path,
        ["FP1-F7", "F7-T7", "T7-P7"],
        [("Fp1-F7", "EOG", "mV"), ("F7-T7", "EMG", "uV"), ("t7-p7", "ECG", "uV")],
    )

    with caplog.at_level(logging.INFO, logger="root"):
        rec = Recording.from_file(path)

    assert list(rec.channels) == ["FP1-F7", "F7-T7", "T7-P7"]  # header labels kept
    assert rec.channels["FP1-F7"]["channel_type"] == "EOG"
    assert rec.channels["FP1-F7"]["physical_dimension"] == "mV"
    np.testing.assert_allclose(rec.signals["FP1-F7"].to_numpy(), data[0] * 1e-3, atol=1e-5)
    assert rec.channels["F7-T7"]["channel_type"] == "EMG"
    assert rec.channels["T7-P7"]["channel_type"] == "ECG"
    assert _report(rec)["matched_case_insensitive"] == {"Fp1-F7": "FP1-F7", "t7-p7": "T7-P7"}
    assert _report(rec)["converted"] == 1
    assert "only when case is ignored" in caplog.text


def test_case_only_difference_applies_when_streaming(tmp_path):
    path, _ = _recording(
        tmp_path,
        ["FP1-F7", "F7-T7"],
        [("Fp1-F7", "EOG", "mV"), ("F7-T7", "EMG", "uV")],
    )

    in_memory = Recording.from_file(path)
    streamed = _streamed(path, tmp_path)

    root = dict(zarr.open_group(str(tmp_path / "streamed.zarr"), mode="r").attrs)
    assert root["channels_tsv_units"] == _report(in_memory)
    assert root["channels_tsv_units"]["matched_case_insensitive"] == {"Fp1-F7": "FP1-F7"}
    assert streamed.channels["FP1-F7"]["channel_type"] == "EOG"
    assert streamed.channels["FP1-F7"]["physical_dimension"] == "mV"
    np.testing.assert_allclose(
        streamed.signals["FP1-F7"].to_numpy(), in_memory.signals["FP1-F7"].to_numpy(), atol=1e-4
    )


def test_exact_match_wins_over_a_case_match(tmp_path):
    """A row naming the channel exactly claims it; a case-variant row is not applied."""
    path, _ = _recording(
        tmp_path,
        ["FP1-F7", "Cz"],
        [("FP1-F7", "EOG", "uV"), ("Fp1-F7", "EMG", "uV"), ("Cz", "EEG", "uV")],
    )
    for rec in (Recording.from_file(path), _streamed(path, tmp_path)):
        assert rec.channels["FP1-F7"]["channel_type"] == "EOG"
    assert "matched_case_insensitive" not in _report(Recording.from_file(path))


@pytest.mark.parametrize(
    ("labels", "rows"),
    [
        # Two channels fold to the row's name: which one it means is unknowable.
        (["FP1-F7", "Fp1-f7", "Cz"], [("fp1-F7", "EOG", "mV"), ("Cz", "EEG", "uV")]),
        # Two rows fold to one channel: which one describes it is unknowable.
        (["FP1-F7", "Cz"], [("Fp1-F7", "EOG", "mV"), ("fp1-f7", "EMG", "mV"), ("Cz", "EEG", "uV")]),
    ],
    ids=["two-channels", "two-rows"],
)
def test_ambiguous_case_match_is_not_applied(tmp_path, caplog, labels, rows):
    path, data = _recording(tmp_path, labels, rows)

    with caplog.at_level(logging.WARNING, logger="root"):
        in_memory = Recording.from_file(path)
    streamed = _streamed(path, tmp_path)

    for rec in (in_memory, streamed):
        for label in labels[:-1]:
            assert rec.channels[label]["channel_type"] != "EOG"
            assert rec.channels[label]["physical_dimension"] == "uV"
        assert rec.channels["Cz"]["channel_type"] == "EEG"
    np.testing.assert_allclose(in_memory.signals[labels[0]].to_numpy(), data[0], atol=0.01)
    assert "matched_case_insensitive" not in _report(in_memory)
    assert "ambiguous when case is ignored" in caplog.text
