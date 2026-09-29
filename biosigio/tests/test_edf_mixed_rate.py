"""Mixed per-channel sampling-rate EDF handling (found converting NEMAR datasets).

EDF/BDF allow each signal its own sampling rate (polysomnography is the classic
case: EEG/EOG/EMG ~100-256 Hz alongside SpO2/respiration ~10-25 Hz). biosigIO
stores one uniform grid, so such a file cannot be loaded onto it as-is. The
importer must therefore either refuse loudly (default) or, on explicit opt-in,
upsample the slower channels onto the fastest channel's grid for a derived
serving copy.

These tests build REAL mixed-rate EDF files with pyedflib (no mocks).
"""

import os
import tempfile

import numpy as np
import pyedflib
import pytest

from biosigio import Recording


def _write_edf(path: str, signals: list[tuple[str, float, np.ndarray]]) -> None:
    """Write a real EDF with per-signal sampling rates.

    `signals` is a list of (label, sample_frequency, data); each data length must
    be sample_frequency * duration for one shared duration (whole datarecords).
    """
    headers = []
    data_list = []
    for label, sf, data in signals:
        headers.append(
            {
                "label": label,
                "dimension": "uV",
                "sample_frequency": sf,
                "physical_max": float(np.max(data)),
                "physical_min": float(np.min(data)),
                "digital_max": 32767,
                "digital_min": -32768,
                "prefilter": "n/a",
                "transducer": "n/a",
            }
        )
        data_list.append(data)
    writer = pyedflib.EdfWriter(path, len(signals))
    try:
        writer.setSignalHeaders(headers)
        writer.writeSamples(data_list)
    finally:
        writer.close()


def _mixed_edf(path: str) -> None:
    """2 s recording: EEG at 100 Hz (200 samples) + SpO2 at 25 Hz (50 samples)."""
    duration = 2.0
    t_fast = np.arange(int(100 * duration)) / 100.0
    t_slow = np.arange(int(25 * duration)) / 25.0
    eeg = 30.0 * np.sin(2 * np.pi * 10 * t_fast)  # 10 Hz, within both Nyquists
    spo2 = 95.0 + 2.0 * np.sin(2 * np.pi * 0.2 * t_slow)  # slow ~0.2 Hz drift
    _write_edf(path, [("EEG", 100.0, eeg), ("SpO2", 25.0, spo2)])


def _single_rate_edf(path: str) -> None:
    """2 s recording, both channels at 100 Hz (no mix)."""
    t = np.arange(200) / 100.0
    a = 30.0 * np.sin(2 * np.pi * 10 * t)
    b = 10.0 * np.cos(2 * np.pi * 5 * t)
    _write_edf(path, [("EEG1", 100.0, a), ("EEG2", 100.0, b)])


class _TmpEDF:
    """Context manager yielding a written EDF path, cleaned up after."""

    def __init__(self, builder):
        self.builder = builder

    def __enter__(self) -> str:
        fd, self.path = tempfile.mkstemp(suffix=".edf")
        os.close(fd)
        self.builder(self.path)
        return self.path

    def __exit__(self, *_exc) -> None:
        os.unlink(self.path)


def test_mixed_rate_errors_by_default():
    """A mixed-rate EDF refuses to load without an explicit opt-in."""
    with _TmpEDF(_mixed_edf) as path:
        with pytest.raises(ValueError, match="mixed per-channel sampling rates"):
            Recording.from_file(path)
        # The importer default matches from_file's default.
        with pytest.raises(ValueError, match="mixed per-channel sampling rates"):
            Recording.from_file(path, mixed_rate="error")


def test_mixed_rate_resample_unifies_grid():
    """`mixed_rate='resample'` lifts the slow channel onto the fast channel's grid."""
    with _TmpEDF(_mixed_edf) as path:
        rec = Recording.from_file(path, mixed_rate="resample")
        # Both channels now share one 100 Hz, 200-sample grid.
        assert rec.signals.shape == (200, 2)
        assert rec.get_sampling_frequency() == 100.0
        assert rec.channels["EEG"]["sample_frequency"] == 100.0
        assert rec.channels["SpO2"]["sample_frequency"] == 100.0
        # The fast channel is untouched (no original_sample_frequency stamped).
        assert "original_sample_frequency" not in rec.channels["EEG"]
        # The upsampled channel keeps its true native rate for provenance.
        assert rec.channels["SpO2"]["original_sample_frequency"] == 25.0
        assert rec.get_metadata("mixed_rate_resampled") is True
        assert rec.get_metadata("mixed_rate_target_hz") == 100.0


def test_resampled_slow_channel_preserves_amplitude():
    """Upsampling must preserve the slow channel's physical level, not corrupt it."""
    with _TmpEDF(_mixed_edf) as path:
        rec = Recording.from_file(path, mixed_rate="resample")
        spo2 = rec.signals["SpO2"].to_numpy()
        assert len(spo2) == 200
        # SpO2 oscillates around 95 with amplitude ~2; resampling keeps that range.
        assert 92.0 < float(np.mean(spo2)) < 98.0
        assert float(np.max(spo2)) < 98.5
        assert float(np.min(spo2)) > 91.5


def test_invalid_mixed_rate_value_rejected():
    """A typo'd policy is rejected up front (not silently treated as a default)."""
    with _TmpEDF(_mixed_edf) as path:
        with pytest.raises(ValueError, match="mixed_rate must be one of"):
            Recording.from_file(path, mixed_rate="resmaple")


def test_single_rate_edf_unaffected_by_policy():
    """A single-rate EDF loads identically regardless of the mixed_rate policy."""
    with _TmpEDF(_single_rate_edf) as path:
        rec_default = Recording.from_file(path)
        rec_resample = Recording.from_file(path, mixed_rate="resample")
        for rec in (rec_default, rec_resample):
            assert rec.signals.shape == (200, 2)
            assert rec.get_sampling_frequency() == 100.0
        # No resample happened, so no derived-view flag is stamped.
        assert rec_resample.get_metadata("mixed_rate_resampled") is None
        assert "original_sample_frequency" not in rec_resample.channels["EEG1"]


def test_resampled_recording_exports_to_zarr():
    """The real NEMAR use: a resampled mixed-rate recording exports to a Zarr store."""
    pytest.importorskip("zarr", reason="Zarr export requires the optional 'zarr' extra")
    with _TmpEDF(_mixed_edf) as path:
        rec = Recording.from_file(path, mixed_rate="resample")
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "rec.zarr")
            rec.to_zarr(store)
            assert os.path.exists(os.path.join(store, "zarr.json"))
