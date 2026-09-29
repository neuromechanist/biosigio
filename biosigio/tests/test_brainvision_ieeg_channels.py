"""Channel-count fidelity for BrainVision iEEG through the Zarr serving export (issue #120).

NEMAR reported that every store built from ``nm000182``/``nm000183`` carried
``n_channels: 1`` and suspected the read or export path of collapsing the channel
dimension (the failure mode of #110/#111 on the EEGLAB importer). It is not a
collapse: those two datasets are Nejedly et al. 2020 segmented graphoelement
clips, and each BIDS run holds clips from exactly ONE source SEEG contact, so the
``.vhdr`` itself declares ``NumberOfChannels=1``. One channel in, one channel out
is the faithful answer there.

These tests pin both halves of that so a future regression is unambiguous:
a genuinely single-channel recording must stay at one channel (no phantom
padding), and a multi-channel recording in the SAME pybv header shape must keep
every channel on both export paths (in-memory ``to_zarr`` and ``stream_to_zarr``).
Had the collapse been real, the multi-channel cases below would fail.

NO MOCKS: the fixtures are real BrainVision triples written to disk in the exact
shape pybv 0.7.6 produced for those datasets (IEEE_FLOAT_32, MULTIPLEXED,
``SamplingInterval=200.0``, per-channel resolution in µV), read back through the
real MNE-backed importer.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from biosigio import Recording

pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")
zarr = pytest.importorskip("zarr", reason="Zarr serving format requires the optional 'zarr' extra")

from biosigio import stream_to_zarr  # noqa: E402

# The per-modality rate caps the NEMAR converter passes to every export; IEEG is
# capped at 1000 Hz, so a 5000 Hz recording lands in an `ieeg_1000hz` group.
MODALITY_RATES = {"EEG": 250, "MEG": 250, "IEEG": 1000, "EMG": 1000}
SFREQ = 5000.0
RESOLUTION = 0.1  # µV per stored unit, as pybv wrote it for nm000182/nm000183


def _write_brainvision(
    directory: pathlib.Path, stem: str, data_uv: np.ndarray, sfreq: float = SFREQ
) -> pathlib.Path:
    """Write a real BrainVision triple in the pybv 0.7.6 shape; return the .vhdr.

    ``data_uv`` is ``(n_channels, n_samples)`` in µV. pybv stores
    ``value / resolution`` (see ``pybv.io._write_bveeg_file``), so the reader
    recovers µV by multiplying the raw float back by the declared resolution;
    writing the values verbatim instead would bake in a silent 10x error.
    """
    n_ch, n_samples = data_uv.shape
    labels = [f"LMacro_{i + 1:02d}" for i in range(n_ch)]

    # MULTIPLEXED = ch1pt1, ch2pt1, ...: column-major ravel of (n_ch, n_samples).
    (data_uv / RESOLUTION).astype("<f4").ravel(order="F").tofile(directory / f"{stem}.eeg")

    channel_infos = "\n".join(
        f"Ch{i + 1}={label},,{RESOLUTION},µV" for i, label in enumerate(labels)
    )
    vhdr = directory / f"{stem}.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "; Written using pybv 0.7.6\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={stem}.eeg\n"
        f"MarkerFile={stem}.vmrk\n"
        "DataFormat=BINARY\n"
        "; Data orientation: MULTIPLEXED=ch1,pt1, ch2,pt1 ...\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={n_ch}\n"
        "; Sampling interval in microseconds\n"
        f"SamplingInterval={1e6 / sfreq}\n\n"
        "[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n\n"
        "[Channel Infos]\n"
        f"{channel_infos}\n\n"
        "[Comment]\n",
        encoding="utf-8",
    )
    (directory / f"{stem}.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={stem}.eeg\n\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20200101000000000000\n",
        encoding="utf-8",
    )
    return vhdr


def _signal(n_ch: int, n_samples: int, sfreq: float = SFREQ) -> np.ndarray:
    """Distinct per-channel sine in µV, so a dropped or duplicated row is visible."""
    t = np.arange(n_samples) / sfreq
    return np.vstack([(20.0 + 10 * c) * np.sin(2 * np.pi * (3 + c) * t) for c in range(n_ch)])


def _serving_group(store: str) -> tuple[str, dict]:
    """Return the (name, attrs) of the store's single signal group."""
    root = zarr.open_group(store=zarr.storage.LocalStore(store), mode="r")
    groups = {
        name: dict(group.attrs)
        for name, group in root.groups()
        if dict(group.attrs).get("n_channels") is not None
    }
    assert len(groups) == 1, f"expected one signal group, got {sorted(groups)}"
    return next(iter(groups.items()))


def _convert_like_nemar(vhdr: pathlib.Path, store: pathlib.Path) -> tuple[str, dict]:
    """The in-memory path of the NEMAR converter: import, force the BIDS datatype
    onto every channel, export int16 under the modality rate caps."""
    rec = Recording.from_file(str(vhdr), mixed_rate="resample")
    for label in rec.channels:
        rec.channels[label]["modality"] = "ieeg"
    rec.to_zarr(str(store), dtype="int16", modality_rates=MODALITY_RATES)
    return _serving_group(str(store))


def test_single_channel_brainvision_ieeg_imports_one_channel(tmp_path):
    """The nm000182/nm000183 shape: NumberOfChannels=1 imports as exactly one channel."""
    data = _signal(1, 10000)  # 2 s @ 5000 Hz
    vhdr = _write_brainvision(tmp_path, "sub-000_task-clips_run-001_ieeg", data)

    rec = Recording.from_file(str(vhdr), mixed_rate="resample")

    assert len(rec.channels) == 1
    assert list(rec.signals.columns) == ["LMacro_01"]
    assert rec.signals.shape == (10000, 1)
    assert rec.channels["LMacro_01"]["sample_frequency"] == SFREQ


def test_single_channel_brainvision_ieeg_exports_one_channel(tmp_path):
    """One channel in, one channel out: `n_channels: 1` is faithful, not a collapse.

    This is the exact store shape issue #120 reported as a defect; it reproduces
    from a one-channel source, which is what those recordings actually contain.
    """
    data = _signal(1, 10000)
    vhdr = _write_brainvision(tmp_path, "sub-000_task-clips_run-001_ieeg", data)

    name, attrs = _convert_like_nemar(vhdr, tmp_path / "one.zarr")

    assert name == "ieeg_1000hz"
    assert attrs["n_channels"] == 1
    assert attrs["n_samples"] == 2000  # 2 s at the 1000 Hz IEEG cap
    assert [c["label"] for c in attrs["channels"]] == ["LMacro_01"]


def test_multi_channel_brainvision_ieeg_keeps_every_channel(tmp_path):
    """Same header shape with 10 channels keeps all 10 -- the guard against a real collapse."""
    data = _signal(10, 10000)
    vhdr = _write_brainvision(tmp_path, "sub-000_task-clips_run-002_ieeg", data)

    rec = Recording.from_file(str(vhdr), mixed_rate="resample")
    assert rec.signals.shape == (10000, 10)

    name, attrs = _convert_like_nemar(vhdr, tmp_path / "many.zarr")

    assert name == "ieeg_1000hz"
    assert attrs["n_channels"] == 10
    assert [c["label"] for c in attrs["channels"]] == [f"LMacro_{i + 1:02d}" for i in range(10)]


def test_multi_channel_brainvision_ieeg_streaming_keeps_every_channel(tmp_path):
    """The bounded-memory path takes recordings over the size threshold (some of
    nm000183's runs), so it needs the same guarantee as the in-memory path."""
    data = _signal(10, 10000)
    vhdr = _write_brainvision(tmp_path, "sub-000_task-clips_run-003_ieeg", data)
    store = tmp_path / "stream.zarr"

    stream_to_zarr(
        str(vhdr),
        str(store),
        force_modality="ieeg",
        modality_rates=MODALITY_RATES,
        dtype="int16",
    )

    name, attrs = _serving_group(str(store))
    assert name == "ieeg_1000hz"
    assert attrs["n_channels"] == 10
    assert [c["label"] for c in attrs["channels"]] == [f"LMacro_{i + 1:02d}" for i in range(10)]


def test_brainvision_resolution_is_applied_once(tmp_path):
    """The declared per-channel resolution is honoured exactly once on read.

    Dropping it (or applying it twice) leaves the waveform intact but the
    amplitude off by 10x, which no channel-count check would catch.
    """
    data = _signal(2, 5000)
    vhdr = _write_brainvision(tmp_path, "sub-000_task-clips_run-004_ieeg", data)

    rec = Recording.from_file(str(vhdr), mixed_rate="resample")

    # No sibling channels.tsv, so the importer reports MNE's native SI volts.
    assert {i["physical_dimension"] for i in rec.channels.values()} == {"V"}
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(
            rec.signals[label].to_numpy(), data[i] * 1e-6, rtol=1e-5, atol=1e-12
        )
