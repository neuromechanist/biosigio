"""BrainVision headers naming pre-BIDS-rename files (NEMAR on005520, on002158).

BIDS renames the ``.eeg``/``.vmrk`` on disk but not the ``DataFile=``/``MarkerFile=``
references inside the ``.vhdr``; the renamed files sit beside the header with its
stem. Both the in-memory importer and the streaming ``_MneSource`` path must read
such a triplet, leave a correct header alone, and still fail when no sibling exists.
Real tiny triplets are written by hand in each test. NO MOCKS.
"""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")

from biosigio import Recording  # noqa: E402
from biosigio.exporters.zarr_stream import _open_stream_source  # noqa: E402
from biosigio.importers.brainvision import resolved_vhdr  # noqa: E402

N_CH, N_SAMPLES, SFREQ = 3, 500, 250.0
STEM = "sub-01_task-MOBAgame_eeg"


def _write_triplet(
    directory,
    data_ref: str,
    marker_ref: str,
    *,
    data_ext: str = ".eeg",
    newline: str = "\n",
    codepage: str = "UTF-8",
    encoding: str = "utf-8",
) -> tuple[str, np.ndarray]:
    """Write ``<STEM>.vhdr`` referencing ``data_ref``/``marker_ref``, plus the
    correctly named ``<STEM><data_ext>`` and ``<STEM>.vmrk`` beside it.

    Returns the header path and the int16 data (channels x samples) written.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, size=(N_CH, N_SAMPLES), dtype=np.int16)
    data.T.tofile(os.path.join(directory, STEM + data_ext))  # MULTIPLEXED
    vmrk = [
        "Brain Vision Data Exchange Marker File, Version 1.0",
        "[Common Infos]",
        f"Codepage={codepage}",
        f"DataFile={data_ref}",
        "[Marker Infos]",
        "Mk1=Stimulus,S  1,125,1,0",
        "Mk2=Stimulus,S  2,250,1,0",
    ]
    with open(os.path.join(directory, STEM + ".vmrk"), "w", encoding=encoding, newline="") as f:
        f.write(newline.join(vmrk) + newline)
    channels = [f"Ch{i + 1}=C{i + 1}é,,0.1,µV" for i in range(N_CH)]
    vhdr = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; Écrit à la main",
        "[Common Infos]",
        f"Codepage={codepage}",
        f"DataFile={data_ref}",
        f"MarkerFile={marker_ref}",
        "DataFormat=BINARY",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={N_CH}",
        f"SamplingInterval={1e6 / SFREQ:g}",
        "[Binary Infos]",
        "BinaryFormat=INT_16",
        "[Channel Infos]",
        *channels,
    ]
    vhdr_path = os.path.join(directory, STEM + ".vhdr")
    with open(vhdr_path, "w", encoding=encoding, newline="") as f:
        f.write(newline.join(vhdr) + newline)
    return vhdr_path, data


def _snapshot(directory) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in pathlib.Path(directory).iterdir()}


def _assert_loaded(rec: Recording, data: np.ndarray) -> None:
    assert rec.signals is not None
    assert rec.signals.shape == (N_SAMPLES, N_CH)
    assert {i["sample_frequency"] for i in rec.channels.values()} == {SFREQ}
    # 0.1 uV resolution -> volts.
    np.testing.assert_allclose(rec.signals.to_numpy().T, data * 0.1e-6)
    assert rec.events is not None and len(rec.events) == 2
    np.testing.assert_allclose(rec.events["onset"].to_numpy(), [124 / SFREQ, 249 / SFREQ])


def _assert_streams(vhdr_path: str, data: np.ndarray) -> None:
    source = _open_stream_source(vhdr_path, None)
    try:
        assert source.sfreq == SFREQ
        assert source.n_samples == N_SAMPLES
        assert len(source.channels) == N_CH
        np.testing.assert_allclose(source.read(list(range(N_CH)), 0, N_SAMPLES), data * 0.1e-6)
        np.testing.assert_allclose(source.read([1], 100, 200), data[1:2, 100:200] * 0.1e-6)
    finally:
        source.close()


def test_matching_header_is_read_unchanged(tmp_path):
    vhdr, data = _write_triplet(tmp_path, f"{STEM}.eeg", f"{STEM}.vmrk")
    with resolved_vhdr(vhdr) as used:
        assert used == vhdr
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


@pytest.mark.parametrize(
    ("data_ref", "marker_ref"),
    [
        ("s13_run2_060717.eeg", "s13_run2_060717.vmrk"),
        ("sub-01_task-MOBA_game_eeg.eeg", "sub-01_task-MOBA_game_eeg.vmrk"),
        (f"{STEM}.eeg", "sub-01_task-MOBA_game_eeg.vmrk"),  # only the marker is stale
    ],
)
def test_stale_references_resolve_to_same_stem_siblings(tmp_path, data_ref, marker_ref):
    vhdr, data = _write_triplet(tmp_path, data_ref, marker_ref)
    before = _snapshot(tmp_path)
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)
    assert _snapshot(tmp_path) == before  # the dataset files are never touched


def test_legacy_dat_sibling_is_tolerated(tmp_path):
    vhdr, data = _write_triplet(tmp_path, "old_name.dat", "old_name.vmrk", data_ext=".dat")
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


@pytest.mark.parametrize(
    ("newline", "codepage", "encoding"),
    [("\r\n", "ANSI", "cp1252"), ("\n", "UTF-8", "utf-8")],
)
def test_patched_copy_keeps_encoding_and_line_endings(tmp_path, newline, codepage, encoding):
    vhdr, data = _write_triplet(
        tmp_path, "old.eeg", "old.vmrk", newline=newline, codepage=codepage, encoding=encoding
    )
    original = open(vhdr, "rb").read()
    with resolved_vhdr(vhdr) as used:
        assert used != vhdr
        patched = open(used, "rb").read()
    assert not os.path.exists(used)  # the temporary copy is cleaned up
    expected = original.replace(
        b"DataFile=old.eeg", f"DataFile={tmp_path / (STEM + '.eeg')}".encode(encoding)
    ).replace(b"MarkerFile=old.vmrk", f"MarkerFile={tmp_path / (STEM + '.vmrk')}".encode(encoding))
    assert patched == expected
    _assert_loaded(Recording.from_file(vhdr), data)


def test_missing_sibling_still_errors(tmp_path):
    vhdr, _ = _write_triplet(tmp_path, "s13_run2_060717.eeg", "s13_run2_060717.vmrk")
    os.remove(tmp_path / f"{STEM}.eeg")
    with pytest.raises(Exception, match="s13_run2_060717.eeg"):
        Recording.from_file(vhdr)
    with pytest.raises(FileNotFoundError, match="s13_run2_060717.eeg"):
        _open_stream_source(vhdr, None)
