"""EDF files that repeat a channel label (NEMAR nm000110, CHB-MIT).

EDF does not require unique labels: CHB-MIT declares ``T8-P8`` twice and uses
``-`` as a placeholder for several unused inputs. A Recording is keyed by
label, so before this fix the second ``T8-P8`` silently replaced the first and
the store came up short of the dataset's ``channels.tsv``. The importer now
suffixes repeats the way MNE does (``T8-P8-0``, ``T8-P8-1``), which is the
naming the dataset's MNE-BIDS ``channels.tsv`` already uses.

These tests write REAL EDF files with pyedflib (no mocks).
"""

import os

import numpy as np
import pyedflib
import pytest

from biosigio import Recording
from biosigio.importers.edf import _unique_channel_labels

# The CHB-MIT montage shape, trimmed: T8-P8 twice, three "-" and two "."
# placeholders, and one unique channel between the repeats.
DUP_LABELS = ["Fp1-F7", "T8-P8", "-", "FT10-T8", "T8-P8", "-", ".", "-", ".", "VNS"]
DUP_EXPECTED = [
    "Fp1-F7",
    "T8-P8-0",
    "--0",
    "FT10-T8",
    "T8-P8-1",
    "--1",
    ".-0",
    "--2",
    ".-1",
    "VNS",
]


def _write_edf(path: str, labels: list[str], rate: float = 256.0, seconds: int = 2) -> np.ndarray:
    """Write a real EDF with the given labels; return the (n_ch, n) signal."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    data = np.vstack([(20.0 + 3 * c) * np.sin(2 * np.pi * (2 + c) * t) for c in range(len(labels))])
    headers = [
        {
            "label": label,
            "dimension": "uV",
            "sample_frequency": rate,
            "physical_max": 100.0,
            "physical_min": -100.0,
            "digital_max": 32767,
            "digital_min": -32768,
            "prefilter": "n/a",
            "transducer": "n/a",
        }
        for label in labels
    ]
    writer = pyedflib.EdfWriter(path, len(labels))
    try:
        writer.setSignalHeaders(headers)
        writer.writeSamples(list(data))
    finally:
        writer.close()
    return data


def test_duplicate_labels_keep_every_channel_with_mne_suffixes(tmp_path):
    path = os.path.join(tmp_path, "dup.edf")
    data = _write_edf(path, DUP_LABELS)

    rec = Recording.from_file(path)

    with pyedflib.EdfReader(path) as reader:
        n_header = reader.signals_in_file
    assert n_header == len(DUP_LABELS)
    assert list(rec.channels) == DUP_EXPECTED
    assert list(rec.signals.columns) == DUP_EXPECTED
    assert len(rec.channels) == n_header
    # Each suffixed channel carries its own on-disk signal, in header order.
    for i, label in enumerate(DUP_EXPECTED):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], atol=0.01)
        assert rec.channels[label]["physical_min"] == -100.0


def test_suffixes_match_mne(tmp_path):
    mne = pytest.importorskip("mne", reason="MNE parity check needs the 'meg' extra")
    path = os.path.join(tmp_path, "dup.edf")
    _write_edf(path, DUP_LABELS)

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")

    assert list(Recording.from_file(path).channels) == raw.ch_names


def test_unique_labels_are_unchanged(tmp_path):
    labels = ["Fp1-F7", "F7-T7", "T7-P7", "EMG1"]
    path = os.path.join(tmp_path, "plain.edf")
    data = _write_edf(path, labels)

    rec = Recording.from_file(path)

    assert list(rec.channels) == labels
    for i, label in enumerate(labels):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], atol=0.01)


def test_suffix_that_collides_with_a_real_label_falls_through():
    # "A-0" already exists, so the first "A" becomes "A-a" (MNE's rule), and
    # "A-1" is free for the second.
    assert _unique_channel_labels(["A", "A-0", "A"]) == ["A-a", "A-0", "A-1"]


def test_stream_store_uses_the_same_suffixed_labels(tmp_path):
    pytest.importorskip("zarr", reason="streaming Zarr export requires the 'zarr' extra")
    pytest.importorskip("mne", reason="streaming Zarr export requires the 'meg' extra (mne)")
    import zarr

    from biosigio import stream_to_zarr

    path = os.path.join(tmp_path, "dup.edf")
    _write_edf(path, DUP_LABELS)
    store = stream_to_zarr(path, os.path.join(tmp_path, "s.zarr"), force_modality="EEG")

    root = zarr.open_group(store, mode="r")
    groups = [root[name] for name in root.group_keys() if "channels" in root[name].attrs]
    labels = [c["label"] for g in groups for c in g.attrs["channels"]]  # ty: ignore[not-iterable]
    assert labels == DUP_EXPECTED


def test_add_channel_refuses_a_duplicate_label():
    rec = Recording()
    rec.add_channel("T8-P8", np.zeros(10), 256, "uV", "EEG")

    with pytest.raises(ValueError, match="'T8-P8' already exists"):
        rec.add_channel("T8-P8", np.ones(10), 256, "uV", "EEG")

    # The original channel is untouched.
    assert list(rec.channels) == ["T8-P8"]
    assert float(rec.signals["T8-P8"].abs().sum()) == 0.0
