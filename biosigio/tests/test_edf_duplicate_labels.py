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
from biosigio.importers._labels import unique_channel_labels

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
    assert unique_channel_labels(["A", "A-0", "A"]) == ["A-a", "A-0", "A-1"]


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


def test_zarr_store_with_duplicate_labels_reimports(tmp_path, caplog):
    """A store the streaming path published before this fix still loads.

    Such a store lists the same label on two rows (the stuck NEMAR stores are
    exactly that). The store is written for real and its channel rows are then
    edited on disk to the pre-fix shape, the way the fallback tests byte-patch
    an EDF header; re-import suffixes the repeat and warns instead of refusing.
    """
    zarr = pytest.importorskip("zarr", reason="Zarr serving format requires the 'zarr' extra")
    t = np.arange(1000)
    rec = Recording()
    rec.add_channel("T8-P8", np.sin(t / 5.0), 250, "uV", "EEG")
    rec.add_channel("FT10-T8", np.cos(t / 7.0), 250, "uV", "EEG")
    rec.add_channel("T8-P8x", np.sin(t / 11.0), 250, "uV", "EEG")
    store = rec.to_zarr(str(tmp_path / "pre_fix"), dtype="float32")

    grp = zarr.open_group(store=zarr.storage.LocalStore(store), mode="r+")["eeg_250hz"]
    rows = [dict(row) for row in grp.attrs["channels"]]  # ty: ignore[not-iterable]
    rows[2]["label"] = "T8-P8"
    grp.attrs["channels"] = rows

    with caplog.at_level("WARNING", logger="biosigio.importers._labels"):
        rt = Recording.from_file(store)

    assert list(rt.channels) == ["T8-P8-0", "FT10-T8", "T8-P8-1"]
    assert "Zarr store channel labels are not unique" in caplog.text
    # Each suffixed channel still reads its own row of the store.
    signals = rec.signals
    assert signals is not None
    np.testing.assert_allclose(rt.signals["T8-P8-0"].to_numpy(), signals["T8-P8"].to_numpy())
    np.testing.assert_allclose(rt.signals["T8-P8-1"].to_numpy(), signals["T8-P8x"].to_numpy())


def test_tolerant_fallback_keeps_duplicate_labels(tmp_path):
    """A duplicate-label file that also needs the tolerant fallback loads.

    MNE renames the repeats ``EEG-0``/``EEG-1``; the fallback pairs header rows
    by position, so each repeat keeps its own row's scaling and data, and the
    labels match what a normal (pyedflib) read of the same file produces. The
    file is EDF+ with an annotation, so the hidden annotations channel sits in
    the header too and must be skipped when pairing.
    """
    pytest.importorskip("mne", reason="the tolerant fallback reads through MNE")
    from biosigio.tests.test_edf_fallback import _channel, _patch_signal_field
    from biosigio.tests.test_edf_fallback import _write_edf as _write_edf_plus

    path = os.path.join(tmp_path, "dup_degenerate.edf")
    rng = np.random.default_rng(133)
    n = 300
    channels = [
        _channel("EEG", 100.0, -200.0, 200.0),
        _channel("REF", 100.0, -200.0, 200.0),
        _channel("EEG", 100.0, -500.0, 500.0),
    ]
    data = [rng.uniform(-200, 200, n), rng.uniform(-200, 200, n), rng.uniform(-500, 500, n)]
    _write_edf_plus(path, channels, data, annotations=[(0.5, 0.0, "stim")])

    # Normal read first, while the file is still compliant: the reference
    # labels and the ground-truth samples by header position.
    normal = Recording.from_file(path)
    assert list(normal.channels) == ["EEG-0", "REF", "EEG-1"]
    with pyedflib.EdfReader(path) as reader:
        truth = [reader.readSignal(i) for i in range(reader.signals_in_file)]

    _patch_signal_field(path, 1, "physical_max", b"-200")
    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)

    rec = Recording.from_file(path, importer="edf")

    assert rec.metadata["edf_tolerant_read"] is True
    assert list(rec.channels) == ["EEG-0", "REF", "EEG-1"]
    np.testing.assert_allclose(rec.signals["EEG-0"].to_numpy(), truth[0], rtol=1e-9)
    np.testing.assert_allclose(rec.signals["EEG-1"].to_numpy(), truth[2], rtol=1e-9)
    assert rec.channels["EEG-0"]["physical_max"] == 200.0
    assert rec.channels["EEG-1"]["physical_max"] == 500.0
    np.testing.assert_array_equal(rec.signals["REF"].to_numpy(), np.full(n, -200.0))
    assert list(rec.events["description"]) == ["stim"]

    # The streaming source goes through the same fallback and names alike.
    from biosigio.exporters.zarr_stream import _EdfSource

    source = _EdfSource(path, force_modality="EEG")
    try:
        assert [c["label"] for c in source.channels] == ["EEG-0", "REF", "EEG-1"]
    finally:
        source.close()


def test_more_than_26_repeats_use_running_numbers(tmp_path):
    # The letter fallback only covers a collision; the running number itself
    # is unbounded, so 30 "-" placeholders become "--0" .. "--29".
    labels = ["-"] * 30 + ["VNS"]
    path = os.path.join(tmp_path, "many.edf")
    data = _write_edf(path, labels, rate=64.0)

    rec = Recording.from_file(path)

    expected = [f"--{i}" for i in range(30)] + ["VNS"]
    assert list(rec.channels) == expected
    # Channels past the 26th keep their own data (the writer's +-100 uV range
    # clips the helper's larger-amplitude channels, so check one inside it).
    np.testing.assert_allclose(rec.signals["--26"].to_numpy(), data[26], atol=0.01)


def test_every_suffix_colliding_raises():
    # "A-0" and "A-a" .. "A-z" all exist already, so the first "A" has no
    # free name; refusing beats silently dropping or merging a channel.
    taken = ["A-0", *(f"A-{c}" for c in "abcdefghijklmnopqrstuvwxyz")]
    with pytest.raises(ValueError, match="Could not de-duplicate EDF channel label 'A'"):
        unique_channel_labels(["A", "A", *taken])


def test_bdf_duplicate_labels_keep_every_channel(tmp_path):
    labels = ["T8-P8", "Cz", "T8-P8"]
    path = os.path.join(tmp_path, "dup.bdf")
    n = 512
    t = np.arange(n) / 256.0
    data = np.vstack([(20.0 + 3 * c) * np.sin(2 * np.pi * (2 + c) * t) for c in range(3)])
    headers = [
        {
            "label": label,
            "dimension": "uV",
            "sample_frequency": 256.0,
            "physical_max": 100.0,
            "physical_min": -100.0,
            "digital_max": 8388607,
            "digital_min": -8388608,
            "prefilter": "n/a",
            "transducer": "n/a",
        }
        for label in labels
    ]
    writer = pyedflib.EdfWriter(path, len(labels), file_type=pyedflib.FILETYPE_BDFPLUS)
    try:
        writer.setSignalHeaders(headers)
        writer.writeSamples(list(data))
    finally:
        writer.close()

    rec = Recording.from_file(path)

    assert list(rec.channels) == ["T8-P8-0", "Cz", "T8-P8-1"]
    np.testing.assert_allclose(rec.signals["T8-P8-0"].to_numpy(), data[0], atol=1e-3)
    np.testing.assert_allclose(rec.signals["T8-P8-1"].to_numpy(), data[2], atol=1e-3)


def test_channels_tsv_on_suffixed_names_applies_in_memory(tmp_path):
    """An MNE-BIDS sidecar names the repeats ``EEG1-0``/``EEG1-1``; the
    in-memory importer suffixes the same way, so each row reaches its own
    channel (the streaming counterpart is in test_zarr_stream_channels_tsv)."""
    from biosigio.tests.test_bids_channels_units import write_channels_tsv

    stem = "sub-01_task-rest"
    path = os.path.join(tmp_path, f"{stem}_eeg.edf")
    data = _write_edf(path, ["EEG1", "EEG1", "Cz"])
    write_channels_tsv(
        tmp_path,
        stem,
        [("EEG1-0", "EOG", "mV"), ("EEG1-1", "EMG", "uV"), ("Cz", "EEG", "uV")],
    )

    rec = Recording.from_file(path)
    ignored = Recording.from_file(path, bids_channels="off")

    assert list(rec.channels) == ["EEG1-0", "EEG1-1", "Cz"]
    assert rec.channels["EEG1-0"]["channel_type"] == "EOG"
    assert rec.channels["EEG1-0"]["physical_dimension"] == "mV"
    assert rec.channels["EEG1-1"]["channel_type"] == "EMG"
    assert rec.channels["EEG1-1"]["physical_dimension"] == "uV"
    # The unit change converts only the row it names: uV -> mV is 1e-3.
    np.testing.assert_allclose(
        rec.signals["EEG1-0"].to_numpy(), ignored.signals["EEG1-0"].to_numpy() * 1e-3
    )
    np.testing.assert_allclose(rec.signals["EEG1-1"].to_numpy(), data[1], atol=0.01)
