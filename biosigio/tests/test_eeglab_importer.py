import os
import tempfile

import numpy as np
import pytest
import scipy.io

from ..importers.eeglab import EEGLABImporter


@pytest.fixture
def sample_eeglab_set():
    """Create a sample EEGLAB .set file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".set", delete=False) as f:
        temp_path = f.name

    # Create sample data
    n_channels = 32
    n_samples = 1000
    sampling_freq = 1000  # Hz

    # Create synthetic EMG data
    data = np.random.randn(n_channels, n_samples) * 0.1  # Small amplitude noise

    # Add some EMG-like signals to a few channels
    t = np.arange(n_samples) / sampling_freq
    for i in range(5):
        # Add 50Hz sine wave (EMG-like) with random phase and amplitude
        phase = np.random.rand() * 2 * np.pi
        amp = np.random.rand() * 0.5 + 0.5
        data[i, :] += amp * np.sin(2 * np.pi * 50 * t + phase)

    # Create channel locations
    chanlocs = np.zeros(
        (1, n_channels),
        dtype=[
            ("labels", "O"),
            ("type", "O"),
            ("X", "O"),
            ("Y", "O"),
            ("Z", "O"),
            ("sph_theta", "O"),
            ("sph_phi", "O"),
            ("sph_radius", "O"),
            ("theta", "O"),
            ("radius", "O"),
            ("ref", "O"),
            ("urchan", "O"),
        ],
    )

    # Fill channel info
    for i in range(n_channels):
        chanlocs[0, i] = (
            np.array([f"emg{i}_left" if i < 16 else f"emg{i - 16}_right"]),  # labels
            np.array(["EMG"]),  # type
            np.array([]),  # X
            np.array([]),  # Y
            np.array([]),  # Z
            np.array([[0]]),  # sph_theta
            np.array([[0]]),  # sph_phi
            np.array([[0]]),  # sph_radius
            np.array([[0]]),  # theta
            np.array([[0]]),  # radius
            np.array([]),  # ref
            np.array([]),  # urchan
        )

    # Create events
    events = np.zeros(
        (1, 5),
        dtype=[
            ("latency", "O"),
            ("duration", "O"),
            ("sample", "O"),
            ("trial_type", "O"),
            ("type", "O"),
            ("urevent", "O"),
        ],
    )

    # Fill event info
    for i in range(5):
        events[0, i] = (
            np.array([[i * 200 + 100]]),  # latency
            np.array([[0]]),  # duration
            np.array([[i * 200 + 100]]),  # sample
            np.array([f"key/{chr(97 + i)}"]),  # trial_type (key/a, key/b, etc.)
            np.array([f"{i + 1}"]),  # type
            np.array([[i + 1]]),  # urevent
        )

    # Create EEGLAB .set file structure
    eeglab_data = {
        "setname": np.array(["test_emg"]),
        "filename": np.array(["test_emg.set"]),
        "filepath": np.array([""]),
        "subject": np.array(["test_subject"]),
        "group": np.array(["test_group"]),
        "condition": np.array(["test_condition"]),
        "session": np.array(["001"]),
        "comments": np.array(["Test EMG data"]),
        "nbchan": np.array([[n_channels]]),
        "trials": np.array([[1]]),
        "pnts": np.array([[n_samples]]),
        "srate": np.array([[sampling_freq]]),
        "xmin": np.array([[0.0]]),
        "xmax": np.array([[n_samples / sampling_freq]]),
        "times": np.array([np.arange(n_samples) / sampling_freq]),
        "data": data,
        "chanlocs": chanlocs,
        "event": events,
    }

    # Save to .set file
    scipy.io.savemat(temp_path, eeglab_data)

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_eeglab_importer(sample_eeglab_set):
    """Test EEGLAB importer with sample data."""
    importer = EEGLABImporter()
    rec = importer.load(sample_eeglab_set)

    # Check if metadata was loaded
    assert rec.get_metadata("subject") == "test_subject"
    assert rec.get_metadata("srate") == 1000
    assert rec.get_metadata("device") == "EEGLAB"

    # Check if signals were loaded
    assert rec.signals is not None
    assert rec.signals.shape[0] == 1000  # 1000 samples
    assert rec.signals.shape[1] == 32  # 32 channels

    # Check if channel info was loaded
    assert len(rec.channels) == 32
    for _ch_name, ch_info in rec.channels.items():
        assert ch_info["channel_type"] == "EMG"
        assert ch_info["sample_frequency"] == 1000
        assert ch_info["physical_dimension"] == "uV"

    # Events are loaded into the events table (onset/duration in seconds), not metadata.
    assert "events" not in rec.metadata
    assert not rec.events.empty
    assert len(rec.events) == 5
    assert list(rec.events["description"]) == [f"{i + 1}" for i in range(5)]
    # EEGLAB latency is 1-based samples -> seconds at 1000 Hz: (i*200+100 - 1) / 1000.
    expected_onsets = [(i * 200 + 100 - 1) / 1000 for i in range(5)]
    assert rec.events["onset"].tolist() == pytest.approx(expected_onsets)
    assert (rec.events["duration"] == 0.0).all()


def test_eeglab_file_not_found():
    """Test error handling for non-existent file."""
    importer = EEGLABImporter()
    with pytest.raises(ValueError):
        importer.load("nonexistent.set")


def test_eeglab_metadata_extraction(sample_eeglab_set):
    """Test metadata extraction from EEGLAB file."""
    importer = EEGLABImporter()
    rec = importer.load(sample_eeglab_set)

    # Test basic metadata
    assert rec.get_metadata("setname") == "test_emg"
    assert rec.get_metadata("subject") == "test_subject"
    assert rec.get_metadata("group") == "test_group"
    assert rec.get_metadata("condition") == "test_condition"
    assert rec.get_metadata("session") == "001"
    assert rec.get_metadata("comments") == "Test EMG data"

    # Test recording parameters
    assert rec.get_metadata("srate") == 1000
    assert rec.get_metadata("nbchan") == 32
    assert rec.get_metadata("trials") == 1
    assert rec.get_metadata("pnts") == 1000
    assert rec.get_metadata("xmin") == 0.0
    assert rec.get_metadata("xmax") == 1.0


def test_eeglab_channel_type_detection(sample_eeglab_set):
    """Test channel type detection from EEGLAB file."""
    importer = EEGLABImporter()
    rec = importer.load(sample_eeglab_set)

    # All channels should be EMG type
    for _ch_name, ch_info in rec.channels.items():
        assert ch_info["channel_type"] == "EMG"

    # Test channel naming
    for i in range(16):
        assert f"emg{i}_left" in rec.channels
    for i in range(16):
        assert f"emg{i}_right" in rec.channels


def test_eeglab_event_processing(sample_eeglab_set):
    """Test event processing from EEGLAB file."""
    importer = EEGLABImporter()
    rec = importer.load(sample_eeglab_set)

    # Events are loaded into the events table (onset/duration in seconds), not metadata.
    assert "events" not in rec.metadata
    assert not rec.events.empty
    assert len(rec.events) == 5
    # The EEGLAB event `type` becomes the description.
    assert list(rec.events["description"]) == [f"{i + 1}" for i in range(5)]
    # latency (1-based samples) -> seconds at 1000 Hz: (i*200+100 - 1) / 1000.
    expected_onsets = [(i * 200 + 100 - 1) / 1000 for i in range(5)]
    assert rec.events["onset"].tolist() == pytest.approx(expected_onsets)


def _write_set_with_fdt(directory, stem, data, srate, embedded_name):
    """Write a real EEGLAB .set whose data lives in a sibling float32 .fdt.

    ``data`` is (nbchan, pnts) for continuous data or (nbchan, pnts, trials) for
    epoched data; the .fdt is written MATLAB column-major. ``EEG.data`` stores
    ``embedded_name`` (the .fdt's original, possibly pre-BIDS-rename name),
    exercising the sibling-by-path resolution.
    """
    nbchan, pnts = data.shape[0], data.shape[1]
    trials = data.shape[2] if data.ndim == 3 else 1
    fdt_path = os.path.join(directory, f"{stem}.fdt")
    data.astype(np.float32).flatten(order="F").tofile(fdt_path)
    set_path = os.path.join(directory, f"{stem}.set")
    scipy.io.savemat(
        set_path,
        {
            "setname": np.array(["fdt_test"]),
            "nbchan": np.array([[nbchan]]),
            "trials": np.array([[trials]]),
            "pnts": np.array([[pnts]]),
            "srate": np.array([[srate]]),
            "xmin": np.array([[0.0]]),
            "xmax": np.array([[pnts / srate]]),
            "data": np.array([embedded_name]),  # filename string, not the matrix
        },
    )
    return set_path


def test_eeglab_reads_separate_fdt(tmp_path):
    """A .set whose EEG.data is a .fdt filename loads the sibling .fdt matrix."""
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((8, 500)) * 10).astype(np.float32)  # (nbchan, pnts)
    set_path = _write_set_with_fdt(str(tmp_path), "sub-09_eeg", data, 200, "sub-09_eeg.fdt")

    rec = EEGLABImporter().load(set_path)

    assert "eeglab_fdt_recovered" not in rec.metadata  # the named file was read
    assert rec.signals.shape == (500, 8)  # samples x channels
    for i, label in enumerate(rec.signals.columns):
        # Recording stores samples x channels, so column i is data row i.
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)


def test_eeglab_fdt_resolves_sibling_despite_renamed_embedded_name(tmp_path, caplog):
    """BIDS renames the .fdt on disk but not EEG.data; the sibling wins."""
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((4, 256)) * 5).astype(np.float32)
    # On-disk sibling is sub-09_task-rest_eeg.fdt, but EEG.data says the
    # pre-rename "Merged.fdt" (which does not exist) -- the importer must still
    # find the sibling by the .set path.
    set_path = _write_set_with_fdt(str(tmp_path), "sub-09_task-rest_eeg", data, 256, "Merged.fdt")

    with caplog.at_level("INFO", logger="biosigio.importers.eeglab"):
        rec = EEGLABImporter().load(set_path)
    assert rec.signals.shape == (256, 4)
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)
    # The substitution is recorded and logged, not silent.
    assert rec.metadata["eeglab_fdt_recovered"] == {
        "referenced": "Merged.fdt",
        "used": "sub-09_task-rest_eeg.fdt",
    }
    assert "names data file Merged.fdt" in caplog.text


def test_eeglab_reads_epoched_fdt(tmp_path):
    """Epoched (trials > 1) .fdt reshapes to concatenated continuous samples."""
    rng = np.random.default_rng(2)
    nbchan, pnts, trials = 4, 100, 3
    data = (rng.standard_normal((nbchan, pnts, trials)) * 7).astype(np.float32)
    set_path = _write_set_with_fdt(str(tmp_path), "sub-09_eeg", data, 100, "sub-09_eeg.fdt")

    rec = EEGLABImporter().load(set_path)

    # trials are concatenated: pnts * trials continuous samples per channel.
    assert rec.signals.shape == (pnts * trials, nbchan)
    for i, label in enumerate(rec.signals.columns):
        # Channel i continuous series = [trial0, trial1, trial2] = column-major flatten.
        expected = data[i].flatten(order="F")
        np.testing.assert_allclose(rec.signals[label].to_numpy(), expected, rtol=0, atol=1e-5)


def test_eeglab_fdt_missing_raises(tmp_path):
    """A .set pointing at a .fdt with no resolvable file errors clearly."""
    set_path = os.path.join(str(tmp_path), "sub-09_eeg.set")
    scipy.io.savemat(
        set_path,
        {
            "nbchan": np.array([[4]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[256]]),
            "srate": np.array([[256]]),
            "data": np.array(["nowhere.fdt"]),
        },
    )
    with pytest.raises(ValueError, match="separate .fdt"):
        EEGLABImporter().load(set_path)


def test_eeglab_fdt_size_mismatch_raises(tmp_path):
    """A .fdt whose element count disagrees with the header is rejected."""
    fdt_path = os.path.join(str(tmp_path), "sub-09_eeg.fdt")
    np.zeros(8 * 500 - 7, dtype=np.float32).tofile(fdt_path)  # wrong count
    set_path = os.path.join(str(tmp_path), "sub-09_eeg.set")
    scipy.io.savemat(
        set_path,
        {
            "nbchan": np.array([[8]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[500]]),
            "srate": np.array([[200]]),
            "data": np.array(["sub-09_eeg.fdt"]),
        },
    )
    with pytest.raises(ValueError, match="float32 samples"):
        EEGLABImporter().load(set_path)


def _nested_eeg_struct(n_channels, n_samples, srate, data_field):
    """Build the dict a real EEGLAB save produces: one top-level ``EEG`` struct.

    ``scipy.io.savemat(path, {'EEG': {...}})`` writes the whole dataset as a
    single MATLAB struct variable named ``EEG`` (the real-world form), so on
    reload every field lands under ``data['EEG']`` rather than at the top level.
    ``data_field`` is either the numeric matrix (inline data) or the ``.fdt``
    filename char array.
    """
    chanlocs = np.zeros(
        (1, n_channels),
        dtype=[("labels", "O"), ("type", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")],
    )
    for i in range(n_channels):
        chanlocs[0, i] = (
            np.array([f"Ch{i + 1}"]),  # labels
            np.array([]),  # type (empty -> modality falls back, exercised by ERP CORE)
            np.array([]),  # X
            np.array([]),  # Y
            np.array([]),  # Z
        )

    events = np.zeros((1, 2), dtype=[("latency", "O"), ("duration", "O"), ("type", "O")])
    events[0, 0] = (np.array([[100]]), np.array([[0]]), np.array(["stim"]))
    events[0, 1] = (np.array([[150]]), np.array([[0]]), np.array(["resp"]))

    return {
        "setname": np.array(["nested_test"]),
        "nbchan": np.array([[n_channels]]),
        "trials": np.array([[1]]),
        "pnts": np.array([[n_samples]]),
        "srate": np.array([[srate]]),
        "xmin": np.array([[0.0]]),
        "xmax": np.array([[n_samples / srate]]),
        "data": data_field,
        "chanlocs": chanlocs,
        "event": events,
    }


def test_eeglab_nested_eeg_struct_inline_data(tmp_path):
    """A real-form .set ({'EEG': struct}) with inline data loads signals + channels.

    Regression for the bug where _extract_metadata/_load read fields from the
    loadmat top level, so a real EEGLAB export (everything nested under ``EEG``)
    silently returned an empty Recording.
    """
    n_channels, n_samples, srate = 3, 200, 256
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)
    set_path = str(tmp_path / "nested_inline_eeg.set")
    scipy.io.savemat(set_path, {"EEG": _nested_eeg_struct(n_channels, n_samples, srate, data)})

    rec = EEGLABImporter().load(set_path)

    assert rec.signals is not None
    assert rec.signals.shape == (n_samples, n_channels)  # samples x channels
    assert len(rec.channels) == n_channels
    assert rec.get_metadata("srate") == srate
    assert rec.get_metadata("nbchan") == n_channels
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)
    # Events under the nested struct still parse.
    assert len(rec.events) == 2
    assert list(rec.events["description"]) == ["stim", "resp"]


def test_eeglab_nested_eeg_struct_with_fdt(tmp_path):
    """A real-form .set whose EEG.data is a .fdt filename loads the sibling .fdt."""
    n_channels, n_samples, srate = 3, 200, 256
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)
    stem = "sub-01_task-x_eeg"
    (tmp_path / f"{stem}.fdt").write_bytes(b"")  # placeholder, overwritten below
    data.flatten(order="F").tofile(str(tmp_path / f"{stem}.fdt"))
    set_path = str(tmp_path / f"{stem}.set")
    scipy.io.savemat(
        set_path,
        {"EEG": _nested_eeg_struct(n_channels, n_samples, srate, np.array([f"{stem}.fdt"]))},
    )

    rec = EEGLABImporter().load(set_path)

    assert rec.signals.shape == (n_samples, n_channels)
    assert len(rec.channels) == n_channels
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)


def test_eeglab_nested_eeg_struct_to_zarr_non_empty(tmp_path):
    """The full path that was broken: real-form .set -> Recording -> non-empty Zarr store."""
    pytest.importorskip("zarr", reason="Zarr serving format requires the optional 'zarr' extra")
    import zarr

    n_channels, n_samples, srate = 4, 500, 256
    rng = np.random.default_rng(2)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)
    set_path = str(tmp_path / "nested_zarr_eeg.set")
    scipy.io.savemat(set_path, {"EEG": _nested_eeg_struct(n_channels, n_samples, srate, data)})

    rec = EEGLABImporter().load(set_path)
    out = rec.to_zarr(str(tmp_path / "rec"))

    assert os.path.exists(os.path.join(out, "zarr.json"))  # root metadata written
    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="r")
    assert root.attrs["channel_groups"]  # at least one channel group serialized


def test_eeglab_to_edf_export(sample_eeglab_set):
    """Test exporting EEGLAB data to EDF format."""
    importer = EEGLABImporter()
    rec = importer.load(sample_eeglab_set)

    # Export to EDF
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        edf_path = f.name

    try:
        rec.to_edf(edf_path)

        # Check if EDF file was created
        assert os.path.exists(edf_path) or os.path.exists(os.path.splitext(edf_path)[0] + ".bdf")

        # Check if channels.tsv was created
        channels_tsv_path = os.path.splitext(edf_path)[0] + "_channels.tsv"
        assert os.path.exists(channels_tsv_path)
    finally:
        # Cleanup
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(os.path.splitext(edf_path)[0] + ".bdf"):
            os.unlink(os.path.splitext(edf_path)[0] + ".bdf")
        if os.path.exists(channels_tsv_path):
            os.unlink(channels_tsv_path)


def _column_form_set(tmp_path, n_channels=6, n_samples=300, srate=250):
    """Write a flat-form .set whose chanlocs/event struct arrays are COLUMN
    vectors ((N, 1) instead of (1, N)), the shape real EEGLAB exports also use
    (e.g. OpenNeuro ds002718). Returns the .set path and the data matrix.
    """
    rng = np.random.default_rng(7)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)

    chanlocs = np.zeros(
        (n_channels, 1),
        dtype=[("labels", "O"), ("type", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")],
    )
    for i in range(n_channels):
        chanlocs[i, 0] = (
            np.array([f"EEG{i + 1:03d}"]),
            np.array(["EEG"]),
            np.array([[float(i)]]),
            np.array([[0.0]]),
            np.array([[1.0]]),
        )

    events = np.zeros((3, 1), dtype=[("latency", "O"), ("duration", "O"), ("type", "O")])
    for i in range(3):
        events[i, 0] = (np.array([[i * 50 + 25]]), np.array([[0]]), np.array([f"ev{i}"]))

    set_path = str(tmp_path / "column_form.set")
    scipy.io.savemat(
        set_path,
        {
            "setname": np.array(["column_form"]),
            "nbchan": np.array([[n_channels]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[n_samples]]),
            "srate": np.array([[srate]]),
            "data": data,
            "chanlocs": chanlocs,
            "event": events,
        },
    )
    return set_path, data


def test_eeglab_column_form_chanlocs_loads_all_channels(tmp_path):
    """Column-vector ((N, 1)) chanlocs/event struct arrays load every channel.

    Regression for the one-channel bug (nemarDatasets/on002718#1): iterating
    ``chanlocs[0]`` on a column-form file yielded a single channel-info entry,
    and the loader then silently truncated the 74-channel data matrix to one
    channel. The raveled walk must see every channel and every event.
    """
    n_channels, n_samples = 6, 300
    set_path, data = _column_form_set(tmp_path, n_channels, n_samples)

    rec = EEGLABImporter().load(set_path)

    assert rec.signals.shape == (n_samples, n_channels)
    assert list(rec.signals.columns) == [f"EEG{i + 1:03d}" for i in range(n_channels)]
    assert len(rec.channels) == n_channels
    for info in rec.channels.values():
        assert info["channel_type"] == "EEG"
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)
    assert len(rec.events) == 3
    assert list(rec.events["description"]) == ["ev0", "ev1", "ev2"]


def test_eeglab_short_chanlocs_never_truncates_data(tmp_path):
    """A chanlocs shorter than the data matrix pads labels instead of dropping rows.

    The data matrix is authoritative for the channel count: a malformed or
    partial chanlocs must warn and keep every signal row, never silently
    truncate (the failure mode behind nemarDatasets/on002718#1).
    """
    n_channels, n_samples, srate = 5, 200, 250
    rng = np.random.default_rng(11)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)

    chanlocs = np.zeros((1, 2), dtype=[("labels", "O"), ("type", "O")])
    chanlocs[0, 0] = (np.array(["C1"]), np.array(["EEG"]))
    chanlocs[0, 1] = (np.array(["C2"]), np.array(["EEG"]))

    set_path = str(tmp_path / "short_chanlocs.set")
    scipy.io.savemat(
        set_path,
        {
            "nbchan": np.array([[n_channels]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[n_samples]]),
            "srate": np.array([[srate]]),
            "data": data,
            "chanlocs": chanlocs,
        },
    )

    with pytest.warns(UserWarning, match="keeping all data rows"):
        rec = EEGLABImporter().load(set_path)

    assert rec.signals.shape == (n_samples, n_channels)
    assert list(rec.signals.columns)[:2] == ["C1", "C2"]
    assert len(rec.channels) == n_channels
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)


def test_eeglab_header_nbchan_mismatch_warns(tmp_path):
    """A header nbchan that disagrees with the data matrix warns and keeps the data."""
    n_channels, n_samples, srate = 4, 100, 250
    rng = np.random.default_rng(3)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)

    chanlocs = np.zeros((1, n_channels), dtype=[("labels", "O"), ("type", "O")])
    for i in range(n_channels):
        chanlocs[0, i] = (np.array([f"C{i + 1}"]), np.array(["EEG"]))

    set_path = str(tmp_path / "nbchan_mismatch.set")
    scipy.io.savemat(
        set_path,
        {
            "nbchan": np.array([[99]]),  # header lies; data matrix is authoritative
            "trials": np.array([[1]]),
            "pnts": np.array([[n_samples]]),
            "srate": np.array([[srate]]),
            "data": data,
            "chanlocs": chanlocs,
        },
    )

    with pytest.warns(UserWarning, match="nbchan=99 disagrees"):
        rec = EEGLABImporter().load(set_path)

    assert rec.signals.shape == (n_samples, n_channels)
    assert len(rec.channels) == n_channels


def test_eeglab_padded_label_collision_stays_unique(tmp_path):
    """A padded default label colliding with a real channel name is disambiguated.

    A real channel literally named "Channel4" must not merge with the padded
    default for data row 4: duplicate labels clobber rec.channels entries and
    break per-channel column slicing in the exporters.
    """
    n_channels, n_samples, srate = 5, 100, 250
    rng = np.random.default_rng(5)
    data = (rng.standard_normal((n_channels, n_samples)) * 10).astype(np.float32)

    chanlocs = np.zeros((1, 2), dtype=[("labels", "O"), ("type", "O")])
    chanlocs[0, 0] = (np.array(["C1"]), np.array(["EEG"]))
    chanlocs[0, 1] = (np.array(["Channel4"]), np.array(["EEG"]))

    set_path = str(tmp_path / "label_collision.set")
    scipy.io.savemat(
        set_path,
        {
            "nbchan": np.array([[n_channels]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[n_samples]]),
            "srate": np.array([[srate]]),
            "data": data,
            "chanlocs": chanlocs,
        },
    )

    with pytest.warns(UserWarning):
        rec = EEGLABImporter().load(set_path)

    labels = list(rec.signals.columns)
    assert len(labels) == n_channels
    assert len(set(labels)) == n_channels  # unique -> no dict clobber, no 2-D slice
    assert len(rec.channels) == n_channels
    assert labels[:2] == ["C1", "Channel4"]
    # Every column still slices to a 1-D series with faithful values.
    for i, label in enumerate(labels):
        col = rec.signals[label].to_numpy()
        assert col.ndim == 1
        np.testing.assert_allclose(col, data[i], rtol=0, atol=1e-5)


def test_classic_missing_srate_defaults_with_a_warning(tmp_path, caplog):
    """A classic .set with no srate loads at the 1000 Hz default, logged, not silent."""
    path = str(tmp_path / "no_srate.set")
    scipy.io.savemat(
        path,
        {
            "nbchan": np.array([[1]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[4]]),
            "data": np.zeros((1, 4), dtype=np.float32),
            "event": np.array(
                [(np.array([[101.0]]), np.array(["stim"]))],
                dtype=[("latency", "O"), ("type", "O")],
            ),
        },
    )

    with caplog.at_level("WARNING", logger="biosigio.importers.eeglab"):
        rec = EEGLABImporter().load(path)

    assert "has no sampling rate" in caplog.text and "assuming 1000 Hz" in caplog.text
    assert rec.events["onset"].tolist() == pytest.approx([0.1])  # (101 - 1) / 1000
