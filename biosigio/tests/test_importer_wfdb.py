import os
import shutil

# import numpy as np # Not directly used here
import pandas as pd
import pytest

from biosigio.core.emg import Recording
from biosigio.importers.wfdb import WFDBImporter

# Directory containing the test WFDB files (relative to project root)
EXAMPLE_DIR = "examples"
WFDB_RECORD_NAME = "100"


@pytest.fixture
def wfdb_data(tmp_path):
    """Fixture to provide WFDB test files in a temporary directory."""
    source_dir = os.path.abspath(EXAMPLE_DIR)
    dest_dir = tmp_path

    # Files needed for record '100'
    files_to_copy = [
        f"{WFDB_RECORD_NAME}.hea",
        f"{WFDB_RECORD_NAME}.dat",
        f"{WFDB_RECORD_NAME}.atr",
    ]

    copied_files = {}
    missing_files = []

    for filename in files_to_copy:
        source_path = os.path.join(source_dir, filename)
        dest_path = dest_dir / filename
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            copied_files[filename] = dest_path
        else:
            missing_files.append(filename)

    if missing_files:
        pytest.skip(f"Missing WFDB example files in '{EXAMPLE_DIR}': {missing_files}")

    return {
        "dir": dest_dir,
        "hea": copied_files.get(f"{WFDB_RECORD_NAME}.hea"),
        "dat": copied_files.get(f"{WFDB_RECORD_NAME}.dat"),
        "atr": copied_files.get(f"{WFDB_RECORD_NAME}.atr"),
    }


@pytest.fixture
def wfdb_importer():
    """Fixture to provide a WFDBImporter instance."""
    return WFDBImporter()


# --- Test Cases ---


def test_wfdb_load_basic(wfdb_importer, wfdb_data):
    """Test basic loading of a WFDB record (.hea file)."""
    rec = wfdb_importer.load(str(wfdb_data["hea"]))

    assert isinstance(rec, Recording)
    assert rec.signals is not None
    assert isinstance(rec.signals, pd.DataFrame)
    # Record 100 has 2 signals
    assert rec.signals.shape[1] == 2
    # Check if time index is created
    assert isinstance(rec.signals.index, pd.Index)

    # Check metadata
    assert rec.get_metadata("record_name") == WFDB_RECORD_NAME
    assert rec.get_metadata("sampling_frequency") == 360.0  # Known frequency for record 100
    assert rec.get_metadata("source_file") == str(wfdb_data["hea"])

    # Check channels
    assert len(rec.channels) == 2
    assert "MLII" in rec.channels
    assert "V5" in rec.channels
    assert rec.channels["MLII"]["physical_dimension"] == "mV"
    assert rec.channels["V5"]["physical_dimension"] == "mV"
    assert rec.channels["MLII"]["sample_frequency"] == 360.0
    assert rec.channels["V5"]["sample_frequency"] == 360.0


def test_wfdb_load_annotations(wfdb_importer, wfdb_data):
    """Test loading WFDB record with annotations (.atr file present)."""
    rec = wfdb_importer.load(str(wfdb_data["hea"]))

    assert rec.events is not None
    assert isinstance(rec.events, pd.DataFrame)
    assert not rec.events.empty
    assert list(rec.events.columns) == ["onset", "duration", "description"]
    # Check specific values for record 100 annotations
    assert len(rec.events) == 2274  # Known number of annotations for record 100.atr
    assert rec.events.iloc[0]["onset"] == pytest.approx(18 / 360.0)
    assert rec.events.iloc[0]["duration"] == 0
    assert rec.events.iloc[0]["description"] == "WFDB Annotation: +"


def test_wfdb_load_no_annotations_file(wfdb_importer, wfdb_data):
    """Test loading WFDB record when the annotation file is missing."""
    # Remove the .atr file
    os.remove(wfdb_data["atr"])

    rec = wfdb_importer.load(str(wfdb_data["hea"]))

    # Events DataFrame should be empty
    assert rec.events is not None
    assert isinstance(rec.events, pd.DataFrame)
    assert rec.events.empty

    # Check metadata status
    assert rec.get_metadata("annotation_status") == "Annotation file (.atr) not found."
    assert rec.get_metadata("annotation_error") is None


def test_wfdb_load_file_not_found(wfdb_importer, tmp_path):
    """Test loading a non-existent WFDB record."""
    non_existent_path = str(tmp_path / "non_existent.hea")
    with pytest.raises(FileNotFoundError) as excinfo:
        wfdb_importer.load(non_existent_path)
    # Check for the new specific message from the importer
    assert "WFDB header file not found" in str(excinfo.value)


def test_wfdb_load_dat_missing(wfdb_importer, wfdb_data):
    """Test loading when the data file (.dat) is missing."""
    # Remove the .dat file
    os.remove(wfdb_data["dat"])

    with pytest.raises(ValueError) as excinfo:
        wfdb_importer.load(str(wfdb_data["hea"]))
    # Check for the specific ValueError raised when .dat is missing during read
    assert "Error reading WFDB record" in str(excinfo.value)
    assert "Data file missing or unreadable" in str(excinfo.value)


def test_wfdb_importer_handles_record_name_input(wfdb_importer, wfdb_data):
    """Test that the importer can load using just the record name within the directory."""
    # Provide the path to the directory and the base record name
    record_base_path = os.path.join(wfdb_data["dir"], WFDB_RECORD_NAME)
    rec = wfdb_importer.load(record_base_path)

    assert isinstance(rec, Recording)
    assert rec.get_metadata("record_name") == WFDB_RECORD_NAME
    assert not rec.events.empty  # Check annotations loaded correctly too


def test_wfdb_repeated_sig_name_keeps_every_signal(tmp_path):
    """A header repeating a sig_name loads every signal under a suffixed label.

    wfdb's writer refuses repeated names, so the record is written with unique
    names and its header then edited on disk; wfdb's reader accepts the result.
    Before the fix the second signal silently replaced the first.
    """
    import numpy as np
    import wfdb

    sig = np.random.default_rng(0).normal(size=(500, 3))
    wfdb.wrsamp(
        "dup",
        fs=250,
        units=["mV"] * 3,
        sig_name=["EMG1", "EMG2", "ECG"],
        p_signal=sig,
        fmt=["16"] * 3,
        write_dir=str(tmp_path),
    )
    hea = tmp_path / "dup.hea"
    hea.write_text(hea.read_text().replace("EMG2", "EMG1"))
    assert wfdb.rdrecord(str(tmp_path / "dup")).sig_name == ["EMG1", "EMG1", "ECG"]

    rec = WFDBImporter().load(str(hea))

    assert list(rec.channels) == ["EMG1-0", "EMG1-1", "ECG"]
    for i, label in enumerate(["EMG1-0", "EMG1-1", "ECG"]):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), sig[:, i], atol=1e-3)
        assert "adc_gain" in rec.channels[label]
