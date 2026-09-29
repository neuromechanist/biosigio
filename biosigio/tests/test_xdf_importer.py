"""Tests for XDF importer functionality."""

import os
import tempfile

import numpy as np
import pytest

from ..core.emg import Recording
from ..importers.xdf import XDFImporter, XDFStreamInfo, XDFSummary, summarize_xdf

# Path to the sample XDF file
SAMPLE_XDF_PATH = "examples/test.xdf"


def test_summarize_xdf():
    """Test summarize_xdf function with sample data.

    Note: The summarize_xdf function uses memory-efficient parsing that reads
    metadata from StreamFooter chunks without loading signal data. The sample
    count in the footer is typically slightly lower than the actual data count
    because the footer is written before all samples are flushed. However, in
    some cases it may also be slightly higher due to recording software quirks.
    We allow a tolerance of ±2 samples to account for this variance.
    """
    summary = summarize_xdf(SAMPLE_XDF_PATH)

    # Check summary structure
    assert isinstance(summary, XDFSummary)
    assert summary.filepath == SAMPLE_XDF_PATH
    assert len(summary.streams) == 1

    # Check stream info
    stream = summary.streams[0]
    assert isinstance(stream, XDFStreamInfo)
    assert stream.name == "obci_neeg1"
    assert stream.stream_type == "EEG"
    assert stream.channel_count == 8
    assert stream.nominal_srate == 1000.0
    # Sample count from footer may differ by ±2 samples from actual data
    assert abs(stream.sample_count - 9520) <= 2
    assert stream.duration_seconds > 9.0


def test_summarize_xdf_str_representation():
    """Test string representation of XDF summary."""
    summary = summarize_xdf(SAMPLE_XDF_PATH)

    # Convert to string
    summary_str = str(summary)

    # Check key elements are in string
    assert "test.xdf" in summary_str
    assert "obci_neeg1" in summary_str
    assert "EEG" in summary_str
    assert "8" in summary_str  # channel count
    assert "1000" in summary_str  # sample rate


def test_summarize_xdf_stream_lookup():
    """Test stream lookup methods in XDFSummary."""
    summary = summarize_xdf(SAMPLE_XDF_PATH)

    # Test get_streams_by_type
    eeg_streams = summary.get_streams_by_type("EEG")
    assert len(eeg_streams) == 1
    assert eeg_streams[0].name == "obci_neeg1"

    # Test get_streams_by_type with non-existent type
    emg_streams = summary.get_streams_by_type("EMG")
    assert len(emg_streams) == 0

    # Test get_stream_by_name
    stream = summary.get_stream_by_name("obci_neeg1")
    assert stream is not None
    assert stream.stream_type == "EEG"

    # Test get_stream_by_name with non-existent name
    stream = summary.get_stream_by_name("nonexistent")
    assert stream is None


def test_xdf_importer_basic():
    """Test XDF importer with sample data."""
    importer = XDFImporter()
    rec = importer.load(SAMPLE_XDF_PATH)

    # Check if channels were loaded
    assert len(rec.channels) == 8

    # Check channel properties
    first_channel = list(rec.channels.keys())[0]
    assert rec.channels[first_channel]["sample_frequency"] == 1000.0

    # Check data shape
    assert rec.signals.shape == (9520, 8)

    # Check metadata
    assert rec.get_metadata("device") == "XDF"
    assert rec.get_metadata("source_file") == SAMPLE_XDF_PATH
    assert rec.get_metadata("stream_count") == 1


def test_xdf_importer_from_emg_class():
    """Test XDF import through Recording.from_file."""
    rec = Recording.from_file(SAMPLE_XDF_PATH)

    # Check basic loading
    assert rec.signals is not None
    assert len(rec.channels) == 8
    assert rec.signals.shape[0] == 9520


def test_xdf_importer_stream_selection_by_type():
    """Test XDF importer with stream type selection."""
    importer = XDFImporter()

    # Select by type - should match the stream
    rec = importer.load(SAMPLE_XDF_PATH, stream_types=["EEG"])
    assert len(rec.channels) == 8

    # Select by non-matching type - should raise error
    with pytest.raises(ValueError, match="No matching streams"):
        importer.load(SAMPLE_XDF_PATH, stream_types=["EMG"])


def test_xdf_importer_stream_selection_by_name():
    """Test XDF importer with stream name selection."""
    importer = XDFImporter()

    # Select by name (case-insensitive)
    rec = importer.load(SAMPLE_XDF_PATH, stream_names=["obci_neeg1"])
    assert len(rec.channels) == 8

    # Select by non-matching name
    with pytest.raises(ValueError, match="No matching streams"):
        importer.load(SAMPLE_XDF_PATH, stream_names=["nonexistent"])


def test_xdf_importer_channel_labels():
    """Test channel labeling in XDF importer."""
    importer = XDFImporter()
    rec = importer.load(SAMPLE_XDF_PATH)

    # The sample stream has no channel labels, so each is named <stream name>_Ch<n>
    channel_names = list(rec.channels.keys())
    assert all("obci_neeg1" in name for name in channel_names)


def test_xdf_importer_timestamps():
    """Test timestamp handling in XDF importer."""
    importer = XDFImporter()
    rec = importer.load(SAMPLE_XDF_PATH)

    # Check time index starts at 0
    assert rec.signals.index[0] == pytest.approx(0.0, abs=1e-6)

    # Check duration matches expected
    duration = rec.signals.index[-1]
    assert duration > 9.0  # Should be about 9.5 seconds


def test_xdf_export_roundtrip():
    """Test XDF import and EDF export roundtrip."""
    # Load XDF file
    rec = Recording.from_file(SAMPLE_XDF_PATH)

    # Export to EDF
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        temp_path = f.name

    try:
        rec.to_edf(temp_path, format="edf")

        # Reload from EDF
        rec_reloaded = Recording.from_file(temp_path)

        # Check basic structure preserved
        assert len(rec_reloaded.channels) == len(rec.channels)

        # Check data is similar (allowing for EDF precision loss)
        for orig_ch, reload_ch in zip(
            rec.channels.keys(), rec_reloaded.channels.keys(), strict=False
        ):
            orig_data = rec.signals[orig_ch].values
            reload_data = rec_reloaded.signals[reload_ch].values

            # Trim to same length (EDF may have different sample count)
            min_len = min(len(orig_data), len(reload_data))
            correlation = np.corrcoef(orig_data[:min_len], reload_data[:min_len])[0, 1]
            assert correlation > 0.99, f"Correlation for {orig_ch} is {correlation}"

    finally:
        # Cleanup
        os.unlink(temp_path)
        # Also cleanup the channels.tsv file if it exists
        tsv_path = temp_path.replace(".edf", "_channels.tsv")
        if os.path.exists(tsv_path):
            os.unlink(tsv_path)


def test_xdf_file_not_found():
    """Test error handling for non-existent file."""
    importer = XDFImporter()
    # pyxdf raises a generic Exception for missing files; allow common error messages
    with pytest.raises(
        (FileNotFoundError, OSError, Exception),
        match=r"(does not exist|No such file or directory|not found)",
    ):
        importer.load("nonexistent.xdf")


def test_xdf_default_channel_type():
    """Test default channel type assignment."""
    importer = XDFImporter()
    rec = importer.load(SAMPLE_XDF_PATH, default_channel_type="EEG")

    # Since the stream has no explicit channel types in desc,
    # channels should get the default type
    for channel_info in rec.channels.values():
        assert channel_info["channel_type"] == "EEG"


# ============================================================================
# Multi-stream XDF tests
# ============================================================================

MULTI_STREAM_XDF_PATH = "examples/multi_stream_test.xdf"


def test_multistream_summarize():
    """Test summarize_xdf with multi-stream file."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    # Check we have 4 streams
    assert len(summary.streams) == 4

    # Check stream types
    stream_types = {s.stream_type for s in summary.streams}
    assert "EEG" in stream_types
    assert "EMG" in stream_types
    assert "Mocap" in stream_types
    assert "Markers" in stream_types


def test_multistream_import_all():
    """Test importing all numeric streams from multi-stream file."""
    rec = Recording.from_file(MULTI_STREAM_XDF_PATH)

    # Should have channels from EEG, EMG, and Mocap (not Markers - string stream)
    # EEG: 8, EMG: 2, Mocap: 6 = 16 total
    assert len(rec.channels) == 16


def test_multistream_import_by_type():
    """Test importing specific stream types from multi-stream file."""
    # Import only EMG
    rec = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EMG"])
    assert len(rec.channels) == 2
    channel_names = list(rec.channels.keys())
    assert "EMG_L" in channel_names
    assert "EMG_R" in channel_names

    # Import only EEG
    eeg = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EEG"])
    assert len(eeg.channels) == 8
    assert all("EEG" in name for name in eeg.channels.keys())

    # Import only Mocap
    mocap = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_types=["Mocap"])
    assert len(mocap.channels) == 6
    assert all("Marker" in name for name in mocap.channels.keys())


def test_multistream_import_by_name():
    """Test importing specific streams by name from multi-stream file."""
    rec = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_names=["TestEMG"])
    assert len(rec.channels) == 2


def test_multistream_import_multiple_types():
    """Test importing multiple stream types from multi-stream file."""
    rec = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EEG", "EMG"])

    # Should have EEG (8) + EMG (2) = 10 channels
    assert len(rec.channels) == 10


def test_multistream_sampling_rates():
    """Test that different sampling rates are handled correctly."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    # Check expected sampling rates
    eeg_stream = summary.get_stream_by_name("TestEEG")
    assert eeg_stream.nominal_srate == 256.0

    emg_stream = summary.get_stream_by_name("TestEMG")
    assert emg_stream.nominal_srate == 2048.0

    mocap_stream = summary.get_stream_by_name("TestMocap")
    assert mocap_stream.nominal_srate == 120.0

    marker_stream = summary.get_stream_by_name("TestMarkers")
    assert marker_stream.nominal_srate == 0.0  # Irregular


def test_multistream_marker_stream_info():
    """Test that marker (string) streams are summarized correctly."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    marker_stream = summary.get_stream_by_name("TestMarkers")
    assert marker_stream is not None
    assert marker_stream.stream_type == "Markers"
    assert marker_stream.sample_count == 5
    assert marker_stream.channel_format == "string"


def test_multistream_channel_labels():
    """Test that channel labels are correctly extracted from multi-stream file."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    eeg_stream = summary.get_stream_by_name("TestEEG")
    assert "EEG1" in eeg_stream.channel_labels
    assert len(eeg_stream.channel_labels) == 8

    emg_stream = summary.get_stream_by_name("TestEMG")
    assert "EMG_L" in emg_stream.channel_labels
    assert "EMG_R" in emg_stream.channel_labels


def test_multistream_uses_highest_sample_rate():
    """Test that multi-stream loading uses highest sampling rate as reference.

    This is critical to avoid data loss from downsampling high-frequency signals.
    When loading EEG (256 Hz) and EMG (2048 Hz), the result should have the
    sample count matching the EMG stream (the highest rate).
    """
    # Get expected sample counts from summary
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)
    emg_stream = summary.get_stream_by_name("TestEMG")  # 2048 Hz - highest

    # Load both streams (EEG at 256 Hz, EMG at 2048 Hz)
    combined = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EEG", "EMG"])

    # The number of samples should match the highest rate stream (EMG at 2048 Hz)
    n_samples = len(combined.signals)
    assert n_samples == emg_stream.sample_count, (
        f"Expected {emg_stream.sample_count} samples (from 2048 Hz stream), got {n_samples}"
    )

    # Verify the metadata reflects the highest sample rate
    assert combined.get_metadata("srate") == 2048.0


def test_multistream_reference_stream_parameter():
    """Test that reference_stream parameter allows user to specify time base."""
    from biosigio.importers.xdf import XDFImporter

    importer = XDFImporter()

    # Get expected sample counts
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)
    eeg_stream = summary.get_stream_by_name("TestEEG")  # 256 Hz

    # Force use of EEG as reference (lower rate)
    combined = importer.load(
        MULTI_STREAM_XDF_PATH,
        stream_types=["EEG", "EMG"],
        reference_stream="TestEEG",
    )

    # Should use EEG sample count (lower rate, will lose EMG resolution)
    n_samples = len(combined.signals)
    assert n_samples == eeg_stream.sample_count, (
        f"Expected {eeg_stream.sample_count} samples (from EEG reference), got {n_samples}"
    )


def test_multistream_reference_stream_not_found():
    """Test error when specified reference stream doesn't exist."""
    from biosigio.importers.xdf import XDFImporter

    importer = XDFImporter()

    with pytest.raises(ValueError, match="Reference stream 'NonExistent' not found"):
        importer.load(
            MULTI_STREAM_XDF_PATH,
            stream_types=["EEG", "EMG"],
            reference_stream="NonExistent",
        )


# ============================================================================
# Timestamp channel tests
# ============================================================================


def test_include_timestamps_single_stream():
    """Test include_timestamps option with single stream."""
    importer = XDFImporter()
    rec = importer.load(SAMPLE_XDF_PATH, include_timestamps=True)

    # Should have original channels plus one timestamp channel
    channel_names = list(rec.channels.keys())

    # Find the timestamp channel
    ts_channels = [ch for ch in channel_names if "_LSL_timestamps" in ch]
    assert len(ts_channels) == 1

    # Check timestamp channel properties
    ts_channel = ts_channels[0]
    assert rec.channels[ts_channel]["physical_dimension"] == "s"
    assert rec.channels[ts_channel]["channel_type"] == "MISC"

    # Verify timestamp values are reasonable (should be monotonically increasing)
    ts_values = rec.signals[ts_channel].values
    assert np.all(np.diff(ts_values) >= 0), "Timestamps should be monotonically increasing"


def test_include_timestamps_multistream():
    """Test include_timestamps option with multi-stream file."""
    rec = Recording.from_file(MULTI_STREAM_XDF_PATH, include_timestamps=True)

    channel_names = list(rec.channels.keys())

    # Should have timestamp channels for each numeric stream (EEG, EMG, Mocap)
    ts_channels = [ch for ch in channel_names if "_LSL_timestamps" in ch]
    assert len(ts_channels) == 3  # EEG, EMG, and Mocap (not Markers)

    # Verify each timestamp channel
    for ts_ch in ts_channels:
        assert rec.channels[ts_ch]["physical_dimension"] == "s"
        assert rec.channels[ts_ch]["channel_type"] == "MISC"


def test_include_timestamps_by_type():
    """Test include_timestamps with stream type selection."""
    rec = Recording.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EMG"], include_timestamps=True)

    channel_names = list(rec.channels.keys())

    # Should have EMG channels plus one timestamp channel
    ts_channels = [ch for ch in channel_names if "_LSL_timestamps" in ch]
    assert len(ts_channels) == 1
    assert "TestEMG_LSL_timestamps" in ts_channels[0]


def test_include_timestamps_default_false():
    """Test that timestamps are not included by default."""
    rec = Recording.from_file(SAMPLE_XDF_PATH)

    channel_names = list(rec.channels.keys())
    ts_channels = [ch for ch in channel_names if "_LSL_timestamps" in ch]

    assert len(ts_channels) == 0, "Timestamp channels should not be included by default"


def test_include_timestamps_export_roundtrip():
    """Test that timestamp channels survive EDF export and reimport."""
    import os
    import tempfile

    # Load with timestamps
    rec = Recording.from_file(SAMPLE_XDF_PATH, include_timestamps=True)

    # Find timestamp channel
    ts_channels = [ch for ch in rec.channels.keys() if "_LSL_timestamps" in ch]
    assert len(ts_channels) == 1
    ts_channel = ts_channels[0]
    original_ts = rec.signals[ts_channel].values.copy()

    # Export to EDF
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        temp_path = f.name

    try:
        rec.to_edf(temp_path, format="edf")

        # Reload from EDF
        rec_reloaded = Recording.from_file(temp_path)

        # Check timestamp channel exists in reloaded data
        # Note: EDF has 16-char limit, so "_LSL_timestamps" may be truncated to "_LSL_t"
        reloaded_channels = list(rec_reloaded.channels.keys())
        reloaded_ts_channels = [ch for ch in reloaded_channels if "_LSL_t" in ch]
        assert len(reloaded_ts_channels) == 1

        # Check values are preserved (with some tolerance for EDF precision)
        reloaded_ts = rec_reloaded.signals[reloaded_ts_channels[0]].values
        min_len = min(len(original_ts), len(reloaded_ts))
        correlation = np.corrcoef(original_ts[:min_len], reloaded_ts[:min_len])[0, 1]
        assert correlation > 0.99, f"Timestamp correlation is {correlation}"

    finally:
        os.unlink(temp_path)
        tsv_path = temp_path.replace(".edf", "_channels.tsv")
        if os.path.exists(tsv_path):
            os.unlink(tsv_path)
