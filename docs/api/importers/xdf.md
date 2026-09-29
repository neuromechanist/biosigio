# XDF Importer

The `XDFImporter` class handles importing data from XDF (Extensible Data Format) files, the native format for Lab Streaming Layer (LSL) recordings. It supports multi-stream files with different sampling rates and data types.

## Class Documentation

::: biosigio.importers.xdf
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Examples

### Basic Loading

```python
from biosigio import Recording
from biosigio.importers.xdf import XDFImporter

# Method 1: Using Recording.from_file (recommended)
rec = Recording.from_file('recording.xdf')

# Method 2: Using the importer directly
importer = XDFImporter()
rec = importer.load('recording.xdf')
```

### Exploring File Contents

Before loading, explore what streams are available:

```python
from biosigio.importers.xdf import summarize_xdf

summary = summarize_xdf('recording.xdf')
print(summary)

# Output example:
# XDF File: recording.xdf
# ----------------------------------------
# Stream 1: MyEEG (EEG)
#   Channels: 8, Rate: 256.0 Hz
#   Samples: 15360, Duration: 60.0s
# Stream 2: MyEMG (EMG)
#   Channels: 2, Rate: 2048.0 Hz
#   Samples: 122880, Duration: 60.0s
# Stream 3: Markers (Markers)
#   Channels: 1, Rate: 0.0 Hz (irregular)
#   Samples: 10
```

### Selective Stream Loading

```python
# Load only specific stream types
rec = Recording.from_file('recording.xdf', stream_types=['EMG'])

# Load multiple types
rec = Recording.from_file('recording.xdf', stream_types=['EMG', 'EEG'])

# Load by stream name
rec = Recording.from_file('recording.xdf', stream_names=['MyEMGDevice'])

# Load by stream ID
rec = Recording.from_file('recording.xdf', stream_ids=[2])
```

### Setting Default Channel Type

```python
# For streams without explicit channel type metadata
rec = Recording.from_file('recording.xdf', default_channel_type='EMG')
```

### Preserving LSL Timestamps

```python
# Include original LSL timestamps as additional channels
rec = Recording.from_file('recording.xdf', include_timestamps=True)

# Each stream gets a "{stream_name}_LSL_timestamps" channel
# Useful for synchronization with other LSL-recorded data
```

## File Format Support

The XDF importer supports:

1. Single-stream and multi-stream XDF files
2. Compressed XDF files (.xdfz)
3. Numeric data types: float32, float64, int8, int16, int32, int64
4. Different sampling rates across streams (with resampling)
5. Channel labels from stream descriptors

## Stream Selection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stream_names` | `list[str]` | Filter by stream names (case-insensitive) |
| `stream_types` | `list[str]` | Filter by stream types (e.g., "EMG", "EEG") |
| `stream_ids` | `list[int]` | Filter by stream IDs |
| `default_channel_type` | `str` | Default type for channels without explicit type |
| `include_timestamps` | `bool` | If True, add LSL timestamp channels for each stream |

## Return Values

The `load()` method returns a `Recording` object with:

### Signals (pandas.DataFrame)
- Time-indexed signal data
- Channels as columns
- Resampled to common time base if multiple streams

### Channels (dict)
For each channel:
- `channel_type`: Inferred or default type
- `physical_dimension`: Unit (default "a.u.")
- `sample_frequency`: Effective sampling rate
- `prefilter`: Pre-filtering string (default "n/a")
- `modality`: Coarse modality inferred from the channel type

### Metadata (dict)
- `device`: "XDF"
- `source_file`: Path to the XDF file
- `stream_count`: Number of selected streams
- `srate`: Sampling rate of the reference stream (base sample rate)

## Helper Classes

### XDFSummary

Provides an overview of the XDF file:

```python
summary = summarize_xdf('recording.xdf')

# Access all streams
for stream in summary.streams:
    print(f"{stream.name}: {stream.channel_count} channels")

# Find streams by type
emg_streams = summary.get_streams_by_type('EMG')

# Find stream by name
stream = summary.get_stream_by_name('MyDevice')
```

### XDFStreamInfo

Contains metadata for a single stream:

- `stream_id`: Unique stream identifier
- `name`: Stream name
- `stream_type`: Stream type (EEG, EMG, etc.)
- `channel_count`: Number of channels
- `nominal_srate`: Declared sampling rate
- `effective_srate`: Actual measured sampling rate
- `channel_format`: Data format (float32, string, etc.)
- `source_id`: Source identifier
- `hostname`: Recording machine hostname
- `sample_count`: Number of samples
- `duration_seconds`: Recording duration
- `channel_labels`: List of channel names

## Implementation Notes

1. **String/Marker Streams:** Streams with `channel_format='string'` are excluded from signal loading but appear in summaries.

2. **Time Alignment:** When loading multiple streams, timestamps are aligned to start at 0.

3. **Resampling:** Multiple streams with different rates are resampled using linear interpolation to the highest rate.

4. **Channel Naming:** Each channel keeps its own label from the stream header, without a stream-name prefix;
   a channel with no label gets `<stream name>_Ch<n>`.
   When a label repeats, the first occurrence keeps it and later ones become `<label>_1`, `<label>_2`, and so on,
   never a label another channel genuinely has
   (see [Repeated channel labels](../../formats/xdf.md#repeated-channel-labels)).
