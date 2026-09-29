# XDF Format

biosigIO supports loading data stored in the Extensible Data Format (XDF), the native format for Lab Streaming Layer (LSL) recordings. XDF files can contain multiple synchronized data streams from various devices, making them common in multi-modal experiments.

## File Structure

XDF files are self-contained binary files that can hold:

- **Multiple Streams:** Each stream represents a data source (e.g., EEG, EMG, motion capture, markers)
- **Stream Headers:** Metadata for each stream including name, type, channel count, sampling rate, and channel labels
- **Time Series Data:** The actual samples with timestamps
- **Stream Footers:** Summary information like sample counts and time ranges

Common file extensions:

- `.xdf`: Standard XDF file
- `.xdfz`: Compressed XDF file (gzip)

## Exploring XDF Files

Before loading, you can explore the contents of an XDF file using `summarize_xdf`:

```python
from biosigio.importers.xdf import summarize_xdf

# Get a summary of all streams in the file
summary = summarize_xdf('recording.xdf')
print(summary)

# Find streams by type
emg_streams = summary.get_streams_by_type('EMG')
eeg_streams = summary.get_streams_by_type('EEG')

# Find a specific stream by name
stream = summary.get_stream_by_name('MyEMGDevice')
if stream:
    print(f"Channels: {stream.channel_count}")
    print(f"Sample rate: {stream.nominal_srate} Hz")
    print(f"Duration: {stream.duration_seconds:.1f}s")
```

## Loading Data

### Basic Loading

Load all numeric streams from an XDF file:

```python
from biosigio.core.emg import Recording

# Load all numeric streams (excludes marker/string streams)
rec = Recording.from_file('recording.xdf')
```

### Selective Loading

XDF files often contain many streams. You can select specific streams:

```python
# Load only EMG streams
rec = Recording.from_file('recording.xdf', stream_types=['EMG'])

# Load specific stream types
rec = Recording.from_file('recording.xdf', stream_types=['EMG', 'EXG'])

# Load by stream name
rec = Recording.from_file('recording.xdf', stream_names=['MyEMGDevice'])

# Load by stream ID
rec = Recording.from_file('recording.xdf', stream_ids=[1, 3])
```

## Multi-Stream Handling

When loading multiple streams with different sampling rates, biosigIO:

1. Resamples all streams to a common time base
2. Uses the highest sampling rate among selected streams
3. Keeps each channel's own label from the stream header; a channel with no label gets `<stream name>_Ch<n>`

```python
# Load EEG and EMG together (different sample rates)
rec = Recording.from_file('recording.xdf', stream_types=['EEG', 'EMG'])

print(list(rec.channels.keys()))
# ['Fp1', 'Fp2', ..., 'Bicep', 'Tricep']
```

### Repeated channel labels

Streams often share labels: two amplifiers that each call their first channel `Ch1`, or two streams with the same name.
A `Recording` is keyed by label, so no two channels may share one.
The first occurrence keeps its label, and each later one, in stream order, becomes `<label>_1`, `<label>_2`, and so on.
A suffix never takes a label that some stream really uses:
with streams `A: [Ch1, Ch1]` and `B: [Ch1_1]`, the channels are `Ch1`, `Ch1_2` and `Ch1_1`, and `Ch1_1` is stream B's own channel.

```python
rec = Recording.from_file('two_amplifiers.xdf')
print(list(rec.channels.keys()))
# ['Ch1', 'Ch2', 'Ch1_1', 'Ch2_1']
print(rec.get_metadata('channel_labels_deduplicated'))
# {'Ch1_1': 'Ch1', 'Ch2_1': 'Ch2'}
```

A warning names each renamed channel with the stream it came from,
and the renames are recorded in the metadata under `channel_labels_deduplicated` as `{new_label: original_label}`, the same key the EDF/BDF, WFDB, EEGLAB, neo and Zarr importers use.
A recording whose labels are already unique carries no such entry.

This is the scheme XDF import has always used, so no channel that imported under its own label before is renamed now.
It keeps the `_1` suffix rather than the `-0`, `-1` suffixes of [EDF repeated channel labels](edf.md#repeated-channel-labels),
and it does not prefix the stream name: a prefix would rename channels that import unchanged today,
and it still could not separate a label repeated within one stream, or two streams that share a name.

## Stream Types

Common LSL stream types you might encounter:

| Type | Description |
|------|-------------|
| `EEG` | Electroencephalography |
| `EMG` | Electromyography |
| `EXG` | Generic ExG (bio-electric signals) |
| `ECG` | Electrocardiography |
| `EOG` | Electrooculography |
| `Mocap` | Motion capture |
| `Markers` | Event markers (string data) |
| `Gaze` | Eye tracking |
| `Audio` | Audio signals |

## Channel Type Detection

The XDF importer attempts to infer channel types from:

1. Stream type in the XDF header
2. Channel labels (e.g., "EMG_Bicep" -> EMG)
3. A default type you can specify

```python
# Set default channel type for streams without explicit type info
rec = Recording.from_file('recording.xdf', default_channel_type='EMG')
```

## Metadata

Loaded XDF files include metadata such as:

- `device`: Set to "XDF"
- `source_file`: Path to the XDF file
- `stream_count`: Number of selected streams
- `srate`: Sampling rate of the reference (highest-rate) stream

```python
rec = Recording.from_file('recording.xdf')
print(rec.get_metadata('stream_count'))
print(rec.get_metadata('srate'))
```

## Preserving LSL Timestamps

XDF files store per-sample timestamps from the Lab Streaming Layer clock. When exporting to formats like EDF that require regular sampling, these timestamps are normally lost during resampling.

To preserve the original LSL timestamps, use the `include_timestamps` option:

```python
# Load with timestamp preservation
rec = Recording.from_file('recording.xdf', include_timestamps=True)

# Each stream gets a timestamp channel named "{stream_name}_LSL_timestamps"
print(list(rec.channels.keys()))
# ['MyEMG_Ch1', 'MyEMG_Ch2', 'MyEMG_LSL_timestamps']
```

Timestamp channels:

- Are created once per stream, so two streams that share a name each keep their timestamps; the second becomes `{stream_name}_LSL_timestamps_1`, and a data channel that already carries a timestamp channel's name keeps it (see [Repeated channel labels](#repeated-channel-labels))
- Contain the original LSL timestamps in seconds
- Are marked with `channel_type='MISC'` and `physical_dimension='s'`
- Are resampled along with the data when multiple streams have different rates
- Can be exported to EDF/BDF for later synchronization analysis (EDF/BDF export requires all channels to share one sampling rate)

This is useful for:

- Synchronizing with other data sources recorded with LSL
- Analyzing timing jitter in the original recording
- Post-hoc alignment with marker streams

## Marker Streams

Marker/event streams contain string data and are not loaded as signal channels. Use `summarize_xdf()` to inspect marker streams:

```python
summary = summarize_xdf('recording.xdf')
marker_stream = summary.get_stream_by_name('EventMarkers')
if marker_stream:
    print(f"Format: {marker_stream.channel_format}")  # 'string'
    print(f"Events: {marker_stream.sample_count}")
```

## Requirements

The XDF importer requires the `pyxdf` package. It is included as a core
dependency when installing biosigIO, so no additional installation is required.
