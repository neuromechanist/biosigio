# XDF Example

This example demonstrates how to work with XDF files from Lab Streaming Layer (LSL) recordings, including exploring multi-stream files and selective loading.

## Exploring XDF Contents

XDF files often contain multiple streams. Before loading, explore what's available:

```python
from biosigio.importers.xdf import summarize_xdf

# Summarize all streams in the file
summary = summarize_xdf('examples/multi_stream_test.xdf')
print(summary)
```

Output:
```
XDF File: examples/multi_stream_test.xdf
Number of streams: 4

Stream 1: TestEEG
  Type: EEG
  Channels: 8
  Nominal srate: 256.0 Hz
  Effective srate: 256.20 Hz
  Samples: 1280
  Duration: 5.00 s
  Format: float32
  Channel labels: EEG1, EEG2, EEG3, EEG4, EEG5, ... (+3 more)

Stream 2: TestEMG
  Type: EMG
  Channels: 2
  Nominal srate: 2048.0 Hz
  Effective srate: 2048.20 Hz
  Samples: 10240
  Duration: 5.00 s
  Format: float32
  Channel labels: EMG_L, EMG_R

Stream 3: TestMocap
  Type: Mocap
  Channels: 6
  Nominal srate: 120.0 Hz
  Effective srate: 120.20 Hz
  Samples: 600
  Duration: 4.99 s
  Format: float32
  Channel labels: Marker1_X, Marker1_Y, Marker1_Z, Marker2_X, Marker2_Y, ... (+1 more)

Stream 4: TestMarkers
  Type: Markers
  Channels: 1
  Nominal srate: 0.0 Hz
  Effective srate: 1.25 Hz
  Samples: 5
  Duration: 4.00 s
  Format: string
  Channel labels: Ch1
```

## Finding Specific Streams

```python
# Find all EMG streams
emg_streams = summary.get_streams_by_type('EMG')
for stream in emg_streams:
    print(f"Found EMG stream: {stream.name} with {stream.channel_count} channels")

# Find a specific stream by name
mocap = summary.get_stream_by_name('TestMocap')
if mocap:
    print(f"Mocap rate: {mocap.nominal_srate} Hz")
    print(f"Mocap channels: {mocap.channel_labels}")
```

## Loading All Numeric Data

```python
from biosigio import Recording

# Load all numeric streams (EEG, EMG, Mocap - excludes Markers)
rec = Recording.from_file('examples/multi_stream_test.xdf')

print(f"Total channels: {len(rec.channels)}")
print(f"Channel names: {list(rec.channels.keys())}")
```

## Selective Stream Loading

### Load by Stream Type

```python
# Load only EMG data
emg_data = Recording.from_file('examples/multi_stream_test.xdf', stream_types=['EMG'])
print(f"EMG channels: {list(emg_data.channels.keys())}")
# Output: ['EMG_L', 'EMG_R']

# Load EEG and EMG together
combined = Recording.from_file('examples/multi_stream_test.xdf', stream_types=['EEG', 'EMG'])
print(f"Combined channels: {len(combined.channels)}")
# Output: 10 (8 EEG + 2 EMG)
```

### Load by Stream Name

```python
# Load specific streams by name
emg_data = Recording.from_file('examples/multi_stream_test.xdf', stream_names=['TestEMG'])
```

## Working with Multi-Rate Data

When loading streams with different sampling rates, all channels are interpolated
onto a common time base. By default the highest-rate stream is the reference (to
avoid downsampling), but each channel keeps its own original `sample_frequency`
in `channels`, so a recording loaded this way is mixed-rate.

```python
# Load EEG (256 Hz) and EMG (2048 Hz)
combined = Recording.from_file('examples/multi_stream_test.xdf', stream_types=['EEG', 'EMG'])

# Each channel reports its own original sampling frequency
for ch, info in combined.channels.items():
    print(f"{ch}: {info['sample_frequency']} Hz")
# EEG channels report 256.0 Hz, EMG channels report 2048.0 Hz
```

Note: a mixed-rate recording cannot be exported with `to_edf` (it requires a
single rate across all channels and raises `ValueError` otherwise). Use
`select_channels` to export one rate group at a time, or `resample` to a common
rate first.

## Preserving LSL Timestamps

XDF files contain per-sample LSL timestamps. To preserve these for synchronization:

```python
# Load with timestamp channels
rec = Recording.from_file('examples/multi_stream_test.xdf',
                    stream_types=['EMG'],
                    include_timestamps=True)

# Each stream gets a timestamp channel
print(list(rec.channels.keys()))
# ['EMG_L', 'EMG_R', 'TestEMG_LSL_timestamps']

# Access the original LSL timestamps
ts = rec.signals['TestEMG_LSL_timestamps']
print(f"First timestamp: {ts.iloc[0]:.6f}s")
print(f"Last timestamp: {ts.iloc[-1]:.6f}s")
```

## Exporting to EDF

After loading, export to EDF/BDF format:

```python
# Load EMG streams with timestamps for synchronization
rec = Recording.from_file('examples/multi_stream_test.xdf',
                    stream_types=['EMG'],
                    include_timestamps=True)

# Export to EDF (timestamps are preserved as a channel)
rec.to_edf('output_emg.edf')

# Verify the export
rec_reloaded = Recording.from_file('output_emg.edf')
print(f"Exported channels: {list(rec_reloaded.channels.keys())}")
```

## Complete Workflow Example

```python
from biosigio import Recording
from biosigio.importers.xdf import summarize_xdf

# 1. Explore the file
summary = summarize_xdf('recording.xdf')
print(summary)

# 2. Identify streams of interest
emg_streams = summary.get_streams_by_type('EMG')
print(f"Found {len(emg_streams)} EMG streams")

# 3. Load selected data
rec = Recording.from_file('recording.xdf', stream_types=['EMG'])

# 4. Check loaded data
print(f"Channels: {list(rec.channels.keys())}")
print(f"Duration: {rec.signals.index[-1]:.1f}s")
print(f"Sample rate: {rec.channels[list(rec.channels.keys())[0]]['sample_frequency']} Hz")

# 5. Plot signals
rec.plot_signals(time_range=(0, 5))

# 6. Export
rec.to_edf('emg_export.edf', verify=True)
```

## Notes

- Marker streams (string data) are not loaded as signal channels
- Channels keep their own labels and are not prefixed with the stream name;
  when a label repeats, the first occurrence keeps it and later ones become `<label>_1`, `<label>_2`,
  never a label another channel genuinely has (see [Repeated channel labels](../formats/xdf.md#repeated-channel-labels))
- Time indices are normalized to start at 0
- The `pyxdf` package is used internally for reading XDF files
