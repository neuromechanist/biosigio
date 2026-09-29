# EEGLAB Examples

This page provides examples for working with EEGLAB `.set` files using biosigIO. Classic (v5/v7) MATLAB `.set` files are read with `scipy.io.loadmat`; MATLAB v7.3 (HDF5) `.set` files are read with h5py and need the `hdf5` extra (`uv sync --extra hdf5`). See the [EEGLAB format page](../formats/eeglab.md) for details.

## Basic EEGLAB Example

```python
import os
from biosigio import Recording
import matplotlib.pyplot as plt

# Load data from an EEGLAB .set file
data_path = 'path_to_your_eeglab_file.set'
rec = Recording.from_file(data_path, importer='eeglab')

# Print metadata
print("\nMetadata:")
print("-" * 50)
for key in ['subject', 'session', 'condition', 'srate', 'nbchan', 'pnts']:
    if key in rec.metadata:
        print(f"{key}: {rec.get_metadata(key)}")

# Print available channels
print("\nAvailable channels:")
print("-" * 50)
channel_types = rec.get_channel_types()
for ch_type in channel_types:
    channels = rec.get_channels_by_type(ch_type)
    print(f"{ch_type} channels ({len(channels)}):")
    for i, ch_name in enumerate(channels[:5]):  # Print first 5 channels of each type
        ch_info = rec.channels[ch_name]
        print(f"  - {ch_name} (Sampling rate: {ch_info['sample_frequency']} Hz, "
              f"Unit: {ch_info['physical_dimension']})")
    if len(channels) > 5:
        print(f"  ... and {len(channels) - 5} more {ch_type} channels")

# Plot EMG channels
emg_channels = rec.get_channels_by_type('EMG')
if emg_channels:
    # Select EMG channels only
    emg_only = rec.select_channels(emg_channels)
    
    # Plot the first 5 seconds
    plt.figure(figsize=(12, 8))
    emg_only.plot_signals(time_range=(0, 5), title="EMG Signals from EEGLAB")
    plt.tight_layout()
    plt.show()
```

## Working with Events

EEGLAB files often contain event markers. Here's how to access and work with them:

```python
from biosigio import Recording
import numpy as np
import matplotlib.pyplot as plt

# Load EEGLAB data with events
rec = Recording.from_file('data_with_events.set', importer='eeglab')

# EEGLAB events are loaded into rec.events (a DataFrame: onset/duration in
# seconds, description = the EEGLAB event 'type'). The latency (samples) is
# converted to an onset in seconds on import.
if not rec.events.empty:
    print(f"Found {len(rec.events)} events")

    # Print the first 5 events
    print(rec.events.head(5).to_string(index=False))

    # Extract specific event types by description
    movement_events = rec.events[rec.events['description'] == 'movement']
    print(f"Found {len(movement_events)} movement events")

    # Plot signals around the first movement event
    if not movement_events.empty:
        # Onsets are already in seconds.
        event_time = movement_events.iloc[0]['onset']

        # Plot 2 seconds before and after the event. Pass show=False so the
        # event marker can be overlaid before displaying.
        window = 2  # seconds
        rec.plot_signals(
            time_range=(event_time - window, event_time + window),
            title=f"EMG around movement event at {event_time:.2f}s",
            show=False
        )

        # Add a vertical line at the event time
        plt.axvline(x=event_time, color='r', linestyle='--', label='Movement Event')
        plt.legend()
        plt.show()
```

## Inspecting Epoch Metadata

EEGLAB stores the number of epochs in the `trials` field and the samples per
epoch in `pnts`. These are read into the recording metadata, so you can inspect
them after loading. `get_metadata` returns `None` when a key is absent, so guard
for that before comparing:

```python
from biosigio import Recording
import matplotlib.pyplot as plt

# Load EEGLAB data
rec = Recording.from_file('epoched_data.set', importer='eeglab')

# trials > 1 indicates epoched data; get_metadata returns None if absent
trials = rec.get_metadata('trials')
is_epoched = trials is not None and trials > 1
print(f"Data is {'epoched' if is_epoched else 'continuous'}")

if is_epoched:
    n_epochs = rec.get_metadata('trials')
    epoch_length = rec.get_metadata('pnts')
    fs = rec.get_sampling_frequency()
    epoch_duration = epoch_length / fs

    print(f"Number of epochs: {n_epochs}")
    print(f"Epoch length: {epoch_length} samples ({epoch_duration:.2f} seconds)")

# Plot the first few seconds (plot_signals manages its own figure)
rec.plot_signals(time_range=(0, 5), title="EEGLAB Signals")
```

## Exporting EEGLAB Data to EDF/BDF

Converting EEGLAB data to EDF/BDF format:

```python
from biosigio import Recording

# Load EEGLAB data
rec = Recording.from_file('data.set', importer='eeglab')

# Export all channels to EDF/BDF
output_path = 'eeglab_all_channels'
rec.to_edf(output_path)  # Format (EDF/BDF) selected automatically

# Export only EMG channels
emg_only = rec.select_channels(channel_type='EMG')
output_path = 'eeglab_emg_only'
emg_only.to_edf(output_path)

print("Conversion complete!")
```

This example demonstrates loading EEGLAB `.set` files, exploring their structure, working with events, handling epoched data, and exporting to EDF/BDF format. 