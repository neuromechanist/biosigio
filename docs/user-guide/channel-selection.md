# Channel Selection

EMG recordings often contain various channel types (EMG, accelerometer, gyroscope, etc.) or channels you may want to exclude from analysis. biosigIO provides flexible methods for selecting and working with specific channels.

## Getting Channel Information

Before selecting channels, you may want to explore the available channels:

```python
# Get a dictionary of all channels and their properties
channels = rec.channels

# Print channel information
for ch_name, ch_info in rec.channels.items():
    print(f"Channel: {ch_name}")
    print(f"  Type: {ch_info['channel_type']}")
    print(f"  Unit: {ch_info['physical_dimension']}")
    print(f"  Sampling rate: {ch_info['sample_frequency']} Hz")
```

## Selecting Channels by Name

You can create a new Recording object with only the channels you specify:

```python
# Select specific channels by name
subset_emg = rec.select_channels(['EMG1', 'EMG2', 'ACC1'])

# The original Recording object remains unchanged
print(f"Original channels: {len(rec.channels)}")
print(f"Selected channels: {len(subset_emg.channels)}")
```

Each name may appear once: `select_channels(['EMG1', 'EMG1'])` raises `ValueError`,
because a second copy of a column would leave the new recording's signals and channel table disagreeing.

## Selecting Channels by Type

biosigIO allows you to select channels based on their type:

```python
# Get available channel types
channel_types = rec.get_channel_types()
print(f"Available channel types: {channel_types}")

# Get channel names of a specific type
emg_channels = rec.get_channels_by_type('EMG')
print(f"EMG channels: {emg_channels}")
acc_channels = rec.get_channels_by_type('ACC')
print(f"ACC channels: {acc_channels}")

# Select all channels of a specific type
emg_only = rec.select_channels(channel_type='EMG')
acc_only = rec.select_channels(channel_type='ACC')
```

## Selecting Channels by Regular Expression

You can also select channels using regular expressions:

```python
import re

# Select all EMG channels with numbers 1-5
pattern = re.compile(r'EMG[1-5]')
selected_channels = [ch for ch in rec.channels if pattern.match(ch)]
subset_emg = rec.select_channels(selected_channels)
```

## Working with Selected Channels

After selecting channels, you can work with the new Recording object just like the original:

```python
# Plot the selected channels
subset_emg.plot_signals()

# Export only the selected channels
subset_emg.to_edf('output_subset')
```

## Combining Selection Methods

You can combine different selection methods for more complex filtering:

```python
# Get all EMG channels
emg_channels = rec.get_channels_by_type('EMG')

# Filter to only include those from a specific muscle group
bicep_channels = [ch for ch in emg_channels if 'Bicep' in ch]

# Create a new Recording object with just these channels
bicep_emg = rec.select_channels(bicep_channels)
```

## Channel Selection Best Practices

- Always verify the selected channels by checking `rec.channels` after selection
- When working with multiple EMG devices, use channel selection to group channels by device
- For large datasets, selecting only necessary channels can improve performance
- Consider creating utility functions for commonly used channel selections in your workflow 