# Trigno Format

The Delsys Trigno wireless EMG system exports data in a CSV format. This page describes the format and how biosigIO works with it.

## Format Description

Trigno CSV files typically have the following structure:

- A header section with metadata
- A data section with columns for each channel
- EMG and accelerometer data in separate columns

### Example File Structure

A Trigno CSV export begins with one metadata line per channel, each starting
with `Label:` and carrying that channel's sampling frequency and unit. The
data section that follows is introduced by a header line containing `X[s]`
time columns (one per channel). For example:

```
Label: EMG1  Sampling frequency: 1925.926 Unit: V  Domain: Time
Label: ACC1.X  Sampling frequency: 148.148 Unit: g  Domain: Time
...
X[s],EMG1,X[s].1,ACC1.X,...
0.000,0.01,0.000,0.10,...
0.001,0.02,0.007,0.10,...
```

The importer detects the start of the data section by finding the line that
contains `X[s]`, and reads each channel's sampling frequency and unit from the
preceding `Label:` lines.

## Channel Types

biosigIO assigns channel types from the channel label:

- **EMG** - Electromyography channels (label contains `EMG`)
- **ACC** - Accelerometer channels (label contains `ACC`)
- **GYRO** - Gyroscope channels (label contains `GYRO`)
- **OTHER** - Any channel whose label matches none of the above

## Importer Implementation

The Trigno importer in biosigIO (`biosigio.importers.trigno`) processes these files by:

1. Reading the `Label:` header lines to extract each channel's sampling frequency and unit
2. Locating the data section via the `X[s]` header line and parsing it as a pandas DataFrame
3. Identifying channel types based on naming conventions
4. Setting the per-channel units and sampling frequencies parsed from the header

## Code Example

```python
from biosigio import Recording

# Load data from Trigno CSV file
rec = Recording.from_file('data.csv', importer='trigno')

# Print identified channel types
channel_types = rec.get_channel_types()
print(f"Identified channel types: {channel_types}")

# Get EMG channels
emg_channels = rec.get_channels_by_type('EMG')
print(f"EMG channels: {emg_channels}")

# Get accelerometer channels
acc_channels = rec.get_channels_by_type('ACC')
print(f"ACC channels: {acc_channels}")
```

## Notes and Limitations

- Trigno systems can have different sampling rates for EMG and accelerometer channels
- The importer reads each channel's sampling frequency from its `Label:` header line
- Column names may vary between different Trigno system versions
- Some metadata may be missing depending on the export settings
- Each data column is matched to its `Label:` line by label, so a file whose `Label:` lines repeat a label raises `ValueError` rather than dropping the repeated channel. Trigno's own labels carry the sensor number, so its exports do not repeat one
- Channels in a Trigno export often have different sampling rates (e.g. EMG vs ACC); EDF/BDF export requires a single rate, so resample to a common rate before exporting