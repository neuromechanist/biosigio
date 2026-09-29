# Trigno Importer

The `TrignoImporter` class is responsible for importing EMG data from Delsys Trigno CSV files.

## Class Documentation

::: biosigio.importers.trigno
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording
from biosigio.importers.trigno import TrignoImporter

# Method 1: Using Recording.from_file (recommended)
rec = Recording.from_file('data.csv', importer='trigno')

# Method 2: Using the importer directly
rec = TrignoImporter().load('data.csv')
```

## File Format Requirements

The importer expects a Delsys Trigno CSV export with:

1. A per-channel metadata header. Each channel is described by a line of the
   form `Label: <name> Sampling frequency: <Hz> ... Unit: <unit> Domain: ...`,
   from which the importer parses the channel name, its sampling frequency, and
   its physical unit.
2. A data header row whose first column is `X[s]` (the time axis in seconds);
   the importer treats the line containing `X[s]` as the start of the numeric
   data section.
3. The numeric data section that follows, with the time column first and one
   column per channel.

Because Trigno records each sensor at its own rate, the parsed
`sample_frequency` is stored per channel.

## Channel Type Detection

The Trigno importer automatically detects channel types based on channel names:

- Channels whose name contains `EMG` are classified as `EMG`
- Channels whose name contains `ACC` are classified as `ACC`
- Channels whose name contains `GYRO` are classified as `GYRO`
- Other channels are classified as `OTHER`

## Parameters

`TrignoImporter().load(filepath)` takes:

- **filepath (str)**: Path to the Trigno CSV file.

Per-channel sampling frequency and units are read from the file's `Label: ...`
header lines, so no manual header/delimiter arguments are required.
A label that repeats is matched to its columns by position, each column taking its own `Label:` line;
`load()` raises `ValueError` if such a file's header line does not have one field per data column
(see [the Trigno format page](../../formats/trigno.md#notes-and-limitations)).

## Return Value

The `load()` method returns a single `Recording` object populated with the
parsed signals, per-channel information (type, unit, sampling frequency), and
recording metadata (including `source_file` and `device`).