# biosigIO

[![PyPI version](https://badge.fury.io/py/biosigio.svg)](https://badge.fury.io/py/biosigio)
[![Tests](https://github.com/neuromechanist/biosigio/actions/workflows/tests.yml/badge.svg)](https://github.com/neuromechanist/biosigio/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/neuromechanist/biosigio/branch/main/graph/badge.svg?token=63EDIA9TWD)](https://codecov.io/gh/neuromechanist/biosigio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20534173.svg)](https://doi.org/10.5281/zenodo.20534173)

A Python package for biosignal import/export and manipulation across modalities (EEG, EMG, iEEG, MEG, and behavioral/marker streams). biosigIO provides a unified `Recording` interface for loading data from many acquisition systems and archives (EEGLAB, Delsys Trigno, OTB, EDF/BDF, WFDB, XDF, MEG/CTF/KIT/4D-BTi and BrainVision via MNE, MEF3 iEEG via MNE+pymef, and proprietary electrophysiology such as Intan/Blackrock via python-neo) and exporting it to standardized and serving formats (EDF/BDF, Parquet, Arrow, Zarr) with harmonized metadata.

The determination of the EDF/BDF format is based on the dynamic range of the data. If the data is within the range of 16-bit integers (~90dB), the EDF format is used. Otherwise, the BDF format is used. This is to ensure that the data is stored in the most efficient format possible. This determination is made automatically using SVD decomposition and/or FFT to determine the dynamic range of the data. (Alternatively, the user can override the format selection by explicitly indicating their desired format).

## Documentation

The documentation including installation instructions, examples, and API reference is available at [https://neuromechanist.github.io/biosigio/](https://neuromechanist.github.io/biosigio/).

What changed in each release is in [CHANGELOG.md](https://github.com/neuromechanist/biosigio/blob/main/CHANGELOG.md).

**Breaking in 1.2.10:** biosigio 1.2.10 is a patch release that contains breaking changes:
`Recording.select_channels` raises `ValueError` for a name listed twice (and now accepts any iterable of names);
the Parquet/Arrow importer raises `ValueError` for a Feather/Arrow table that repeats a column name;
EDF/BDF export numbers labels that collide once truncated to 16 characters;
XDF, neo and EEGLAB suffixes for repeated labels never take a label the file genuinely uses, so a label such as `Ch1_1` can now point at a different signal;
a `channels.tsv` row matched to its channel case-insensitively now applies its type and unit, which can convert the signal;
and, for anyone upgrading from before 1.2.9, `Recording.add_channel` still raises for a label that already exists (below).
To adapt, de-duplicate `select_channels` arguments and re-check stores regenerated from files with repeated labels;
to defer, pin `biosigio==1.2.9` (a `~=1.2.9` or `<1.3` pin takes 1.2.10 automatically).
Details are in the [changelog](https://github.com/neuromechanist/biosigio/blob/main/CHANGELOG.md).

**Breaking in 1.2.9:** `Recording.add_channel` raises `ValueError` for a label that already exists instead of silently replacing that channel; use `set_channel` to change metadata or assign `rec.signals[label]` to replace samples.

## Features

- Import biosignal recordings from many systems and archives:
  - EEGLAB set files (supported, classic and MATLAB v7.3; v7.3 needs the `hdf5` extra)
  - Delsys Trigno (supported)
  - OTB Systems (supported)
  - EDF/BDF(+) (supported, including annotations)
  - WFDB (supported, including annotations)
  - XDF/Lab Streaming Layer (supported, multi-stream)
  - MEG: `.fif`, CTF `.ds`, KIT/Yokogawa `.con`/`.sqd`/`.kdf`, and 4D Neuroimaging/BTi (a directory, detected by content) via MNE (supported; `meg` extra)
  - BrainVision `.vhdr` via MNE (supported; `meg` extra)
  - MEF3 iEEG `.mefd` via MNE (supported; `mef3` extra -- mne>=1.12 plus pymef)
  - Proprietary electrophysiology via python-neo: Intan, Blackrock, Spike2, Plexon, Micromed, Neuralynx (supported; `neo` extra)
  - Generic CSV (supported with auto-detection)
  - Noraxon (planned)
  
- Smart import:
  - Automatic file format detection based on extension
  - Specialized format detection for CSV files
  - Custom importers for system-specific formats
  - Automatic annotation/event loading (WFDB, EDF+/BDF+, and EEGLAB .set) into the events table, embedded back on EDF+/BDF+ export and carried in the Parquet/Arrow/Zarr serialization formats
  - LSL timestamp preservation for XDF files (for synchronization)
  
- Export to standardized formats:
  - EDF/BDF(+) with channels.tsv metadata (automatically selects format based on signal properties, preserves annotations)

- Serialization & serving (see [docs](https://neuromechanist.github.io/biosigio/formats/serialization/)):
  - Parquet and Arrow/Feather: lossless columnar round-trip (analytics, fast IPC); requires the `arrow` extra
  - Zarr: cloud-native serving store (one store serves viewing, inference, and training), a derived downsampled copy; requires the `zarr` extra
  
- Data manipulation:
  - Channel selection
  - Metadata handling
  - Event/Annotation handling (access, add)
  - Basic signal visualization
  - Raw data access and modification

## Installation

biosigIO uses [UV](https://docs.astral.sh/uv/) for Python environment and package management.

### From PyPI (recommended)

```bash
uv pip install biosigio
```

(If your own project is uv-managed, use `uv add biosigio` to track it as a dependency.)

### From source

```bash
git clone https://github.com/neuromechanist/biosigio.git
cd biosigio
uv pip install .
```

## Usage

### Basic Example

```python
from biosigio import Recording

# Load data with automatic format detection
rec = Recording.from_file('data.csv')  # Format detected from file extension

# Load data with explicit importer
rec = Recording.from_file('data.csv', importer='trigno')

# Plot specific channels
rec.plot_signals(['EMG1', 'EMG2'])

# Export to EDF or BDF (format automatically determined)
rec.to_edf('output.edf')
```

### Generic CSV Import

```python
# Import a generic CSV file
rec = Recording.from_file('data.csv', importer='csv',
                   sample_frequency=1000,  # Required if no time column
                   has_header=True,        # Whether file has header row
                   channel_names=['EMG_L', 'EMG_R', 'ACC_X'])
```

### Channel Selection

```python
# Select specific channels
subset_emg = rec.select_channels(['EMG1', 'EMG2', 'ACC1'])

# Select all channels of a specific type
emg_only = rec.select_channels(channel_type='EMG')

# Plot selected channels
subset_emg.plot_signals()
```

### Metadata Handling

```python
# Set metadata
rec.set_metadata('subject', 'S001')
rec.set_metadata('condition', 'resting')

# Get metadata
subject = rec.get_metadata('subject')
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/neuromechanist/biosigio.git
cd biosigio
```

2. Install for development (editable install with dev dependencies):
```bash
uv sync --extra dev
```

### Running Tests

```bash
uv run pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgment
This project is partially supported by a Meta Reality Labs gift to @sccn and NIH 5R01NS047293.