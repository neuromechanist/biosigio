# EEGLAB Format

EEGLAB is a MATLAB toolbox for processing EEG, MEG, and other electrophysiological data. biosigIO can import EEGLAB `.set` files to work with biosignal data stored in this format.

## Format Description

EEGLAB `.set` files are MATLAB files that contain:

- Signal data in a matrix format
- Channel information (names, locations, types)
- Event markers
- Metadata about the recording
- Processing history

### Structure

A typical EEGLAB `.set` file contains these fields:

| Field | Description |
|-------|-------------|
| `data` | Signal data matrix (channels × time points) |
| `chanlocs` | Channel information (names, types, locations) |
| `srate` | Sampling rate in Hz |
| `xmin` | Time of first data point (seconds) |
| `xmax` | Time of last data point (seconds) |
| `times` | Time points vector |
| `event` | Event markers |
| `epoch` | Epoch information (if epoched) |
| `subject` | Subject identifier |
| `condition` | Condition name or description |

## Importer Implementation

EEGLAB saves `.set` files in two MATLAB container formats, both read by
`biosigio.importers.eeglab.EEGLABImporter` under the same `.set` extension:

- **Classic MATLAB v5/v7** (a zlib-compressed MAT container), via
  `scipy.io.loadmat`.
- **MATLAB v7.3** (an HDF5 container -- MATLAB/EEGLAB switch to this
  automatically once a variable exceeds ~2 GB, or on an explicit `-v7.3`
  save), via [h5py](https://www.h5py.org/), an optional dependency (the
  `hdf5` extra: `uv sync --extra hdf5`).

The importer tells the two apart by sniffing the file's leading header text
(`b"MATLAB 7.3 MAT-file"` vs. the classic `"MATLAB 5.0 MAT-file"`), not by
extension -- both forms are `.set`.

### Classic (v5/v7) path

1. Loading the `.set` file using `scipy.io.loadmat`
2. Normalizing the two EEGLAB save forms to a flat field map. Real EEGLAB saves a
   dataset as a single MATLAB struct named `EEG`, so `loadmat` returns
   `{'EEG': struct}` with every field nested one level down; the legacy form
   writes the fields at the top level. biosigIO unwraps the nested `EEG` struct
   (and accepts the flat form) before reading any field, so a real-world `.set`
   loads correctly instead of silently importing as an empty recording
3. Extracting the EEG structure with signal data and metadata. When the matrix is
   stored in a separate float32 `.fdt` file (EEGLAB's default for large
   recordings, where `EEG.data` holds the `.fdt` filename), the sibling `.fdt`
   next to the `.set` is loaded and reshaped to `(nbchan, pnts * trials)`
4. Converting channel information to biosigIO's channel format
5. Creating appropriate metadata dictionary
6. Loading event markers into the recording's `events` table (onset/duration in seconds)

### MATLAB v7.3 (HDF5) path

1. Opening the file with h5py and reading the top-level `EEG` group's fields
   directly (`nbchan`, `srate`, `pnts`, `trials`, `data`, ...). A v7.3 file can
   also carry the struct's fields flat at the HDF5 root with no `EEG` group;
   when the root holds `nbchan`, `srate`, `pnts` and `data`, the root is read
   as the struct (an `EEG` group wins when both are present). A file with
   neither layout raises `FileReadError` saying it is not an EEGLAB v7.3
   dataset (not `CorruptFileError`: the HDF5 container itself is valid). Header scalars
   round-trip through HDF5 as 1x1 float arrays rather than true scalars, so
   they are flattened and coerced to int/float explicitly
2. Reading the signal matrix from `EEG.data`. HDF5 stores the matrix
   TRANSPOSED relative to MATLAB -- `(n_samples, n_channels)` instead of
   biosigIO's channel-major `(n_channels, n_samples)` -- so it is transposed
   back explicitly; getting this backwards would silently swap samples and
   channels. When `EEG.data` holds a `.fdt` filename (a char array) instead of
   the numeric matrix, the sibling `.fdt` is resolved the same way as the
   classic path
3. Resolving `chanlocs`/`event` struct-array fields (labels, types, X/Y/Z,
   latency, ...), each stored as an array of HDF5 object references into a
   `#refs#` group rather than as flat values. Every field is dereferenced
   through `#refs#`, and char arrays are decoded from their Unicode
   code-point representation. An empty MATLAB value (`[]`, stored as a small
   `uint64` dimensions array flagged `MATLAB_empty`) is read as no value, the
   same as a null reference, rather than as characters or a number
4. Raising the same `NotContinuousRecordingError` the rest of biosigIO uses
   when `EEG.trials > 1` (an epoched file), instead of silently flattening
   epochs into a fake continuous stream

## Code Example

```python
from biosigio import Recording

# Load data from EEGLAB .set file
rec = Recording.from_file('data.set', importer='eeglab')

# Print metadata
print(f"Subject: {rec.get_metadata('subject')}")
print(f"Condition: {rec.get_metadata('condition')}")
print(f"Sampling rate: {rec.get_metadata('srate')} Hz")

# Print channel information
print(f"Number of channels: {rec.get_n_channels()}")
channel_types = rec.get_channel_types()
print(f"Channel types: {channel_types}")

# Plot data
rec.plot_signals(time_range=(0, 5))
```

## Channel Type Mapping

EEGLAB doesn't always explicitly designate channel types. biosigIO's EEGLAB importer uses the following rules to assign channel types:

1. A recognized `chanlocs.type` is used directly: `EEG`, `EMG`, `ECG`, `EKG`, `EOG`, `SEEG` and `ECOG` pass through; `ACC`/`ACCELEROMETER` become `ACC`, `GYRO`/`GYROSCOPE` become `GYRO`, and `TRIG`/`TRIGGER` become `TRIG`
2. Otherwise the label is searched, in this order, for 'EMG', 'ECG' or 'EKG' (type `ECG`), 'EOG', 'ACC', 'GYRO' and 'TRIG'
3. Other channels are assigned type 'OTHER'; the label alone never makes a channel `EEG`, so a BIDS `channels.tsv` is the way to type unlabeled EEG channels

## Notes and Limitations

- Both EEGLAB save forms load: the standard single `EEG` struct (`loadmat` returns `{'EEG': struct}`, which is what real EEGLAB writes) and the legacy flat layout. The importer normalizes them before reading, so a nested-struct file no longer imports as an empty recording
- Both classic (v5/v7) and MATLAB v7.3 (HDF5) `.set` files are supported; the v7.3 path requires the optional `hdf5` extra (`uv sync --extra hdf5`) and raises a clear install-hint error if h5py is missing
- Signal data stored inline in the `.set` or in a separate float32 `.fdt` file is supported for both save forms; the sibling `.fdt` is resolved by the `.set` path (so BIDS-renamed files load even though `EEG.data` keeps the original `.fdt` name)
- Event markers are loaded into the `events` table (`rec.events`): EEGLAB event latency/duration (samples) are converted to onset/duration in seconds and the event `type` becomes the description
- Channel locations (if available) are preserved in the channel information
- Channel labels are made unique: a repeated `chanlocs` label (or a padded default label that collides with a real one) is renamed with a numeric suffix (`Fz`, `Fz_2`, `Fz_3`, ...) and a warning names it
- An epoched MATLAB v7.3 file (`EEG.trials > 1`) raises `NotContinuousRecordingError` rather than being read as a fake continuous recording. The classic path's separate, pre-existing behavior for an epoched `.fdt` is unchanged here: it concatenates trials into one continuous series instead (see `test_eeglab_reads_epoched_fdt`). The two paths therefore disagree on what an epoched file should become; that inconsistency predates this change and is tracked separately rather than resolved here.
- Some EEGLAB-specific information may not be fully preserved in the conversion
- Time information is properly handled to maintain accurate timing in the imported data 