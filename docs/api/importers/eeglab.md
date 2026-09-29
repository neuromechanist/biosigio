# EEGLAB Importer

The `EEGLABImporter` class is responsible for importing EMG (and other biopotential) data from EEGLAB `.set` files.

## Class Documentation

::: biosigio.importers.eeglab
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording
from biosigio.importers.eeglab import EEGLABImporter

# Method 1: Using Recording.from_file (recommended)
rec = Recording.from_file('data.set', importer='eeglab')

# Method 2: Using the importer directly
rec = EEGLABImporter().load('data.set')
```

## File Format Support

The EEGLAB importer dispatches on the file's actual content (a leading
`"MATLAB 7.3 MAT-file"` header vs. a classic MAT header), not the `.set`
extension, so it supports:

1. Classic (pre-v7.3, non-HDF5) MATLAB `.set` files, via `scipy.io.loadmat`.
2. MATLAB v7.3 (HDF5) `.set` files, via
   [h5py](https://www.h5py.org/) -- an optional dependency (the `hdf5` extra:
   `uv sync --extra hdf5`); a clear `ImportError` with an install hint is
   raised if a v7.3 file is loaded without it installed. Both the `EEG`-group
   layout and a flat layout (the struct's fields at the HDF5 root) are read.
   An epoched v7.3 file (`EEG.trials > 1`) raises
   `NotContinuousRecordingError` instead of being read as a fake continuous
   recording.
3. Signal data inline in the `.set` or in a sibling `.fdt` file.
4. Multiple channel types (EMG, EEG, ACC, etc.)
5. Event markers (loaded into `Recording.events`)

## Channel Type Detection

The EEGLAB importer attempts to detect channel types based on:

1. Channel labels in the EEGLAB `chanlocs` structure
2. Channel type information if available
3. Naming conventions (e.g., channels with 'EMG' in the name are classified as 'EMG')

## Parameters

`EEGLABImporter().load(filepath)` takes:

- **filepath (str)**: Path to the EEGLAB `.set` file.

## Return Value

The `load()` method returns a single `Recording` object with:

1. **signals**: signal data with channels as columns.
2. **channels**: per-channel information including:
   - `channel_type`: type of channel (EMG, EEG, etc.)
   - `physical_dimension`: physical unit (defaults to `'uV'`)
   - `sample_frequency`: sampling rate in Hz
   - `X`/`Y`/`Z`: channel coordinates when available in `chanlocs`
3. **metadata**: fields parsed from the EEGLAB file, which may include:
   - `setname`, `filename`, `filepath`
   - `subject`, `group`, `condition`, `session`, `comments`
   - `srate`: sampling rate
   - `nbchan`, `trials`, `pnts`
   - `xmin`/`xmax`: time limits
   - `device`: set to `'EEGLAB'`
   - `source_file`: the path that was loaded
4. **events**: a DataFrame with `onset` and `duration` in seconds (converted
   from EEGLAB's 1-based sample latencies) and the event `type` as
   `description`.

## Notes

- Event markers are loaded into `Recording.events`, not into metadata.
- Channel coordinates are preserved in the channel information when available.
- A repeated channel label is renamed with a numeric suffix (`Fz_2`, ...)
  and a warning is issued, so every channel is kept.
  The first occurrence keeps its label, and a suffix never takes a label another channel genuinely has:
  `Fz, Fz, Fz_2` imports as `Fz, Fz_3, Fz_2`
  (see [the EEGLAB format page](../../formats/eeglab.md)).
  The renames are recorded in the metadata as `channel_labels_deduplicated`.