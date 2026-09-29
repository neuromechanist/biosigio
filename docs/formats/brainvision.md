# BrainVision (.vhdr)

biosigIO supports loading data stored in the BrainVision Core Data Format, the native format of BrainVision Recorder by Brain Products. It is a common distribution format for electroencephalography (EEG) and intracranial EEG (iEEG) datasets, including many BIDS (Brain Imaging Data Structure) datasets on OpenNeuro.

Reading is handled by MNE-Python, which reads the BrainVision triplet natively. MNE-Python is an optional, heavy dependency shared with the MEG (magnetoencephalography) importer, so it is installed through the `meg` extra:

```bash
uv sync --extra meg
# or, for an existing install:
uv pip install 'biosigio[meg]'
```

If MNE-Python is not installed, importing a BrainVision file raises an `ImportError` with this install hint.

## File Structure

A BrainVision recording is a triplet of files that share the same base name (e.g., `subject01`):

- **Header File (`.vhdr`):** A text file describing the recording: channel names, units, sampling interval, data orientation, and pointers to the marker and binary data files.
- **Marker File (`.vmrk`):** A text file listing time-stamped markers (events) such as stimulus and response triggers, segment boundaries, and recording-control notes.
- **Binary Data File (`.eeg` or `.dat`):** The actual signal samples in binary form, as specified in the header.

You always pass the **header (`.vhdr`) file** to biosigIO. MNE-Python resolves the sibling `.vmrk` and `.eeg`/`.dat` files from the `DataFile=` and `MarkerFile=` entries recorded in the header, so the three files must stay together in the same directory.

### Renamed files and stale header references

Renaming a BrainVision triplet (as BIDS conversion does) changes the file names on disk but not the `DataFile=`/`MarkerFile=` entries inside the header, so the header can name a `.eeg` or `.vmrk` that no longer exists. When a referenced file is missing and a sibling with the header's own stem exists (`<stem>.eeg`, then `<stem>.dat`, for the data; `<stem>.vmrk` for the markers), biosigIO reads a patched temporary copy of the header that points at the sibling. Because the copy lives in a temporary directory, every non-empty `DataFile=` and `MarkerFile=` entry in its `[Common Infos]` section is rewritten to an absolute path, not only the stale one; the rest of the header, its encoding and its line endings are kept. If the header's codepage cannot spell such a path (for example, a Greek or CJK directory name under a cp1252 header), the copy is written as UTF-8 and its `Codepage=` entry is changed to `UTF-8` to match. The dataset's own files are never modified. The streaming Zarr export (`stream_to_zarr`) uses the same resolution.

A header whose references all exist is passed to MNE unchanged, and so is one where neither the named file nor a same-stem sibling exists, so that MNE reports the missing file as before.

## Loading Data

Provide the path to the `.vhdr` header file to `Recording.from_file`. The `.vhdr` extension is recognized automatically, so the importer is inferred:

```python
from biosigio import Recording

# Load a BrainVision recording (pass the .vhdr header)
rec = Recording.from_file('path/to/subject01.vhdr')
```

You can also select the importer explicitly with the `importer` argument:

```python
rec = Recording.from_file('path/to/subject01.vhdr', importer='brainvision')
```

## Channel Types and Units

Channels are read through the channel-type and unit mapping shared by all MNE-backed importers (MEG and BrainVision). Each MNE channel type maps to a distinct biosigIO channel type, so different sensor streams are preserved rather than collapsed together. The mapping includes, among others:

| MNE type | biosigIO channel type |
|----------|----------------------|
| `eeg` | `EEG` |
| `seeg` | `SEEG` |
| `ecog` | `ECOG` |
| `dbs` | `DBS` |
| `eog` | `EOG` |
| `ecg` | `ECG` |
| `emg` | `EMG` |
| `stim` | `TRIG` |
| `resp` | `RESP` |
| `misc`, `bio`, `syst` | `MISC` |

Any MNE type not in the mapping becomes `OTHER`.

Physical units are derived from the FIFF unit code that MNE assigns to each channel. Voltage channels are reported with a physical dimension of `V`; channels without a recognized unit code use `n/a`.

## Marker (Event) Handling

MNE-Python exposes the `.vmrk` markers as annotations on the loaded recording. biosigIO reads these markers into the `rec.events` pandas DataFrame, sorted by onset, with the following columns:

- `onset`: The time of the marker in seconds from the start of the recording.
- `duration`: The duration of the marker in seconds (0 for instantaneous markers).
- `description`: A string describing the marker, taken from the BrainVision marker label.

```python
rec = Recording.from_file('subject01.vhdr')

# Inspect the markers read from the .vmrk file
print(rec.events.head())
```

If the recording contains no markers, `rec.events` is left at its default empty value.

See the [Metadata Handling](../user-guide/metadata.md#annotations-events) guide for more details on accessing and working with the `events` DataFrame.

## Metadata

Loaded BrainVision recordings include metadata such as:

- `source_file`: Path to the `.vhdr` header file passed to `from_file`.
- `number_of_signals`: The number of channels read from the recording.

```python
rec = Recording.from_file('subject01.vhdr')
print(rec.get_metadata('source_file'))
print(rec.get_metadata('number_of_signals'))
```

## Requirements

The BrainVision importer requires MNE-Python, installed through the `meg` extra (`uv sync --extra meg`). This extra is shared with the MEG importer, so installing it enables both. No additional package is needed to read the BrainVision triplet; MNE reads it natively.
