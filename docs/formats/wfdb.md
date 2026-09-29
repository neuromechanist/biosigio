# WFDB Format

biosigIO supports loading data stored in the Waveform Database (WFDB) format, commonly used for physiological signals like ECG and EMG, especially in datasets available on PhysioNet.

## File Structure

Loading WFDB data typically involves three files sharing the same base name (e.g., `record1`):

- **Header File (`.hea`):** Contains metadata about the recording, such as signal names, units, sampling frequency, signal resolution, start time, and pointers to the data file(s).
- **Data File(s) (`.dat`):** Contains the actual signal samples in binary format as specified in the header.
- **Annotation File (`.atr`, optional):** Contains time-stamped event annotations associated with the recording. Common annotation files use the `.atr` extension, but other extensions might exist.

## Loading Data

To load a WFDB record, provide the path to the **header (`.hea`) file** to `Recording.from_file`. A `.hea`, `.dat`, or `.atr` extension is auto-detected as WFDB; an extension-less base record name needs an explicit `importer='wfdb'`:

```python
from biosigio.core.emg import Recording

# Load using the header file path (extension auto-detected)
rec = Recording.from_file('path/to/your/record.hea')

# Or load using just the base name (no extension); pass importer explicitly
# rec = Recording.from_file('record', importer='wfdb')
```

biosigIO uses the `wfdb` library (PyPI package `wfdb`) internally. It ships as a core dependency, so no separate installation is required.

## Repeated Signal Names

The `wfdb` writer refuses a repeated signal name, but its reader accepts one from a hand-edited or foreign header. A `Recording` is keyed by channel label, so repeated names get the same running suffixes as repeated EDF labels (`II-0`, `II-1`; see [EDF repeated channel labels](edf.md#repeated-channel-labels)) and a warning names them. Every signal is kept.

## Annotation Handling

If an annotation file (e.g., `record.atr`) exists in the same directory and shares the same base name as the header file, the `WFDBImporter` will **automatically load** these annotations.

Loaded annotations are stored in the `rec.events` pandas DataFrame with the following columns:

- `onset`: The time of the event in seconds from the start of the recording.
- `duration`: The duration of the event in seconds (typically 0 for standard WFDB point annotations).
- `description`: A string describing the event, usually derived from the WFDB annotation symbol (e.g., `"WFDB Annotation: N"`).

See the [Metadata Handling](../user-guide/metadata.md#annotations-events) guide for more details on accessing and working with the `events` DataFrame.

If the annotation file is not found, a message will be stored in the `rec.metadata['annotation_status']` field. 