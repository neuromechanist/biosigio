# EDF/BDF Format

EDF (European Data Format) and BDF (BioSemi Data Format) are standard formats for storing multichannel biological and physical signals. biosigIO supports both importing from and exporting to these formats.

## Format Description

### EDF Format

EDF (European Data Format) is a simple and flexible format for exchange and storage of multichannel biological and physical signals:

- 16-bit resolution (range: ±32767)
- Supports varying sampling rates between channels
- Includes metadata in the header
- Supports annotations and events

### BDF Format

BDF (BioSemi Data Format) is an extension of EDF that uses 24-bit resolution instead of 16-bit:

- 24-bit resolution (range: ±8,388,607)
- Higher dynamic range for signals with larger amplitude variations
- Otherwise identical to EDF

### File Structure

Both formats consist of:

1. **Header Record**: Contains metadata about the recording:
   - Version information
   - Patient and recording information
   - Date and time
   - Number of channels
   - Sampling rates
   - Calibration values
   
2. **Data Records**: Contains the actual signal data, organized in blocks.

## biosigIO's Approach to EDF/BDF

biosigIO can both import from and export to EDF/BDF formats. When exporting, biosigIO automatically determines which format to use based on the dynamic range of the data:

- If the signal's dynamic range is within 16-bit resolution (~90dB), it uses EDF
- If the signal requires greater precision, it uses BDF

This determination is made using:
- SVD (Singular Value Decomposition) analysis to determine the effective rank and energy distribution
- FFT (Fast Fourier Transform) analysis to evaluate the noise floor and signal-to-noise ratio

## Importer Implementation

The EDF importer in biosigIO (`biosigio.importers.edf`) uses the `pyedflib` package to:

1. Read the EDF/BDF file header to extract metadata
2. Load the signal data for all channels
3. Convert channel information to biosigIO's format
4. Read EDF+/BDF+ annotations into the `events` DataFrame

### EDF+D and BDF+D files with gaps

An EDF+D or BDF+D file gives every data record its own start time, and the records need not be contiguous.
MNE-Python concatenates them, which would put every sample after a gap earlier on the time axis than it was recorded while the annotations keep their true times (issue #180).
So biosigIO reads the time-keeping annotation of every record itself, whenever the header's reserved field says EDF+D or BDF+D, and compares each record's start with the end of the one before it (tolerance 1 microsecond).

- A gap, an overlap or a record out of order raises `DiscontinuousRecordingError` (code `discontinuous_recording`, a `FileReadError`) naming the file and every gap as `(start, length)` seconds on the file's own time axis, in its `gaps` attribute.
  The in-memory importer and `stream_to_zarr` both raise it, before MNE is touched, so the MNE-backed fallback is not needed to get the error.
- A file labeled discontinuous whose records are contiguous is read as above and flagged `edf_discontinuous_verified_contiguous`.
- A time-keeping annotation that cannot be read raises `FileReadError`, since continuity cannot be shown.

Converting a gapped recording is not supported yet; it waits for a gap-aware design.

### Tolerant fallback for non-compliant-but-readable files

`pyedflib`'s compliance checker rejects some real-world EDF/BDF files outright
even though the data is byte-intact and other tools (MNE-Python) read them
without complaint. The importer recovers three such conditions instead of
discarding the recording:

- a channel with `physical_min == physical_max` (a reference electrode in a
  referential montage, constant by construction; issue #109)
- a numeric header field padded with NUL bytes instead of ASCII spaces
- a file correctly marked EDF+D (discontinuous), but only when its data records are in fact contiguous; see [EDF+D and BDF+D files with gaps](#edfd-and-bdfd-files-with-gaps)

A fourth, a late first time-keeping onset, is recovered by `pyedflib` itself and needs no MNE; see [Late first time-keeping onset](#late-first-time-keeping-onset).

When one of these three is detected, biosigIO re-reads the file through MNE and
rescales its SI-volt output back to the same native physical units a normal
`pyedflib` read would have produced, so the recovered values are numerically
identical to what `pyedflib` would report for the same channel. A channel with
a degenerate `physical_min == physical_max` has no defined scale, so its output
is the constant `physical_min` (== `physical_max`) throughout -- exactly zero in
the common grounded/referential case. Recovering this way requires the `meg`
extra (MNE); without it, such a file raises an `ImportError` that names the
`meg` extra, with pyedflib's own error as its `__cause__`. It is deliberately
not a `BiosigIOError`: installing the extra makes the same file readable.

A channel MNE auto-detects as a trigger/status channel (by default, one
literally named `"Status"` or `"Trigger"`) is read with that detection
turned off (`stim_channel=False`): MNE would otherwise give such a channel
its own uncalibrated, bitmasked treatment unrelated to the header's declared
physical range, which would silently disagree with a normal `pyedflib` read
of that channel. With detection off, it is calibrated the same way as every
other channel, preserving the same unit-parity guarantee for it too.

A recovered read is flagged in the recording's metadata:

- `edf_tolerant_read` (`True`) and `edf_tolerant_read_reason` (one of
  `degenerate_physical_range`, `malformed_numeric_field`,
  `discontinuous_datarecords`, `late_timekeeping_onset`)
- each affected channel additionally carries `degenerate_physical_range: True`

A read of a file labeled EDF+D or BDF+D whose data records were checked and found contiguous also sets `edf_discontinuous_verified_contiguous` (`True`).

A genuinely truncated or corrupt file is unaffected by this: only the four
conditions are treated as recoverable, and a size check runs before any
recovery attempt so a file that is *also* truncated still raises
`CorruptFileError` rather than being silently read short.

**Scope of an MNE-backed recovered read** (the first three conditions): all signal-level data is fully populated, namely
per-channel units, physical/digital min/max, transducer, prefilter, sample
rate, channel type, and EDF+/BDF+ annotations/events. The one thing NOT
populated is the patient/recording free-text metadata a normal read parses
out of the header's `patient_id`/`recording_id` fields (`patientcode`,
`gender`, `birthdate`, `patient_name`, `admincode`, `technician`, `equipment`,
`recording_additional`): these keys are simply absent on a recovered read,
not defaulted to empty strings, since biosigIO does not re-implement
`pyedflib`'s own sub-parsing of those two free-text fields for the fallback
path.

A file with two on-disk channels sharing the same label is covered too.
MNE renames the repeats (`EEG-0`, `EEG-1`), so the fallback pairs each channel
with its header row by position rather than by label: MNE lists the signals in
header order, minus the EDF+/BDF+ annotations channel. Each pairing is still
checked by name, and a channel that cannot be paired fails loud rather than
taking another row's scaling. The recovered channels carry the same suffixed
labels a normal read produces (see [Repeated channel labels](#repeated-channel-labels)).

### Late first time-keeping onset

An EDF+ or BDF+ file keeps a time-stamped annotation list (TAL) in every data record.
The first TAL of a record is its time-keeping onset, the record's start in seconds from the start of the recording.
The specification starts the first record at 0 s, and EDFlib (the C library inside `pyedflib`) refuses a file whose first onset is 1 s or more, saying only that the file "contains format errors".
A recording cut out of a longer session at a data-record boundary keeps its session-relative onsets, so the cut file starts late (at +40 s, for example) and is refused although every byte is valid.

The refusal comes from the walk over the annotation lists, which runs only when annotations are read.
biosigIO therefore reads such a file with `pyedflib` itself, with that walk switched off: the samples and every header field are exactly those of a compliant file, and MNE is not involved.
It does so only after proving that the late onset is the file's sole defect.
The refusal must be EDFlib's generic one, the header's reserved field must start with `EDF+C` or `BDF+C`, the file must open with annotations off (which still checks the header and the file size), and record 0 must open with a time-keeping onset of 1 s or more.
Then a small copy holding only the annotation signals, with every onset brought back by whole seconds, must pass EDFlib's full annotation walk, so every other annotation list and the record-to-record continuity are checked as for a compliant file.
The copy is a temporary file, a small fraction of the recording's size, and is deleted.
Any other defect, such as a gap between records, a malformed later annotation, a bad header field or a truncated file, still raises `CorruptFileError`, and so does a first onset under 1 s with a defect elsewhere.
A first onset that would carry the start date past year 9999, or that a float or an integer cannot hold, is declined the same way, on both read paths.
If the operating system cannot create, write or read back the temporary copy (a missing or read-only temporary directory, a full disk), biosigIO raises the system's own `OSError` on both read paths, not `CorruptFileError`: the file is not at fault, and a caller can retry.

**Time convention.**
Event onsets and the signals are relative to the first record, as for a compliant file, so the first record is at 0 s.
`startdate` is the time of the first record: the header's start date and time plus the first onset, added as a datetime, so an onset that crosses a minute, an hour or a day carries over (a header of 23:59:45 and a first onset of 40 s gives 00:00:25 on the next day).
A fractional first onset is the corner a reader cannot verify against `pyedflib`: its `getStartdatetime` (0.1.42) divides EDFlib's 100 ns sub-second by 100, so the exact sum biosigIO reports matches a native read only for whole-second onsets.
`edf_first_record_onset` keeps the first onset in seconds, the offset of this file from the start of the session it was cut from, so the header's own start is `startdate` minus that offset.

A recovered read is flagged with `edf_tolerant_read` (`True`), `edf_tolerant_read_reason` (`late_timekeeping_onset`) and `edf_first_record_onset`.
The `mixed_rate` policy is unchanged: a mixed-rate file raises `MixedSamplingRateError` by default and loads with `mixed_rate="resample"`.

The streaming Zarr export (`stream_to_zarr`) handles the file the same way and keeps its bounded-memory windowed read, unlike the MNE-backed recoveries, which preload the recording.
It records the same three flags in the store's `recording_metadata`; the streaming path has never recorded `startdate`.

## Repeated channel labels

EDF does not require unique labels; CHB-MIT, for example, declares `T8-P8`
twice and uses `-` for several unused inputs. A `Recording` is keyed by label,
so every occurrence of a repeated label gets a running suffix, the same way
MNE names them: `T8-P8-0`, `T8-P8-1`. A suffix that would collide with an
existing label falls through to `-a`, `-b`, and so on. A warning names the
repeated labels and the file. The in-memory importer, the streaming Zarr export and the
tolerant fallback all name the channels of one file identically, so a
`channels.tsv` written by MNE-BIDS matches by name.

Every rename is also recorded in the recording's metadata under
`channel_labels_deduplicated`, as `{new_label: original_label}` (for example
`{"T8-P8-0": "T8-P8", "T8-P8-1": "T8-P8"}`). Both Zarr export paths write it
into the store's `recording_metadata`, so it survives a round trip; a file
whose labels are already unique carries no such entry.

For a single repeated label the names match MNE's exactly. When several
repeated labels interact (one label's suffixed name colliding with another's),
biosigIO resolves them in first-occurrence order, which is deterministic;
MNE's order is not guaranteed, so the two can differ in that case.

## Exporter Implementation

The EDF exporter in biosigIO (`biosigio.exporters.edf`) also uses `pyedflib` to:

1. Automatically determine whether to use EDF or BDF based on signal characteristics
2. Generate appropriate header information
3. Scale the signals correctly for storage
4. Write the `events` DataFrame back as EDF+/BDF+ annotations
5. Create a sidecar `<output>_channels.tsv` file with detailed channel metadata (BIDS-compatible)

## Code Example

```python
from biosigio import Recording

# Import from EDF file
rec = Recording.from_file('input.edf', importer='edf')

# Print basic information
print(f"Number of channels: {rec.get_n_channels()}")
print(f"Sampling frequency: {rec.get_sampling_frequency()} Hz")
print(f"Recording duration: {rec.get_duration()} seconds")

# Export to EDF/BDF (format selected automatically)
rec.to_edf('output')  # Will add .edf or .bdf extension automatically

# Force a specific format
rec.to_edf('output_edf', format='edf')  # Forces 16-bit EDF
rec.to_edf('output_bdf', format='bdf')  # Forces 24-bit BDF

# Export with verification to ensure data integrity 
rec.to_edf('output_verified', verify=True)
```

## Signal Verification

When exporting EMG data to EDF/BDF format, you can optionally verify the integrity of the exported data by setting the `verify` parameter to `True`:

```python
# Export with verification
verification_results = rec.to_edf('output', verify=True)

# Export with custom verification settings
verification_results = rec.to_edf(
    'output',
    verify=True,
    verify_tolerance=0.001,  # Absolute tolerance for signal comparison
    verify_channel_map={'EMG1': 'CH1'},  # Custom channel mapping
    verify_plot=True  # Generate visualization of comparison
)
```

The verification process:
1. Exports the data to the specified file
2. Immediately reloads the file using the EDF importer
3. Compares the original and reloaded signals using normalization-based metrics
4. Logs a detailed verification report
5. Optionally generates a visual comparison if `verify_plot=True`
6. Returns the verification results as a dictionary

## Notes and Limitations

- The BIDS-compatible `<output>_channels.tsv` sidecar file includes detailed channel information
- EDF+/BDF+ annotations are loaded into the `events` DataFrame (columns `onset`, `duration`, `description`) and written back as EDF+/BDF+ annotations on export
- When importing from EDF, biosigIO attempts to identify channel types based on labels and signal characteristics
- When exporting to EDF/BDF, biosigIO automatically handles scaling to maximize precision, with a physical window of its own for every channel
- An exported file reads back within half a least significant bit (LSB) of the given values.
  The LSB is taken from the header as stored, `(physical_max - physical_min) / (digital_max - digital_min)`, and the exporter rounds each sample to the nearest digital level of that scale.
  A reader scales by the header text, and the header keeps each physical bound as at most 8 characters of fixed-notation text: a bound below 1e-4 in magnitude holds a few digits (`0.00001`, not `1e-05`), and a negative bound one digit fewer than a positive one.
  The exporter rounds each bound outward, writes the header to a scratch file, reads it back, and moves a bound outward in the last place if it is stored inside the data window (`95382.65` is stored as `95382.64`, `-53820.7` as `-53820.6`).
  If no stored bound brackets the data, export raises `ValueError`, so rescale the channel to a unit that suits its magnitude.
  A value outside the bounds, which `clip_outliers` allows, saturates at the end of the digital range.
  The smallest range the header can hold around zero is `[0, 1e-6]` (data that is not negative), `[-1e-5, 0]` (data that is not positive) or `[-1e-5, 1e-6]`, so a channel whose values are far below 1e-5 in its unit, such as MEG in tesla (about 1e-10), keeps few levels; rescale it to a smaller unit such as fT.
  Export warns when a channel's data span is above zero and under 4 stored LSB, since its samples then fall on a few digital levels or one.
  The warning names the channel as the recording calls it (not its EDF label), with its span, its span in LSB and its LSB, for up to five channels, then a count.
  The rule is the span alone, with no relative floor: a channel that varies by float residue (values of 0 and 1e-12) is reported in EDF, where it loses its variation, and a constant channel is never reported, since it has none to lose.
  A channel far below 1e-5 in its unit is told to rescale to a smaller unit; a small variation on a large offset (410986 mV, give or take 5e-6) is told to subtract the offset before export, because the span centred on zero would be resolved.
  A constant channel is not mapped onto an exact digital level (#172), so one whose value is under half an LSB reads back as the stored bound at 0, within half an LSB of the source but not equal to it.
  The scratch header used to read back the stored bounds is a hidden `.biosigio-bounds-*` file beside the output, removed on success and on failure; if the process is killed outright (SIGKILL), the file is left in the output directory and is safe to delete.
- EDF/BDF export requires a single sampling rate across all channels; if per-channel rates differ, export raises `ValueError`. Resample channels to a common rate first
- With `format='auto'`, `to_edf('output')` selects either `.edf` (16-bit) or `.bdf` (24-bit) based on signal analysis and appends the matching extension
- EDF stores a channel label in 16 characters. A longer label is truncated, and when truncation would give two channels the same label (every Delsys Trigno channel of one sensor, `Mini sensor 10: ACC.X 10`, `Mini sensor 10: ACC.Y 10`, ..., shares its first 16 characters), those channels are numbered within the 16 characters instead (`Mini sensor 10-0`, `Mini sensor 10-1`, ...) and a warning lists them. In the `_channels.tsv` sidecar, only those numbered channels are named by the label written to the file, so their rows match the file on re-import; every other channel keeps its full label there, including one that is truncated but stays unique (as in earlier releases, such a row does not match the truncated label on re-import)
