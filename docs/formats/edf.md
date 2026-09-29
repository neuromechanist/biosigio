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

### Tolerant fallback for non-compliant-but-readable files

`pyedflib`'s compliance checker rejects some real-world EDF/BDF files outright
even though the data is byte-intact and other tools (MNE-Python) read them
without complaint. The importer recovers three such conditions instead of
discarding the recording:

- a channel with `physical_min == physical_max` (a reference electrode in a
  referential montage, constant by construction; issue #109)
- a numeric header field padded with NUL bytes instead of ASCII spaces
- a file correctly marked EDF+D (discontinuous)

When one of these is detected, biosigIO re-reads the file through MNE and
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
  `discontinuous_datarecords`)
- each affected channel additionally carries `degenerate_physical_range: True`

A genuinely truncated or corrupt file is unaffected by this: only the three
conditions above are treated as recoverable, and a size check runs before any
recovery attempt so a file that is *also* truncated still raises
`CorruptFileError` rather than being silently read short.

**Scope of a recovered read:** all signal-level data is fully populated --
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
- When exporting to EDF/BDF, biosigIO automatically handles scaling to maximize precision
- EDF/BDF export requires a single sampling rate across all channels; if per-channel rates differ, export raises `ValueError`. Resample channels to a common rate first
- With `format='auto'`, `to_edf('output')` selects either `.edf` (16-bit) or `.bdf` (24-bit) based on signal analysis and appends the matching extension
- EDF has limitations on channel naming (maximum 16 characters)
