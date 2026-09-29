# Metadata Handling

biosigIO provides comprehensive metadata management capabilities, allowing you to work with recording session information, subject details, and other contextual data. Proper metadata handling is particularly important when working with research data that needs to be shared or archived.

The `Recording` object stores various pieces of metadata loaded from the source file or added manually.

## Accessing Metadata

When you load data into biosigIO, any available metadata from the source file is automatically imported:

```python
# Load data
rec = Recording.from_file('data.set', importer='eeglab')

# Access all metadata
all_metadata = rec.metadata
print(all_metadata)

# Access specific metadata field
subject = rec.get_metadata('subject')
print(f"Subject: {subject}")

# Check if a metadata field exists
if rec.has_metadata('condition'):
    condition = rec.get_metadata('condition')
    print(f"Condition: {condition}")
```

You can access the general metadata dictionary directly or use helper methods:

```python
# Access the entire metadata dictionary
all_metadata = rec.metadata
print(all_metadata)

# Get a specific metadata value
subject_id = rec.get_metadata('subject')
fs = rec.get_metadata('sampling_frequency')
print(f"Subject: {subject_id}, Fs: {fs}")

# Set or update a metadata value
rec.set_metadata('task', 'Isometric Contraction')
```

The specific keys available in `rec.metadata` depend on the data format and the information present in the source file header. Common keys might include `sampling_frequency`, `subject_id`, `recording_date`, device information, etc.

## Channel-Specific Information

Information specific to each channel (like its type, physical dimension/unit, or prefiltering details) is stored within the `rec.channels` dictionary, keyed by the channel label:

```python
for channel_name, channel_info in rec.channels.items():
    print(f"Channel: {channel_name}")
    print(f"  Type: {channel_info.get('channel_type', 'N/A')}")
    print(f"  Unit: {channel_info.get('physical_dimension', 'N/A')}")
    print(f"  Sampling Freq: {channel_info.get('sample_frequency', 'N/A')}")
    print(f"  Prefilter: {channel_info.get('prefilter', 'N/A')}")

# Access info for a specific channel
emg1_info = rec.channels['EMG1']
print(f"EMG1 Unit: {emg1_info['physical_dimension']}")
```

To change a channel's metadata, use `rec.set_channel()`, which validates the channel type and modality and re-derives the modality when only the type changes:

```python
rec.set_channel('EMG1', channel_type='EMG', physical_dimension='mV')
```

`set_channel` never touches the samples. Channel labels are unique, so `rec.add_channel()` with an existing label raises `ValueError`; to replace a channel's samples, assign `rec.signals['EMG1'] = new_data` (same length as the recording).

## Annotations / Events

Time-stamped annotations or events associated with the recording are stored in the `rec.events` attribute as a pandas DataFrame.

This DataFrame has the following standard columns:

*   `onset`: The start time of the event in seconds, relative to the beginning of the recording.
*   `duration`: The duration of the event in seconds.
*   `description`: A string describing the event.

**Loading Annotations:**

*   **EDF/BDF:** Annotations stored in the EDF+/BDF+ annotation channel are automatically loaded into `rec.events` by the `EDFImporter`.
*   **WFDB:** Annotations stored in a corresponding `.atr` (or similar) file are automatically loaded into `rec.events` by the `WFDBImporter` when loading the `.hea` file.
*   **EEGLAB `.set`:** EEGLAB events are loaded into `rec.events` by the `EEGLABImporter`; the event latency/duration (samples) are converted to onset/duration in seconds and the event `type` becomes the description.
*   **Other Formats:** For formats that don't have standardized annotation support within the file (like CSV, Trigno, OTB), annotations are typically not loaded automatically. You may need to load them from a separate file and add them manually.

**Accessing Annotations:**

```python
# Check if any events were loaded
if rec.events is not None and not rec.events.empty:
    print(f"Loaded {len(rec.events)} events.")
    print("First 5 events:")
    print(rec.events.head())

    # Filter events by description
    marker_events = rec.events[rec.events['description'].str.contains('Marker')]
    print("\nMarker Events:")
    print(marker_events)
else:
    print("No events/annotations loaded.")
```

**Adding Annotations Manually:**

You can add events programmatically using the `rec.add_event()` method:

```python
rec.add_event(onset=10.5, duration=5.0, description="Rest Period")
rec.add_event(onset=15.5, duration=0.0, description="Stimulus Onset")

print("\nEvents after adding manually:")
print(rec.events)
```

**Exporting Annotations:**

When exporting data using `rec.to_edf()`, the events stored in `rec.events` are automatically written to the EDF+/BDF+ annotation channel.

## Setting Metadata

You can add or modify metadata fields:

```python
# Set a single metadata field
rec.set_metadata('subject', 'S001')

# Set several fields with repeated set_metadata calls
rec.set_metadata('condition', 'resting')
rec.set_metadata('experimenter', 'John Doe')
rec.set_metadata('recording_date', '2023-01-15')

# Or update the metadata dictionary directly
rec.metadata.update({
    'subject': 'S001',
    'condition': 'resting',
    'experimenter': 'John Doe',
    'recording_date': '2023-01-15'
})
```

## Common Metadata Fields

While biosigIO is flexible about what metadata you can store, some common fields include:

| Field | Description | Example |
|-------|-------------|---------|
| `subject` | Subject identifier | `"S001"` |
| `session` | Session identifier | `"1"` |
| `condition` | Experimental condition | `"rest"` |
| `recording_date` | Date of recording | `"2023-01-15"` |
| `device` | Recording device | `"Trigno Wireless"` |
| `srate` | Sampling rate in Hz | `2000` |

## Metadata in Exported Files

When exporting to EDF/BDF, biosigIO automatically includes metadata in the file header and generates a sidecar channels.tsv file with channel-specific metadata following BIDS conventions:

```python
# Export to EDF with metadata
rec.to_edf('output')

# This will create:
# - output.edf or output.bdf (depending on format selection)
# - output_channels.tsv (channel metadata in tab-separated format, BIDS naming)
```

The channels.tsv file will include information like:

```
name    type    units   sampling_frequency    ...
EMG1    EMG     µV      2000                  ...
EMG2    EMG     µV      2000                  ...
ACC1    ACC     g       2000                  ...
```

## Copying Metadata Between Recording Objects

When working with multiple Recording objects, you can copy metadata between them:

```python
# Create a subset with only EMG channels
emg_only = rec.select_channels(channel_type='EMG')

# Copy all metadata from original to subset
emg_only.metadata = rec.metadata.copy()

# Or selectively copy metadata
emg_only.set_metadata('subject', rec.get_metadata('subject'))
```

## Best Practices for Metadata

1. **Consistency**: Establish conventions for metadata fields and stick to them
2. **Completeness**: Include all relevant information about the recording context
3. **Standardization**: Use standard units and nomenclature
4. **Validation**: Verify metadata accuracy before export
5. **Documentation**: Document your metadata structure for collaborators

By maintaining good metadata practices, you ensure that your EMG data remains interpretable and useful for future analysis. 