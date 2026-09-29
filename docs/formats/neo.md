# Proprietary Electrophysiology (Neo)

biosigIO can import continuous recordings from a wide range of proprietary acquisition systems through [python-neo](https://neo.readthedocs.io/), a library that provides a unified reader layer over many vendor formats. This lets you bring data from amplifiers such as Intan, Blackrock, and Spike2 into a `Recording` without writing format-specific code.

python-neo is an optional, heavy dependency, so it is not installed by default. Install it with the `neo` extra:

```bash
uv sync --extra neo
# or, for an existing install:
uv pip install 'biosigio[neo]'
```

If you attempt to load a neo-backed format without the extra installed, biosigIO raises an `ImportError` with this install hint rather than failing obscurely.

## Supported Formats

The Neo importer is selected automatically from the file extension for the following formats:

| System | Extensions |
|--------|------------|
| Intan | `.rhd`, `.rhs` |
| Blackrock | `.ns1`, `.ns2`, `.ns3`, `.ns4`, `.ns5`, `.ns6` |
| Spike2 / Cambridge Electronic Design (CED) | `.smr`, `.smrx` |
| Plexon | `.plx`, `.pl2` |
| Micromed | `.trc` |
| Neuralynx | `.ncs` |

Some neo readers operate on a directory rather than a single file (for example folder-based acquisition systems). In those cases pass the directory path to `Recording.from_file` with an explicit `importer='neo'`.

## File Structure

python-neo models a recording as a Block containing one or more Segments (trials), and each Segment contains one or more AnalogSignals. Each AnalogSignal is one **signal stream**: a group of channels that share a single sampling rate. A common example is an Intan recording where the amplifier bank streams at 30 kHz while an auxiliary analog-to-digital converter (ADC) streams at a lower rate.

biosigIO holds a single time grid, and that grid carries exactly one sampling rate. This leads to a key model for how streams are mapped:

- **Same-rate streams merge.** If a file contains several streams that share the same sampling rate (and the same length and start time), they are merged onto one time grid as a single `Recording`. Channel labels are kept distinct, so a name that would collide across streams is disambiguated as `name_0`, `name_1`, and so on. The first occurrence keeps its name, a suffix never takes a name some stream really uses, and the renames are logged and recorded in the metadata as `channel_labels_deduplicated` (`{new_label: original_label}`).
- **Multi-rate files require a selector.** If a file holds streams at more than one sampling rate, biosigIO does not silently collapse them to a single rate. Instead it raises a clear error that lists every stream with its index, name, rate, and channel count, and asks you to pass `stream=` to choose one. The same applies when same-rate streams have differing lengths or start times that cannot share one grid.

## Loading Data

Provide the path to the acquisition file to `Recording.from_file`. The importer is inferred from the extension, so no `importer=` argument is needed for the supported formats:

```python
from biosigio import Recording

# Single-rate Intan recording (all same-rate streams merged onto one grid)
rec = Recording.from_file('rec.rhd')

# Multi-rate Blackrock file: select the stream by name (or by integer index)
rec = Recording.from_file('rec.ns5', stream='lfp')

# Label every channel with a BIDS type at load time
rec = Recording.from_file('rec.rhd', channel_type='SEEG')
```

### Load Arguments

The Neo importer accepts the following keyword arguments through `Recording.from_file`:

- `stream`: Which signal stream to import, given as an integer index or a stream name. This is required only when the file holds streams at more than one sampling rate; otherwise all same-rate streams are merged. If a name matches more than one stream, biosigIO asks you to select by integer index instead.
- `segment`: Segment (trial) index for multi-segment recordings (default `0`). When a file has more than one segment, biosigIO warns which segment it imported and reminds you that `segment=` selects another.
- `channel_type`: BIDS channel type applied to every imported channel (default `'OTHER'`). See [Channel Types](#channel-types) below.

## Channel Types

python-neo carries signal values, physical units, sampling rates, channel names, and events, but it does **not** carry a Brain Imaging Data Structure (BIDS) channel *type*. An Intan `.rhd` file, for example, does not distinguish stereo-electroencephalography (SEEG) channels from electroencephalography (EEG) channels. For that reason every channel imported through the Neo importer defaults to type `OTHER`.

You have two ways to assign meaningful types:

```python
# 1. Label the whole recording at load time
rec = Recording.from_file('rec.rhd', channel_type='SEEG')

# 2. Place a sibling BIDS _channels.tsv next to the file; its per-channel
#    type/units are applied automatically (bids_channels='auto', the default)
rec = Recording.from_file('rec.rhd')
```

A sibling `_channels.tsv` gives per-channel control and overrides the importer's defaults, whereas `channel_type=` applies a single type to the entire recording. To disable the automatic `_channels.tsv` lookup, pass `bids_channels='off'`.

## Annotation Handling

Annotations come from two kinds of neo objects in the imported segment:

- **Events** are instantaneous markers. Each event time becomes an entry with `duration` of `0`.
- **Epochs** are intervals that carry a duration. Each epoch becomes an entry with its `onset` and `duration` preserved.

neo times are expressed in the recording's absolute time base, which may not start at zero. The imported signal grid is 0-based, so all event and epoch onsets are shifted by the stream's start time (`t_start`) to stay aligned with the signal. The discarded absolute origin is kept in metadata (see `t_start_s` below) for downstream temporal alignment, for example when writing BIDS `events.tsv`.

Loaded annotations are stored in the `rec.events` pandas DataFrame with the columns `onset`, `duration`, and `description`. When a neo Event or Epoch carries no labels, descriptions are generated from the object name (for example `event_0`, `event_1`). See the [Metadata Handling](../user-guide/metadata.md#annotations-events) guide for more on the `events` DataFrame.

## Metadata

A recording loaded through the Neo importer includes the following metadata:

- `source_file`: Path to the acquisition file.
- `neo_io`: The neo reader class used (for example `IntanIO`).
- `t_start_s`: The absolute start time of the imported stream in seconds, retained for temporal alignment.
- `number_of_signals`: The number of imported channels.

```python
rec = Recording.from_file('rec.rhd')
print(rec.get_metadata('neo_io'))
print(rec.get_metadata('t_start_s'))
```

## Limitations

- Recordings without continuous analog signals (events-only or spikes-only files) are not supported; biosigIO raises a clear error in that case.
- A single time grid holds one sampling rate, so multi-rate files must be imported one stream at a time using `stream=`.
- Channel types are not present in the source formats and default to `OTHER` until you set them.

## Requirements

The Neo importer requires the optional `neo` extra. Install it with `uv sync --extra neo` or `uv pip install 'biosigio[neo]'`.
