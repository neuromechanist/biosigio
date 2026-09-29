# BIDS Helpers

Helpers for the Brain Imaging Data Structure (BIDS): locate a sibling
`_channels.tsv` / `_events.tsv` next to a recording, apply its per-channel
types/units (`apply_channels_tsv`), and load its authoritative event table into
`rec.events` (`apply_events_tsv`). `apply_channels_tsv` is applied automatically
by `Recording.from_file` (unless `bids_channels="off"`); `apply_events_tsv` is
called explicitly when the sidecar events should override the data file's own
markers.

`apply_channels_tsv_to_stream` is the same thing for a recording that is never
loaded into a `Recording` at all: the streaming Zarr exporter
(`stream_to_zarr`, also `bids_channels="auto"` by default) has only a channel
table and reads the samples window by window afterwards, so it takes a
per-channel conversion factor to apply later instead of rescaling a column now.
Both functions decide every per-channel question with the same internal table,
so the two export paths cannot disagree about what a sidecar means.

## Choosing the sidecar: `bids_channels`

`Recording.from_file` and `stream_to_zarr` take the same `bids_channels`
argument with the same meanings, resolved by the same `resolve_channels_tsv`, so
a caller can move a recording between the two paths without its units changing:

| Value | Meaning |
|---|---|
| `"auto"` (default) | resolve the sibling `_channels.tsv` via `find_channels_tsv`; a recording with no sidecar is left as the importer read it |
| `"off"`, or `None` | do not look for a sidecar at all |
| a path (`str` or `PathLike`) | use this sidecar, wherever it lives |
| a `pandas.DataFrame` | use this table, already loaded |

The path and DataFrame forms exist for a recording whose sidecar is not next to
it: a converter that filters or rewrites a recording into a scratch directory
still has to apply the *original* recording's `channels.tsv`, and must pass it
explicitly, because `"auto"` looks beside the file it is given.

A trailing slash on a directory-valued recording (a CTF `.ds`, a 4D/BTi
directory) is stripped before the lookup, so `.ds` and `.ds/` resolve the same
sidecar on both paths.

`apply_channels_tsv` itself accepts either a path or a DataFrame, and treats
them identically (missing cells in a supplied frame become the empty cell that
means "declares nothing", matching how the TSV is read).

## Units are converted, not relabelled

The `units` column describes the numbers, so adopting it **rescales the
samples** (see [Physical Units](units.md)). An MNE-backed importer returns SI
volts; a sidecar declaring `uV` therefore multiplies that channel by 10^6 and
then sets the label, leaving values and unit in agreement. It is a no-op
wherever the importer already reports the sidecar's unit (EDF via pyedflib).

What "idempotent" guarantees precisely: applying the *same* sidecar any number
of times converts at most once and warns at most once. The second pass finds
each channel's `physical_dimension` already equal to the declared unit (or its
`bids_unit` already recorded) and does nothing. It is not a claim about applying
two *different* sidecars in sequence -- those compose, so a channel read as `V`
and then given a `mV` sidecar followed by a `uV` one ends up in `uV`, scaled by
10^6 overall.

Some channels are never rescaled, whatever the sidecar declares:

- **discrete types** (`TRIG`, `SYSCLOCK`, `CTRL`) hold codes rather than a
  measured quantity. MNE labels stim channels with the FIFF volts code while
  they carry integer event codes, so a sidecar declaring `mV` would turn codes
  5/3/7 into 5000/3000/7000;
- **channels with no samples**: metadata without a data column has no numbers
  for a new label to agree with, so the label does not move either;
- **units that are not convertible** -- different quantities, or a spelling
  neither side can parse (`n/a`, `a.u.`).

In each case the importer's values *and* its label are kept and the sidecar's
claim is recorded as `channels[label]["bids_unit"]` with a warning, so nothing
is relabelled without being converted. Adopting a unit later clears any
`bids_unit` left by an earlier disagreement.

A per-file summary lands in `rec.metadata["channels_tsv_units"]`:

```python
{"converted": 2, "relabelled": 0, "kept_importer_unit": 1, "units_column_present": True}
```

`units_column_present` separates "the sidecar declared no units at all" from
"the units were already correct", which the counters alone cannot.

### Matching rows to channels

A row describes the channel whose label equals its `name` exactly.
A row with no exact match falls back to ignoring case, but only when that is unambiguous:
exactly one channel label equals the row's name ignoring case,
no row names that channel exactly,
and no other unmatched row names it ignoring case.
This covers a sidecar that spells a label differently from the data file,
such as `Fp1-F7` for an EDF header's `FP1-F7`, as in CHB-MIT.
The channel keeps the data file's label; the match is logged at info level
and listed in the summary as `matched_case_insensitive`, `{sidecar_name: channel_label}`:

```python
{"converted": 0, "relabelled": 0, "kept_importer_unit": 0, "units_column_present": True,
 "matched_case_insensitive": {"Fp1-F7": "FP1-F7"}}
```

The key is present only when such a match happened.
An ambiguous row, such as `fp1-f7` against channels `FP1-F7` and `Fp1-F7`, is left unapplied with a warning.
The in-memory and streaming paths apply the same rule.

"Ignoring case" means full Unicode case folding (Python's `str.casefold`), which is broader than lower-casing:
a row `STRASSE` matches a channel `Straße`, because `ß` folds to `ss`.
Unicode normalization is not applied,
so a precomposed `é` (NFC) and an `e` followed by a combining accent (NFD) are different names.

## Module Documentation

::: biosigio.bids
    options:
      show_root_heading: true
      show_source: true
      members: true
