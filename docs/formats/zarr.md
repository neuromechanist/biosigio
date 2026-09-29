# Zarr Serving Store

biosigIO can export a recording to a biosigIO Zarr serving store: a single, cloud-native store that serves three downstream jobs from one conversion, namely fast interactive viewing on thin clients, edge and batch inference, and training-time streaming. This page documents the store contract and the serving model so the consumers of the store (a web viewer, an inference service, a training loader, a batch converter) can read it correctly. Those consumers are built elsewhere; biosigIO only produces and reads the store.

## Overview

The store is a sharded Zarr version 3 (v3) store with one group per (modality, native rate) pair. Level 0 of every group is the anti-aliased inference signal, resampled to a per-modality canonical rate, with a min/max view pyramid above it for rendering.

The store is a derived serving copy, not an archive. The Brain Imaging Data Structure (BIDS) dataset, and the European Data Format (EDF) files alongside it, remain the authoritative source of truth and the citable artifact. The Zarr store is reproducible from them, and the downsampling parameters are recorded in the store attributes so the derivation is documented.

Zarr is an optional dependency (the `zarr` extra, which installs zarr v3) and is imported lazily. Install it with one of:

```bash
uv sync --extra zarr           # development install
uv pip install 'biosigio[zarr]'   # existing environment
```

Write a store with `Recording.to_zarr`:

```python
from biosigio import Recording

rec = Recording.from_file('recording.edf')

# Defaults: int16 storage, per-modality rate caps, min/max view pyramid
store_path = rec.to_zarr('recording.zarr')
print(store_path)  # 'recording.zarr'
```

## Store layout

One Zarr v3 root group holds the whole recording. Both the writer and the reader honor this layout:

```
/                              root group
  attrs: biosigio_version, format="biosigio-zarr", format_version,
         source_format, modality_rates, dtype, view_downsample,
         view_chunk_columns, anti_alias_filter, channel_groups,
         recording_metadata, channels_tsv_units (only when a BIDS
         channels.tsv was applied), created_utc, note

  <modality>_<rate>hz/         one group per (modality, native rate)
    attrs: modality, rate, original_rate, n_channels, n_samples, channels[],
           n_view_levels, view_levels[], view_downsample, view_chunk_columns,
           chunk_seconds, shard_seconds
    0                          (n_ch, n_time) level-0 signal, sharded
      attrs: level=0, rate, source_rate_hz, downsample_factor=1, kind="signal",
             chunk_samples, shard_samples, anti_aliased, usable_for_inference,
             scale[], offset[],
             physical_formula="physical = digital * scale + offset"
    view/                      min/max render pyramid (not sharded)
      1, 2, ...                (2, n_ch, n_time_L), axis0 = [min, max]
        attrs: level, downsample_factor, rate_effective, chunk_columns,
               kind="minmax_envelope", usable_for_inference=false

  events/
    onset (f64), duration (f64), code (i32)
    attrs: label_map {code: description}, n_events
```

The group name encodes the served (target) rate, not the source rate, for example `eeg_250hz` even when the source was recorded at 500 Hz. The two rates are both kept in the group attributes:

- `group.attrs['rate']` is the served target rate (the rate of level 0).
- `group.attrs['original_rate']` is the integer-rounded native rate used as the grouping key.
- Each entry in `group.attrs['channels'][].original_rate` is the true per-channel acquisition rate as a float, so a non-integer source rate (for example 511.7 Hz) is preserved even though the group key rounds it (metadata loss is data loss).

The level-0 array carries the per-channel `scale` and `offset`. Reconstruct physical units with:

```
physical = digital * scale + offset
```

Per-channel metadata lives in each group's `channels` attribute, one object per channel with these keys:

`label`, `channel_type`, `modality`, `unit`, `prefilter`, `original_rate`, `target_rate`, `anti_aliased`, `usable_for_inference`, `scale`, `offset`, `row_index`.

The `row_index` maps a channel to its row in the `(n_ch, n_time)` level-0 array.

Two keys are present only when they apply: `nonfinite_samples`, the count of NaN or inf samples that were zero-filled, and `bids_unit`, a unit a BIDS `channels.tsv` declared that the exporter recorded rather than adopted (see [BIDS units in a store](#bids-units-in-a-store)).

## Level 0 vs view/*

The two array tiers exist for different jobs and are downsampled by deliberately different rules:

- **Level 0** is the anti-aliased inference signal, one sample per time step at the group's canonical rate. Inference and training read this tier. Its `usable_for_inference` flag is `true` for groups that contain at least one continuous channel and `false` for a group made up entirely of discrete channels.
- **view/\*** are min/max render envelopes. Each level stores two values per downsample bin, the minimum and the maximum, on axis 0 (`[min, max]`). Envelopes preserve visible transients so brief spikes do not vanish when zoomed out, which is why they are nonlinear and are flagged `usable_for_inference=false`. Never train or run inference on the view tier.

The two downsampling philosophies coexist on purpose: anti-aliased polyphase resampling for level 0, min/max binning for view/*.

## Chunk geometry and the declared pyramid

The two tiers are also chunked by different rules, for the same reason they are downsampled by different rules.

- **Level 0 is chunked on time.** Its inner chunk spans `chunk_seconds` (4 s by default) and its shard spans `shard_seconds` (300 s), rounded to a whole number of chunks. Inference and training read a time span, so a time-based grain is the right unit. The array attributes record the resolved lengths in samples as `chunk_samples` and `shard_samples`, and `source_rate_hz` alongside `rate` so the native acquisition rate is explicit next to the served (capped) one.
- **view/\* levels are chunked on a constant column count**, `view_chunk_columns` (1024 by default), capped by the level's own length so a short level is a single chunk. Each level's resolved width is in its `chunk_columns` attribute. A viewport needs roughly 1000 to 2500 columns whatever level it picks, so columns, not seconds, are the unit a render request is measured in. A time-based rule would shrink the chunk by the pyramid factor at every level and shatter one screenful into hundreds of tiny requests: on the 40-minute, 129-channel, 250 Hz store the `nemarOrg/nemar-cli#1178` transfer-efficiency audit measured, a whole-recording render at level 4 was 594 requests for 1.16 MB (about 2 KB each) and the level-6 minimap was 148 requests for 77 KB. At 1024 columns the same reads are 3 requests and 1 request. The chunk counts are pinned by a test; the byte figures are the audit's.

Each channel group also **declares its pyramid**, so a reader learns which `view/*` paths exist without probing for a 404:

- `n_view_levels` is the number of `view/*` arrays actually written for that group.
- `view_levels` lists them, always contiguous from 1, for example `[1, 2, 3]`.
- `view_downsample`, `view_chunk_columns`, `chunk_seconds`, and `shard_seconds` describe the geometry to expect, so a client knows the shape of the store before it fetches anything.

The declaration replaces existence probing, not array opening: a client still opens each level's array to read its exact shape and its resolved `chunk_columns`, which is `view_chunk_columns` capped by that level's length.

These are additive attributes plus a change of chunk shape, and every Zarr v3 reader takes chunk shapes from the array metadata it must already parse, so `format_version` stays at 2.

## Per-modality rates and dtype

**Rate caps.** Level 0 is resampled to `target = min(native, cap)` per modality, and the store never upsamples. The default caps are:

| Modality | Default cap (Hz) |
|---|---|
| EEG | 250 |
| MEG | 250 |
| iEEG (SEEG, ECoG, DBS) | 1000 |
| EMG | 1000 |
| Other (BEH, MISC, ...) | native (no cap) |

A modality absent from the cap table keeps its native rate. Override the caps per call with the `modality_rates` argument:

```python
# Keep MEG high-gamma by raising its cap; EEG stays at the default
rec.to_zarr('recording.zarr', modality_rates={'MEG': 1000, 'EEG': 250})
```

**Storage dtype.** The default `dtype="int16"` stores each channel with a per-channel `scale` and `offset` (half the bytes of float32; consumers cast on read). int16 cannot represent NaN or inf, so the exporter rejects non-finite samples rather than silently corrupting them; use `dtype="float32"` (lossless, and it keeps NaN) when a channel carries gaps:

```python
rec.to_zarr('recording.zarr', dtype='float32')
```

**Discrete channels.** Trigger and clock channel types (`TRIG`, `SYSCLOCK`, `CTRL`) are resampled by nearest sample with no anti-alias filter so step edges survive, and they are flagged `usable_for_inference=false`.

**Heterogeneous channels.** Channels are grouped by (modality, native rate), so each array stays length-consistent. A common single-rate recording yields one group per modality; a genuinely mixed-rate source yields one group per rate, which is faithful rather than silently resampled together.

## BIDS units in a store

When the recording sits in a BIDS layout, the sibling `_channels.tsv` is authoritative for each channel's `type` and `units`, and adopting a declared unit **converts the samples** into it rather than relabelling them (see [BIDS Helpers](../api/bids.md)). Both export paths apply it and by the same rules, so which path built a store is not something a consumer has to know:

- `Recording.from_file` applies the sidecar on import, and `to_zarr` then exports the converted values.
- `stream_to_zarr`, the bounded-memory path for recordings too large to load, folds each channel's conversion into its single streaming pass.

Both take the same `bids_channels` argument, with the same values and the same resolution: `"auto"` (default), `"off"` or `None`, an explicit path, or a DataFrame. See [`bids_channels`](../api/bids.md#choosing-the-sidecar-bids_channels) for what each means and when a converter has to pass a sidecar explicitly.

That parity is the contract: one recording and one sidecar produce the same `unit`, `channel_type`, `scale`, `offset` and values whichever path ran, so a dataset whose recordings straddle a size threshold cannot serve microvolts for its small runs and volts for its large ones.

Two attributes record what the sidecar did:

- `channels[].bids_unit` on a channel whose declared unit was recorded rather than adopted (a discrete trigger channel, or a unit that is not convertible from the importer's), so the BIDS claim survives without being asserted over values that contradict it.
- `channels_tsv_units` on the root group, the per-file summary (`converted`, `relabelled`, `kept_importer_unit`, `units_column_present`). It is absent when no sidecar was applied.

Both are additive, so `format_version` stays at 2.

`force_modality` (streaming only) still wins over the sidecar for **grouping**: it pins every channel to one modality, matching the BIDS datatype suffix a converter is working from, while the sidecar's `type` still sets each channel's `channel_type`.

## Reading a store

Read a store back into a `Recording` with `Recording.from_file`. The `.zarr` extension is auto-detected, or pass `importer='zarr'` explicitly:

```python
from biosigio import Recording

# Extension-inferred
rec = Recording.from_file('recording.zarr')

# Explicit importer
rec = Recording.from_file('recording.zarr', importer='zarr')
```

The importer reads level 0 of one group, applies the `physical = digital * scale + offset` dequantization, and restores channels, events, and recording metadata. The view pyramid is render-only and is never read here.

A store published by the streaming exporter before repeated EDF labels were suffixed can list the same `label` twice in a group. The importer reads it anyway, suffixing the repeats the same way the EDF importer does (`T8-P8-0`, `T8-P8-1`; see [EDF repeated channel labels](edf.md#repeated-channel-labels)) and warning, so no channel is dropped. Regenerating such a store from its source EDF writes the suffixed labels into the store itself, which changes its channel names.

Because the groups in a store can sit at different rates that cannot share biosigIO's single time grid, the importer reconstructs one group at a time. When a store holds a single group it is selected automatically; when it holds more than one, pass the `group=` selector by group name. Calling `from_file` on a multi-group store without `group=` raises an error that lists the available groups:

```python
# Multi-group store: choose which (modality, rate) group to reconstruct
rec = Recording.from_file('recording.zarr', importer='zarr', group='eeg_250hz')
```

Reconstruction is at the store's canonical (possibly downsampled) level-0 rate, not the original full-rate signal, because the store is the serving copy and BIDS remains the source of truth.

## Format version and provenance

The root group attributes carry the format tag and version so a reader can recognize and version-check a store:

- `format` is `"biosigio-zarr"`.
- `format_version` is currently `2`. In version 2, `recording_metadata` is a native JSON object, with non-JSON-native values such as datetimes carried in typed envelopes, so a browser or zarrita reader can consume it without a second parse. Version 1 stored `recording_metadata` as a JSON string; the reader still accepts version 1 stores.

A reader should reject a store whose `format_version` is newer than the one it supports rather than guess at an unknown layout. The biosigIO importer does exactly this: it raises if the store's version exceeds the version this build reads.

### Extended attributes

Downstream tools may attach their own attributes to the root group, a modality group, or the `events` group, for example sensor positions, a line-noise frequency, or richer per-code descriptions alongside the `events` group's `label_map`. The biosigIO reader reads only the attributes it defines and ignores any it does not recognize, so these domain-specific extensions are forward-compatible: they enrich the store for a specialized consumer (such as a scalp-topography viewer) without changing the `format_version` or breaking the store contract.

## Serving model (external)

The viewing, inference, training, and batch-conversion layers live outside biosigIO. biosigIO only writes and reads the store; the sections below summarize the intended serving model so those external consumers can be built against the same contract.

**Reads need no backend.** A Zarr v3 store is objects plus JSON, so a browser reader (zarrita) streams it directly from object storage over HTTPS using ranged GETs. The viewing path has no decode service.

**Cross-Origin Resource Sharing (CORS) is the one hard requirement.** The bucket must allow browser GET, HEAD, and Range requests, and expose the ETag and Content-Length headers.

**A Content Delivery Network (CDN) is recommended for a public production viewer.** A viewport pulls several small chunk objects; an edge CDN with HTTP/2 or HTTP/3 multiplexing shares one connection across those fetches, caches the small, shared coarse pyramid levels at the edge, and collapses request count and origin egress. The viewer code is identical with or without a CDN, since zarrita reads object URLs; ship on object storage plus CORS, then front it with a CDN by changing the base URL. The hot viewing path stays CDN-friendly because the view levels are not sharded.

**Reader contracts** for the three consumers:

- **Viewer:** read the root and group attributes, take the list of `view/*` levels from the group's `view_levels` or `n_view_levels` rather than probing for a 404, then open those levels and pick the one whose `rate_effective` puts roughly one sample per screen pixel. Fetch the chunks covering the viewport and render the min/max band using each channel's `scale` and `offset`. At maximum zoom, read level 0 directly. Group the display by modality and overlay events from the `events` group.
- **Inference:** read level 0 of the relevant group, apply scale and offset, and window the signal. Never read view/*. For a lower analysis rate, derive it on read from level 0 with an anti-aliased resample rather than persisting a third tier.
- **Training:** iterate shards of level 0 sequentially, one object per shard and many windows per object, shuffling at the shard level plus a within-shard buffer.

## Default parameters

The exporter defaults follow the store specification. Override any of them as keyword arguments to `to_zarr`:

| Parameter | Default | Meaning |
|---|---|---|
| `modality_rates` | EEG 250, MEG 250, iEEG 1000, EMG 1000 | Per-modality canonical rate cap |
| `dtype` | `int16` | Storage type; `float32` for lossless (keeps NaN) |
| `view_downsample` | 4 | Time decimation factor between pyramid levels |
| `min_view_samples` | 512 | Pyramid floor; see the stopping rule below |
| `max_view_levels` | 12 | Hard cap on pyramid depth |
| `chunk_seconds` | 4 | Level-0 random-access grain |
| `shard_seconds` | 300 | Level-0 sequential-read grain, rounded to whole chunks |
| `view_chunk_columns` | 1024 | Columns per `view/*` chunk, capped by the level's length |
| `compressor_level` | 5 | zstd compression level (Blosc codec) |

**The pyramid stopping rule.** A level is built, and then the loop stops once the level just built is at or below `min_view_samples`. So the floor bounds the level a further one would have been built *from*, not the shortest level in the store: the final level can itself be shorter than `min_view_samples`. A 30000-sample level 0 at the defaults gives 7500, 1875, and 468; the 468 is written because 1875 was still above the floor, and no level 4 follows because 468 is not.

```python
# A lossless store with a shorter sequential-read grain
rec.to_zarr('recording.zarr', dtype='float32', shard_seconds=120)
```
