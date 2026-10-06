# Zarr Exporter (serving store)

Exports a `Recording` to a sharded Zarr v3 serving store: an anti-aliased,
per-modality-resampled `level 0` inference signal plus a min/max view pyramid for
rendering. A derived serving copy, not an archival source. Requires the `zarr`
extra (zarr v3).

See [Zarr Serving Store](../../formats/zarr.md) for the on-disk store contract and
serving model.

## Module Documentation

::: biosigio.exporters.zarr
    options:
      show_root_heading: true
      show_source: true
      members: true

## Subject information

`exclude_subject_info=True` removes the members named in `SUBJECT_INFO_KEYS` from
the store's `recording_metadata` (matched by normalized name at any depth, outside
biosigIO's label-keyed maps), reduces `source_file` and `bti_pdf_file` to their final
path component, and sets the root attribute `subject_info_excluded: true`, which
records that the option ran rather than that the store was verified clean. It keeps
recording dates and times, events, channel labels, units and prefilter text. See
[Subject information](../../formats/zarr.md#subject-information) for the full list and
the limits of matching by name.

::: biosigio.exporters.subject_info
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

rec = Recording.from_file("data.edf")
rec.to_zarr("out.zarr")                 # int16 by default (per-channel scale/offset)
rec.to_zarr("lossless.zarr", dtype="float32")
rec.to_zarr("no-subject.zarr", exclude_subject_info=True)
```
