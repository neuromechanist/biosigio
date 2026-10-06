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

`exclude_subject_info=True` leaves subject information out of the store; see
[Subject information](../../formats/zarr.md#subject-information).

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
