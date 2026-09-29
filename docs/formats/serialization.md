# Serialization & Serving Formats

Besides EDF/BDF, biosigIO can serialize a `Recording` to three columnar or array
formats: Parquet, Arrow/Feather, and Zarr. Each carries enough information to be
read back without any side files, and all of them round-trip through the same
entry point, `Recording.from_file`, which auto-detects the format from the file
extension (`.parquet`, `.feather`, `.arrow`, `.zarr`).

```python
from biosigio import Recording

rec = Recording.from_file('recording.edf')

rec.to_parquet('recording.parquet')   # analytics
rec.to_arrow('recording.feather')     # fast IPC
rec.to_zarr('recording.zarr')         # cloud-native serving
```

## When to Use Which

The three formats solve different jobs. Parquet and Arrow/Feather are
lossless columnar tables for analysis and interchange; Zarr is a derived,
downsampled and quantized serving store, not an archival source.

| Format | Best for | Lossless | Notes |
|--------|----------|----------|-------|
| Parquet | Analytics (DuckDB, Polars, pandas, Spark) | Yes | Self-describing columnar table; one file per recording |
| Arrow/Feather | Fast zero-copy inter-process communication (IPC) | Yes | Same schema as Parquet, optimized for speed over compression |
| Zarr | Cloud-native serving: one store serves viewing, inference, and training | No | Derived, downsampled and quantized serving copy; the Brain Imaging Data Structure (BIDS) and EDF stay authoritative |

Parquet and Arrow keep the full signal: channels become columns, the time index
is preserved, and channels, events, and recording metadata travel in the file's
schema. Use them whenever you want the recording back exactly as it was.

Zarr is the serving format. A single conversion produces one cloud-native store
that all three downstream consumers read directly from object storage: a viewer
streams a min/max render pyramid, inference reads the anti-aliased base signal,
and training streams shards of it. Because it is downsampled per modality and
stored as scaled `int16` by default, it is a derived copy and not the source of
truth. The original BIDS dataset (or EDF) remains the authoritative, citable
artifact. See the [Zarr store contract](zarr.md) for the full layout.

## Lossless vs Derived Round-Trip

The two families round-trip differently, and the difference is intentional.

**Parquet and Arrow are lossless.** Reading a `.parquet` or `.feather` file back
restores the `Recording` exactly: signals (with their time index), channels,
events, and recording metadata. The file holds everything needed to rebuild the
object, so import is a faithful inverse of export.

**Zarr is reconstructed, not restored.** The importer rebuilds a `Recording`
from the store's canonical `level 0` signal, which is the anti-aliased,
per-modality downsampled inference signal, so the reconstructed sampling rate is
the store's canonical rate rather than the original acquisition rate. The
per-channel Zarr-only attributes (the `scale`/`offset` quantization parameters,
the `usable_for_inference` flags, the render pyramid) are applied during
dequantization but are not surfaced back onto the reconstructed channels. The
min/max view pyramid (`view/*`) is render-only and is never read on import.

For both families, `source_file` is reset to the path you read from, while the
original `source_format` provenance is preserved (a re-imported serialization
file keeps the format it was originally converted from rather than being
relabeled as `tabular` or `zarr`).

## Self-Describing Schema

Both families embed a versioned `biosigio` metadata blob so a reader can
recognize and version-check the file before trusting it.

- **Parquet and Arrow** carry the blob in the Arrow schema metadata
  (`tabular_schema.FORMAT = "biosigio-tabular"`, `FORMAT_VERSION = 1`). It holds
  the recording metadata, per-channel info, and events as one JSON object.
  Each signal column is one channel label, so a Feather/Arrow file edited to
  name two columns alike raises `ValueError` on import (Parquet itself refuses
  to read such a file back).
- **Zarr** carries an equivalent blob in the store's root attributes
  (`FORMAT = "biosigio-zarr"`, `FORMAT_VERSION = 2`), reusing the same metadata
  encoding so both formats record state the same way.

The encoding is lossless for values that are not native to JSON: datetimes and
dates become a typed envelope and are reconstructed to their original Python type
on read, and numpy scalars/arrays are converted to their Python equivalents.
Metadata that cannot be serialized raises an error rather than being silently
coerced to a string, because metadata loss is data loss.

## Optional Dependencies

These formats are not part of the core install; install the matching extra.

| Format | Extra | Install |
|--------|-------|---------|
| Parquet, Arrow/Feather | `arrow` (pyarrow) | `uv sync --extra arrow` or `uv pip install 'biosigio[arrow]'` |
| Zarr | `zarr` (zarr v3) | `uv sync --extra zarr` or `uv pip install 'biosigio[zarr]'` |

If the extra is missing, the exporter and importer raise an `ImportError` with
the exact install command, so you never get a partial or silent failure.

## Round-Trip Examples

### Parquet

```python
from biosigio import Recording

rec = Recording.from_file('recording.edf')
rec.to_parquet('recording.parquet')

# Auto-detected from the .parquet extension; restored exactly.
restored = Recording.from_file('recording.parquet')
```

### Arrow/Feather

```python
from biosigio import Recording

rec = Recording.from_file('recording.edf')
rec.to_arrow('recording.feather')

restored = Recording.from_file('recording.feather')
```

### Zarr

```python
from biosigio import Recording

rec = Recording.from_file('recording.edf')
rec.to_zarr('recording.zarr')

# Reconstructed at the store's canonical (downsampled) level-0 rate.
served = Recording.from_file('recording.zarr')
```

A Zarr store can hold several `(modality, rate)` groups that cannot share a
single time grid. When there is more than one group, pass `group=` to choose
which to reconstruct:

```python
served = Recording.from_file('recording.zarr', group='eeg_250hz')
```

Keep these examples minimal; for the full store layout, tuning knobs, and the
serving model, see the [Zarr store contract](zarr.md), and for end-to-end
walk-throughs see the [serialization examples](../examples/serialization.md).

## See Also

- [Zarr Serving Store](zarr.md): the on-disk contract, signal rules, and serving model.
- [Serialization Examples](../examples/serialization.md): worked round-trip examples.
