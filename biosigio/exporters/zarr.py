"""Zarr exporter for biosignal recordings (biosigIO serving format).

Writes a single, cloud-native store that serves three jobs from one conversion:
fast edge viewing, edge/batch inference, and training-time streaming. The design
follows a few correctness rules:

- ``level 0`` of every channel group is the **canonical inference signal**,
  anti-aliased and resampled to a per-modality rate (250 Hz for EEG/MEG, 1000 Hz
  for iEEG/EMG by default), never upsampled.
- A **min/max view pyramid** sits above level 0 (``<group>/view/<L>``). These
  levels are nonlinear envelopes for rendering only and are flagged
  ``usable_for_inference=False`` so nobody trains on them by mistake.
- Heterogeneous channels (mixed types/units, and the rarer mixed native rates
  that BIDS allows within a modality) are grouped by ``(modality, native rate)``
  so every array stays internally length-consistent. Trigger/clock channels are
  resampled without anti-aliasing (nearest sample) to preserve step edges.
- Signals are stored ``int16`` with a per-channel ``scale``/``offset`` by default
  (half the bytes of float32; ML casts on read). Pass ``dtype="float32"`` for a
  lossless store.

The original full-rate recording is never the source of truth here; that stays in
the BIDS archive. This store is a derived serving copy, and the downsampling
parameters are recorded in attrs so it is reproducible. zarr is an optional
dependency (the ``zarr`` extra), imported lazily.

Layout (one root group per recording)::

    /                         root attrs: provenance, modality_rates, metadata
      <modality>_<rate>hz/    one group per (modality, native rate); attrs declare
                              the pyramid and the chunk geometry
        0                     (n_ch, n_time) base signal, anti-aliased. Chunked and
                              sharded on TIME (``chunk_seconds`` / ``shard_seconds``),
                              the granularity inference and training reads want.
        view/1, view/2, ...   (2, n_ch, n_time_L) min/max envelopes. Chunked on a
                              constant COLUMN count (``view_chunk_columns``), NOT on
                              time: a viewport needs ~1-2k columns at whatever level
                              it picks, so a time-based chunk would shrink fourfold
                              per level and turn one screen into hundreds of tiny
                              requests. On a 40-minute 129-channel 250 Hz EEG
                              store measured in a transfer-efficiency audit, a
                              whole-recording view at LEVEL 4 was 594 requests for
                              1.16 MB and the LEVEL 6 minimap 148 for 77 KB; at 1024
                              columns those are 3 requests and 1. The chunk counts
                              are pinned by a test here, the byte figures are the
                              audit's and are not re-measured.
      events/                 onset, duration, code arrays + label_map attr

Every channel group declares its pyramid and geometry in attrs -- ``n_view_levels``,
``view_levels``, ``view_downsample``, ``view_chunk_columns``, ``chunk_seconds``,
``shard_seconds`` -- so a reader plans its reads instead of probing ``view/1``,
``view/2``, ... until a 404. ``format_version`` stays at 2 for these: they are
additive attrs plus a chunk-shape change, and every Zarr v3 reader takes chunk
shapes from the array metadata it must already parse.
"""

from __future__ import annotations

import datetime as _dt
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..core.channel_types import DISCRETE_CHANNEL_TYPES
from ..core.emg import Recording
from ..tabular_schema import metadata_to_mapping
from ..version import __version__ as _BIOSIGIO_VERSION

# Format tag/version for the root attrs, so a reader can recognize and
# version-check a biosigIO Zarr store. v2 stores ``recording_metadata`` as a
# native JSON object (v1 stored it as a JSON string); the reader accepts both.
FORMAT = "biosigio-zarr"
FORMAT_VERSION = 2

# Per-modality canonical inference rate (Hz). target = min(native, cap); a
# modality absent from this map keeps its native rate.
DEFAULT_MODALITY_RATES: dict[str, int] = {
    "EEG": 250,
    "MEG": 250,
    "IEEG": 1000,
    "EMG": 1000,
}

# Channel types that are discrete/event-like rather than continuous signals. They
# are resampled by nearest sample (no anti-alias low-pass) so step edges survive,
# and they are never marked inference-usable. Defined in core.channel_types
# because the BIDS sidecar path must exempt exactly the same set from unit
# rescaling (issue #122); re-exported under the historical name.
_DISCRETE_TYPES = DISCRETE_CHANNEL_TYPES

_INT16_MIN, _INT16_MAX = -32768, 32767


def require_zarr():
    """Import zarr lazily, raising a clear install hint when it is absent."""
    try:
        import zarr  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Zarr serving-format export/import requires zarr (v3), an optional "
            "dependency. Install it with: uv sync --extra zarr  (or, for an existing "
            "install, uv pip install 'biosigio[zarr]')."
        ) from e
    import zarr

    return zarr


def _target_rate(native_rate: float, modality: str, modality_rates: dict[str, int]) -> float:
    """Canonical rate for a channel: capped by modality, never upsampled."""
    cap = modality_rates.get(modality.upper())
    if cap is None:
        return float(native_rate)
    return float(min(native_rate, cap))


def _resample_channel(
    x: np.ndarray, native_rate: float, target_rate: float, *, discrete: bool
) -> np.ndarray:
    """Resample one channel to ``target_rate``.

    Continuous channels use polyphase resampling (anti-aliased). Discrete channels
    use nearest-sample selection so trigger edges are not smeared.
    """
    x = np.asarray(x, dtype=np.float64)
    if float(target_rate) == float(native_rate):
        return x
    n_out = int(round(len(x) * target_rate / native_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)
    if discrete:
        idx = np.minimum(
            (np.arange(n_out) * native_rate / target_rate).round().astype(int), len(x) - 1
        )
        return x[idx]
    frac = Fraction(int(round(target_rate)), int(round(native_rate)))
    y = resample_poly(x, frac.numerator, frac.denominator)
    # resample_poly length can be off by one versus the nominal target; trim or
    # pad to the exact ratio length so a group's channels stay aligned.
    if len(y) > n_out:
        y = y[:n_out]
    elif len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)), mode="edge")
    return y


def _quantize_int16_channel(row: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Map one channel (1-D float) to int16 with scale/offset.

    ``physical = digital * scale + offset``. A constant (or empty) channel gets
    scale=1 and the constant in offset.

    Non-finite samples (NaN/inf) cannot be represented in int16. Rather than fail
    the whole recording for a single bad channel -- a NaN in one MISC/aux channel
    (an accelerometer, force plate, or sync line in a MoBI dataset) would otherwise
    sink the entire recording, including every good EEG channel -- the non-finite
    samples are zero-filled: the physical range (scale/offset) is computed over the
    FINITE samples only, then each non-finite sample is written as the digital code
    for physical 0 (clamped into range). The count of filled samples is returned so
    the caller can flag the channel (``nonfinite_samples``, and demote it to
    ``usable_for_inference=False``). The fill is lossy, so it is documented rather
    than silent, and the BIDS source remains authoritative. Callers that must keep
    NaN exactly (e.g. for training) should export with ``dtype="float32"``.

    Operates one channel at a time so the exporter can quantize straight into the
    output array without materializing a float64 copy of the whole group (#95).
    """
    # Quantize in float64 regardless of the caller's dtype (the old batch path
    # cast the whole block up front); a no-op for the float64 resample output.
    row = np.asarray(row, dtype=np.float64)
    finite_mask = np.isfinite(row)
    n_nonfinite = int(row.size - int(np.count_nonzero(finite_mask))) if row.size else 0
    finite = row[finite_mask] if n_nonfinite else row
    if finite.size == 0:  # empty, or every sample non-finite: flat zero channel
        return np.zeros(row.shape, dtype=np.int16), 1.0, 0.0, n_nonfinite
    pmin = float(np.min(finite))
    pmax = float(np.max(finite))
    if pmax == pmin:  # constant channel: store the constant in offset
        return np.zeros(row.shape, dtype=np.int16), 1.0, pmin, n_nonfinite
    digital_span = _INT16_MAX - _INT16_MIN
    s = (pmax - pmin) / digital_span
    o = pmin - _INT16_MIN * s
    if n_nonfinite:  # replace NaN/inf with physical 0 before quantizing
        row = np.where(finite_mask, row, 0.0)
    digital = np.round((row - o) / s)
    return np.clip(digital, _INT16_MIN, _INT16_MAX).astype(np.int16), s, o, n_nonfinite


def _build_minmax_pyramid(
    base: np.ndarray, factor: int, min_samples: int, max_levels: int
) -> list[np.ndarray]:
    """Return a list of (2, n_ch, n_time_L) min/max envelope levels.

    Level L is built from level L-1 (min of mins, max of maxs), so the envelope is
    exact at every zoom. Axis 0 is [min, max].
    """
    levels: list[np.ndarray] = []
    cur_min = base
    cur_max = base
    n_ch = base.shape[0]
    for _ in range(max_levels):
        n_time = cur_min.shape[1]
        if n_time <= min_samples:
            break
        n_out = n_time // factor
        if n_out < 1:
            break
        trim = n_out * factor
        lvl_min = cur_min[:, :trim].reshape(n_ch, n_out, factor).min(axis=2)
        lvl_max = cur_max[:, :trim].reshape(n_ch, n_out, factor).max(axis=2)
        levels.append(np.stack([lvl_min, lvl_max], axis=0))
        cur_min, cur_max = lvl_min, lvl_max
    return levels


def _chunk_shard_time(
    n_time: int, rate: float, chunk_seconds: float, shard_seconds: float
) -> tuple[int, int]:
    """Pick a time chunk and a shard that is a multiple of it."""
    chunk_t = max(1, min(n_time, int(round(chunk_seconds * rate))))
    k = max(1, int(round(shard_seconds / chunk_seconds)))
    n_chunks = -(-n_time // chunk_t)  # ceil
    shard_t = chunk_t * min(k, n_chunks)
    return chunk_t, shard_t


def _view_chunk_columns(n_time_level: int, view_chunk_columns: int) -> int:
    """Time-chunk width (columns) for one view-pyramid level.

    A constant column count, capped by the level's own length so a short level is
    a single chunk. Deliberately NOT time-based: view levels are read by column
    budget (a viewport wants ~1-2k columns at whatever level it picks), so a
    seconds-based rule would shrink the chunk fourfold per level and shatter one
    screenful into hundreds of tiny requests (about 2 KB each on the 40-minute,
    129-channel, 250 Hz EEG store a transfer-efficiency audit measured). Shared
    by both exporters (the
    in-memory one here and :mod:`biosigio.exporters.zarr_stream`) so their
    geometry cannot drift.
    """
    return max(1, min(int(n_time_level), int(view_chunk_columns)))


class ZarrExporter:
    """Exporter to a sharded Zarr v3 store with a min/max view pyramid."""

    @staticmethod
    def export(
        rec: Recording,
        filepath: str,
        *,
        modality_rates: dict[str, int] | None = None,
        dtype: str = "int16",
        view_downsample: int = 4,
        min_view_samples: int = 512,
        max_view_levels: int = 12,
        chunk_seconds: float = 4.0,
        shard_seconds: float = 300.0,
        view_chunk_columns: int = 1024,
        compressor_level: int = 5,
        events_df=None,
    ) -> str:
        """Write ``rec`` to a Zarr store at ``filepath``.

        Args:
            rec: Source recording.
            filepath: Output store path (``.zarr`` appended if missing).
            modality_rates: Per-modality canonical rate cap. Defaults to
                :data:`DEFAULT_MODALITY_RATES`. MEG defaults to 250 Hz with EEG;
                raise its cap if you need MEG high-gamma.
            dtype: ``"int16"`` (default, scaled, half the bytes) or ``"float32"``
                (lossless).
            view_downsample: Time decimation factor between pyramid levels.
            min_view_samples: Pyramid floor. For the exact stopping rule -- and
                why the last level written can be shorter than this -- see
                :func:`~biosigio.exporters.zarr_stream._pyramid_level_lengths`.
            max_view_levels: Hard cap on pyramid depth.
            chunk_seconds: Time span per chunk of ``level 0`` (random-access
                granularity). View levels are chunked by column count instead;
                see ``view_chunk_columns``.
            shard_seconds: Time span per shard of ``level 0`` (sequential-read
                granularity for training); rounded to a whole number of chunks.
            view_chunk_columns: Columns per chunk of every ``view/*`` level
                (capped by the level's length). Must be >= 1. The default 1024
                is about one viewport, so the columns a viewport needs, at
                whichever level it picks, are a handful of requests rather than
                hundreds.
            compressor_level: zstd level for the Blosc codec.
            events_df: Optional events table; falls back to ``rec.events``.

        Returns:
            The store path written.
        """
        zarr = require_zarr()
        from zarr.codecs import BloscCodec

        if rec.signals is None or len(rec.channels) == 0:
            raise ValueError("No signals loaded")
        if dtype not in ("int16", "float32"):
            raise ValueError("dtype must be 'int16' or 'float32'")
        if int(view_chunk_columns) < 1:
            raise ValueError("view_chunk_columns must be >= 1")

        rates = dict(DEFAULT_MODALITY_RATES if modality_rates is None else modality_rates)
        if not filepath.endswith(".zarr"):
            filepath = filepath + ".zarr"

        store = zarr.storage.LocalStore(filepath)
        root = zarr.create_group(store=store, overwrite=True)
        compressors = [BloscCodec(cname="zstd", clevel=compressor_level)]

        # Group channels by (modality, native rate) so each array is length-
        # consistent. Preserve channel order within a group.
        groups: dict[tuple[str, int], list[str]] = {}
        for label in rec.signals.columns:
            info = rec.channels[label]
            key = (str(info.get("modality", "MISC")), int(round(info["sample_frequency"])))
            groups.setdefault(key, []).append(label)

        written_groups: list[str] = []
        for (modality, native_rate), labels in groups.items():
            target_rate = _target_rate(native_rate, modality, rates)

            # Stream channels one at a time straight into the output-dtype array.
            # The previous code held a float64 list of every channel, a float64
            # vstack copy, and the int16 copy at once (~4.5x the int16 output);
            # a 5 GB recording peaked at ~26 GB RSS and OOM'd every free CI
            # runner (#95). n_time is the post-resample length, identical for
            # every channel in the group (they share native_rate).
            n_ch = len(labels)
            n_time = max(0, int(round(len(rec.signals) * target_rate / native_rate)))
            base = np.empty((n_ch, n_time), dtype=np.int16 if dtype == "int16" else np.float32)
            scale = np.ones(n_ch, dtype=np.float64)
            offset = np.zeros(n_ch, dtype=np.float64)
            chan_meta = []
            for i, label in enumerate(labels):
                info = rec.channels[label]
                ctype = str(info.get("channel_type", "MISC")).upper()
                discrete = ctype in _DISCRETE_TYPES
                y = _resample_channel(
                    rec.signals[label].values, native_rate, target_rate, discrete=discrete
                )
                n_nonfinite = 0
                if dtype == "int16":
                    base[i], scale[i], offset[i], n_nonfinite = _quantize_int16_channel(y)
                else:
                    base[i] = y.astype(np.float32)
                del y
                meta = {
                    "label": label,
                    "channel_type": ctype,
                    "modality": modality,
                    "unit": info.get("physical_dimension", "n/a"),
                    "prefilter": info.get("prefilter", "n/a"),
                    # The true per-channel rate, not the rounded group key, so a
                    # non-integer acquisition rate is not lost (metadata loss is data loss).
                    "original_rate": float(info["sample_frequency"]),
                    "target_rate": float(target_rate),
                    "anti_aliased": bool((not discrete) and target_rate < native_rate),
                    # A channel whose NaN/inf gaps were zero-filled is lossy: keep it
                    # viewable but never inferable (int16 fill is not the real signal).
                    "usable_for_inference": (not discrete) and n_nonfinite == 0,
                    "scale": float(scale[i]),
                    "offset": float(offset[i]),
                    "row_index": i,
                }
                if n_nonfinite:
                    meta["nonfinite_samples"] = n_nonfinite
                # A unit the BIDS sidecar declared but the values contradict is
                # kept as metadata rather than asserted (see
                # biosigio.bids._decide_unit); the streaming exporter carries it
                # the same way, so the two stores stay comparable channel for
                # channel.
                if info.get("bids_unit"):
                    meta["bids_unit"] = info["bids_unit"]
                chan_meta.append(meta)

            gname = f"{modality.lower()}_{int(round(target_rate))}hz"
            # Disambiguate the rare collision of two native rates -> same target.
            if gname in written_groups:
                candidate = f"{gname}_from{int(native_rate)}"
                if candidate in written_groups:
                    raise ValueError(
                        f"Zarr group name collision: {candidate!r} is already used. Three or more "
                        f"native rates in modality {modality!r} map to the same target rate; "
                        "split the recording or widen the modality rate cap."
                    )
                gname = candidate
            written_groups.append(gname)
            grp = root.create_group(gname)

            # Min/max view pyramid (rendering only), built in digital space so it
            # shares the base scale/offset; never inference-usable. Built before the
            # group attrs so the group can DECLARE its depth (n_view_levels /
            # view_levels) and a reader never has to probe view/1..n until a 404.
            pyramid = _build_minmax_pyramid(
                base, view_downsample, min_view_samples, max_view_levels
            )
            chunk_t, shard_t = _chunk_shard_time(n_time, target_rate, chunk_seconds, shard_seconds)
            grp.attrs.update(
                {
                    "modality": modality,
                    "rate": float(target_rate),
                    "original_rate": float(native_rate),
                    "n_channels": int(n_ch),
                    "n_samples": int(n_time),
                    "channels": chan_meta,
                    "n_view_levels": len(pyramid),
                    "view_levels": list(range(1, len(pyramid) + 1)),
                    "view_downsample": int(view_downsample),
                    "view_chunk_columns": int(view_chunk_columns),
                    "chunk_seconds": float(chunk_seconds),
                    "shard_seconds": float(shard_seconds),
                }
            )

            a0 = grp.create_array(
                "0",
                shape=(n_ch, n_time),
                chunks=(n_ch, chunk_t),
                shards=(n_ch, shard_t),
                dtype=base.dtype,
                compressors=compressors,
            )
            a0[:] = base
            a0.attrs.update(
                {
                    "level": 0,
                    "rate": float(target_rate),
                    # The native acquisition rate this group came from, so a reader
                    # can tell the effective (capped) rate from the source rate
                    # without re-deriving it from the per-channel metadata.
                    "source_rate_hz": float(native_rate),
                    "downsample_factor": 1,
                    "kind": "signal",
                    "chunk_samples": int(chunk_t),
                    "shard_samples": int(shard_t),
                    "anti_aliased": any(m["anti_aliased"] for m in chan_meta),
                    # A group of only discrete channels is not inference-usable.
                    "usable_for_inference": any(m["usable_for_inference"] for m in chan_meta),
                    "scale": scale.tolist(),
                    "offset": offset.tolist(),
                    "physical_formula": "physical = digital * scale + offset",
                }
            )

            view = grp.create_group("view")
            for li, lvl in enumerate(pyramid, start=1):
                eff_factor = view_downsample**li
                ct = _view_chunk_columns(lvl.shape[2], view_chunk_columns)
                av = view.create_array(
                    str(li),
                    shape=lvl.shape,
                    chunks=(2, n_ch, ct),
                    dtype=lvl.dtype,
                    compressors=compressors,
                )
                av[:] = lvl
                av.attrs.update(
                    {
                        "level": li,
                        "downsample_factor": eff_factor,
                        "rate_effective": float(target_rate) / eff_factor,
                        "kind": "minmax_envelope",
                        "axis0": ["min", "max"],
                        "chunk_columns": int(ct),
                        "usable_for_inference": False,
                    }
                )

        # Events as onset/duration/code arrays plus a portable code->label map.
        ev = rec.events if events_df is None else events_df
        eg = root.create_group("events")
        if ev is not None and len(ev) > 0:
            onset = ev["onset"].to_numpy(dtype=np.float64)
            duration = ev["duration"].to_numpy(dtype=np.float64)
            descs = ev["description"].astype(str).to_numpy()
            uniques = list(dict.fromkeys(descs.tolist()))
            code_of = {d: i for i, d in enumerate(uniques)}
            codes = np.array([code_of[d] for d in descs], dtype=np.int32)
            for name, arr in (("onset", onset), ("duration", duration), ("code", codes)):
                a = eg.create_array(name, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
                a[:] = arr
            eg.attrs["label_map"] = {str(i): d for i, d in enumerate(uniques)}
            eg.attrs["n_events"] = int(len(ev))
        else:
            eg.attrs["label_map"] = {}
            eg.attrs["n_events"] = 0

        # What applying the BIDS channels.tsv did to this recording's units, at
        # the root rather than buried in recording_metadata -- and written by the
        # streaming exporter too, which has no Recording to carry it.
        units_report = rec.metadata.get("channels_tsv_units")
        root.attrs.update(
            {
                "biosigio_version": _BIOSIGIO_VERSION,
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "source_format": rec.metadata.get("source_format", "n/a"),
                "modality_rates": rates,
                "dtype": dtype,
                "view_downsample": view_downsample,
                "view_chunk_columns": int(view_chunk_columns),
                "anti_alias_filter": "scipy.signal.resample_poly (polyphase FIR)",
                "channel_groups": written_groups,
                # Native JSON object (datetimes/numpy as typed envelopes), shared
                # with the tabular schema, so a browser/zarrita reader can consume
                # it directly without a second parse and without a lossy str() dump.
                "recording_metadata": metadata_to_mapping(rec.metadata),
                **({} if units_report is None else {"channels_tsv_units": units_report}),
                "created_utc": _dt.datetime.now(_dt.UTC).isoformat(),
                "note": (
                    "Derived serving copy. level 0 of each group is the anti-aliased "
                    "inference signal; view/* are min/max render envelopes (not for "
                    "inference). BIDS source remains authoritative."
                ),
            }
        )
        return filepath
