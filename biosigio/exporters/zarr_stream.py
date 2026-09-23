"""Streaming Zarr export for large recordings (bounded RAM).

The in-memory path (:meth:`ZarrExporter.export` via ``Recording.from_file`` +
``to_zarr``) holds the whole recording at float64: MNE ``preload=True``, then
``raw.get_data()``, then a pandas DataFrame, then a de-fragmenting ``.copy()`` --
peaking at 2-3x the recording size. An 18 GB iEEG recording (BrainVision) needs
~150 GB that way and OOMs the converter (worse under the Hallu cron's ``--jobs``
fan-out). The #95 fix only streamed the *exporter*; the *importer* still loads
everything.

This path keeps peak RAM bounded by ~one channel, independent of recording size:

1. Open the source LAZILY with MNE (``preload=False``) -- works for the formats
   NEMAR serves (``.fif`` / ``.vhdr`` / ``.set`` / ``.edf`` / ``.bdf`` / CTF
   ``.ds`` / MEF3 ``.mefd`` / 4D-BTi). MEF3 recordings in particular can be
   multi-GB iEEG sessions, so bounded-memory streaming is not optional for them.
2. Pass 1: stream the source in time-windows (``raw.get_data(start, stop)``) into
   a channel-major float32 memmap on scratch. RAM = one time-window; disk I/O is a
   single sequential pass (a per-channel read of a multiplexed source would instead
   scan the file once per channel).
3. Pass 2: for each channel, read its full row from the memmap, apply the BIDS
   unit conversion, resample + quantize it, and write it (base row + its slice of
   each min/max pyramid level) into output memmaps on scratch. RAM = one channel.
4. Pass 3: copy the output memmaps into the Zarr arrays in shard- and
   chunk-aligned blocks, so each shard and chunk is written once and never read
   back. Writing rows straight into the store instead makes every row a partial
   write of every shard, which zarr serves by re-encoding the whole shard: n_ch
   rewrites per shard and gigabytes in flight (issue #129). RAM = one block.

Between passes 1 and 2 there is nothing to rescale in memory, so the BIDS
``_channels.tsv`` is applied to the *channel table* instead (``bids_channels``,
default ``"auto"``, mirroring ``Recording.from_file``): types and modalities move
immediately, and each declared unit becomes a per-channel factor pass 2 folds in
exactly once. Without that, a dataset whose recordings straddle a size threshold
served its small runs in the sidecar's unit and its large ones in the importer's
native unit -- self-consistent stores that disagree with each other by 10^6
(issue #127).

The resulting store is structurally identical to :meth:`ZarrExporter.export` and
numerically equal within int16 quantization: same layout, same attrs, same chunk
geometry -- ``level 0`` chunked and sharded on time (``chunk_seconds`` /
``shard_seconds``), every ``view/*`` level chunked on a constant column count
(``view_chunk_columns``, shared with the in-memory path through
:func:`~biosigio.exporters.zarr._view_chunk_columns` so the two cannot drift), and
each channel group declaring its pyramid via ``n_view_levels`` / ``view_levels``.
See :mod:`biosigio.exporters.zarr` for the layout block and the rationale. The
downsampling parameters and provenance are recorded in attrs, same as the
in-memory path. Requires both the ``zarr`` and ``meg`` (MNE) extras.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import tempfile
from typing import TYPE_CHECKING, cast

import numpy as np

from ..bids import apply_channels_tsv_to_stream, resolve_channels_tsv
from ..core.modality import infer_modality_from_channel_type
from ..importers._mne_common import _FIFF_UNIT_TO_DIM, _MNE_TYPE_TO_biosigIO, require_mne
from ..tabular_schema import metadata_to_mapping
from ..version import __version__ as _BIOSIGIO_VERSION
from .zarr import (
    _DISCRETE_TYPES,
    DEFAULT_MODALITY_RATES,
    FORMAT,
    FORMAT_VERSION,
    _build_minmax_pyramid,
    _chunk_shard_time,
    _quantize_int16_channel,
    _resample_channel,
    _target_rate,
    _view_chunk_columns,
    require_zarr,
)

if TYPE_CHECKING:
    # Only for the `bids_channels` annotation: a DataFrame is a valid sidecar,
    # but nothing here touches pandas at runtime (biosigio.bids does).
    import pandas as pd


def _scratch_array(path: str, dtype, shape: tuple[int, ...]) -> np.ndarray:
    """A file-backed scratch array for pass 2's output (see ``stream_to_zarr``).

    A memmap keeps the group's quantized output out of the process's anonymous
    memory: page cache, not heap, and reclaimable under pressure. numpy cannot
    map a zero-length file, so an empty shape gets a plain (empty) array.
    """
    if 0 in shape:
        return np.zeros(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def _pyramid_level_lengths(
    n_time: int, factor: int, min_samples: int, max_levels: int
) -> list[int]:
    """Output length of each min/max pyramid level for a base of ``n_time`` samples.

    Mirrors the level-stopping rule in :func:`_build_minmax_pyramid` so the view
    Zarr arrays can be created up front (before any channel is written) and then
    filled one channel-row at a time.

    **The stopping rule, stated once for the whole module.** A level is built, and
    then the loop stops once the level just built is at or below ``min_samples``.
    So ``min_samples`` bounds the level a further one would have been built *from*,
    not the shortest level in the store: the final level can itself be shorter than
    the floor. At the defaults (factor 4, floor 512), ``n_time=30000`` gives
    ``[7500, 1875, 468]`` -- the 468 is written because 1875 was still above the
    floor, and nothing follows it because 468 is not. ``max_levels`` caps the depth
    independently.
    """
    lengths: list[int] = []
    n = n_time
    for _ in range(max_levels):
        if n <= min_samples:
            break
        n_out = n // factor
        if n_out < 1:
            break
        lengths.append(n_out)
        n = n_out
    return lengths


# --- Lazy stream sources (#944) -----------------------------------------------
# stream_to_zarr needs, per recording: sfreq, n_samples, and per-channel
# (label / biosigIO type / unit) metadata, plus windowed float64 blocks. MNE
# (preload=False) covers .fif/.vhdr/.set/CTF/.mefd/4D-BTi. EDF/BDF go through
# pyedflib INSTEAD -- biosigIO's in-memory path reads EDF via pyedflib, and MNE
# disagrees on EDF unit scaling (an EDF whose physical dimension is unknown: MNE
# assumes Volts, pyedflib keeps the file's dimension), so streaming EDF via MNE
# would NOT match a re-run on the in-memory path. pyedflib.readSignal(chn, start, n)
# is numerically identical to readSignal(chn) sliced, so the streamed store matches
# the in-memory store exactly. (EDF-via-stream is not used in production today, so
# switching its reader changes no existing store.) When pyedflib itself refuses to
# open the file (issue #109 and friends -- see ``..importers._edf_tolerant``),
# ``_EdfSource`` falls back to the SAME tolerant MNE-backed, unit-rescaled read the
# in-memory importer uses, so the two paths keep agreeing even on a recovered file.
#
# .mefd and 4D/BTi need their OWN reader call rather than MNE's generic
# ``read_raw()`` dispatch: .mefd additionally needs the MEF3 version/pymef gate
# (see ``..importers.mef3.require_mne_mef``), and a BTi directory has no
# extension for ``read_raw()`` to dispatch on at all (see
# ``..importers.meg._resolve_bti_reader_kwargs``, the same content-based
# detection AND config/hs_file sidecar resolution the in-memory importer uses --
# shared so the BTi precedence rule lives in exactly one place). Both still end
# up as a plain MNE ``Raw`` with ``preload=False``, so ``_MneSource`` accepts an
# already-opened ``raw`` (and an optional ``extra_metadata`` dict, e.g. which
# BTi processed-data file was actually read) and skips opening it again -- the
# windowed-read/channel-metadata code below is identical either way.


class _MneSource:
    """MNE-backed lazy source for the formats MNE reads faithfully
    (.fif/.vhdr/.set/CTF/.mefd/4D-BTi)."""

    def __init__(
        self,
        filepath: str,
        force_modality: str | None,
        raw=None,
        extra_metadata: dict | None = None,
    ):
        if raw is None:
            mne = require_mne()
            raw = mne.io.read_raw(filepath, preload=False, verbose="ERROR")
        self._raw = raw
        self._closed = False
        self.extra_metadata = dict(extra_metadata or {})
        self.sfreq = float(self._raw.info["sfreq"])
        self.n_samples = int(self._raw.n_times)
        types = self._raw.get_channel_types()
        self.channels: list[dict] = []
        for i, name in enumerate(self._raw.ch_names):
            ctype = _MNE_TYPE_TO_biosigIO.get(types[i], "OTHER")
            self.channels.append(
                {
                    "idx": i,
                    "label": name,
                    "channel_type": ctype,
                    "modality": force_modality or infer_modality_from_channel_type(ctype),
                    "unit": _FIFF_UNIT_TO_DIM.get(int(self._raw.info["chs"][i]["unit"]), "n/a"),
                }
            )

    def read(self, picks: list[int], start: int, stop: int) -> np.ndarray:
        return self._raw.get_data(picks=picks, start=start, stop=stop)

    def close(self) -> None:
        # MNE's lazy Raw owns no handle to release, but the flag is still set:
        # `closed` is the interface both sources share, and a caller (or a test)
        # asking whether a source was released must get the same answer whichever
        # one it holds.
        self._closed = True

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed


class _EdfSource:
    """pyedflib-backed lazy source for .edf/.bdf, matching biosigIO's in-memory
    EDF importer (physical-dimension units, label/transducer channel typing).

    Uniform per-channel rate ONLY: a mixed-rate EDF raises MixedSamplingRateError
    (same as the importer's default), so it stays on the in-memory resample path
    rather than being silently mis-gridded by a single-rate streaming window.

    Three conditions pyedflib's compliance checker rejects outright (a
    ``physical_min == physical_max`` channel, a NUL-padded numeric header field,
    a correctly-marked-discontinuous EDF+D) are recovered the same way the
    in-memory importer recovers them -- see ``importers._edf_tolerant`` for the
    unit-parity rationale. That path preloads the whole recording (there is no
    windowed pyedflib reader for a file pyedflib itself refuses to open), so it
    trades this module's bounded-memory guarantee for correctness on these rare,
    already-exceptional files; ``read()`` still serves windows from the
    preloaded array so callers see the same interface either way."""

    def __init__(self, filepath: str, force_modality: str | None):
        import pyedflib

        from ..exceptions import MixedSamplingRateError
        from ..importers._edf_tolerant import classify_pyedflib_error, read_edf_tolerant
        from ..importers.edf import EDFImporter

        self.extra_metadata: dict = {}
        self._reader = None
        self._closed = False
        self._fallback_data: np.ndarray | None = None
        typer = EDFImporter()._determine_channel_type

        try:
            reader = pyedflib.EdfReader(filepath)
        except Exception as open_exc:
            reason = classify_pyedflib_error(open_exc)
            if reason is None:
                raise
            # ImportError (MNE/`meg` extra missing) is deliberately NOT caught
            # here: it propagates with its own clear install hint, same as the
            # in-memory importer degrading only when it also can't recover.
            fallback = read_edf_tolerant(filepath, reason)
            self.extra_metadata["edf_tolerant_read"] = True
            self.extra_metadata["edf_tolerant_read_reason"] = reason

            rates = {ch.sample_frequency for ch in fallback.channels}
            if len(rates) > 1:
                raise MixedSamplingRateError(
                    "EDF/BDF recording has mixed per-channel sampling rates "
                    f"({sorted(rates)} Hz); the streaming path needs a uniform "
                    'grid. The in-memory path handles this with mixed_rate="resample".'
                ) from None

            self.sfreq = float(fallback.channels[0].sample_frequency) if fallback.channels else 0.0
            self.n_samples = len(fallback.channels[0].data) if fallback.channels else 0
            self._fallback_data = (
                np.stack([ch.data for ch in fallback.channels])
                if fallback.channels
                else np.empty((0, 0))
            )
            self.channels = []
            for i, ch in enumerate(fallback.channels):
                ctype = typer(ch.label, ch.transducer)
                self.channels.append(
                    {
                        "idx": i,
                        "label": ch.label,
                        "channel_type": ctype,
                        "modality": force_modality or infer_modality_from_channel_type(ctype),
                        "unit": ch.physical_dimension or "n/a",
                    }
                )
            return

        self._reader = reader
        headers = self._reader.getSignalHeaders()
        nsamps = self._reader.getNSamples()
        rates = {float(h["sample_frequency"]) for h in headers}
        if len(rates) > 1:
            self._reader.close()
            raise MixedSamplingRateError(
                "EDF/BDF recording has mixed per-channel sampling rates "
                f"({sorted(rates)} Hz); the streaming path needs a uniform grid. The "
                'in-memory path handles this with mixed_rate="resample".'
            )
        self.sfreq = float(headers[0]["sample_frequency"]) if headers else 0.0
        self.n_samples = int(nsamps[0]) if len(nsamps) else 0
        self.channels = []
        for i, h in enumerate(headers):
            label = cast(str, h["label"]).strip()
            transducer = cast(str, h.get("transducer", "")).strip()
            ctype = typer(label, transducer)
            self.channels.append(
                {
                    "idx": i,
                    "label": label,
                    "channel_type": ctype,
                    "modality": force_modality or infer_modality_from_channel_type(ctype),
                    "unit": cast(str, h.get("dimension", "")).strip() or "n/a",
                }
            )

    def read(self, picks: list[int], start: int, stop: int) -> np.ndarray:
        if self._fallback_data is not None:
            return self._fallback_data[picks, start:stop].astype(np.float64)
        assert self._reader is not None  # only unset when _fallback_data is set instead
        n = stop - start
        return np.stack([self._reader.readSignal(i, start, n) for i in picks]).astype(np.float64)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._closed = True

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run (and with it pyedflib's file handle)."""
        return self._closed


def _open_stream_source(filepath: str, force_modality: str | None):
    """Open a lazy source, reading EDF/BDF via pyedflib (importer parity) and
    everything else via MNE. #944

    ``.mefd`` and 4D/BTi each need a specific MNE reader call (see the module
    docstring above) rather than MNE's generic ``read_raw()`` dispatch, so they are
    opened here and handed to ``_MneSource`` as an already-built ``raw``.
    """
    stripped = filepath.rstrip("/\\")
    ext = os.path.splitext(stripped)[1].lower()
    if ext in (".edf", ".bdf"):
        return _EdfSource(filepath, force_modality)
    if ext == ".mefd":
        from ..importers.mef3 import require_mne_mef

        mne = require_mne_mef()
        raw = mne.io.read_raw_mef(filepath, preload=False, verbose="ERROR")
        return _MneSource(filepath, force_modality, raw=raw)
    if ext == "" and os.path.isdir(stripped):
        # Raises UnsupportedFormatError with a clear message if this doesn't
        # look like a BTi directory (same check the in-memory importer uses),
        # rather than silently falling through to _MneSource's generic
        # read_raw(), which cannot dispatch an extension-less directory anyway
        # and would only produce a less specific error.
        from ..importers.meg import _resolve_bti_reader_kwargs

        kwargs = _resolve_bti_reader_kwargs(stripped)
        mne = require_mne()
        raw = mne.io.read_raw_bti(preload=False, verbose="ERROR", **kwargs)
        return _MneSource(
            stripped,
            force_modality,
            raw=raw,
            extra_metadata={"bti_pdf_file": kwargs["pdf_fname"]},
        )
    return _MneSource(filepath, force_modality)


def stream_to_zarr(
    filepath: str,
    store_path: str,
    *,
    force_modality: str | None = None,
    modality_rates: dict[str, int] | None = None,
    dtype: str = "int16",
    events_df=None,
    bids_channels: str | os.PathLike | pd.DataFrame | None = "auto",
    recording_metadata: dict | None = None,
    view_downsample: int = 4,
    min_view_samples: int = 512,
    max_view_levels: int = 12,
    chunk_seconds: float = 4.0,
    shard_seconds: float = 300.0,
    view_chunk_columns: int = 1024,
    compressor_level: int = 5,
    read_chunk_seconds: float = 30.0,
    scratch_dir: str | None = None,
) -> str:
    """Stream ``filepath`` to a Zarr serving store with bounded peak memory.

    Args:
        filepath: Source recording MNE can open (``.fif``/``.vhdr``/``.set``/
            ``.edf``/``.bdf``/CTF ``.ds``/MEF3 ``.mefd``/a 4D-BTi directory).
        store_path: Output store path (``.zarr`` appended if missing).
        force_modality: If set (e.g. ``"IEEG"`` from a BIDS suffix), assign every
            channel this modality so the recording lands in one coherent group at
            the modality's rate cap -- matching the NEMAR driver's suffix-driven
            grouping. If None, modality is inferred per channel from its type.
            **Precedence with the sidecar:** ``force_modality`` always wins for
            *grouping*, and a ``channels.tsv`` never moves a channel out of the
            forced modality; the sidecar's ``type`` still sets each channel's
            ``channel_type`` (and, when ``force_modality`` is None, the modality
            derived from it).
        modality_rates: Per-modality rate cap (defaults to
            :data:`~biosigio.exporters.zarr.DEFAULT_MODALITY_RATES`).
        dtype: ``"int16"`` (scaled) or ``"float32"`` (lossless).
        events_df: Optional events table (onset/duration/description).
        bids_channels: The BIDS ``_channels.tsv`` to apply over the importer's
            per-channel guesses, mirroring
            :meth:`~biosigio.core.emg.Recording.from_file`'s parameter of the same
            name so both export paths read the same sidecar by default. ``"auto"``
            (default) looks for the sibling sidecar next to ``filepath``; ``"off"``
            (or None) disables it; a path or a DataFrame is used as given. Adopting
            a declared unit **converts the samples** into it, so a store built here
            and one built via ``Recording.from_file(...).to_zarr(...)`` agree on
            units, types and values (issue #127). See
            :func:`~biosigio.bids.apply_channels_tsv_to_stream`.
        recording_metadata: Optional metadata dict stored in the store root attrs.
        min_view_samples: Pyramid floor. For the exact stopping rule -- and why the
            last level written can be shorter than this -- see
            :func:`_pyramid_level_lengths`.
        view_chunk_columns: Columns per chunk of every ``view/*`` level (capped by
            the level's length). Must be >= 1; see
            :meth:`~biosigio.exporters.zarr.ZarrExporter.export`.
        read_chunk_seconds: Time-window size for the streaming transpose pass.
        scratch_dir: Directory for the temporary memmaps (defaults to the system
            temp dir): the float32 channel-major transpose, then pass 2's output
            (about 1.7x the output dtype's size per sample, base plus pyramid),
            which replaces it before pass 3. Point this at fast local scratch.

    Returns:
        The store path written.
    """
    if dtype not in ("int16", "float32"):
        raise ValueError("dtype must be 'int16' or 'float32'")
    if int(view_chunk_columns) < 1:
        raise ValueError("view_chunk_columns must be >= 1")
    zarr = require_zarr()
    from zarr.codecs import BloscCodec

    rates = dict(DEFAULT_MODALITY_RATES if modality_rates is None else modality_rates)
    if not store_path.endswith(".zarr"):
        store_path = store_path + ".zarr"
    out_np = np.int16 if dtype == "int16" else np.float32

    # EDF/BDF read via pyedflib (importer parity), everything else lazily via MNE.
    src = _open_stream_source(filepath, force_modality)
    sfreq = src.sfreq
    n_samples = src.n_samples
    if n_samples == 0 or len(src.channels) == 0:
        src.close()
        raise ValueError("No signals loaded")

    # The BIDS sidecar is authoritative for per-channel type and units, exactly as
    # it is on the in-memory path (Recording.from_file's bids_channels). Applied
    # BEFORE the grouping below, because adopting a type can change a channel's
    # modality and therefore which group it belongs to. Each channel's declared
    # unit becomes a `unit_factor` that pass 2 multiplies in once.
    #
    # A sidecar is caller-supplied input and can be absent, a directory, or
    # unparsable, so this is the one step between opening the source and the
    # `with` block below that can raise. The source holds an OS file handle
    # (pyedflib) or an MNE reader, so it is closed on the way out rather than
    # left to the garbage collector.
    try:
        channels_tsv = resolve_channels_tsv(filepath, bids_channels)
        units_report = (
            None
            if channels_tsv is None
            else apply_channels_tsv_to_stream(
                src.channels, channels_tsv, force_modality=force_modality
            )
        )
    except BaseException:
        src.close()
        raise

    # Per-channel metadata; group by (modality, native rate). The source is
    # single-rate (a mixed-rate EDF is rejected in _EdfSource), so every channel
    # shares `sfreq`; force_modality collapses to one group.
    groups: dict[tuple[str, int], list[dict]] = {}
    for info in src.channels:
        groups.setdefault((str(info["modality"]), int(round(sfreq))), []).append(info)

    store = zarr.storage.LocalStore(store_path)
    root = zarr.create_group(store=store, overwrite=True)
    compressors = [BloscCodec(cname="zstd", clevel=compressor_level)]
    written_groups: list[str] = []

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        for (modality, native_rate), members in groups.items():
            target_rate = _target_rate(native_rate, modality, rates)
            n_ch = len(members)
            picks = [ci["idx"] for ci in members]

            # Pass 1: stream the source into a channel-major float32 memmap on scratch.
            # RAM here is one read window (n_ch x read_chunk), not the whole signal.
            mm_path = os.path.join(tmp, f"{modality}_{native_rate}.f32")
            mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n_ch, n_samples))
            read_chunk = max(1, int(round(read_chunk_seconds * native_rate)))
            for s in range(0, n_samples, read_chunk):
                e = min(n_samples, s + read_chunk)
                block = src.read(picks, s, e)  # (n_ch, e-s) float64
                mm[:, s:e] = block.astype(np.float32)
            mm.flush()

            n_time = max(0, int(round(n_samples * target_rate / native_rate)))
            gname = f"{modality.lower()}_{int(round(target_rate))}hz"
            if gname in written_groups:  # single native rate per modality -> shouldn't collide
                raise ValueError(f"Zarr group name collision: {gname!r}")

            grp = root.create_group(gname)
            chunk_t, shard_t = _chunk_shard_time(n_time, target_rate, chunk_seconds, shard_seconds)
            a0 = grp.create_array(
                "0",
                shape=(n_ch, n_time),
                chunks=(n_ch, chunk_t),
                shards=(n_ch, shard_t),
                dtype=out_np,
                compressors=compressors,
            )
            view = grp.create_group("view")
            level_lengths = _pyramid_level_lengths(
                n_time, view_downsample, min_view_samples, max_view_levels
            )
            view_arrays = []
            for li, length in enumerate(level_lengths, start=1):
                eff = view_downsample**li
                # Constant column count, not a constant time span: same rule as the
                # in-memory exporter (see _view_chunk_columns). Pass 3 writes each
                # chunk exactly once, so the width trades request count against
                # chunk size and never costs memory beyond one chunk.
                ct = _view_chunk_columns(length, view_chunk_columns)
                av = view.create_array(
                    str(li),
                    shape=(2, n_ch, length),
                    chunks=(2, n_ch, ct),
                    dtype=out_np,
                    compressors=compressors,
                )
                view_arrays.append((li, eff, ct, av))

            # Pass 2: one channel at a time -- resample, quantize, build its pyramid.
            # Peak RAM is a single channel. The results go to scratch memmaps, NOT
            # straight into the store: a row is a partial write of every level-0
            # shard (n_ch x shard_t) and every view chunk (2 x n_ch x ct), and zarr
            # serves a partial write by reading, decoding, re-encoding and
            # rewriting the whole shard. Row by row, that re-compressed level 0
            # n_ch times over and held up to async.concurrency shards in memory
            # per row (issue #129: a 461 MB, 128-channel EDF peaked at +5.1 GB
            # RSS). Pass 3 below copies the memmaps into the store block by
            # block, so each shard and chunk is written once and never read.
            q_mm = _scratch_array(
                os.path.join(tmp, f"{modality}_{native_rate}.q"), out_np, (n_ch, n_time)
            )
            view_mms = [
                _scratch_array(
                    os.path.join(tmp, f"{modality}_{native_rate}.v{li}"),
                    out_np,
                    (2, n_ch, av.shape[2]),
                )
                for li, _eff, _ct, av in view_arrays
            ]
            scale = np.ones(n_ch, dtype=np.float64)
            offset = np.zeros(n_ch, dtype=np.float64)
            chan_meta = []
            for i, ci in enumerate(members):
                ctype = str(ci["channel_type"]).upper()
                discrete = ctype in _DISCRETE_TYPES
                # The channels.tsv conversion, applied here and only here: in
                # float64, on the whole channel, before the resample -- the same
                # order the in-memory path uses (rescale the column, then export),
                # so the two stores' values agree to within int16 quantization.
                factor = float(ci.get("unit_factor", 1.0))
                # np.array, not np.asarray: the in-place multiply below must land
                # on this pass's own copy, never back through a view into the
                # scratch memmap. (The float32 -> float64 change already forces a
                # copy today; stating it means a future dtype change cannot
                # silently turn the conversion into a write to scratch.)
                x = np.array(mm[i], dtype=np.float64)
                if factor != 1.0:
                    x *= factor
                y = _resample_channel(x, native_rate, target_rate, discrete=discrete)
                n_nonfinite = 0
                if dtype == "int16":
                    q, scale[i], offset[i], n_nonfinite = _quantize_int16_channel(y)
                else:
                    q = y.astype(np.float32)
                q_mm[i, :] = q
                if view_arrays:
                    pyramid = _build_minmax_pyramid(
                        q[np.newaxis, :], view_downsample, min_view_samples, max_view_levels
                    )
                    for vmm, lvl in zip(view_mms, pyramid, strict=False):
                        vmm[:, i, :] = lvl[:, 0, :]
                meta = {
                    "label": ci["label"],
                    "channel_type": ctype,
                    "modality": modality,
                    "unit": ci["unit"],
                    "prefilter": "n/a",
                    "original_rate": float(native_rate),
                    "target_rate": float(target_rate),
                    "anti_aliased": bool((not discrete) and target_rate < native_rate),
                    # A zero-filled NaN/inf channel is lossy: viewable, never inferable.
                    "usable_for_inference": (not discrete) and n_nonfinite == 0,
                    "scale": float(scale[i]),
                    "offset": float(offset[i]),
                    "row_index": i,
                }
                if n_nonfinite:
                    meta["nonfinite_samples"] = n_nonfinite
                # A unit the sidecar declared but the values contradict is kept as
                # metadata rather than asserted (see biosigio.bids._decide_unit);
                # the in-memory exporter carries it the same way.
                if ci.get("bids_unit"):
                    meta["bids_unit"] = ci["bids_unit"]
                chan_meta.append(meta)
                del x, y, q
            # The float32 transpose is spent; drop it (and its scratch file) before
            # pass 3 so the group's peak scratch is the output, not input + output.
            del mm
            with contextlib.suppress(OSError):
                os.remove(mm_path)

            # Pass 3: whole-shard and whole-chunk writes. Every block below covers
            # all channels of its column range, so zarr writes it once without
            # reading anything back. RAM is one block (for level 0,
            # n_ch x shard_t output samples), independent of n_ch x duration.
            for s in range(0, n_time, shard_t):
                e = min(n_time, s + shard_t)
                a0[:, s:e] = q_mm[:, s:e]
            for (_li, _eff, ct, av), vmm in zip(view_arrays, view_mms, strict=True):
                length = av.shape[2]
                for s in range(0, length, ct):
                    e = min(length, s + ct)
                    av[:, :, s:e] = vmm[:, :, s:e]
            del q_mm, view_mms

            grp.attrs.update(
                {
                    "modality": modality,
                    "rate": float(target_rate),
                    "original_rate": float(native_rate),
                    "n_channels": int(n_ch),
                    "n_samples": int(n_time),
                    "channels": chan_meta,
                    "n_view_levels": len(level_lengths),
                    "view_levels": list(range(1, len(level_lengths) + 1)),
                    "view_downsample": int(view_downsample),
                    "view_chunk_columns": int(view_chunk_columns),
                    "chunk_seconds": float(chunk_seconds),
                    "shard_seconds": float(shard_seconds),
                }
            )
            a0.attrs.update(
                {
                    "level": 0,
                    "rate": float(target_rate),
                    "source_rate_hz": float(native_rate),
                    "downsample_factor": 1,
                    "kind": "signal",
                    "chunk_samples": int(chunk_t),
                    "shard_samples": int(shard_t),
                    "anti_aliased": any(m["anti_aliased"] for m in chan_meta),
                    "usable_for_inference": any(m["usable_for_inference"] for m in chan_meta),
                    "scale": scale.tolist(),
                    "offset": offset.tolist(),
                    "physical_formula": "physical = digital * scale + offset",
                }
            )
            for li, eff, ct, av in view_arrays:
                av.attrs.update(
                    {
                        "level": li,
                        "downsample_factor": int(eff),
                        "rate_effective": float(target_rate) / eff,
                        "kind": "minmax_envelope",
                        "axis0": ["min", "max"],
                        "chunk_columns": int(ct),
                        "usable_for_inference": False,
                    }
                )
            written_groups.append(gname)

        # Events as onset/duration/code arrays + a portable code->label map.
        eg = root.create_group("events")
        ev = events_df
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

        meta = dict(recording_metadata or {})
        for key, value in src.extra_metadata.items():
            meta.setdefault(key, value)
        meta.setdefault("source_file", filepath)
        meta.setdefault("number_of_signals", len(src.channels))
        meta.setdefault("streamed", True)
        if units_report is not None:
            # Where the in-memory path finds it: apply_channels_tsv leaves this in
            # rec.metadata, which ZarrExporter copies into recording_metadata.
            meta["channels_tsv_units"] = units_report
        root.attrs.update(
            {
                "biosigio_version": _BIOSIGIO_VERSION,
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "source_format": meta.get("source_format", "n/a"),
                "modality_rates": rates,
                "dtype": dtype,
                "view_downsample": view_downsample,
                "view_chunk_columns": int(view_chunk_columns),
                "anti_alias_filter": "scipy.signal.resample_poly (polyphase FIR)",
                "channel_groups": written_groups,
                "recording_metadata": metadata_to_mapping(meta),
                **({} if units_report is None else {"channels_tsv_units": units_report}),
                "created_utc": _dt.datetime.now(_dt.UTC).isoformat(),
                "note": (
                    "Derived serving copy (streamed, bounded-memory conversion). level 0 "
                    "of each group is the anti-aliased inference signal; view/* are min/max "
                    "render envelopes (not for inference). BIDS source remains authoritative."
                ),
            }
        )
    src.close()
    return store_path
