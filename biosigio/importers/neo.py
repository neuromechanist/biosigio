"""python-neo-backed importer for proprietary electrophysiology acquisition formats.

Reads continuous recordings from the many acquisition systems covered by
python-neo's unified reader layer (Intan .rhd/.rhs, Blackrock .ns1-.ns6,
Spike2/CED .smr/.smrx, Plexon .plx/.pl2, Micromed .trc, Neuralynx .ncs, ...)
into a :class:`~biosigio.core.emg.Recording`.

neo models a recording as Block -> Segment -> AnalogSignal, where each
AnalogSignal is one *signal stream*: a group of channels that share a sampling
rate (e.g. an Intan amplifier bank at 30 kHz versus an auxiliary ADC). biosigio's
single time grid holds one rate, so streams that share a rate are merged into
one Recording, while streams at different rates require an explicit ``stream``
selector. A multi-rate file is never silently collapsed to a single rate.

neo carries signal values, physical units, sampling rates, channel names, and
events, but not BIDS channel *types* (a .rhd file does not distinguish SEEG from
EEG). Channels therefore default to type ``OTHER``; pass ``channel_type=`` to
label an entire recording, or rely on a sibling BIDS ``_channels.tsv`` (applied
by :meth:`Recording.from_file`) for per-channel types.

neo is an optional, heavy dependency, imported lazily via :func:`require_neo`.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ..exceptions import is_resource_exhaustion
from ._labels import DEDUPLICATED_LABELS_KEY, suffix_repeated_labels
from .base import BaseImporter

logger = logging.getLogger(__name__)


def require_neo():
    """Import python-neo lazily, raising a clear install hint when it is absent."""
    try:
        import neo  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Neo-backed import of proprietary electrophysiology formats requires "
            "python-neo, an optional dependency. Install it with: "
            "uv sync --extra neo  (or, for an existing install, "
            "uv pip install 'biosigio[neo]')."
        ) from e
    import neo

    return neo


def _physical_dimension(signal) -> str:
    """Map a neo AnalogSignal's units to an EDF-style physical-dimension string."""
    dim = str(signal.units.dimensionality)
    return dim if dim and dim != "dimensionless" else "n/a"


def _channel_names(signal, stream_index: int, n_channels: int) -> list[str]:
    """Channel names from neo array annotations, falling back to generated names.

    Some neo readers (and round-trips through formats that drop array
    annotations) leave ``channel_names`` empty or mismatched; in that case names
    are generated from the stream name so every channel is still labelled.
    """
    names = signal.array_annotations.get("channel_names")
    if names is not None and len(names) == n_channels:
        return [str(n) for n in names]
    base = signal.name or f"stream{stream_index}"
    return [f"{base}_{i}" for i in range(n_channels)]


class NeoImporter(BaseImporter):
    """Importer for proprietary electrophysiology formats via python-neo."""

    def load(
        self,
        filepath: str,
        *,
        stream: int | str | None = None,
        segment: int = 0,
        channel_type: str = "OTHER",
    ) -> Recording:
        """Load a neo-readable recording into a :class:`Recording`.

        Args:
            filepath: Path to the acquisition file (or directory, for folder-based
                formats such as Open Ephys / CTF).
            stream: Which signal stream (neo AnalogSignal) to import, by integer
                index or by stream name. Required only when the file holds streams
                at more than one sampling rate; otherwise all same-rate streams are
                merged onto one time grid.
            segment: Segment (trial) index for multi-segment recordings (default 0).
            channel_type: BIDS channel type applied to every imported channel.
                neo does not carry channel types, so this defaults to ``OTHER``;
                set it (e.g. ``"SEEG"``, ``"EEG"``, ``"EMG"``) to label the whole
                recording, or supply a sibling ``_channels.tsv``.
        """
        neo = require_neo()
        try:
            reader = neo.io.get_io(filepath)
            block = reader.read_block(lazy=False)
        except Exception as e:
            # Resource exhaustion is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            raise ValueError(f"Error reading neo file {filepath}: {e}") from e

        segments = block.segments
        if not segments:
            raise ValueError(f"No segments found in {filepath}")
        if not 0 <= segment < len(segments):
            raise ValueError(
                f"segment {segment} out of range: {filepath} has {len(segments)} segment(s)"
            )
        if len(segments) > 1:
            warnings.warn(
                f"{filepath} has {len(segments)} segments; importing segment {segment}. "
                "Pass segment= to choose another.",
                stacklevel=2,
            )
        seg = segments[segment]

        selected = self._select_streams(seg.analogsignals, stream, filepath)

        rec = Recording()
        arrays = []
        for sig in selected:
            data = np.asarray(sig.magnitude)
            arrays.append(data[:, np.newaxis] if data.ndim == 1 else data)
        # Merged streams can repeat a name. The first occurrence keeps it and later
        # ones become <name>_0, <name>_1, ..., never a name some stream really uses,
        # chosen over every stream at once so a later stream's genuine name is
        # not taken by an earlier stream's suffix.
        stream_names = [
            _channel_names(sig, idx, data.shape[1])
            for idx, (sig, data) in enumerate(zip(selected, arrays, strict=True))
        ]
        labels, renames = suffix_repeated_labels(
            [name for names in stream_names for name in names], separator="_", start=0
        )
        if renames:
            logger.warning(
                "neo channel names are not unique in %s; renamed %s",
                filepath,
                ", ".join(f"{new!r} ({old!r})" for new, old in renames.items()),
            )
            rec.set_metadata(DEDUPLICATED_LABELS_KEY, renames)
        next_label = iter(labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            for sig, data, names in zip(selected, arrays, stream_names, strict=True):
                fs = float(sig.sampling_rate.rescale("Hz").magnitude)
                dimension = _physical_dimension(sig)
                for col in range(len(names)):
                    label = next(next_label)
                    rec.add_channel(
                        label=label,
                        data=data[:, col],
                        sample_frequency=fs,
                        physical_dimension=dimension,
                        channel_type=channel_type,
                    )
        if rec.signals is not None:
            rec.signals = rec.signals.copy()  # de-fragment after many inserts (#66)

        # Align events to the imported signal's time origin (neo times are in the
        # recording's absolute time base, which may not start at zero). The 0-based
        # signal grid discards the absolute origin, so keep it in metadata for
        # downstream (e.g. BIDS) temporal alignment.
        t0 = float(selected[0].t_start.rescale("s").magnitude)
        self._add_events(rec, seg, t0)

        rec.set_metadata("source_file", filepath)
        rec.set_metadata("neo_io", type(reader).__name__)
        rec.set_metadata("t_start_s", t0)
        rec.set_metadata("number_of_signals", len(rec.channels))
        return rec

    @staticmethod
    def _select_streams(analogsignals, stream, filepath):
        """Resolve which AnalogSignal stream(s) to import (see class docstring)."""
        if not analogsignals:
            raise ValueError(
                f"No continuous analog signals found in {filepath}; events/spikes-only "
                "recordings are not supported by the neo importer."
            )

        def describe(signals):
            return "; ".join(
                f"[{i}] name={s.name!r} rate={float(s.sampling_rate.rescale('Hz').magnitude)}Hz "
                f"channels={s.shape[1] if s.ndim > 1 else 1}"
                for i, s in enumerate(signals)
            )

        if stream is not None:
            if isinstance(stream, int):
                if not 0 <= stream < len(analogsignals):
                    raise ValueError(
                        f"stream index {stream} out of range; available streams: "
                        f"{describe(analogsignals)}"
                    )
                return [analogsignals[stream]]
            matches = [s for s in analogsignals if s.name == stream]
            if not matches:
                raise ValueError(
                    f"No stream named {stream!r}; available streams: {describe(analogsignals)}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Stream name {stream!r} is ambiguous ({len(matches)} streams share it); "
                    f"select by integer index instead. Streams: {describe(analogsignals)}"
                )
            return matches

        if len(analogsignals) == 1:
            return list(analogsignals)

        rates = {float(s.sampling_rate.rescale("Hz").magnitude) for s in analogsignals}
        if len(rates) > 1:
            raise ValueError(
                f"{filepath} has {len(analogsignals)} signal streams at different sampling "
                f"rates; pass stream= (index or name) to choose one. Streams: "
                f"{describe(analogsignals)}"
            )
        lengths = {s.shape[0] for s in analogsignals}
        if len(lengths) > 1:
            raise ValueError(
                f"{filepath} has same-rate streams of differing lengths that cannot share "
                f"one time grid; pass stream= to choose one. Streams: {describe(analogsignals)}"
            )
        # Same rate and length but a different t_start would place samples at the
        # wrong relative time on the shared 0-based grid -- refuse to merge silently.
        t_starts = {round(float(s.t_start.rescale("s").magnitude), 9) for s in analogsignals}
        if len(t_starts) > 1:
            raise ValueError(
                f"{filepath} has same-rate streams with differing t_start values that cannot "
                f"share one time grid; pass stream= to choose one. Streams: "
                f"{describe(analogsignals)}"
            )
        return list(analogsignals)

    @staticmethod
    def _add_events(rec: Recording, seg, t0: float) -> None:
        """Add neo Events (instantaneous) and Epochs (with duration) as biosigio events."""

        def labels_for(obj, times):
            labels = [str(label) for label in np.asarray(obj.labels)]
            if len(labels) != len(times):  # neo objects may carry no labels
                base = obj.name or "event"
                labels = [f"{base}_{i}" for i in range(len(times))]
            return labels

        for ev in seg.events:
            times = np.asarray(ev.times.rescale("s").magnitude)
            for onset, label in zip(times, labels_for(ev, times), strict=True):
                rec.add_event(onset=float(onset) - t0, duration=0.0, description=label)
        for ep in seg.epochs:
            times = np.asarray(ep.times.rescale("s").magnitude)
            durations = np.asarray(ep.durations.rescale("s").magnitude)
            for onset, dur, label in zip(times, durations, labels_for(ep, times), strict=True):
                rec.add_event(onset=float(onset) - t0, duration=float(dur), description=label)
