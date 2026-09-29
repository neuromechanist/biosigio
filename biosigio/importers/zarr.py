"""Importer for the biosigIO Zarr serving format (verification / training reads).

Reconstructs a :class:`~biosigio.core.emg.Recording` from a biosigIO Zarr store
written by :class:`~biosigio.exporters.zarr.ZarrExporter`: it reads ``level 0`` of a
channel group, applies the per-channel ``physical = digital * scale + offset``
dequantization, and restores channel metadata, events, and recording metadata.

This is the *serving copy*, so reconstruction is at the store's canonical
(possibly downsampled) ``level 0`` rate, not the original full-rate signal -- the
BIDS archive remains the source of truth. The view pyramid (``view/*``) is never
read here; it is render-only.

A store can hold several ``(modality, rate)`` groups at different rates, which
cannot share biosigio's single time grid; the importer reconstructs one group at a
time (auto when there is only one) and requires an explicit ``group=`` selector
otherwise, mirroring the neo importer's per-stream selection. zarr is an optional
dependency (the ``zarr`` extra), imported lazily.
"""

import warnings

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ..exceptions import is_resource_exhaustion
from ..exporters.zarr import FORMAT, FORMAT_VERSION, require_zarr
from ..tabular_schema import metadata_from_mapping
from ._labels import DEDUPLICATED_LABELS_KEY, unique_channel_labels
from .base import BaseImporter


class ZarrImporter(BaseImporter):
    """Importer for biosigIO Zarr stores (``.zarr``)."""

    def load(self, filepath: str, *, group: str | None = None) -> Recording:
        """Reconstruct a Recording from a biosigIO Zarr store.

        Args:
            filepath: Path to the ``.zarr`` store (a directory).
            group: Which ``(modality, rate)`` group to reconstruct, by name (e.g.
                ``"eeg_250hz"``). Required only when the store holds more than one
                group; otherwise the single group is used.
        """
        zarr = require_zarr()
        try:
            root = zarr.open_group(store=zarr.storage.LocalStore(filepath), mode="r")
        except Exception as e:
            # Resource exhaustion is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            raise ValueError(f"Error reading Zarr store {filepath}: {e}") from e

        root_attrs = dict(root.attrs)
        if root_attrs.get("format") != FORMAT:
            raise ValueError(
                f"Not a biosigIO Zarr store: root 'format' is {root_attrs.get('format')!r}, "
                f"expected {FORMAT!r}. Only stores written by biosigio's Zarr exporter can be read."
            )
        # Accept an integral version (int, or a float like 2.0 from a hand-edited
        # or non-Python writer); reject non-numeric or a newer store.
        version = root_attrs.get("format_version", 1)
        try:
            is_supported = int(version) == version and int(version) <= FORMAT_VERSION
        except (TypeError, ValueError):
            is_supported = False
        if not is_supported:
            raise ValueError(
                f"Unsupported biosigIO Zarr store version {version!r}; this build reads up to "
                f"version {FORMAT_VERSION}. Upgrade biosigio to read this store."
            )

        signal_groups = [name for name in root.keys() if name != "events"]
        selected = self._select_group(signal_groups, group, filepath)

        rec = Recording()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            grp = root[selected]
            arr = grp["0"]
            a0_attrs = dict(arr.attrs)
            digital = np.asarray(arr[:])  # (n_channels, n_time)
            scale = np.asarray(a0_attrs["scale"], dtype=np.float64)
            offset = np.asarray(a0_attrs["offset"], dtype=np.float64)
            rate = float(dict(grp.attrs)["rate"])
            channel_rows = dict(grp.attrs)["channels"]
            # A store published by the streaming path before repeated EDF labels
            # were suffixed can hold the same label twice; a Recording is keyed by
            # label, so suffix repeats (with a warning) rather than refuse it.
            labels, renames = unique_channel_labels(
                [str(meta["label"]) for meta in channel_rows],
                source="Zarr store",
                filepath=filepath,
            )
            for label, meta in zip(labels, channel_rows, strict=True):
                i = int(meta["row_index"])
                physical = digital[i].astype(np.float64) * scale[i] + offset[i]
                rec.add_channel(
                    label=label,
                    data=physical,
                    sample_frequency=rate,
                    physical_dimension=str(meta.get("unit", "n/a")),
                    channel_type=str(meta.get("channel_type", "OTHER")),
                    modality=meta.get("modality"),
                    prefilter=str(meta.get("prefilter", "n/a")),
                )
        if rec.signals is not None:
            rec.signals = rec.signals.copy()  # de-fragment after many inserts (#66)

        self._add_events(rec, root)

        meta_blob = root_attrs.get("recording_metadata")
        if meta_blob is not None:
            # Accepts both a native object (v2) and a legacy JSON string (v1).
            rec.metadata = metadata_from_mapping(meta_blob)
        if renames:
            # Renames made on THIS read (a store published before repeated labels
            # were suffixed) join any the store already records from its source.
            recorded = rec.metadata.get(DEDUPLICATED_LABELS_KEY) or {}
            rec.set_metadata(DEDUPLICATED_LABELS_KEY, {**recorded, **renames})
        rec.set_metadata("source_file", filepath)
        return rec

    @staticmethod
    def _select_group(signal_groups, group, filepath):
        """Pick which channel group to reconstruct (see class docstring)."""
        if not signal_groups:
            raise ValueError(f"No channel groups found in Zarr store {filepath}")
        if group is not None:
            if group not in signal_groups:
                raise ValueError(
                    f"No group named {group!r} in {filepath}; available groups: {signal_groups}"
                )
            return group
        if len(signal_groups) > 1:
            raise ValueError(
                f"{filepath} has {len(signal_groups)} channel groups at different rates that "
                f"cannot share one time grid; pass group= to choose one: {signal_groups}"
            )
        return signal_groups[0]

    @staticmethod
    def _add_events(rec: Recording, root) -> None:
        """Reconstruct events from the store's events group (code -> label map)."""
        if "events" not in root.keys():
            return
        eg = root["events"]
        attrs = dict(eg.attrs)
        if not attrs.get("n_events", 0):
            return
        onset = np.asarray(eg["onset"][:], dtype=np.float64)
        duration = np.asarray(eg["duration"][:], dtype=np.float64)
        codes = np.asarray(eg["code"][:])
        label_map = attrs.get("label_map", {})
        for o, d, c in zip(onset, duration, codes, strict=True):
            description = label_map.get(str(int(c)), str(int(c)))
            rec.add_event(onset=float(o), duration=float(d), description=description)
