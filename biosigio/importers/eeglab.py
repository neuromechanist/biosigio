"""Importer for EEGLAB ``.set`` files.

EEGLAB saves two on-disk forms under the same ``.set`` extension:

* Classic MATLAB v5/v7 (a zlib-compressed MAT container) -- read via
  ``scipy.io.loadmat`` in :meth:`EEGLABImporter._load`, unchanged by the v7.3
  support added here.
* MATLAB v7.3 (an HDF5 container, which MATLAB/EEGLAB switches to automatically
  once a variable exceeds ~2 GB, or on an explicit ``-v7.3`` save) --
  ``scipy.io.loadmat`` (and MNE's EEGLAB reader) refuse these outright
  ("Please use HDF reader for matlab v7.3 files, e.g. h5py"), so they are read
  via h5py in :meth:`EEGLABImporter._load_v73` instead. :func:`_is_matlab_v73`
  tells the two forms apart by sniffing the leading header text, since the
  ``.set`` extension alone cannot.

h5py is an optional dependency (the ``hdf5`` extra), imported lazily via
:func:`require_h5py`; classic ``.set`` files need no extra at all.
"""

import logging
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..core.emg import Recording
from ..core.modality import infer_modality_from_channel_type
from ..exceptions import (
    FileReadError,
    NotContinuousRecordingError,
    classify_read_error,
    is_resource_exhaustion,
)
from .base import BaseImporter

# MATLAB .mat files (any version) open with a 128-byte descriptive text header.
# A v7.3 (HDF5) file's header literally starts with this text; classic v5/v7
# files start with "MATLAB 5.0 MAT-file" instead. Sniffing just this prefix is
# format-declared and cheap -- unlike trying an h5py open as control flow, it
# never mistakes a truncated/corrupt classic file for a v7.3 one.
_MATLAB_V73_MAGIC = b"MATLAB 7.3 MAT-file"
_MAGIC_SNIFF_BYTES = 128
# Sampling rate assumed when a .set header carries none (absent, empty or zero).
_DEFAULT_SRATE = 1000.0
# Recording-metadata key recording that the signal was read from a ``.fdt`` other
# than the one ``EEG.data`` names (the BIDS-renamed same-stem sibling).
FDT_RECOVERED_KEY = "eeglab_fdt_recovered"

logger = logging.getLogger(__name__)


def require_h5py():
    """Import h5py lazily, raising a clear install hint when it is absent."""
    try:
        import h5py  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Reading a MATLAB v7.3 (HDF5) EEGLAB .set file requires h5py, an "
            "optional dependency. Install it with: uv sync --extra hdf5  (or, for "
            "an existing install, uv pip install 'biosigio[hdf5]')."
        ) from e
    import h5py

    return h5py


def _is_matlab_v73(filepath: str) -> bool:
    """Sniff whether ``filepath`` is a MATLAB v7.3 (HDF5) ``.set`` file.

    Both v5/v7 and v7.3 EEGLAB exports use the ``.set`` extension, so the
    extension cannot distinguish them; the leading header text can (see
    ``_MATLAB_V73_MAGIC``). Any I/O failure (missing file, permissions) is
    treated as "not v7.3" here -- the caller's own open of the file surfaces
    the real error with a clearer message.
    """
    try:
        with open(filepath, "rb") as fh:
            header = fh.read(_MAGIC_SNIFF_BYTES)
    except OSError:
        return False
    return header.startswith(_MATLAB_V73_MAGIC)


class EEGLABImporter(BaseImporter):
    """Importer for EEGLAB .set files containing EMG data."""

    def _extract_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract metadata from EEGLAB .set file.

        Args:
            data: Dictionary containing EEGLAB .set file data

        Returns:
            dict: Dictionary containing extracted metadata
        """
        metadata = {}

        # Extract basic recording information
        if "setname" in data:
            metadata["setname"] = str(data["setname"][0]) if data["setname"].size > 0 else ""

        if "filename" in data:
            metadata["filename"] = str(data["filename"][0]) if data["filename"].size > 0 else ""

        if "filepath" in data:
            metadata["filepath"] = str(data["filepath"][0]) if data["filepath"].size > 0 else ""

        # Extract subject information
        if "subject" in data:
            metadata["subject"] = str(data["subject"][0]) if data["subject"].size > 0 else ""

        if "group" in data:
            metadata["group"] = str(data["group"][0]) if data["group"].size > 0 else ""

        if "condition" in data:
            metadata["condition"] = str(data["condition"][0]) if data["condition"].size > 0 else ""

        if "session" in data:
            metadata["session"] = str(data["session"][0]) if data["session"].size > 0 else ""

        if "comments" in data:
            metadata["comments"] = str(data["comments"][0]) if data["comments"].size > 0 else ""

        # Extract recording parameters
        if "srate" in data:
            metadata["srate"] = float(data["srate"][0][0]) if data["srate"].size > 0 else 0

        if "nbchan" in data:
            metadata["nbchan"] = int(data["nbchan"][0][0]) if data["nbchan"].size > 0 else 0

        if "trials" in data:
            metadata["trials"] = int(data["trials"][0][0]) if data["trials"].size > 0 else 0

        if "pnts" in data:
            metadata["pnts"] = int(data["pnts"][0][0]) if data["pnts"].size > 0 else 0

        if "xmin" in data and data["xmin"].size > 0:
            metadata["xmin"] = float(data["xmin"][0][0])

        if "xmax" in data and data["xmax"].size > 0:
            metadata["xmax"] = float(data["xmax"][0][0])

        # Add device information
        metadata["device"] = "EEGLAB"

        return metadata

    def _determine_channel_type(self, channel_info: dict[str, Any]) -> str:
        """
        Determine channel type based on channel information.

        Args:
            channel_info: Dictionary containing channel information

        Returns:
            str: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        # Check if type is explicitly specified. After _process_channel_info,
        # 'type' and 'label' are plain strings (scalar-flattened), so consume them
        # as strings rather than indexing/.size like the old array form.
        ch_type = channel_info.get("type")
        if ch_type:
            ch_type_upper = ch_type.upper()
            # Neural/physiological BIDS types pass straight through.
            if ch_type_upper in ("EEG", "EMG", "ECG", "EKG", "EOG", "SEEG", "ECOG"):
                return ch_type_upper
            elif ch_type_upper in ["ACC", "ACCELEROMETER"]:
                return "ACC"
            elif ch_type_upper in ["GYRO", "GYROSCOPE"]:
                return "GYRO"
            elif ch_type_upper in ["TRIG", "TRIGGER"]:
                return "TRIG"

        # If type is not specified or not recognized, try to determine from label.
        label = channel_info.get("label", "")
        if label:
            label_upper = label.upper()
            if "EMG" in label_upper:
                return "EMG"
            elif "ECG" in label_upper or "EKG" in label_upper:
                return "ECG"
            elif "EOG" in label_upper:
                return "EOG"
            elif "ACC" in label_upper:
                return "ACC"
            elif "GYRO" in label_upper:
                return "GYRO"
            elif "TRIG" in label_upper:
                return "TRIG"

        # No reliable type info: do NOT assume EMG (avoids modality creep onto
        # EEG/other data). Caller-supplied BIDS channels.tsv types can refine this.
        return "OTHER"

    def _process_channel_info(self, chanlocs: np.ndarray) -> list[dict[str, Any]]:
        """
        Process channel location information.

        Args:
            chanlocs: Array containing channel location information

        Returns:
            list: List of dictionaries containing channel information
        """
        channel_info_list = []

        field_names = chanlocs.dtype.names
        if field_names is None:
            return channel_info_list

        # EEGLAB saves chanlocs as a struct array whose MATLAB shape survives
        # loadmat: (1, N) row form OR (N, 1) column form (real exports use both).
        # Indexing chanlocs[0] assumed the row form and saw exactly ONE channel
        # on column-form files, silently truncating every later channel; ravel
        # makes the walk shape-agnostic.
        for element in np.asarray(chanlocs).ravel():
            channel_info = {}

            # Extract channel fields
            for field in field_names:
                # Get the field value for this channel
                field_value = np.asarray(element[field])
                if field_value.size == 0:
                    continue
                # scipy.io.loadmat returns scalar struct fields as nested
                # (1, 1)-shaped arrays, so field_value[0] is still 1-D and
                # float()/str() on it fails or mis-formats. Flatten to a scalar.
                scalar = field_value.ravel()[0]

                # Process based on field name
                if field == "labels":
                    channel_info["label"] = str(scalar)
                elif field == "type":
                    channel_info["type"] = str(scalar)
                elif field == "X":
                    channel_info["X"] = float(scalar)
                elif field == "Y":
                    channel_info["Y"] = float(scalar)
                elif field == "Z":
                    channel_info["Z"] = float(scalar)

            # Determine channel type
            channel_info["channel_type"] = self._determine_channel_type(channel_info)

            # Add to list
            channel_info_list.append(channel_info)

        return channel_info_list

    def _process_events(self, events: np.ndarray) -> list[dict[str, Any]]:
        """
        Process event information.

        Args:
            events: Array containing event information

        Returns:
            list: List of dictionaries containing event information
        """
        event_list = []

        # Check if events exist
        if events.size == 0:
            return event_list

        field_names = events.dtype.names
        if field_names is None:
            return event_list

        # Same row-vs-column shape hazard as chanlocs: walk the raveled struct
        # array and flatten each field to a scalar instead of assuming (1, 1)
        # nesting.
        for element in np.asarray(events).ravel():
            event_info = {}

            # Extract event fields
            for field in field_names:
                # Get the field value for this event
                field_value = np.asarray(element[field])
                if field_value.size == 0:
                    continue
                scalar = field_value.ravel()[0]

                # Process based on field name
                if field == "latency":
                    event_info["latency"] = float(scalar)
                elif field == "type":
                    event_info["type"] = str(scalar)
                elif field == "duration":
                    event_info["duration"] = float(scalar)

            # Add to list if it has required fields
            if "latency" in event_info and "type" in event_info:
                event_list.append(event_info)

        return event_list

    def load(self, filepath: str) -> Recording:
        """
        Load EMG data from EEGLAB .set file.

        Dispatches on the file's actual content, not its extension: a
        MATLAB v7.3 (HDF5) ``.set`` is read via h5py (:meth:`_load_v73`);
        every other ``.set`` (classic v5/v7) goes through the unchanged
        ``scipy.io.loadmat``-based path (:meth:`_load`). See
        :func:`_is_matlab_v73` for how the two are told apart.

        Args:
            filepath: Path to the EEGLAB .set file

        Returns:
            Recording: Recording object containing the loaded data
        """
        if _is_matlab_v73(filepath):
            return self._load_v73(filepath)
        return self._load(filepath)

    @staticmethod
    def _normalize_eeglab_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Flatten the two on-disk EEGLAB ``.set`` save forms to one flat dict.

        A real EEGLAB export saves the whole dataset as a single MATLAB struct
        variable named ``EEG``, so ``scipy.io.loadmat`` returns
        ``{'EEG': <(1, 1) struct>}`` and every field (``nbchan``, ``data``,
        ``chanlocs``, ``event``, ...) is nested one level under ``data['EEG']``.
        Some files (and the synthetic test fixtures) instead store those fields at
        the top level of the ``.mat`` dict. ``_extract_metadata`` and the
        signal/channel/event reads in :meth:`_load` expect top-level fields, so
        this normalizes the nested form to the flat form before either runs and
        passes the flat form through unchanged.

        For a struct *array* (multiple datasets, shape ``(1, N)``) only the first
        dataset is used; the rest is currently unsupported. Any non-``EEG``
        top-level keys (e.g. loadmat's ``__header__``) are preserved so nothing is
        lost.

        Args:
            data: The raw mapping returned by ``scipy.io.loadmat``.

        Returns:
            A flat dict whose EEGLAB fields are at the top level.
        """
        eeg = data.get("EEG")
        if eeg is None:
            return data
        eeg = np.asarray(eeg)
        # Only unwrap a MATLAB struct (it exposes named fields via dtype.names);
        # a flat-form dict has no "EEG" key, or an "EEG" that is not a struct.
        names = eeg.dtype.names
        if names is None or eeg.size == 0:
            return data

        # A struct array (shape (1, N)) holds several datasets; take the first.
        record = eeg.ravel()[0]

        flat: dict[str, Any] = {name: record[name] for name in names}
        # Keep any non-EEG top-level keys (loadmat's __header__/__version__, or a
        # second variable saved alongside) without letting them clobber EEG fields.
        for key, value in data.items():
            if key != "EEG" and key not in flat:
                flat[key] = value
        return flat

    @staticmethod
    def _read_fdt(
        set_filepath: str,
        data_field: np.ndarray,
        metadata: dict[str, Any],
        rec: Recording | None = None,
    ) -> np.ndarray:
        """Load the signal matrix from a sibling EEGLAB ``.fdt`` float32 file.

        EEGLAB writes the ``[nbchan x pnts x trials]`` matrix to a separate
        ``.fdt`` in MATLAB column-major order when the data is large; ``EEG.data``
        then stores only the ``.fdt``'s original (pre-BIDS-rename) filename. The
        sibling ``.fdt`` next to the ``.set`` is resolved first because BIDS
        renames the file on disk but not the embedded reference, with the
        embedded name as a fallback. Reading a file other than the one the
        header names is logged and, when ``rec`` is given, recorded under
        :data:`FDT_RECOVERED_KEY` as ``{"referenced": ..., "used": ...}``.

        Args:
            set_filepath: Path to the ``.set`` file being loaded.
            data_field: The ``EEG.data`` char array holding the ``.fdt`` name.
            metadata: Extracted header metadata (needs ``nbchan``/``pnts``).
            rec: The Recording being built, to record a substitution on.

        Returns:
            The signal matrix as a ``(nbchan, pnts * trials)`` float32 array.
        """
        nbchan = int(metadata.get("nbchan", 0))
        pnts = int(metadata.get("pnts", 0))
        trials = int(metadata.get("trials", 1)) or 1
        if nbchan <= 0 or pnts <= 0:
            raise ValueError(
                "EEGLAB data is in a separate .fdt file but the .set header is "
                "missing nbchan/pnts needed to reshape it"
            )
        directory = os.path.dirname(set_filepath)
        sibling = os.path.splitext(set_filepath)[0] + ".fdt"
        embedded = "".join(np.atleast_1d(data_field).ravel().astype(str)).strip()
        candidates = [sibling]
        if embedded:
            candidates.append(os.path.join(directory, os.path.basename(embedded)))
        fdt_path = next((p for p in candidates if os.path.isfile(p)), None)
        if fdt_path is None:
            raise FileNotFoundError(
                f"EEGLAB data is in a separate .fdt file but none was found "
                f"(tried sibling {sibling!r} and embedded name {embedded!r})"
            )
        referenced = os.path.basename(embedded)
        used = os.path.basename(fdt_path)
        if referenced and used != referenced:
            logger.info(
                "EEGLAB %s names data file %s; reading the same-stem sibling %s instead.",
                set_filepath,
                referenced,
                used,
            )
            if rec is not None:
                rec.set_metadata(FDT_RECOVERED_KEY, {"referenced": referenced, "used": used})
        raw = np.fromfile(fdt_path, dtype="<f4")
        expected = nbchan * pnts * trials
        if raw.size != expected:
            raise ValueError(
                f"{fdt_path} holds {raw.size} float32 samples but the .set header "
                f"implies {expected} (nbchan {nbchan} x pnts {pnts} x trials {trials})"
            )
        # MATLAB stores column-major, so the matrix is (nbchan, samples).
        return raw.reshape((nbchan, pnts * trials), order="F")

    # -- MATLAB v7.3 (HDF5) support --------------------------------------------
    #
    # The methods below are the h5py-backed equivalents of the scipy/loadmat
    # helpers above. They are kept separate (not a shared code path) so the
    # well-tested classic v5/v7 reader stays untouched by this addition.

    @staticmethod
    def _h5_scalar(group: Any, name: str, default: float) -> float:
        """Read a MATLAB v7.3 header scalar (e.g. ``nbchan``, ``srate``) as a float.

        These round-trip through HDF5 as 1x1 (or 1-D) float arrays, not true
        scalars -- e.g. ``EEG["nbchan"][()]`` comes back as ``129.0``, not the
        plain int ``129`` -- so the value must be flattened to a 1-D array
        before being cast, rather than indexed as if its shape were known.
        Returns ``default`` when the field is absent or empty.
        """
        if name not in group:
            return default
        arr = np.asarray(group[name][()], dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else default

    @staticmethod
    def _h5_scalar_or_none(group: Any, name: str) -> float | None:
        """Like :meth:`_h5_scalar`, for optional fields (e.g. ``xmin``/``xmax``)."""
        if name not in group:
            return None
        arr = np.asarray(group[name][()], dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else None

    @staticmethod
    def _h5_decode_chars(codes: np.ndarray) -> str:
        """Decode a MATLAB v7.3 char array (Unicode code-point integers) to ``str``.

        MATLAB's HDF5 writer stores char arrays as arrays of integer code
        points (uint16), not as an HDF5 string type, so ``chr()`` per element
        is the correct decode; casting the array to a numpy string dtype would
        stringify the *numbers* instead of the characters they encode.
        """
        return "".join(chr(int(c)) for c in np.asarray(codes).ravel())

    def _deref_h5_value(self, h5py_mod: Any, h5file: Any, entry: Any) -> Any:
        """Resolve one struct-array element: dereference a reference, or pass a bare value.

        ``chanlocs``/``event`` are MATLAB struct ARRAYS (one element per
        channel/event). Because elements can differ in type, length, or be
        empty, MATLAB's HDF5 writer stores each field as an array of HDF5
        object references into ``#refs#`` rather than as a flat array of
        values. A reference read raw is an opaque ``h5py.Reference``, not the
        value it points to, so every field needs this dereferencing, not just
        text ones (labels/type).
        """
        if isinstance(entry, h5py_mod.Reference):
            if not entry:  # null reference -- MATLAB's "no value" for this element
                return None
            target = h5file[entry]
            if target.attrs.get("MATLAB_empty"):
                # MATLAB stores an empty `[]` as a uint64 dims dataset (e.g.
                # [0, 0]) flagged MATLAB_empty; it is "no value", not chars.
                return None
            arr = np.asarray(target[()])
            if arr.size == 0:
                return None
            is_char = target.attrs.get("MATLAB_class") == b"char" or arr.dtype.kind == "u"
            return self._h5_decode_chars(arr) if is_char else float(arr.ravel()[0])
        if isinstance(entry, bytes):
            return entry.decode()
        return entry

    def _deref_struct_array(self, h5py_mod: Any, h5file: Any, group: Any) -> list[dict[str, Any]]:
        """Resolve a MATLAB v7.3 struct-array HDF5 group (``chanlocs`` or ``event``).

        Returns one flat dict per element (channel/event), mapping field name
        to its dereferenced value.

        An EMPTY struct array (``EEG.chanlocs = []`` or ``EEG.event = []``) is
        not a group at all: MATLAB stores it as a ``uint64`` dims dataset
        flagged ``MATLAB_empty``, the same marker :meth:`_deref_h5_value`
        treats as "no value". That yields no elements, not an error.
        """
        if not isinstance(group, h5py_mod.Group) or group.attrs.get("MATLAB_empty"):
            return []
        field_values: dict[str, list[Any]] = {}
        n = 0
        for field in group.keys():
            raw = np.asarray(group[field][()]).ravel()
            field_values[field] = [self._deref_h5_value(h5py_mod, h5file, e) for e in raw]
            n = max(n, len(field_values[field]))
        return [
            {field: values[i] for field, values in field_values.items() if i < len(values)}
            for i in range(n)
        ]

    def _process_chanlocs_v73(
        self, h5py_mod: Any, h5file: Any, chanlocs_group: Any
    ) -> list[dict[str, Any]]:
        """v7.3 equivalent of :meth:`_process_channel_info`, from dereferenced records."""
        channel_info_list = []
        n_unlabelled = 0
        for record in self._deref_struct_array(h5py_mod, h5file, chanlocs_group):
            channel_info: dict[str, Any] = {}
            if record.get("labels") is not None:
                channel_info["label"] = str(record["labels"])
            else:
                n_unlabelled += 1
            if record.get("type") is not None:
                channel_info["type"] = str(record["type"])
            for axis in ("X", "Y", "Z"):
                if record.get(axis) is not None:
                    channel_info[axis] = float(record[axis])
            # Same type-determination rule as the classic path (label
            # fallback when `type` is absent/unrecognized).
            channel_info["channel_type"] = self._determine_channel_type(channel_info)
            channel_info_list.append(channel_info)
        if n_unlabelled:
            logger.warning(
                "EEGLAB v7.3 chanlocs in %s: %d channel(s) have an empty label; "
                "they get the default ChannelN label instead.",
                h5file.filename,
                n_unlabelled,
            )
        return channel_info_list

    def _process_events_v73(
        self, h5py_mod: Any, h5file: Any, event_group: Any
    ) -> list[dict[str, Any]]:
        """v7.3 equivalent of :meth:`_process_events`, from dereferenced records."""
        event_list = []
        n_dropped = 0
        for record in self._deref_struct_array(h5py_mod, h5file, event_group):
            if record.get("latency") is None or record.get("type") is None:
                n_dropped += 1
                continue
            duration = record.get("duration")
            event_list.append(
                {
                    "latency": float(record["latency"]),
                    "type": str(record["type"]),
                    "duration": float(duration) if duration is not None else 0.0,
                }
            )
        if n_dropped:
            logger.warning(
                "EEGLAB v7.3 events in %s: dropped %d event(s) with an empty type or latency.",
                h5file.filename,
                n_dropped,
            )
        return event_list

    def _load_v73(self, filepath: str) -> Recording:
        """Load an EEGLAB ``.set`` file saved in MATLAB v7.3 (HDF5) format.

        Mirrors :meth:`_load`'s field extraction (header scalars, channel
        info, events, signal matrix) but reads through h5py instead of
        ``scipy.io.loadmat``, which refuses v7.3 files outright. See the
        module docstring and ``_is_matlab_v73`` for why a separate method.
        """
        h5py = require_h5py()
        try:
            with h5py.File(filepath, "r") as f:
                # EEGLAB can also save the struct's fields FLAT at the HDF5
                # root (no `EEG` group); the classic loadmat path already
                # accepts that layout, so treat the root as the struct when
                # it carries the core fields rather than calling it corrupt. `EEG`
                # wins when both layouts are present.
                if "EEG" in f:
                    eeg = f["EEG"]
                elif all(key in f for key in ("nbchan", "srate", "pnts", "data")):
                    eeg = f
                else:
                    # A valid HDF5 container that simply holds some other MATLAB
                    # variable(s). Raised as the generic typed read error (still
                    # terminal: re-reading will not change it), never with the
                    # word "corrupt", which would make classify_read_error call a
                    # healthy file CorruptFileError.
                    raise FileReadError(
                        f"{filepath} is not an EEGLAB v7.3 dataset (no 'EEG' group "
                        "and no flat nbchan/srate/pnts/data at the root)"
                    )

                # nbchan/srate/pnts/trials come back as float arrays (see
                # `_h5_scalar`), so they are flattened and coerced to the
                # int/float the rest of the importer expects.
                nbchan = int(round(self._h5_scalar(eeg, "nbchan", 0.0)))
                srate = self._h5_scalar(eeg, "srate", 0.0)
                if not srate:
                    logger.warning(
                        "EEGLAB v7.3 file %s has no sampling rate (srate absent, empty "
                        "or zero); assuming %g Hz.",
                        filepath,
                        _DEFAULT_SRATE,
                    )
                    srate = _DEFAULT_SRATE
                pnts = int(round(self._h5_scalar(eeg, "pnts", 0.0)))
                trials = int(round(self._h5_scalar(eeg, "trials", 1.0))) or 1

                # trials > 1 means the file is epoched, not continuous. Raise
                # the same typed error the classic path's classify_read_error
                # signals for other non-continuous derivatives, rather than
                # silently flattening epochs into a fake continuous stream.
                if trials > 1:
                    raise NotContinuousRecordingError(
                        f"EEGLAB v7.3 file holds {trials} epochs, not a "
                        f"continuous recording ({filepath})"
                    )

                rec = Recording()
                rec.set_metadata("device", "EEGLAB")
                rec.set_metadata("source_file", filepath)
                rec.set_metadata("srate", srate)
                rec.set_metadata("nbchan", nbchan)
                rec.set_metadata("trials", trials)
                rec.set_metadata("pnts", pnts)
                for key in ("xmin", "xmax"):
                    value = self._h5_scalar_or_none(eeg, key)
                    if value is not None:
                        rec.set_metadata(key, value)
                for key in ("setname", "subject", "group", "condition", "session", "comments"):
                    if key not in eeg:
                        continue
                    raw_value = np.asarray(eeg[key][()]).ravel()
                    if raw_value.size and raw_value.dtype.kind == "u":
                        rec.set_metadata(key, self._h5_decode_chars(raw_value))

                channel_info_list: list[dict[str, Any]] = []
                if "chanlocs" in eeg:
                    channel_info_list = self._process_chanlocs_v73(h5py, f, eeg["chanlocs"])
                if not channel_info_list:
                    channel_info_list = [
                        {"label": f"Channel{i + 1}", "channel_type": "OTHER"} for i in range(nbchan)
                    ]

                if "event" in eeg:
                    for ev in self._process_events_v73(h5py, f, eeg["event"]):
                        rec.add_event(
                            onset=(ev["latency"] - 1) / srate,
                            duration=ev.get("duration", 0.0) / srate,
                            description=ev["type"],
                        )

                if "data" not in eeg or eeg["data"].size == 0:
                    return rec

                data_ds = eeg["data"]
                if data_ds.dtype.kind == "f":
                    # HDF5/v7.3 stores the array TRANSPOSED relative to MATLAB:
                    # h5py hands back (n_samples, n_channels) -- verified
                    # against a real affected file in issue #113 -- while the
                    # rest of biosigio, and the classic loadmat path above, is
                    # channel-major, (n_channels, n_samples). Getting this
                    # backwards silently swaps samples and channels; for a
                    # near-square channel count it would still "work" and
                    # produce garbage, so transpose explicitly by default
                    # rather than letting shape sort itself out downstream.
                    #
                    # The header's nbchan cross-checks *which* axis is
                    # channels (the same defensive check sccn/eegprep's
                    # pop_loadset_h5 uses) rather than transposing blindly:
                    # if the array is unambiguously already channel-major
                    # (axis 0 already matches nbchan and axis 1 does not),
                    # a second transpose would swap it right back into the
                    # wrong orientation, so that one case is skipped.
                    raw = np.asarray(data_ds[()])
                    if raw.ndim != 2:
                        signal_data = raw.reshape(1, -1)
                    elif raw.shape[0] == nbchan and raw.shape[1] != nbchan:
                        signal_data = raw  # already channel-major
                    else:
                        signal_data = raw.T
                else:
                    # EEG.data holds the companion .fdt's filename as a char
                    # array (Unicode code points), not the numeric matrix.
                    # Decode it to a plain string and reuse the classic
                    # path's sibling-.fdt resolution (same nbchan/pnts/trials
                    # reshape, same sibling-over-embedded-name preference).
                    codes = np.asarray(data_ds[()]).ravel()
                    filename_field = np.array(list(self._h5_decode_chars(codes)))
                    signal_data = self._read_fdt(
                        filepath,
                        filename_field,
                        {"nbchan": nbchan, "pnts": pnts, "trials": trials},
                        rec,
                    )

                time_index = np.arange(signal_data.shape[1]) / srate

                # The data matrix is authoritative for channel count, same
                # rule as the classic path: chanlocs only names the rows, so
                # a short/missing chanlocs must never truncate the signal.
                n_data_ch = signal_data.shape[0]
                if nbchan and nbchan != n_data_ch:
                    warnings.warn(
                        f"EEGLAB v7.3 header nbchan={nbchan} disagrees with the "
                        f"data matrix ({n_data_ch} rows); using the data matrix",
                        stacklevel=3,
                    )
                if len(channel_info_list) != n_data_ch:
                    warnings.warn(
                        f"EEGLAB chanlocs describes {len(channel_info_list)} "
                        f"channel(s) but the data matrix has {n_data_ch} row(s); "
                        f"keeping all data rows",
                        stacklevel=3,
                    )
                    channel_info_list = channel_info_list[:n_data_ch]
                    for i in range(len(channel_info_list), n_data_ch):
                        channel_info_list.append(
                            {"label": f"Channel{i + 1}", "channel_type": "OTHER"}
                        )

                # Labels must be unique (rec.channels is keyed by label, and
                # the exporters slice rec.signals by column name); disambiguate
                # collisions with a numeric suffix and warn, same as _load.
                used_labels: set[str] = set()
                for i, channel_info in enumerate(channel_info_list):
                    label = channel_info.get("label", f"Channel{i + 1}")
                    if label in used_labels:
                        k = 2
                        while f"{label}_{k}" in used_labels:
                            k += 1
                        warnings.warn(
                            f"duplicate EEGLAB channel label {label!r}; renaming to {label}_{k}",
                            stacklevel=3,
                        )
                        label = f"{label}_{k}"
                    used_labels.add(label)
                    channel_info["label"] = label

                labels = [info["label"] for info in channel_info_list]
                rec.signals = pd.DataFrame(
                    signal_data.T, columns=labels, index=time_index, copy=False
                )
                del signal_data

                for channel_info in channel_info_list:
                    channel_label = channel_info["label"]
                    ch_type = channel_info.get("channel_type", "OTHER")
                    rec.channels[channel_label] = {
                        "sample_frequency": srate,
                        "physical_dimension": "uV",  # Default unit for EEG/EMG
                        "prefilter": "n/a",
                        "channel_type": ch_type,
                        "modality": infer_modality_from_channel_type(ch_type),
                    }
                    if "X" in channel_info:
                        rec.channels[channel_label]["X"] = channel_info["X"]
                    if "Y" in channel_info:
                        rec.channels[channel_label]["Y"] = channel_info["Y"]
                    if "Z" in channel_info:
                        rec.channels[channel_label]["Z"] = channel_info["Z"]

                return rec

        except Exception as e:
            # Resource exhaustion (MemoryError, thread/allocation-exhaustion
            # OSError/RuntimeError) is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            raise classify_read_error(e, filepath) from e

    def _load(self, filepath: str) -> Recording:
        """Internal loader (see :meth:`load`)."""
        try:
            # Load the .set file. A real EEGLAB export wraps every field in a
            # single ``EEG`` struct, so unwrap it to the flat top-level form that
            # the metadata/signal/channel/event reads below expect (a no-op for
            # files already saved flat).
            data = self._normalize_eeglab_dict(loadmat(filepath))

            # Create Recording object
            rec = Recording()

            # Extract and store metadata
            metadata = self._extract_metadata(data)
            for key, value in metadata.items():
                rec.set_metadata(key, value)

            # Store source file information
            rec.set_metadata("source_file", filepath)

            # Sampling rate (used for the event onset/duration conversion below and
            # for the signal time index).
            # Fall back to 1000 Hz when srate is absent OR present-but-zero (an
            # empty srate field), so the event/time-index divisions never hit 0.
            srate = float(metadata.get("srate", 0.0))
            if not srate:
                logger.warning(
                    "EEGLAB file %s has no sampling rate (srate absent, empty or "
                    "zero); assuming %g Hz.",
                    filepath,
                    _DEFAULT_SRATE,
                )
                srate = _DEFAULT_SRATE

            # Process channel information
            if "chanlocs" in data and data["chanlocs"].size > 0:
                channel_info_list = self._process_channel_info(data["chanlocs"])
            else:
                # If no channel locations, create default channel info
                channel_info_list = []
                for i in range(metadata.get("nbchan", 0)):
                    channel_info_list.append({"label": f"Channel{i + 1}", "channel_type": "OTHER"})

            # Process event information into the standard events table. EEGLAB
            # stores event latency/duration in samples (latency is 1-based), so
            # convert to seconds; the event ``type`` becomes the description.
            if "event" in data and data["event"].size > 0:
                for ev in self._process_events(data["event"]):
                    rec.add_event(
                        onset=(ev["latency"] - 1) / srate,
                        duration=ev.get("duration", 0.0) / srate,
                        description=ev["type"],
                    )

            # Extract signal data
            if "data" in data and data["data"].size > 0:
                # Get data array. EEGLAB stores large recordings with the signal
                # matrix in a separate float32 ``.fdt`` file; in that case
                # ``EEG.data`` holds the ``.fdt`` filename (a char array) rather
                # than the numeric matrix, so load the sibling ``.fdt`` instead.
                signal_data = data["data"]
                if signal_data.dtype.kind in ("U", "S"):
                    signal_data = self._read_fdt(filepath, signal_data, metadata, rec)

                # Derive the time index in seconds from the sample count. EEGLAB's
                # `times` field is in milliseconds, so dividing it by srate (the old
                # behavior) mis-scaled the index; sample_index / srate is correct and
                # always matches the data length.
                time_index = np.arange(signal_data.shape[1]) / srate

                # The data matrix is authoritative for the channel count: every
                # row ships to the caller. chanlocs only NAMES the rows, so a
                # short/missing chanlocs (or one misread from an odd save form)
                # must never truncate the signal -- pad with default labels
                # instead. A longer-than-data chanlocs is trimmed (labels
                # without signal rows describe nothing).
                n_data_ch = signal_data.shape[0]
                if len(channel_info_list) != n_data_ch:
                    warnings.warn(
                        f"EEGLAB chanlocs describes {len(channel_info_list)} "
                        f"channel(s) but the data matrix has {n_data_ch} row(s); "
                        f"keeping all data rows",
                        stacklevel=3,
                    )
                    channel_info_list = channel_info_list[:n_data_ch]
                    for i in range(len(channel_info_list), n_data_ch):
                        channel_info_list.append(
                            {"label": f"Channel{i + 1}", "channel_type": "OTHER"}
                        )
                header_nbchan = int(metadata.get("nbchan", 0))
                if header_nbchan and header_nbchan != n_data_ch:
                    warnings.warn(
                        f"EEGLAB header nbchan={header_nbchan} disagrees with the "
                        f"data matrix ({n_data_ch} rows); using the data matrix",
                        stacklevel=3,
                    )

                # Labels must be unique: rec.channels is keyed by label and the
                # exporters slice rec.signals by column name, so a duplicate
                # (a real channel literally named "Channel4" colliding with a
                # padded default, or duplicated labels in chanlocs itself)
                # silently clobbers channel metadata and breaks per-channel
                # export slicing. Disambiguate with a numeric suffix and warn.
                used_labels: set[str] = set()
                for i, channel_info in enumerate(channel_info_list):
                    label = channel_info.get("label", f"Channel{i + 1}")
                    if label in used_labels:
                        k = 2
                        while f"{label}_{k}" in used_labels:
                            k += 1
                        warnings.warn(
                            f"duplicate EEGLAB channel label {label!r}; renaming to {label}_{k}",
                            stacklevel=3,
                        )
                        label = f"{label}_{k}"
                    used_labels.add(label)
                    channel_info["label"] = label

                # Build the frame in a single allocation (samples x channels)
                # rather than assigning one column at a time. The per-column path
                # reallocates the block manager O(n_channels) times and roughly
                # doubles peak RAM, which OOMs large / high-channel-count .fdt
                # recordings (#66, #95); the .fdt's native float32 is preserved so
                # a multi-GB recording is not silently upcast to float64.
                labels = [info["label"] for info in channel_info_list]
                rec.signals = pd.DataFrame(
                    signal_data.T, columns=labels, index=time_index, copy=False
                )
                del signal_data

                # Add channel information (the pad/trim above guarantees
                # channel_info_list has exactly n_data_ch entries with unique labels)
                for channel_info in channel_info_list:
                    channel_label = channel_info["label"]

                    # Add channel info (no silent EMG default; carry modality)
                    ch_type = channel_info.get("channel_type", "OTHER")
                    rec.channels[channel_label] = {
                        "sample_frequency": srate,
                        "physical_dimension": "uV",  # Default unit for EEG/EMG
                        "prefilter": "n/a",
                        "channel_type": ch_type,
                        "modality": infer_modality_from_channel_type(ch_type),
                    }

                    # Add additional channel metadata
                    if "X" in channel_info:
                        rec.channels[channel_label]["X"] = channel_info["X"]
                    if "Y" in channel_info:
                        rec.channels[channel_label]["Y"] = channel_info["Y"]
                    if "Z" in channel_info:
                        rec.channels[channel_label]["Z"] = channel_info["Z"]

            return rec

        except Exception as e:
            # Resource exhaustion (MemoryError, thread/allocation-exhaustion
            # OSError/RuntimeError) is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            raise classify_read_error(e, filepath) from e
