"""Subject information in recording metadata, and the export option that drops it.

A biosigIO Zarr store holds the signal, the events, channel names, types and
units, and technical recording metadata. Information about the subject (who was
recorded) and operator or administrative free text belongs at dataset scope, for
example in a BIDS ``participants.tsv``, not inside every recording's store.
Several importers copy such members out of the source header into
``Recording.metadata``, and the Zarr exporters copy that metadata into the store's
``recording_metadata`` root attribute. ``exclude_subject_info=True`` on
:meth:`~biosigio.core.emg.Recording.to_zarr`,
:meth:`~biosigio.exporters.zarr.ZarrExporter.export` and
:func:`~biosigio.exporters.zarr_stream.stream_to_zarr` removes them, through
:func:`strip_subject_info`, and marks the store with
:data:`SUBJECT_INFO_EXCLUDED_ATTR`. The default keeps them, as earlier releases
did.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Member names, lowercase; a metadata key matches when its casefolded form is one
# of these. Each entry is a spelling an importer (or a library biosigIO reads
# through) actually produces; read the importer before adding one.
SUBJECT_INFO_KEYS: frozenset[str] = frozenset(
    {
        # EDF/BDF header, as importers/edf.py names the pyedflib fields it copies.
        # Subject: patient code, sex, birth date, name, additional patient text.
        "patientcode",
        "gender",
        "birthdate",
        "patient_name",
        "patient_additional",
        # Operator and administrative free text from the recording field.
        "admincode",
        "technician",
        "equipment",
        "recording_additional",
        # pyedflib's own getHeader() spellings of the same fields, which a caller
        # passing a header to stream_to_zarr(recording_metadata=...) carries.
        "patientname",
        "sex",
        # EEGLAB EEG.subject (subject code) and EEG.group (subject group, such as
        # patients or controls), both read by importers/eeglab.py.
        "subject",
        "group",
        # Free-text comments: EEGLAB EEG.comments, and the WFDB header comments,
        # which in PhysioNet records carry age, sex, diagnoses and medication.
        "comments",
    }
)

# Root attribute a store written with exclude_subject_info=True carries, set to
# True. Absent otherwise: a store written without the option is unchanged.
SUBJECT_INFO_EXCLUDED_ATTR = "subject_info_excluded"


def _is_subject_info_key(key: object) -> bool:
    """Whether ``key`` names a subject-information member (case-insensitive).

    Only a string key can match.
    """
    return isinstance(key, str) and key.casefold() in SUBJECT_INFO_KEYS


def _strip(value: Any) -> Any:
    """Rebuild ``value`` without subject-information members, at any depth."""
    if isinstance(value, Mapping):
        return {k: _strip(v) for k, v in value.items() if not _is_subject_info_key(k)}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip(v) for v in value)
    return value


def strip_subject_info(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``metadata`` without its subject-information members.

    A member is removed when its name, compared case-insensitively, is in
    :data:`SUBJECT_INFO_KEYS`, at any depth: through nested mappings (older EDF
    stores nested the header fields under ``recording_info``), lists and tuples.
    It is removed whatever its value, an empty one included, so a reader never
    sees the member at all. Every other member is kept, in its original order.

    The input is never modified: mappings, lists and tuples are rebuilt, and the
    remaining leaf values are the caller's own objects, not copies. A mapping
    comes back as a plain ``dict``.

    Matching is by member name only, so it also applies inside maps keyed by
    channel label: a channel literally labeled ``Sex`` loses its entry from a
    map such as ``channel_labels_deduplicated``.

    Args:
        metadata: Recording metadata, such as ``Recording.metadata`` or the
            encoded ``recording_metadata`` of a store.

    Returns:
        A new dict without the subject-information members.

    Raises:
        TypeError: ``metadata`` is not a mapping.

    Example:
        >>> strip_subject_info({"patientcode": "P1", "recording_info": {"Sex": "F",
        ...     "startdate": "2020-01-01"}, "number_of_signals": 2})
        {'recording_info': {'startdate': '2020-01-01'}, 'number_of_signals': 2}
    """
    if not isinstance(metadata, Mapping):
        raise TypeError(f"strip_subject_info expects a mapping, got {type(metadata).__name__!r}")
    return _strip(metadata)
