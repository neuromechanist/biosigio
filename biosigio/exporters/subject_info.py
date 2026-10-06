"""Subject information in recording metadata, and the export option that drops it.

A biosigIO Zarr store holds the signal, the events, channel names, types and
units, and technical recording metadata. Information about the subject (who was
recorded, and their age, sex or other phenotype), operator or administrative free
text, and provenance that can name a person or a local path belong at dataset
scope, for example in a BIDS ``participants.tsv``, not inside every recording's
store. Several importers copy such members out of the source header into
``Recording.metadata``, and the Zarr exporters copy that metadata into the store's
``recording_metadata`` root attribute.

``exclude_subject_info=True`` on :meth:`~biosigio.core.emg.Recording.to_zarr`,
:meth:`~biosigio.exporters.zarr.ZarrExporter.export` and
:func:`~biosigio.exporters.zarr_stream.stream_to_zarr` removes the members named
in :data:`SUBJECT_INFO_KEYS` (through :func:`strip_subject_info`), reduces the
path-valued members in :data:`PATH_PROVENANCE_KEYS` to their final component, and
marks the store with :data:`SUBJECT_INFO_EXCLUDED_ATTR`. It keeps everything
else, including recording start dates and times, events, and per-channel
labels, units and prefilter text. The default writes the metadata as earlier
releases did.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Member names. A metadata key matches when its normalized form (casefolded, every
# non-alphanumeric character dropped) equals the normalized form of one of these,
# so "Sex:", "birth_date" and "Patient Name" match while "subjects" does not. A
# matching member is removed whole, whatever its value. Most entries are
# spellings an importer, or a library biosigIO reads through, produces; the
# generic ones cover what a caller passes in recording_metadata. Recording start
# dates and times (startdate, starttime, meas_date, recording_date) are left out
# on purpose: an acquisition date not linked to a subject is kept.
SUBJECT_INFO_KEYS: frozenset[str] = frozenset(
    {
        # EDF/BDF header fields importers/edf.py copies when non-empty: patient
        # code, sex (stored as "gender"), birth date, additional patient text,
        # and the operator and administrative recording fields.
        "patientcode",
        "gender",
        "birthdate",
        "patient_additional",
        "admincode",
        "technician",
        "equipment",
        "recording_additional",
        # The EDF importer asks pyedflib for "patient_name", which pyedflib never
        # returns (its key is "patientname"), so no importer emits either today;
        # both stay for a caller who passes pyedflib's header, whose own
        # spellings also include "sex".
        "patient_name",
        "patientname",
        "sex",
        # edfio's names for the raw EDF+ patient and recording fields.
        "local_patient_identification",
        "local_recording_identification",
        # MNE's info["subject_info"] (removed whole) and its fields, which a
        # caller may pass flat, plus MNE's info["experimenter"].
        "subject_info",
        "his_id",
        "first_name",
        "last_name",
        "middle_name",
        "birthday",
        "hand",
        "weight",
        "height",
        "experimenter",
        # Generic spellings for the same facts in a caller's metadata.
        "patient",
        "patient_id",
        "participant_id",
        "subject_id",
        "dob",
        "age",
        "handedness",
        "operator",
        "name",
        "description",
        # EEGLAB (importers/eeglab.py): EEG.subject, EEG.group (such as patients
        # or controls), and the set name and file name and path the set was
        # saved under, which often carry a subject's name or a local user path.
        "subject",
        "group",
        "setname",
        "filename",
        "filepath",
        # Free-text comments: EEGLAB EEG.comments, and the WFDB header comments,
        # which in PhysioNet records carry age, sex, diagnoses and medication.
        "comments",
        # WFDB record name, which can encode a subject identifier.
        "record_name",
        # Read-recovery provenance, removed whole: their inner names
        # ("referenced", "used", "DataFile", ...) are too generic to match, and
        # their values are file names that can carry a subject's name.
        "eeglab_fdt_recovered",
        "brainvision_header_recovered",
    }
)

# Path-valued provenance members. With exclude_subject_info=True the writers keep
# only the final path component, so a local directory (and an OS user name in it)
# never reaches the store; the file name itself is kept and can carry a BIDS
# label such as sub-01.
PATH_PROVENANCE_KEYS: tuple[str, ...] = ("source_file", "bti_pdf_file")

# biosigIO's own maps keyed by channel label or sidecar row name. Matching never
# descends into them, so a channel labeled "Sex" or "Group" keeps its entry.
_LABEL_KEYED_MEMBERS = frozenset({"channels_tsv_units", "channel_labels_deduplicated"})

# Root attribute a store written with exclude_subject_info=True carries, set to
# True. Absent otherwise: a store written without the option is unchanged.
SUBJECT_INFO_EXCLUDED_ATTR = "subject_info_excluded"


def _normalize(key: str) -> str:
    """Casefold ``key`` and drop every non-alphanumeric character."""
    return "".join(ch for ch in key.casefold() if ch.isalnum())


_NORMALIZED_KEYS = frozenset(_normalize(k) for k in SUBJECT_INFO_KEYS)


def is_subject_info_key(key: object) -> bool:
    """Whether ``key`` names a member of :data:`SUBJECT_INFO_KEYS`.

    The match is exact on the normalized form (casefolded, every
    non-alphanumeric character dropped), never a substring or prefix: ``"Sex:"``
    and ``"birth_date"`` match, ``"subjects"`` and ``"comments_total"`` do not.
    Only a string key can match.
    """
    return isinstance(key, str) and _normalize(key) in _NORMALIZED_KEYS


def _rebuild(value: Any, *, strip: bool) -> Any:
    """Rebuild mappings, lists and tuples in ``value``; with ``strip``, without
    the subject-information members."""
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if strip and is_subject_info_key(k):
                continue
            out[k] = _rebuild(v, strip=strip and k not in _LABEL_KEYED_MEMBERS)
        return out
    if isinstance(value, list):
        return [_rebuild(v, strip=strip) for v in value]
    if isinstance(value, tuple):
        return tuple(_rebuild(v, strip=strip) for v in value)
    return value


def strip_subject_info(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``metadata`` without its subject-information members.

    A member is removed when :func:`is_subject_info_key` matches its name, at any
    depth: through nested mappings (older EDF stores nested the header fields
    under ``recording_info``), lists and tuples. It is removed whatever its
    value, an empty one or a whole nested object included, so a reader never sees
    the member at all. Every other member is kept, in its original order.

    Matching is by member name only, and it never descends into biosigIO's own
    label-keyed maps, ``channels_tsv_units`` and ``channel_labels_deduplicated``:
    their keys are channel labels or sidecar row names, so a channel labeled
    ``Sex`` keeps its entry. Anywhere else a key spelled like a member is
    removed, whatever it describes, and a subject's details held under an
    unlisted name, or inside a value (a free-text string, an event label, a
    channel label), are not touched. Values are never inspected; path-valued
    members are reduced by the writers, not here.

    The input is never modified: mappings, lists and tuples are rebuilt, and the
    remaining leaf values are the caller's own objects, not copies. A mapping
    comes back as a plain ``dict``.

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
    return _rebuild(metadata, strip=True)


def _final_component(path: str) -> str:
    """The last component of a POSIX or Windows path, ignoring trailing separators."""
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def _reduce_path_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``metadata`` whose top-level :data:`PATH_PROVENANCE_KEYS`
    string values are reduced to their final path component.

    ``/home/alice/study/sub-01_eeg.edf`` becomes ``sub-01_eeg.edf``, and a BTi
    directory ``.../sub-01_task-rest_meg/`` becomes ``sub-01_task-rest_meg``.
    A non-string value is kept as is. The input is not modified.
    """
    out = dict(metadata)
    for key in PATH_PROVENANCE_KEYS:
        value = out.get(key)
        if isinstance(value, str):
            out[key] = _final_component(value)
    return out
