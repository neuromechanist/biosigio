"""Shared helpers for the ``exclude_subject_info`` tests (not a test module).

Used by ``test_zarr_subject_info.py`` and ``test_real_data_nm000181.py``, so a
test module never imports from another. Everything here returns member NAMES,
never values, so an assertion built on it cannot print a header value.
"""

from __future__ import annotations

from biosigio import is_subject_info_key

# biosigIO's own maps keyed by channel label or sidecar row name, which the
# removal deliberately leaves whole.
LABEL_KEYED_MEMBERS = frozenset({"channels_tsv_units", "channel_labels_deduplicated"})

# Every top-level metadata member an importer (or the streaming writer) emits
# that is technical and stays in a store written with exclude_subject_info=True.
# A member an importer emits must be either here or matched by
# is_subject_info_key; the classification tests fail on any other, so a new
# importer key is classified on purpose rather than leaking by default.
KEPT_TECHNICAL = frozenset(
    {
        # Every importer, and Recording.from_file
        "source_file",  # reduced to its final component under the option
        "source_format",
        "number_of_signals",
        # EDF/BDF (importers/edf.py)
        "startdate",
        "filetype",
        "file_duration",
        "datarecord_duration",
        "mixed_rate_resampled",
        "mixed_rate_target_hz",
        "edf_tolerant_read",
        "edf_tolerant_read_reason",
        # Repeated-label renames and the BIDS channels.tsv report (label-keyed)
        "channel_labels_deduplicated",
        "channels_tsv_units",
        # WFDB (importers/wfdb.py)
        "sampling_frequency",
        "starttime",
        "annotation_status",
        "annotation_error",
        # EEGLAB (importers/eeglab.py)
        "device",
        "srate",
        "nbchan",
        "trials",
        "pnts",
        "xmin",
        "xmax",
        "session",
        "condition",
        # 4D/BTi (importers/meg.py), reduced to its final component under the option
        "bti_pdf_file",
        # XDF, OTB, neo, CSV
        "stream_count",
        "signal_resolution",
        "neo_io",
        "t_start_s",
        "file_format",
        # Written by stream_to_zarr
        "streamed",
    }
)


def member_names(value) -> set[str]:
    """Every mapping key at any depth, as written."""
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            names.add(str(key))
            names |= member_names(item)
    elif isinstance(value, list | tuple):
        for item in value:
            names |= member_names(item)
    return names


def subject_members(value) -> set[str]:
    """Names of subject-information members at any depth, outside the
    label-keyed maps (an independent walk, not the code under test)."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if is_subject_info_key(key):
                found.add(str(key))
            elif key not in LABEL_KEYED_MEMBERS:
                found |= subject_members(item)
    elif isinstance(value, list | tuple):
        for item in value:
            found |= subject_members(item)
    return found


def unclassified(metadata: dict) -> set[str]:
    """Top-level members neither kept as technical nor matched as subject
    information."""
    return {
        str(key) for key in metadata if key not in KEPT_TECHNICAL and not is_subject_info_key(key)
    }
