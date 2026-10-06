"""``strip_subject_info``: the removal behind ``exclude_subject_info``.

A Zarr store carries the signal, the events, channel names, types and units, and
technical recording metadata; subject information belongs at dataset scope.
``strip_subject_info`` removes the members named in ``SUBJECT_INFO_KEYS`` from a
metadata mapping, matching each key by its normalized form. The writers that
call it are tested in ``test_zarr_subject_info.py``.
"""

import copy
import datetime
import pathlib
import re
import types

import numpy as np
import pytest

import biosigio
from biosigio import SUBJECT_INFO_KEYS, is_subject_info_key, strip_subject_info

# Every spelling, by where it comes from. Pinned here as the requirement, so
# dropping one from SUBJECT_INFO_KEYS, or adding one, fails a test.
SPELLINGS_BY_SOURCE = {
    "EDF/BDF importer": [
        "patientcode",
        "gender",
        "birthdate",
        "patient_additional",
        "admincode",
        "technician",
        "equipment",
        "recording_additional",
    ],
    "pyedflib getHeader, and the EDF importer's dead lookup": [
        "patientname",
        "patient_name",
        "sex",
    ],
    "edfio": ["local_patient_identification", "local_recording_identification"],
    "MNE info": [
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
    ],
    "generic caller spellings": [
        "patient",
        "patient_id",
        "participant_id",
        "subject_id",
        "dob",
        "age",
        "handedness",
        "operator",
    ],
    "EEGLAB importer": ["subject", "group", "setname", "filename", "filepath", "comments"],
    "WFDB importer": ["comments", "record_name"],
    "read-recovery provenance": ["eeglab_fdt_recovered", "brainvision_header_recovered"],
}
ALL_SPELLINGS = sorted({k for keys in SPELLINGS_BY_SOURCE.values() for k in keys})


@pytest.mark.parametrize("key", ALL_SPELLINGS)
def test_every_documented_spelling_is_removed(key):
    assert strip_subject_info({key: "x", "number_of_signals": 2}) == {"number_of_signals": 2}


def test_constant_names_exactly_the_documented_spellings():
    assert set(ALL_SPELLINGS) == SUBJECT_INFO_KEYS


def test_importers_recovery_keys_are_in_the_constant():
    """The two read-recovery entries are named by the importers' own constants."""
    from biosigio.importers.brainvision import HEADER_RECOVERED_KEY
    from biosigio.importers.eeglab import FDT_RECOVERED_KEY

    assert {HEADER_RECOVERED_KEY, FDT_RECOVERED_KEY} <= SUBJECT_INFO_KEYS


def test_public_names_are_exported():
    expected = {
        "SUBJECT_INFO_KEYS",
        "SUBJECT_INFO_EXCLUDED_ATTR",
        "strip_subject_info",
        "is_subject_info_key",
    }
    assert expected <= set(biosigio.__all__)


@pytest.mark.parametrize(
    "key",
    [
        "sex ",
        "Sex:",
        "SEX",
        "birth_date",
        "Birth-Date",
        "patient-code",
        "Patient Name",
        "PatientName",
        "first name",
        "Record_Name",
        "Subject_Info",
        "D.O.B.",
    ],
)
def test_normalized_spellings_match(key):
    """Casefolded with every non-alphanumeric character dropped, then exact."""
    assert is_subject_info_key(key)
    assert strip_subject_info({key: "x", "srate": 1}) == {"srate": 1}


def test_near_miss_names_are_kept():
    """Exact match on the normalized name, never a substring or a prefix."""
    meta = {
        "subjects": 3,
        "subject_count": 2,
        "channel_groups": ["EEG"],
        "comments_total": 1,
        "sexual_dimorphism": 0,
        "agent": "x",
        "named": True,
        "filename_pattern": "*.edf",
        "group_delay": 0.1,
        "average_reference": True,
    }
    out = strip_subject_info(meta)
    assert out == meta
    assert list(out) == list(meta)


def test_technical_name_and_description_at_depth_are_kept():
    """The OTB importer parses its XML into this shape (a device's model name, a
    channel's electrode description); generic names like these never match."""
    meta = {
        "device": {"name": "Sessantaquattro", "sampling_frequency": 2000, "ad_bits": 16},
        "channels": {"CH1": {"description": "Ch1 HD-sEMG grid", "gain": 150}},
        "name": "caller's own field",
        "description": "caller's own field",
    }
    assert strip_subject_info(meta) == meta


def test_docs_list_exactly_the_constant():
    """docs/formats/zarr.md names every member the option removes, and no other."""
    page = pathlib.Path(__file__).resolve().parents[2] / "docs/formats/zarr.md"
    if not page.exists():
        pytest.skip("docs not present in this install")
    text = page.read_text()
    block = text[text.index("**What the option removes.**") : text.index("A name matches after")]
    documented = set(re.findall(r"`([a-z_]+)`", block)) - {"recording_metadata"}
    assert documented == SUBJECT_INFO_KEYS


@pytest.mark.parametrize("key", ["startdate", "starttime", "meas_date", "recording_date"])
def test_recording_dates_are_kept(key):
    """An acquisition date not linked to a subject stays."""
    assert strip_subject_info({key: "2020-01-02"}) == {key: "2020-01-02"}


def test_flat_members_removed_others_kept_in_order():
    meta = {
        "startdate": datetime.datetime(2020, 1, 2, 3, 4, 5),
        "patientcode": "P1",
        "filetype": 1,
        "birthdate": "02 jan 1970",
        "number_of_signals": 2,
        "equipment": "amp",
        "source_file": "rec.edf",
    }
    out = strip_subject_info(meta)
    assert list(out) == ["startdate", "filetype", "number_of_signals", "source_file"]
    assert out["startdate"] == datetime.datetime(2020, 1, 2, 3, 4, 5)


def test_nested_members_removed_at_any_depth():
    """The older EDF importer nested the header fields under recording_info."""
    meta = {
        "recording_info": {
            "startdate": "2020-01-02",
            "patientcode": "P1",
            "gender": "Male",
            "deeper": {"technician": "T", "rate": 256},
        },
        "file_info": {"filetype": 1},
    }
    assert strip_subject_info(meta) == {
        "recording_info": {"startdate": "2020-01-02", "deeper": {"rate": 256}},
        "file_info": {"filetype": 1},
    }


def test_whole_entries_removed_with_their_contents():
    """MNE's subject_info and the read-recovery records go whole, whatever is
    inside them."""
    meta = {
        "subject_info": {"his_id": "x", "id": 4, "weight": 70.0},
        "eeglab_fdt_recovered": {"referenced": "a.fdt", "used": "b.fdt"},
        "brainvision_header_recovered": {"DataFile": {"referenced": "a.eeg", "used": "b.eeg"}},
        "srate": 250.0,
    }
    assert strip_subject_info(meta) == {"srate": 250.0}


def test_matching_is_case_insensitive():
    meta = {"PatientCode": "P1", "BIRTHDATE": "x", "Sex": "F", "Group": "controls", "Rate": 1}
    assert strip_subject_info(meta) == {"Rate": 1}


def test_label_keyed_maps_are_not_descended():
    """A channel labeled like a member keeps its entry in biosigIO's own maps."""
    meta = {
        "channels_tsv_units": {
            "converted": 0,
            "matched_case_insensitive": {"sex": "Sex", "GROUP": "Group"},
        },
        "channel_labels_deduplicated": {"Age-0": "Age", "Age": "Age"},
        "gender": "F",
    }
    out = strip_subject_info(meta)
    assert out == {
        "channels_tsv_units": meta["channels_tsv_units"],
        "channel_labels_deduplicated": meta["channel_labels_deduplicated"],
    }
    assert out["channels_tsv_units"] is not meta["channels_tsv_units"]


def test_members_inside_lists_and_tuples_removed():
    meta = {
        "streams": [{"Subject": "S1", "task": "rest"}, "plain", 3, [{"equipment": "e"}]],
        "pair": ({"comments": "c", "run": 1}, None),
    }
    assert strip_subject_info(meta) == {
        "streams": [{"task": "rest"}, "plain", 3, [{}]],
        "pair": ({"run": 1}, None),
    }


@pytest.mark.parametrize("value", ["", None, 0, [], {}, {"startdate": "kept with its parent"}])
def test_member_removed_whatever_its_value(value):
    """An empty value is still a member a reader would see, so it goes too; a
    member's whole value goes with it."""
    assert strip_subject_info({"patientcode": value, "filetype": 0}) == {"filetype": 0}


def test_non_dict_values_and_non_string_keys_kept():
    stamp = datetime.date(2020, 1, 2)
    array = np.arange(3)
    meta = {1: "int key", None: "none key", "date": stamp, "array": array, "flag": True}
    out = strip_subject_info(meta)
    assert list(out) == [1, None, "date", "array", "flag"]
    assert out["date"] is stamp
    assert out["array"] is array


def test_non_dict_mappings_are_stripped():
    """Any Mapping, at the top or nested, comes back as a plain dict."""
    top = strip_subject_info(types.MappingProxyType({"patientcode": "x", "a": 1}))
    assert top == {"a": 1}
    assert type(top) is dict
    nested = strip_subject_info({"outer": types.MappingProxyType({"Sex": "F", "a": 1})})
    assert nested == {"outer": {"a": 1}}
    assert type(nested["outer"]) is dict


def test_input_is_never_modified():
    meta = {
        "patientcode": "P1",
        "recording_info": {"birthdate": "x", "startdate": "y"},
        "streams": [{"subject": "S1", "task": "rest"}],
    }
    before = copy.deepcopy(meta)
    out = strip_subject_info(meta)
    assert meta == before
    assert out["recording_info"] is not meta["recording_info"]
    assert out["streams"][0] is not meta["streams"][0]


def test_non_mapping_input_raises():
    with pytest.raises(TypeError, match="expects a mapping"):
        strip_subject_info([("patientcode", "P1")])  # type: ignore[arg-type]
