"""``strip_subject_info``: the removal behind ``exclude_subject_info``.

A Zarr store carries the signal, the events, channel names, types and units, and
technical recording metadata; subject and operator information belongs at dataset
scope. ``strip_subject_info`` removes the members named in ``SUBJECT_INFO_KEYS``
from a metadata mapping. The writers that call it are tested in
``test_zarr_subject_info.py``.
"""

import copy
import datetime

import numpy as np
import pytest

from biosigio import SUBJECT_INFO_KEYS, strip_subject_info

# --- strip_subject_info ---------------------------------------------------------

# Every spelling, by where it comes from. Pinned here as the requirement, so
# dropping one from SUBJECT_INFO_KEYS fails this test.
SPELLINGS_BY_SOURCE = {
    "EDF/BDF importer": [
        "patientcode",
        "gender",
        "birthdate",
        "patient_name",
        "patient_additional",
        "admincode",
        "technician",
        "equipment",
        "recording_additional",
    ],
    "pyedflib getHeader": ["patientname", "sex"],
    "EEGLAB importer": ["subject", "group", "comments"],
    "WFDB importer": ["comments"],
}


@pytest.mark.parametrize(
    "key",
    sorted({k for keys in SPELLINGS_BY_SOURCE.values() for k in keys}),
)
def test_every_importer_spelling_is_removed(key):
    assert strip_subject_info({key: "x", "number_of_signals": 2}) == {"number_of_signals": 2}


def test_constant_names_exactly_the_importer_spellings():
    """No member is removed that no importer produces."""
    expected = {k for keys in SPELLINGS_BY_SOURCE.values() for k in keys}
    assert expected == SUBJECT_INFO_KEYS


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


def test_matching_is_case_insensitive():
    meta = {"PatientCode": "P1", "BIRTHDATE": "x", "Sex": "F", "Group": "controls", "Rate": 1}
    assert strip_subject_info(meta) == {"Rate": 1}


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
