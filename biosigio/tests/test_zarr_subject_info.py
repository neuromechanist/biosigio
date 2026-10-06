"""Subject information left out of a Zarr store on request (``exclude_subject_info``).

A store carries the signal, the events, channel names, types and units, and
technical recording metadata; subject and operator information belongs at dataset
scope. Both Zarr writers remove the members named in ``SUBJECT_INFO_KEYS`` when
``exclude_subject_info=True`` (the removal itself is tested in
``test_subject_info.py``). The default writes exactly what earlier releases wrote.

NO MOCKS: the writer tests convert EDF+ and BDF+ files written to disk by pyedflib
with every patient and recording header field set, the committed WFDB record 100
(MIT-BIH, whose header comments carry age, sex and medication), and the committed
EEGLAB fixture (whose ``EEG.subject`` is set). A real NEMAR recording is exercised
in ``test_real_data_nm000181.py``. Assertions compare member NAMES only, so a
failure never prints a header value.
"""

from __future__ import annotations

import copy
import datetime
import os
import pathlib

import numpy as np
import pyedflib
import pytest

from biosigio import SUBJECT_INFO_EXCLUDED_ATTR, SUBJECT_INFO_KEYS, Recording
from biosigio.tabular_schema import metadata_to_mapping

zarr = pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")

from biosigio import stream_to_zarr  # noqa: E402
from biosigio.exporters.zarr import ZarrExporter  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
WFDB_100 = _REPO / "examples/100.hea"
EEGLAB_SET = _REPO / "examples/wristbandEMG_truncated.set"

# The header members the EDF/BDF importer copies into Recording.metadata when
# they are non-empty. The importer reads pyedflib's "patientname" as
# "patient_name", which pyedflib never returns, so the name is not among them.
EDF_IMPORTER_SUBJECT_KEYS = {
    "patientcode",
    "gender",
    "birthdate",
    "patient_additional",
    "admincode",
    "technician",
    "equipment",
    "recording_additional",
}
# Technical members the importer writes beside them, which must survive.
EDF_TECHNICAL_KEYS = {
    "startdate",
    "filetype",
    "number_of_signals",
    "file_duration",
    "datarecord_duration",
}
# Root attributes biosigIO 1.2.10 wrote, in this order, for a store with no
# channels.tsv applied. The default (and exclude_subject_info=False) must write
# exactly these.
ROOT_ATTRS_1_2_10 = [
    "biosigio_version",
    "format",
    "format_version",
    "source_format",
    "modality_rates",
    "dtype",
    "view_downsample",
    "view_chunk_columns",
    "anti_alias_filter",
    "channel_groups",
    "recording_metadata",
    "created_utc",
    "note",
]


def member_names(value) -> set[str]:
    """Every mapping key at any depth, casefolded (an independent walk)."""
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            names.add(str(key).casefold())
            names |= member_names(item)
    elif isinstance(value, list | tuple):
        for item in value:
            names |= member_names(item)
    return names


def subject_members(value) -> set[str]:
    return member_names(value) & SUBJECT_INFO_KEYS


def write_edf_with_header(path: str, file_type: int) -> None:
    """Write a real two-channel EDF+/BDF+ with every patient and recording header
    field set, and two annotations."""
    rate = 256.0
    t = np.arange(int(rate * 4)) / rate
    data = [30.0 * np.sin(2 * np.pi * 5 * t), 20.0 * np.cos(2 * np.pi * 3 * t)]
    bdf = file_type == pyedflib.FILETYPE_BDFPLUS
    headers = [
        {
            "label": label,
            "dimension": "uV",
            "sample_frequency": rate,
            "physical_max": 100.0,
            "physical_min": -100.0,
            "digital_max": 8388607 if bdf else 32767,
            "digital_min": -8388608 if bdf else -32768,
            "prefilter": "n/a",
            "transducer": "n/a",
        }
        for label in ("EEG Fz", "EEG Cz")
    ]
    writer = pyedflib.EdfWriter(path, len(headers), file_type=file_type)
    try:
        writer.setSignalHeaders(headers)
        writer.setPatientCode("SUBJ-0042")
        writer.setPatientName("Test_Person")
        writer.setSex(1)
        writer.setBirthdate(datetime.date(1970, 1, 2))
        writer.setPatientAdditional("left_handed")
        writer.setAdmincode("ADM-7")
        writer.setTechnician("Operator_A")
        writer.setEquipment("Amplifier_1000")
        writer.setRecordingAdditional("pilot_session")
        writer.setStartdatetime(datetime.datetime(2020, 1, 2, 3, 4, 5))
        writer.writeSamples(data)
        writer.writeAnnotation(1.0, 0.5, "stim")
        writer.writeAnnotation(2.0, -1, "resp")
    finally:
        writer.close()


@pytest.fixture(params=["edf", "bdf"])
def header_file(request, tmp_path) -> str:
    """A real EDF+ or BDF+ file on disk with a fully populated header."""
    path = os.path.join(tmp_path, f"rec.{request.param}")
    file_type = pyedflib.FILETYPE_BDFPLUS if request.param == "bdf" else pyedflib.FILETYPE_EDFPLUS
    write_edf_with_header(path, file_type)
    return path


def root_attrs(store: str) -> dict:
    return dict(zarr.open_group(store=zarr.storage.LocalStore(store), mode="r").attrs)


def without_timestamp(attrs: dict) -> dict:
    return {k: v for k, v in attrs.items() if k != "created_utc"}


def group_and_event_attrs(store: str) -> dict:
    """Every channel group's and the events group's attributes, plus event arrays."""
    root = zarr.open_group(store=zarr.storage.LocalStore(store), mode="r")
    out = {}
    for name in root.keys():
        out[name] = dict(root[name].attrs)
    events = root["events"]
    for name in ("onset", "duration", "code"):
        if name in events.keys():
            out[f"events/{name}"] = np.asarray(events[name][:]).tolist()
    return out


# --- in-memory writer (Recording.to_zarr / ZarrExporter.export) ----------------


def test_in_memory_default_and_off_write_what_1_2_10_wrote(header_file, tmp_path):
    rec = Recording.from_file(header_file)
    default = rec.to_zarr(str(tmp_path / "default"))
    off = rec.to_zarr(str(tmp_path / "off"), exclude_subject_info=False)

    attrs_default, attrs_off = root_attrs(default), root_attrs(off)
    assert list(attrs_default) == ROOT_ATTRS_1_2_10
    assert without_timestamp(attrs_off) == without_timestamp(attrs_default)
    assert SUBJECT_INFO_EXCLUDED_ATTR not in attrs_off
    # 1.2.10 stored metadata_to_mapping(rec.metadata), nothing removed.
    assert attrs_off["recording_metadata"] == metadata_to_mapping(rec.metadata)
    assert subject_members(attrs_off["recording_metadata"]) == EDF_IMPORTER_SUBJECT_KEYS


def test_in_memory_exclude_drops_every_member(header_file, tmp_path):
    rec = Recording.from_file(header_file)
    before = copy.deepcopy(rec.metadata)
    off = rec.to_zarr(str(tmp_path / "off"))
    on = rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True)

    attrs_off, attrs_on = root_attrs(off), root_attrs(on)
    assert subject_members(attrs_on["recording_metadata"]) == set()
    assert attrs_on[SUBJECT_INFO_EXCLUDED_ATTR] is True
    assert EDF_TECHNICAL_KEYS <= set(attrs_on["recording_metadata"])
    kept_off = {
        k: v for k, v in attrs_off["recording_metadata"].items() if k not in SUBJECT_INFO_KEYS
    }
    assert attrs_on["recording_metadata"] == kept_off
    # Nothing else in the store changes: channels, geometry, events.
    assert group_and_event_attrs(on) == group_and_event_attrs(off)
    assert set(attrs_on) - set(attrs_off) == {SUBJECT_INFO_EXCLUDED_ATTR}
    # The Recording the caller holds is untouched.
    assert rec.metadata == before
    assert subject_members(rec.metadata) == EDF_IMPORTER_SUBJECT_KEYS


def test_exporter_called_directly_honors_the_option(header_file, tmp_path):
    rec = Recording.from_file(header_file)
    store = ZarrExporter.export(rec, str(tmp_path / "direct"), exclude_subject_info=True)
    attrs = root_attrs(store)
    assert subject_members(attrs["recording_metadata"]) == set()
    assert attrs[SUBJECT_INFO_EXCLUDED_ATTR] is True


def test_store_written_with_exclude_round_trips(header_file, tmp_path):
    rec = Recording.from_file(header_file)
    off = Recording.from_file(rec.to_zarr(str(tmp_path / "off")))
    on = Recording.from_file(rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))

    assert subject_members(off.metadata) == EDF_IMPORTER_SUBJECT_KEYS
    assert subject_members(on.metadata) == set()
    assert on.metadata["startdate"] == rec.metadata["startdate"]
    np.testing.assert_array_equal(on.signals.to_numpy(), off.signals.to_numpy())
    assert on.channels == off.channels
    assert on.events.equals(off.events)


def test_nested_recording_info_from_an_older_store_is_stripped(tmp_path):
    """Metadata shaped like an older EDF store (fields under recording_info)
    loses them on re-export, and a store holding them still imports."""
    t = np.arange(1000)
    rec = Recording()
    rec.add_channel("C3", np.sin(t / 5.0), 250, "uV", "EEG")
    rec.set_metadata(
        "recording_info",
        {"startdate": datetime.datetime(2020, 1, 2), "patientcode": "P1", "Gender": "F"},
    )
    old = Recording.from_file(rec.to_zarr(str(tmp_path / "old")))
    assert subject_members(old.metadata) == {"patientcode", "gender"}

    on = root_attrs(old.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert on["recording_metadata"]["recording_info"] == {
        "startdate": metadata_to_mapping({"d": datetime.datetime(2020, 1, 2)})["d"]
    }


def test_wfdb_header_comments_left_out(tmp_path):
    """MIT-BIH record 100's header comments hold the subject's age, sex and
    medication; the WFDB importer copies them as ``comments``."""
    rec = Recording.from_file(str(WFDB_100))
    assert "comments" in rec.metadata
    off = root_attrs(rec.to_zarr(str(tmp_path / "off")))
    on = root_attrs(rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert "comments" in off["recording_metadata"]
    assert subject_members(on["recording_metadata"]) == set()
    assert {"record_name", "sampling_frequency"} <= set(on["recording_metadata"])


def test_eeglab_subject_left_out(tmp_path):
    rec = Recording.from_file(str(EEGLAB_SET))
    assert rec.metadata.get("subject")
    off = root_attrs(rec.to_zarr(str(tmp_path / "off")))
    on = root_attrs(rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert "subject" in off["recording_metadata"]
    assert subject_members(on["recording_metadata"]) == set()
    assert {"srate", "nbchan", "setname"} <= set(on["recording_metadata"])


# --- streaming writer (stream_to_zarr) -----------------------------------------


def caller_metadata(header_file: str) -> dict:
    """What a caller might pass: the importer's metadata, pyedflib's raw header
    nested under its own key, and mixed-case members inside a list."""
    meta = dict(Recording.from_file(header_file).metadata)
    reader = pyedflib.EdfReader(header_file)
    try:
        meta["edf_header"] = reader.getHeader()
    finally:
        reader.close()
    meta["sessions"] = [{"Subject": "S1", "task": "rest"}, {"COMMENTS": "", "run": 2}]
    return meta


def test_streaming_default_and_off_write_what_1_2_10_wrote(header_file, tmp_path):
    meta = caller_metadata(header_file)
    default = stream_to_zarr(header_file, str(tmp_path / "default"), recording_metadata=meta)
    off = stream_to_zarr(
        header_file, str(tmp_path / "off"), recording_metadata=meta, exclude_subject_info=False
    )

    attrs_default, attrs_off = root_attrs(default), root_attrs(off)
    assert list(attrs_default) == ROOT_ATTRS_1_2_10
    assert without_timestamp(attrs_off) == without_timestamp(attrs_default)
    assert SUBJECT_INFO_EXCLUDED_ATTR not in attrs_off
    expected = dict(meta)
    expected.setdefault("number_of_signals", 2)
    expected.setdefault("streamed", True)
    assert attrs_off["recording_metadata"] == metadata_to_mapping(expected)
    assert {"patientname", "sex", "subject", "comments"} <= subject_members(
        attrs_off["recording_metadata"]
    )


def test_streaming_exclude_strips_caller_and_source_metadata(header_file, tmp_path):
    meta = caller_metadata(header_file)
    before = copy.deepcopy(meta)
    off = stream_to_zarr(header_file, str(tmp_path / "off"), recording_metadata=meta)
    on = stream_to_zarr(
        header_file, str(tmp_path / "on"), recording_metadata=meta, exclude_subject_info=True
    )

    attrs_off, attrs_on = root_attrs(off), root_attrs(on)
    stored = attrs_on["recording_metadata"]
    assert subject_members(stored) == set()
    assert attrs_on[SUBJECT_INFO_EXCLUDED_ATTR] is True
    assert EDF_TECHNICAL_KEYS <= set(stored)
    assert {"source_file", "streamed"} <= set(stored)
    assert "startdate" in stored["edf_header"]
    assert stored["sessions"] == [{"task": "rest"}, {"run": 2}]
    assert group_and_event_attrs(on) == group_and_event_attrs(off)
    assert set(attrs_on) - set(attrs_off) == {SUBJECT_INFO_EXCLUDED_ATTR}
    assert meta == before


def test_streaming_exclude_without_caller_metadata(header_file, tmp_path):
    """With no caller dict the source contributes only technical members, and
    the flag is still written."""
    attrs = root_attrs(stream_to_zarr(header_file, str(tmp_path / "on"), exclude_subject_info=True))
    assert subject_members(attrs["recording_metadata"]) == set()
    assert attrs[SUBJECT_INFO_EXCLUDED_ATTR] is True
    assert Recording.from_file(str(tmp_path / "on.zarr")).metadata["streamed"] is True
