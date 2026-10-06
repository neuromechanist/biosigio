"""Subject information left out of a Zarr store on request (``exclude_subject_info``).

A store carries the signal, the events, channel names, types and units, and
technical recording metadata; subject information belongs at dataset scope. With
``exclude_subject_info=True`` both Zarr writers remove the members named in
``SUBJECT_INFO_KEYS`` (the removal itself is tested in ``test_subject_info.py``),
reduce ``source_file`` and ``bti_pdf_file`` to their final path component, and
mark the store. The default writes exactly what earlier releases wrote.

NO MOCKS: the writer tests convert EDF+ and BDF+ files written to disk by pyedflib
with every patient and recording header field set, and the committed fixtures
(WFDB record 100, whose header comments carry age, sex and medication; the EEGLAB
sets, whose ``EEG.subject`` is set; the MEG, BrainVision, XDF, OTB and Trigno
recordings). A real NEMAR recording is exercised in
``test_real_data_nm000181.py``.

The header values here are synthetic, written by this module, so some
assertions compare whole metadata dicts and a failure can print them. Only the
real-data module restricts itself to member names.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import io
import os
import pathlib

import numpy as np
import pyedflib
import pytest

from biosigio import SUBJECT_INFO_EXCLUDED_ATTR, Recording, is_subject_info_key
from biosigio.tabular_schema import metadata_to_mapping

from .subject_info_helpers import (
    member_names,
    subject_member_paths,
    subject_members,
    unclassified,
)

zarr = pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")

from biosigio import stream_to_zarr  # noqa: E402
from biosigio.exporters.zarr import ZarrExporter  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
EX = _REPO / "examples"
WFDB_100 = EX / "100.hea"
EEGLAB_SET = EX / "wristbandEMG_truncated.set"
BTI_DIR = EX / "bti/sub-01_task-test_meg"

# The header members the EDF/BDF importer copies into Recording.metadata when
# they are non-empty. The importer asks pyedflib for "patient_name", which
# pyedflib never returns, so the name is not among them.
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
# Root attributes biosigIO 1.2.10 wrote for a store with no channels.tsv
# applied. A later release may add attributes (additive is the normal path, see
# docs/formats/zarr.md), so the tests check these are present, not that they
# are all there is.
ROOT_ATTRS_1_2_10 = {
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
}
# Members stream_to_zarr adds to a caller's recording_metadata when absent.
STREAM_DEFAULTED = {"source_file", "number_of_signals", "streamed"}


def write_edf_with_header(
    path: str, file_type: int, labels: tuple[str, ...] = ("EEG Fz", "EEG Cz")
) -> None:
    """Write a real EDF+/BDF+ with every patient and recording header field set,
    and two annotations."""
    rate = 256.0
    t = np.arange(int(rate * 4)) / rate
    data = [(30.0 - 5 * i) * np.sin(2 * np.pi * (5 + i) * t) for i in range(len(labels))]
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
        for label in labels
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


def kept_after_exclude(meta: dict) -> dict:
    """What exclude_subject_info should leave of an encoded, flat metadata dict:
    subject members dropped, source paths reduced to their final component."""
    out = {k: v for k, v in meta.items() if not is_subject_info_key(k)}
    for key in ("source_file", "bti_pdf_file"):
        if isinstance(out.get(key), str):
            out[key] = pathlib.PurePath(out[key]).name
    return out


def small_recording() -> Recording:
    t = np.arange(1000)
    rec = Recording()
    rec.add_channel("C3", np.sin(t / 5.0), 250, "uV", "EEG")
    return rec


# --- in-memory writer (Recording.to_zarr / ZarrExporter.export) ----------------


def test_in_memory_default_and_off_write_what_1_2_10_wrote(header_file, tmp_path):
    rec = Recording.from_file(header_file)
    attrs_default = root_attrs(rec.to_zarr(str(tmp_path / "default")))
    attrs_off = root_attrs(rec.to_zarr(str(tmp_path / "off"), exclude_subject_info=False))

    assert ROOT_ATTRS_1_2_10 <= set(attrs_default)
    assert SUBJECT_INFO_EXCLUDED_ATTR not in attrs_default
    # 1.2.10 stored metadata_to_mapping(rec.metadata): nothing removed, the
    # source path whole.
    assert attrs_default["recording_metadata"] == metadata_to_mapping(rec.metadata)
    assert attrs_default["recording_metadata"]["source_file"] == header_file
    assert subject_members(attrs_default["recording_metadata"]) == EDF_IMPORTER_SUBJECT_KEYS
    assert without_timestamp(attrs_off) == without_timestamp(attrs_default)


def test_exporter_default_keeps_subject_members(header_file, tmp_path):
    """ZarrExporter.export without the option writes what 1.2.10 wrote."""
    rec = Recording.from_file(header_file)
    attrs = root_attrs(ZarrExporter.export(rec, str(tmp_path / "default")))
    assert SUBJECT_INFO_EXCLUDED_ATTR not in attrs
    assert attrs["recording_metadata"] == metadata_to_mapping(rec.metadata)
    assert subject_members(attrs["recording_metadata"]) == EDF_IMPORTER_SUBJECT_KEYS


def test_in_memory_exclude_drops_every_member(header_file, tmp_path):
    rec = Recording.from_file(header_file)
    before = copy.deepcopy(rec.metadata)
    off = rec.to_zarr(str(tmp_path / "off"))
    on = rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True)

    attrs_off, attrs_on = root_attrs(off), root_attrs(on)
    stored = attrs_on["recording_metadata"]
    assert subject_members(stored) == set()
    assert EDF_IMPORTER_SUBJECT_KEYS & member_names(stored) == set()
    assert attrs_on[SUBJECT_INFO_EXCLUDED_ATTR] is True
    assert EDF_TECHNICAL_KEYS <= set(stored)
    assert stored["source_file"] == os.path.basename(header_file)
    assert stored == kept_after_exclude(attrs_off["recording_metadata"])
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


def test_reexport_without_the_option_does_not_carry_the_flag(header_file, tmp_path):
    """The flag records that the option ran on that export; it is not metadata
    a re-export inherits."""
    rec = Recording.from_file(header_file)
    first = rec.to_zarr(str(tmp_path / "first"), exclude_subject_info=True)
    again = root_attrs(Recording.from_file(first).to_zarr(str(tmp_path / "again")))
    assert SUBJECT_INFO_EXCLUDED_ATTR not in again
    assert subject_members(again["recording_metadata"]) == set()


def test_nested_recording_info_from_an_older_store_is_stripped(tmp_path):
    """Metadata shaped like an older EDF store (fields under recording_info)
    loses them on re-export, and a store holding them still imports."""
    rec = small_recording()
    rec.set_metadata(
        "recording_info",
        {"startdate": datetime.datetime(2020, 1, 2), "patientcode": "P1", "Gender": "F"},
    )
    old = Recording.from_file(rec.to_zarr(str(tmp_path / "old")))
    assert subject_members(old.metadata) == {"patientcode", "Gender"}

    on = root_attrs(old.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert on["recording_metadata"]["recording_info"] == {
        "startdate": metadata_to_mapping({"d": datetime.datetime(2020, 1, 2)})["d"]
    }


def test_wfdb_header_comments_and_record_name_left_out(tmp_path):
    """MIT-BIH record 100's header comments hold the subject's age, sex and
    medication; the WFDB importer copies them as ``comments``."""
    rec = Recording.from_file(str(WFDB_100))
    assert {"comments", "record_name"} <= set(rec.metadata)
    off = root_attrs(rec.to_zarr(str(tmp_path / "off")))
    on = root_attrs(rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert {"comments", "record_name"} <= set(off["recording_metadata"])
    assert {"comments", "record_name"} & set(on["recording_metadata"]) == set()
    assert subject_members(on["recording_metadata"]) == set()
    assert {"sampling_frequency", "source_format"} <= set(on["recording_metadata"])


def test_eeglab_subject_and_provenance_left_out(tmp_path):
    rec = Recording.from_file(str(EEGLAB_SET))
    assert rec.metadata.get("subject")
    eeglab_members = {"subject", "group", "comments", "setname", "filename", "filepath"}
    off = root_attrs(rec.to_zarr(str(tmp_path / "off")))
    on = root_attrs(rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert eeglab_members <= set(off["recording_metadata"])
    assert eeglab_members & set(on["recording_metadata"]) == set()
    assert subject_members(on["recording_metadata"]) == set()
    assert {"srate", "nbchan", "trials", "device"} <= set(on["recording_metadata"])


# --- path-valued provenance -------------------------------------------------------


def test_bti_paths_reduced_to_final_component_in_both_writers(tmp_path):
    """A BTi recording records the directory and the processed-data file it read;
    under the option only their final components reach the store."""
    pytest.importorskip("mne", reason="BTi import requires the optional 'meg' extra (mne)")
    rec = Recording.from_file(str(BTI_DIR))
    assert os.path.isabs(rec.metadata["bti_pdf_file"])

    default = root_attrs(rec.to_zarr(str(tmp_path / "default")))["recording_metadata"]
    assert default["bti_pdf_file"] == rec.metadata["bti_pdf_file"]
    assert default["source_file"] == rec.metadata["source_file"]

    for name, store in (
        ("in-memory", rec.to_zarr(str(tmp_path / "mem"), exclude_subject_info=True)),
        (
            "streaming",
            stream_to_zarr(str(BTI_DIR), str(tmp_path / "st"), exclude_subject_info=True),
        ),
    ):
        meta = root_attrs(store)["recording_metadata"]
        assert meta["bti_pdf_file"] == "c,rfDC", name
        assert meta["source_file"] == BTI_DIR.name, name


def test_source_file_reduced_on_posix_and_windows_paths(tmp_path):
    rec = small_recording()
    for path, expected in (
        ("/home/alice/study/sub-01_eeg.edf", "sub-01_eeg.edf"),
        ("C:\\Users\\alice\\study\\sub-01_eeg.edf", "sub-01_eeg.edf"),
        ("/data/sub-01_task-rest_meg/", "sub-01_task-rest_meg"),
        ("sub-01_eeg.edf", "sub-01_eeg.edf"),
    ):
        rec.set_metadata("source_file", path)
        store = rec.to_zarr(str(tmp_path / "s"), exclude_subject_info=True)
        assert root_attrs(store)["recording_metadata"]["source_file"] == expected
        assert rec.metadata["source_file"] == path


# --- biosigIO's label-keyed maps --------------------------------------------------


def test_channels_named_like_members_keep_their_sidecar_entries(tmp_path):
    """Channels labeled Sex and Group, matched by a channels.tsv only ignoring
    case: the report under recording_metadata keeps both entries, and so agrees
    with the root copy, in both writers."""
    edf = str(tmp_path / "labels.edf")
    write_edf_with_header(edf, pyedflib.FILETYPE_EDFPLUS, labels=("Sex", "Group", "Fz"))
    tsv = tmp_path / "labels_channels.tsv"
    tsv.write_text("name\ttype\tunits\nsex\tEEG\tuV\nGROUP\tEEG\tuV\nFz\tEEG\tuV\n")
    expected_matches = {"sex": "Sex", "GROUP": "Group"}

    rec = Recording.from_file(edf, bids_channels=str(tsv))
    stores = {
        "in-memory": rec.to_zarr(str(tmp_path / "mem"), exclude_subject_info=True),
        "streaming": stream_to_zarr(
            edf, str(tmp_path / "st"), bids_channels=str(tsv), exclude_subject_info=True
        ),
    }
    for name, store in stores.items():
        attrs = root_attrs(store)
        nested = attrs["recording_metadata"]["channels_tsv_units"]
        assert nested == attrs["channels_tsv_units"], name
        assert nested["matched_case_insensitive"] == expected_matches, name
        assert subject_members(attrs["recording_metadata"]) == set(), name


# --- strip before and after encoding ---------------------------------------------


def _object_array_of_dicts() -> np.ndarray:
    arr = np.empty(1, dtype=object)
    arr[0] = {"Subject": "S1", "task": "rest"}
    return arr


def test_member_inside_an_encoded_value_is_stripped(header_file, tmp_path):
    """An object array of dicts only becomes a list of dicts when encoded, so the
    member inside it needs the strip after encoding."""
    rec = Recording.from_file(header_file)
    rec.set_metadata("sessions", _object_array_of_dicts())
    mem = root_attrs(rec.to_zarr(str(tmp_path / "mem"), exclude_subject_info=True))
    st = root_attrs(
        stream_to_zarr(
            header_file,
            str(tmp_path / "st"),
            recording_metadata={"sessions": _object_array_of_dicts()},
            exclude_subject_info=True,
        )
    )
    for name, attrs in (("in-memory", mem), ("streaming", st)):
        assert attrs["recording_metadata"]["sessions"] == [{"task": "rest"}], name


def test_unencodable_value_under_a_dropped_member_does_not_fail(header_file, tmp_path):
    """Bytes cannot be encoded; under a member being dropped they must not fail
    the export, which needs the strip before encoding. Without the option the
    export still refuses them, as 1.2.10 did."""
    rec = Recording.from_file(header_file)
    rec.set_metadata("patientcode", b"\x00raw")
    with pytest.raises(TypeError, match="bytes"):
        rec.to_zarr(str(tmp_path / "default"))
    mem = root_attrs(rec.to_zarr(str(tmp_path / "mem"), exclude_subject_info=True))
    st = root_attrs(
        stream_to_zarr(
            header_file,
            str(tmp_path / "st"),
            recording_metadata={"patientcode": b"\x00raw", "task": "rest"},
            exclude_subject_info=True,
        )
    )
    for name, attrs in (("in-memory", mem), ("streaming", st)):
        assert "patientcode" not in attrs["recording_metadata"], name
        assert attrs[SUBJECT_INFO_EXCLUDED_ATTR] is True, name
    assert st["recording_metadata"]["task"] == "rest"


# --- every importer key is classified --------------------------------------------


def _load_quietly(path: str, **kwargs) -> Recording:
    with contextlib.redirect_stdout(io.StringIO()):
        return Recording.from_file(path, **kwargs)


EEGLAB_MEMBERS = {"subject", "group", "comments", "setname", "filename", "filepath"}

# (fixture, relative path, or None for a file this module writes; extra kwargs;
# needs MNE; the subject-information members the importer emits, as dotted paths
# at any depth outside the label-keyed maps). The last column is the table the
# nested check compares against, so a list entry that matches a technical member
# anywhere in an importer's metadata fails here.
CLASSIFIED_FIXTURES = [
    ("pyedflib EDF+", None, {}, False, EDF_IMPORTER_SUBJECT_KEYS),
    ("pyedflib BDF+", None, {}, False, EDF_IMPORTER_SUBJECT_KEYS),
    ("neo", None, {}, False, set()),
    ("WFDB record 100", "100.hea", {}, False, {"comments", "record_name"}),
    ("EEGLAB set", "wristbandEMG_truncated.set", {}, False, EEGLAB_MEMBERS),
    (
        "EEGLAB BIDS set",
        "bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set",
        {},
        False,
        EEGLAB_MEMBERS,
    ),
    (
        "EDF BIDS EMG",
        "bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf",
        {},
        False,
        set(),
    ),
    ("XDF", "test.xdf", {}, False, set()),
    ("XDF multi-stream", "multi_stream_test.xdf", {}, False, set()),
    ("OTB Sessantaquattro", "one_sessantaquattro_truncated.otb+", {}, False, set()),
    ("OTB two Mouvi", "two_mouvi_truncated.otb+", {}, False, set()),
    ("Trigno", "truncated_trigno_sample.csv", {"importer": "trigno"}, False, set()),
    ("FIF", "bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif", {}, True, set()),
    ("CTF", "ctf/catch-alp-good-f.ds", {}, True, set()),
    ("BTi", "bti/sub-01_task-test_meg", {}, True, set()),
    ("KIT", "kit/sub-01_task-test_meg.sqd", {}, True, set()),
    ("BrainVision", "brainvision/sub-01_task-rest_eeg.vhdr", {}, True, set()),
]


def _write_neo_block(path: str) -> None:
    """A real neo file (NeoMatlabIO round-trips through scipy), two channels."""
    neo = pytest.importorskip("neo", reason="needs the optional 'neo' extra")
    import quantities as pq

    seg = neo.Segment()
    data = np.vstack([np.sin(np.arange(2000) / 7.0), np.cos(np.arange(2000) / 9.0)]).T
    seg.analogsignals.append(
        neo.AnalogSignal(data, units="uV", sampling_rate=1000 * pq.Hz, name="amp")
    )
    block = neo.Block()
    block.segments.append(seg)
    neo.io.NeoMatlabIO(filename=path).write_block(block)


@pytest.mark.parametrize(
    ("label", "rel", "kwargs", "needs_mne", "expected"),
    CLASSIFIED_FIXTURES,
    ids=[f[0] for f in CLASSIFIED_FIXTURES],
)
def test_every_importer_member_is_classified(label, rel, kwargs, needs_mne, expected, tmp_path):
    """Each top-level member an importer emits is either kept as technical or
    removed as subject information, and the members the list matches, at any
    depth, are exactly the expected ones: an unclassified member, or a list
    entry that hits a technical member somewhere inside the metadata, fails."""
    if needs_mne:
        pytest.importorskip("mne", reason="needs the optional 'meg' extra (mne)")
    if label == "neo":
        path = str(tmp_path / "rec.mat")
        _write_neo_block(path)
        kwargs = {"importer": "neo"}
    elif rel is None:
        bdf = "BDF" in label
        path = str(tmp_path / ("rec.bdf" if bdf else "rec.edf"))
        write_edf_with_header(path, pyedflib.FILETYPE_BDFPLUS if bdf else pyedflib.FILETYPE_EDFPLUS)
    else:
        path = str(EX / rel)
        if not os.path.exists(path):
            pytest.skip(f"{label} fixture missing")
    rec = _load_quietly(path, **kwargs)
    assert unclassified(rec.metadata) == set()
    assert subject_member_paths(rec.metadata) == expected
    stored = root_attrs(rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True))
    assert subject_member_paths(stored["recording_metadata"]) == set()
    # Only the matched members are gone: everything else the importer wrote,
    # nested members included, survives.
    kept = kept_after_exclude(metadata_to_mapping(rec.metadata))
    assert stored["recording_metadata"] == kept


@pytest.mark.parametrize("source", ["edf", "bti"])
def test_streaming_source_members_are_classified(source, header_file, tmp_path):
    """What the streaming writer adds from the source, with no caller dict."""
    if source == "bti":
        pytest.importorskip("mne", reason="BTi streaming requires the optional 'meg' extra (mne)")
    path = header_file if source == "edf" else str(BTI_DIR)
    meta = root_attrs(stream_to_zarr(path, str(tmp_path / "st")))["recording_metadata"]
    assert unclassified(meta) == set()
    assert subject_member_paths(meta) == set()


def test_every_pyedflib_header_field_but_the_start_date_is_subject_info(header_file):
    """A header field a future pyedflib adds fails here rather than leaking."""
    reader = pyedflib.EdfReader(header_file)
    try:
        fields = set(reader.getHeader())
    finally:
        reader.close()
    assert {f for f in fields - {"startdate"} if not is_subject_info_key(f)} == set()


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
    stored = attrs_default["recording_metadata"]
    assert ROOT_ATTRS_1_2_10 <= set(attrs_default)
    assert SUBJECT_INFO_EXCLUDED_ATTR not in attrs_default
    # Every member the caller passed is stored exactly as 1.2.10 encoded it,
    # and the writer adds only the members it defaults.
    assert metadata_to_mapping(meta).items() <= stored.items()
    assert set(stored) - set(meta) <= STREAM_DEFAULTED
    assert {"patientname", "sex", "Subject", "COMMENTS"} <= subject_members(stored)
    assert without_timestamp(attrs_off) == without_timestamp(attrs_default)


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
    assert EDF_IMPORTER_SUBJECT_KEYS & member_names(stored) == set()
    assert attrs_on[SUBJECT_INFO_EXCLUDED_ATTR] is True
    assert EDF_TECHNICAL_KEYS <= set(stored)
    assert {"source_file", "streamed"} <= set(stored)
    assert stored["source_file"] == os.path.basename(header_file)
    assert list(stored["edf_header"]) == ["startdate"]
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
    assert attrs["recording_metadata"]["source_file"] == os.path.basename(header_file)
    assert Recording.from_file(str(tmp_path / "on.zarr")).metadata["streamed"] is True
