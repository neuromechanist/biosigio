"""Real-recording check that ``exclude_subject_info`` empties a store of subject
information, on a header written by real acquisition software rather than by a
test.

NEMAR ``nm000181`` (NMT: Neurodiagnostic Montage Template Scalp EEG, public,
CC BY-SA 4.0) is EDF+ whose patient field is populated. A read-only, value-free
scan of the dataset found the same patient name and birth date in every one of
its files, so the fields are placeholders, but they are non-empty: the EDF
importer copies the birth date into ``Recording.metadata``, and pyedflib's own
header carries the name as well. That is the shape of the stores a census of
NEMAR's Zarr copies found carrying patient code, birth date and sex.

Header values never reach the output. Every assertion compares a set of member
NAMES or a boolean computed before the ``assert``, because pytest's failure
explanation prints the operands of the asserted expression.

Opt in and run it
-----------------

The file is 39,608 bytes and is **not** committed, because it is a third-party
recording with a populated patient field. Set ``BIOSIGIO_REAL_DATA=1`` to opt
in; without it the tests skip, so the default suite stays offline::

    BIOSIGIO_REAL_DATA=1 uv run pytest biosigio/tests/test_real_data_nm000181.py

Download, caching and the offline skip are :mod:`biosigio.tests.real_data`'s
job (cache: ``~/.cache/biosigio/real_data``, overridable with
``BIOSIGIO_REAL_DATA_CACHE``). The sha256 below is the one nm000181's published
v1.0.0 manifest lists for this git-annex object.
"""

import pyedflib
import pytest

pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")

import zarr  # noqa: E402

from biosigio import (  # noqa: E402
    SUBJECT_INFO_EXCLUDED_ATTR,
    SUBJECT_INFO_KEYS,
    Recording,
    stream_to_zarr,
)
from biosigio.tests.real_data import fetch_real_recording  # noqa: E402
from biosigio.tests.test_zarr_subject_info import member_names  # noqa: E402

# Pinned to a released version so the bytes cannot change under the assertions.
URL = "https://data.nemar.org/nm000181/v1.0.0/sub-2109/eeg/sub-2109_task-clinical_eeg.edf"
SIZE = 39_608
SHA256 = "8a3b885d9bd00827c231a043c8a097c04c4e168956e0f97b20eb3b841c64cc59"
N_CHANNELS = 21
# The per-modality rate caps the NEMAR converter passes to every export.
MODALITY_RATES = {"EEG": 250, "MEG": 250, "IEEG": 1000, "EMG": 1000}
IN_MEMORY_TECHNICAL = {"startdate", "filetype", "number_of_signals", "file_duration"}
STREAMED_TECHNICAL = {"startdate", "number_of_signals", "streamed"}


class Store:
    """The facts a test asserts about one store, as names and booleans only."""

    def __init__(self, path: str):
        root = zarr.open_group(store=zarr.storage.LocalStore(path), mode="r")
        attrs = dict(root.attrs)
        meta = attrs["recording_metadata"]
        self.root_names = set(attrs)
        self.meta_names = set(meta)
        self.subject = member_names(meta) & SUBJECT_INFO_KEYS
        self.flag = attrs.get(SUBJECT_INFO_EXCLUDED_ATTR)
        self.meta = meta  # compared, never asserted on directly
        self.rest = {name: dict(root[name].attrs) for name in root.keys()}
        events = root["events"]
        for name in ("onset", "duration", "code"):
            if name in events.keys():
                self.rest[f"events/{name}"] = events[name][:].tolist()


@pytest.fixture(scope="module")
def edf():
    return str(fetch_real_recording(URL, min_bytes=SIZE, sha256=SHA256))


@pytest.fixture(scope="module")
def header(edf) -> dict:
    """pyedflib's header of the real file: what a caller copying the header passes."""
    reader = pyedflib.EdfReader(edf)
    try:
        return reader.getHeader()
    finally:
        reader.close()


def test_the_header_really_carries_subject_members(edf, header):
    """Otherwise the exclusion below would pass on an empty header."""
    populated = {k for k, v in header.items() if v not in ("", None)}
    imported = member_names(Recording.from_file(edf).metadata) & SUBJECT_INFO_KEYS
    assert {"patientname", "birthdate"} <= populated
    assert "birthdate" in imported


def test_in_memory_export(edf, tmp_path):
    """The path NEMAR takes for a file this size: from_file, then to_zarr."""
    rec = Recording.from_file(edf)
    options = {"dtype": "int16", "modality_rates": MODALITY_RATES}
    default = Store(rec.to_zarr(str(tmp_path / "default"), **options))
    off = Store(rec.to_zarr(str(tmp_path / "off"), exclude_subject_info=False, **options))
    on_path = rec.to_zarr(str(tmp_path / "on"), exclude_subject_info=True, **options)
    on = Store(on_path)

    off_is_default = off.meta == default.meta
    assert off_is_default, "exclude_subject_info=False changed recording_metadata"
    assert off.root_names == default.root_names
    assert "birthdate" in off.subject
    assert off.flag is None

    assert on.subject == set()
    assert on.flag is True
    assert IN_MEMORY_TECHNICAL <= on.meta_names
    assert on.root_names - off.root_names == {SUBJECT_INFO_EXCLUDED_ATTR}
    rest_unchanged = on.rest == off.rest
    assert rest_unchanged, "exclude_subject_info changed channel or event attributes"

    reread = Recording.from_file(on_path)
    reread_subject = member_names(reread.metadata) & SUBJECT_INFO_KEYS
    assert reread_subject == set()
    assert len(reread.channels) == N_CHANNELS


@pytest.mark.parametrize("caller", ["importer_metadata", "pyedflib_header"])
def test_streaming_export(edf, header, caller, tmp_path):
    """The bounded-memory path, with the subject members arriving through the
    caller's ``recording_metadata`` (the streaming source adds none of its own)."""
    meta = Recording.from_file(edf).metadata if caller == "importer_metadata" else dict(header)
    options = {
        "force_modality": "EEG",
        "modality_rates": MODALITY_RATES,
        "dtype": "int16",
        "recording_metadata": meta,
    }
    off = Store(stream_to_zarr(edf, str(tmp_path / "off"), **options))
    on_path = stream_to_zarr(edf, str(tmp_path / "on"), exclude_subject_info=True, **options)
    on = Store(on_path)

    expected = {"birthdate"} if caller == "importer_metadata" else {"birthdate", "patientname"}
    assert expected <= off.subject
    assert off.flag is None

    assert on.subject == set()
    assert on.flag is True
    assert STREAMED_TECHNICAL <= on.meta_names
    rest_unchanged = on.rest == off.rest
    assert rest_unchanged, "exclude_subject_info changed channel or event attributes"
    reread_subject = member_names(Recording.from_file(on_path).metadata) & SUBJECT_INFO_KEYS
    assert reread_subject == set()
