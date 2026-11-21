"""Real-recording check that a gapped EDF+D file from the NEMAR catalog is refused (issue #180).

The synthetic files in ``test_edfd_gaps.py`` prove the rule on bytes written to the
specification; this proves it on a file a real acquisition system wrote.

The source is NEMAR ``nm000456`` v1.0.2, run 07 of ``sub-PennEPI02152`` (EDF+D, 47
signals, 2 s data records, 4,942 records, 236,753,856 bytes). It is far too large to
download for a test, and the only gapped files in the catalog are of that size, so the
tests fetch an EXCERPT of its real bytes with HTTP Range requests: the 12,288-byte
header and six consecutive data records. Only the header's number-of-data-records field
is patched afterwards (to 6), so the excerpt is a valid file.

* Records 4178 to 4183 straddle the file's one gap: record 4180 starts at 8360.542969 s
  and ends at 8362.542969 s, and record 4181 starts at 8396.445312 s, 33.902343 s later.
* Records 4170 to 4175 are contiguous, a real EDF+D control that must stay readable.

Nothing is committed. Set ``BIOSIGIO_REAL_DATA=1`` to opt in; without it, or offline, the
tests skip, so the default suite stays offline::

    BIOSIGIO_REAL_DATA=1 uv run pytest biosigio/tests/test_real_data_edfd_gaps.py

Download, caching and the skips are :func:`biosigio.tests.real_data.fetch_real_byte_ranges`'s
job (cache: ``~/.cache/biosigio/real_data``, overridable with ``BIOSIGIO_REAL_DATA_CACHE``).
The digests below pin the header bytes and each whole excerpt of the released v1.0.2 file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")

import zarr  # noqa: E402

from biosigio import BiosigIOError, Recording, stream_to_zarr  # noqa: E402
from biosigio.tests.real_data import fetch_real_byte_ranges  # noqa: E402

URL = (
    "https://data.nemar.org/nm000456/v1.0.2/sub-PennEPI02152/eeg/"
    "sub-PennEPI02152_task-monitoring_run-07_eeg.edf"
)
HEADER_BYTES = 12_288  # 256 * (47 signals + 1)
RECORD_BYTES = 47_904
N_RECORDS = 6
HEADER_SHA256 = "2753428681432130b5ffdf2a2a54fde7908e642bb3baef04593fbc49c732f0fa"
GAP_EXCERPT_FIRST_RECORD = 4178
GAP_EXCERPT_SHA256 = "3a86b7fee09d25a6fa56dad615f18c1f5a384464ac40307f8315821d2207e172"
CONTIGUOUS_EXCERPT_FIRST_RECORD = 4170
CONTIGUOUS_EXCERPT_SHA256 = "a67f55ed92cf940f0114ca834c4c3180352a8ef2d7168fad05c58699334a4846"
# The gap: record 4180 ends at 8360.542969 + 2 s; record 4181 starts at 8396.445312 s.
GAP_START_S = 8362.542969
GAP_LENGTH_S = 33.902343


def _excerpt(name: str, first_record: int, sha256: str, tmp_path: Path) -> str:
    """A valid private copy of the excerpt: header plus six records, record count patched."""
    start = HEADER_BYTES + first_record * RECORD_BYTES
    cached = fetch_real_byte_ranges(
        URL,
        [(0, HEADER_BYTES - 1), (start, start + N_RECORDS * RECORD_BYTES - 1)],
        filename=name,
        sha256=sha256,
    )
    data = bytearray(cached.read_bytes())
    assert hashlib.sha256(data[:HEADER_BYTES]).hexdigest() == HEADER_SHA256
    assert data[192:197] == b"EDF+D"
    data[236:244] = f"{N_RECORDS}".ljust(8).encode("ascii")
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


@pytest.fixture
def gapped(tmp_path) -> str:
    return _excerpt(
        "nm000456_run07_gap.edf", GAP_EXCERPT_FIRST_RECORD, GAP_EXCERPT_SHA256, tmp_path
    )


@pytest.fixture
def contiguous(tmp_path) -> str:
    return _excerpt(
        "nm000456_run07_contiguous.edf",
        CONTIGUOUS_EXCERPT_FIRST_RECORD,
        CONTIGUOUS_EXCERPT_SHA256,
        tmp_path,
    )


def _assert_the_real_gap(err: BiosigIOError, path: str) -> None:
    assert err.code == "discontinuous_recording"
    assert len(err.gaps) == 1
    assert err.gaps[0].start == pytest.approx(GAP_START_S, abs=1e-3)
    assert err.gaps[0].length == pytest.approx(GAP_LENGTH_S, abs=1e-3)
    assert path in str(err)


def test_importer_refuses_the_real_gapped_file(gapped):
    with pytest.raises(BiosigIOError) as info:
        Recording.from_file(gapped, importer="edf")
    _assert_the_real_gap(info.value, gapped)


def test_streaming_export_refuses_the_real_gapped_file(gapped, tmp_path):
    store = tmp_path / "store.zarr"
    with pytest.raises(BiosigIOError) as info:
        stream_to_zarr(gapped, str(store))
    _assert_the_real_gap(info.value, gapped)
    assert not store.exists()


def test_importer_reads_the_real_contiguous_edfd_file(contiguous):
    rec = Recording.from_file(contiguous, importer="edf")
    assert rec.metadata["edf_tolerant_read"] is True
    assert rec.metadata["edf_tolerant_read_reason"] == "discontinuous_datarecords"
    assert rec.metadata["edf_discontinuous_verified_contiguous"] is True
    assert rec.metadata["file_duration"] == pytest.approx(N_RECORDS * 2.0)


def test_streaming_export_reads_the_real_contiguous_edfd_file(contiguous, tmp_path):
    store = stream_to_zarr(contiguous, str(tmp_path / "store.zarr"))
    metadata = zarr.open(store, mode="r").attrs["recording_metadata"]
    assert metadata["edf_tolerant_read"] is True
    assert metadata["edf_discontinuous_verified_contiguous"] is True
