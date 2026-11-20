"""Real-recording check that a gapped EDF+D file from the NEMAR catalog is refused (issue #180).

The synthetic files in ``test_edfd_gaps.py`` prove the rule on bytes written to the
specification; this proves it on a file a real acquisition system wrote.

The file is **not** committed. Set ``BIOSIGIO_REAL_DATA=1`` to opt in; without it the
tests skip, so the default suite stays offline::

    BIOSIGIO_REAL_DATA=1 uv run pytest biosigio/tests/test_real_data_edfd_gaps.py

Download, caching and the offline skip are :mod:`biosigio.tests.real_data`'s job (cache:
``~/.cache/biosigio/real_data``, overridable with ``BIOSIGIO_REAL_DATA_CACHE``).

``REAL_EDFD_URL`` is the one thing to fill in: a direct URL to a NEMAR EDF+D recording
whose data records have a gap, pinned to a released dataset version so the bytes cannot
change under the assertions. ``REAL_EDFD_SHA256`` is its digest, once known. While the
URL is ``None`` the tests skip.
"""

from __future__ import annotations

import pytest

pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")

from biosigio import BiosigIOError, Recording, stream_to_zarr  # noqa: E402
from biosigio.tests.real_data import fetch_real_recording  # noqa: E402

REAL_EDFD_URL: str | None = None
REAL_EDFD_SHA256: str | None = None
REAL_EDFD_MIN_BYTES = 1


@pytest.fixture(scope="module")
def real_edfd_path() -> str:
    if REAL_EDFD_URL is None:
        pytest.skip("real-data test skipped: REAL_EDFD_URL is not set yet (biosigIO issue #180)")
    return str(
        fetch_real_recording(REAL_EDFD_URL, min_bytes=REAL_EDFD_MIN_BYTES, sha256=REAL_EDFD_SHA256)
    )


def _reserved_field(path: str) -> bytes:
    with open(path, "rb") as f:
        f.seek(192)
        return f.read(44)


def test_the_real_file_is_labeled_discontinuous(real_edfd_path):
    assert _reserved_field(real_edfd_path).upper().startswith((b"EDF+D", b"BDF+D"))


def test_importer_refuses_the_real_gapped_file(real_edfd_path):
    with pytest.raises(BiosigIOError) as info:
        Recording.from_file(real_edfd_path, importer="edf")
    err = info.value
    assert err.code == "discontinuous_recording"
    assert err.gaps and all(gap.length > 0 for gap in err.gaps)
    assert real_edfd_path in str(err)


def test_streaming_export_refuses_the_real_gapped_file(real_edfd_path, tmp_path):
    store = tmp_path / "store.zarr"
    with pytest.raises(BiosigIOError) as info:
        stream_to_zarr(real_edfd_path, str(store))
    assert info.value.code == "discontinuous_recording"
    assert not store.exists()
