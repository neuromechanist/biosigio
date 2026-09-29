"""Real-recording regression for the STREAMED sidecar path (issue #127).

``test_real_data_nm000182.py`` pins the in-memory half of the unit fix on a real
recording; this pins the streaming half, on the dataset the disagreement was
actually reported on. NEMAR ``nm000183`` (Nejedly et al. 2020 SEEG
graphoelements, public) is 632 BrainVision runs, and exactly one of them --
``sub-012`` ``run-012``, 311 MiB -- sits above NEMAR's 256 MiB streaming
threshold. So the dataset genuinely straddled it: 631 runs converted in memory
and served the sidecar's ``uV``, this one converted by streaming and served
MNE's SI ``V``, six orders of magnitude apart inside one dataset, with the
per-channel ``type`` differing too (``SEEG`` against MNE's ``EEG`` guess).

The assertions are therefore about the run that streams: its store must carry
``uV`` on microvolt-scaled values, and must match an in-memory export of the
*same file* channel for channel.

Opt in and run it
-----------------

The recording is ~311 MB and is **not** committed. Set ``BIOSIGIO_REAL_DATA=1``
to opt in; without it the tests skip, so the default suite stays offline and
fast::

    BIOSIGIO_REAL_DATA=1 uv run pytest biosigio/tests/test_real_data_nm000183.py

Download, caching and the offline skip are :mod:`biosigio.tests.real_data`'s job
(cache: ``~/.cache/biosigio/real_data``, overridable with
``BIOSIGIO_REAL_DATA_CACHE``). A BrainVision recording is four files -- the
``.vhdr`` names its ``.eeg`` and ``.vmrk`` siblings by bare filename, and
``find_channels_tsv`` looks for the ``_channels.tsv`` next to the header -- so
all four are fetched under their own names into that one flat cache directory,
which is the adjacency MNE and BIDS both expect.

The in-memory comparison loads a 4.5-hour single-channel recording at float64
and peaks around 4.5 GB of RAM; that is the point of the comparison (it is the
path the streaming one has to agree with), not an oversight. The streaming
conversion of the same file stays bounded, which is why the threshold exists.

data.nemar.org returns an intermittent HTTP 500 on a small fraction of requests,
independent of file or client, so an occasional skip on a working network is
expected; re-run and the cached files make it moot.
"""

import numpy as np
import pytest

pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")
pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")

import zarr  # noqa: E402

from biosigio import Recording, stream_to_zarr  # noqa: E402
from biosigio.tests.real_data import fetch_real_recording  # noqa: E402
from biosigio.tests.test_zarr_stream_channels_tsv import (  # noqa: E402
    assert_stores_agree,
    channel_facts,
    dequantized,
)

# Pinned to a released version so the bytes cannot change under the assertions
# (the unversioned data.nemar.org path 404s).
BASE_URL = "https://data.nemar.org/nm000183/v1.0.2/sub-012/ieeg/"
STEM = "sub-012_task-MachineLearningEEG_run-012"
CHANNEL = "A12"
RESOLUTION_MICROVOLTS = 0.1  # "Ch1=A12,,0.1,µV" in the real header
# The per-modality rate caps the NEMAR converter passes to every export.
MODALITY_RATES = {"EEG": 250, "MEG": 250, "IEEG": 1000, "EMG": 1000}
# nemar-cli's should_stream() boundary: above this a recording converts through
# stream_to_zarr instead of Recording.from_file + to_zarr.
STREAMING_THRESHOLD_BYTES = 256 * 1024**2

# Suffix -> (size floor, sha256). The floor rejects a truncated download or an
# HTML error page cheaply; the digest is the real check.
#
# Every digest here describes bytes checked against nm000183's published v1.0.2
# manifest, not merely whatever this machine downloaded. The manifest carries two
# checksum kinds: `.vhdr`, `.vmrk` and `.eeg` are git-annex objects listed with
# sha256, which are these values verbatim (the `.vhdr` digest also appears as the
# ETag data.nemar.org serves); `channels.tsv` is git-tracked text listed with a
# git blob SHA-1, which `git hash-object` reproduced
# (32400d135ed6f54c6cbfe172967823c155b3fcb4) before its sha256 was taken from
# those same verified bytes.
FILES = {
    "_ieeg.vhdr": (700, "6078a1c7be383111c698b8d9338109e8ba6f6a0e17d2a31f94875f422541f066"),
    "_ieeg.vmrk": (231_000, "f8b68cfefa6d58a78a82f68f065a423f207c728813e555fb1a5feb5033f228f2"),
    "_channels.tsv": (100, "ecc11233c12a9749f3a40360d2c6ce26282af77be57968ba21c0b689197f9389"),
    "_ieeg.eeg": (
        326_000_000,
        "fa6d881b701001dca59c5ea110f97e1ab66e3403edccc2ae89ec9f9d9acaf6bd",
    ),
}


@pytest.fixture(scope="module")
def vhdr():
    """The cached ``.vhdr``, with its ``.eeg``/``.vmrk``/``channels.tsv`` beside it."""
    paths = {
        suffix: fetch_real_recording(
            f"{BASE_URL}{STEM}{suffix}", min_bytes=min_bytes, sha256=digest
        )
        for suffix, (min_bytes, digest) in FILES.items()
    }
    return paths["_ieeg.vhdr"]


@pytest.fixture(scope="module")
def streamed_store(vhdr, tmp_path_factory) -> str:
    """The run converted the way NEMAR converts anything over the threshold."""
    store = str(tmp_path_factory.mktemp("nm000183") / "streamed.zarr")
    return stream_to_zarr(
        str(vhdr),
        store,
        force_modality="IEEG",
        modality_rates=MODALITY_RATES,
        dtype="int16",
    )


def test_the_run_is_genuinely_over_the_streaming_threshold(vhdr):
    """Otherwise this file would not exercise the path the issue is about."""
    data_file = vhdr.with_name(f"{STEM}_ieeg.eeg")
    assert data_file.stat().st_size > STREAMING_THRESHOLD_BYTES


def test_streamed_store_carries_the_sidecars_microvolts(vhdr, streamed_store):
    """The reported defect: this store used to say ``V`` over volt-valued samples.

    The sidecar declares ``uV`` and ``SEEG``; MNE reads the pybv-written µV as SI
    volts and calls the channel EEG. Both have to move, and the values with them.

    The amplitude is checked against the ``.eeg`` block read straight off disk and
    scaled by the header's own resolution, so the ground truth does not go through
    MNE, biosigIO's importer, or the sidecar logic under test. The comparison is on
    distributions rather than samples because the store is an anti-aliased
    5000 -> 1000 Hz copy.
    """
    root = zarr.open_group(streamed_store, mode="r")
    assert list(root.attrs["channel_groups"]) == ["ieeg_1000hz"]

    fact = channel_facts(streamed_store)[CHANNEL]
    assert fact["unit"] == "uV"
    assert fact["channel_type"] == "SEEG"
    assert fact["modality"] == "IEEG"
    assert "bids_unit" not in root["ieeg_1000hz"].attrs["channels"][0]

    on_disk = np.fromfile(vhdr.with_name(f"{STEM}_ieeg.eeg"), dtype="<f4")
    native_std = float(on_disk.std(dtype=np.float64)) * RESOLUTION_MICROVOLTS
    native_mean = float(on_disk.mean(dtype=np.float64)) * RESOLUTION_MICROVOLTS
    del on_disk

    values = dequantized(streamed_store)[CHANNEL]
    # An SEEG amplitude in microvolts is an order-1e0 number here. Before the fix
    # this same store held ~1e-6 of it, under the label "V".
    assert 0.1 <= float(np.std(values)) <= 100.0
    assert float(np.std(values)) == pytest.approx(native_std, rel=0.05)
    assert float(np.mean(values)) == pytest.approx(native_mean, abs=0.05)

    assert root.attrs["channels_tsv_units"] == {
        "converted": 1,
        "relabelled": 0,
        "kept_importer_unit": 0,
        "units_column_present": True,
    }


def test_streamed_matches_an_in_memory_export_of_the_same_run(vhdr, streamed_store, tmp_path):
    """The whole point of #127: the threshold must not change the answer.

    Loads the 4.5-hour recording in memory (float64, ~3 GB) and exports it the
    way NEMAR exports anything under the threshold, then compares the two stores
    channel for channel -- unit, type, modality, group, quantization range, and
    the dequantized samples within a few int16 steps.
    """
    rec = Recording.from_file(str(vhdr))
    for label in rec.channels:
        rec.channels[label]["modality"] = "IEEG"
    in_memory = rec.to_zarr(
        str(tmp_path / "in_memory.zarr"), dtype="int16", modality_rates=MODALITY_RATES
    )
    del rec

    assert_stores_agree(streamed_store, in_memory)
