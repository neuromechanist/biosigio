"""EEGLAB MATLAB v7.3 (HDF5) ``.set`` reading (issue #113) -- no mocks.

h5py can WRITE a v7.3-shaped ``.set`` file, so every test here synthesizes a
real HDF5 container in ``tmp_path`` -- with the same on-disk shape as a real
MATLAB v7.3 export (a 512-byte MAT text header ahead of the HDF5 superblock,
a top-level ``EEG`` group holding scalar fields directly, and struct-array
fields like ``chanlocs``/``event`` stored as arrays of HDF5 object references
into a ``#refs#`` group) -- and reads it back through
``EEGLABImporter``/``Recording.from_file``, exactly as a real MATLAB-saved
v7.3 ``.set`` would be read. This is real HDF5 I/O, not a stand-in for it.

Two on-disk details are load-bearing and worth calling out because they are
easy to get wrong when hand-building an HDF5 file that must still "look like"
a MATLAB MAT-file to both a magic-byte sniff and an HDF5 library:

* The HDF5 superblock signature must start at a power-of-two byte offset
  h5py/libhdf5 actually probes for (0, then 512, 1024, ...); a plain 128-byte
  MAT header immediately followed by the HDF5 body does NOT open. Real
  MATLAB v7.3 files pad the header out to 512 bytes for this reason, so the
  fixture builder does too (see ``_HEADER_SIZE`` below).
* MATLAB's HDF5 writer stores struct-array fields (``chanlocs.labels``,
  ``event.type``, ...) as arrays of object references into ``#refs#``, one
  per element, and stores char data as arrays of Unicode code-point integers
  (not an HDF5 string type). The fixture builder mirrors both, matching the
  structure verified against a real affected file in issue #113.

Skips cleanly if the optional ``hdf5`` extra (h5py) is not installed.
"""

import os

import numpy as np
import pytest
import scipy.io

h5py = pytest.importorskip(
    "h5py", reason="v7.3 .set reading requires the optional 'hdf5' extra (h5py)"
)

from ..exceptions import (  # noqa: E402
    CorruptFileError,
    FileReadError,
    NotContinuousRecordingError,
    is_resource_exhaustion,
)
from ..importers.eeglab import EEGLABImporter, _is_matlab_v73  # noqa: E402

_HEADER_SIZE = 512  # MAT header text + padding; see module docstring


def _wrap_matlab_v73_header(body: bytes) -> bytes:
    """Prefix an HDF5 byte string with a MATLAB v7.3 MAT-file text header.

    Shared by ``_write_v73_set`` and any fixture that needs a v7.3-magic file
    around a hand-built HDF5 body (e.g. one with no top-level ``EEG`` group,
    to exercise the "v7.3 magic but not an EEGLAB set" error path).
    """
    header_text = b"MATLAB 7.3 MAT-file, Platform: biosigio-test"
    header = header_text[:116].ljust(116, b"\x20")
    subsystem_offset = b"\x00" * 8
    version = (0x0200).to_bytes(2, "little")  # major=2 -> v7.3, per scipy's decoder
    endian_indicator = b"IM"  # little-endian
    prefix = header + subsystem_offset + version + endian_indicator
    assert len(prefix) == 128
    padded_prefix = prefix + b"\x00" * (_HEADER_SIZE - len(prefix))
    return padded_prefix + body


def _write_v73_set(
    path,
    *,
    nbchan,
    pnts,
    srate,
    trials=1,
    data=None,
    data_filename=None,
    labels=None,
    types=None,
    x=None,
    events=None,
    header_nbchan=None,
    already_channel_major=False,
    xmin=None,
    xmax=None,
    setname=None,
    subject=None,
    group_name=None,
    condition=None,
    session=None,
    comments=None,
    flat=False,
    empty_chanlocs=False,
    empty_event=False,
):
    """Write a MATLAB-v7.3-shaped ``.set``: a MAT text header + an HDF5 body.

    Args:
        path: Destination ``.set`` path.
        nbchan: ``EEG.nbchan`` (written as a 1x1 float, matching real files).
        pnts: ``EEG.pnts``.
        srate: ``EEG.srate`` (None leaves the field out).
        trials: ``EEG.trials`` (>1 marks the file epoched).
        data: ``(nbchan, pnts)`` channel-major array. Written to HDF5
            transposed to ``(pnts, nbchan)`` -- the orientation h5py/MATLAB
            v7.3 actually uses -- so the importer's un-transpose is exercised
            for real, not assumed.
        data_filename: If given (instead of ``data``), ``EEG.data`` is written
            as a char array holding this filename, exercising the
            companion-``.fdt`` path.
        labels: Per-channel label strings for ``chanlocs.labels`` (a None
            entry is written as MATLAB's empty ``[]``).
        types: Per-channel type strings for ``chanlocs.type``.
        x: Per-channel ``chanlocs.X`` floats.
        events: List of ``{"latency": float, "type": str}`` dicts for
            ``EEG.event`` (a None type is written as MATLAB's empty ``[]``).
        header_nbchan: If given, overrides the ``nbchan`` value actually
            written to the header (default: ``nbchan``), for exercising a
            header/data mismatch without changing the data matrix itself.
        already_channel_major: If True, writes ``data`` to HDF5 WITHOUT the
            usual transpose -- i.e. already channel-major on disk -- to
            exercise the orientation cross-check's "don't re-transpose an
            already channel-major array" branch. Real v7.3 files are always
            sample-major (see the module docstring); this is purely a
            defensive-code exercise, not a claim about real files.
        xmin, xmax: ``EEG.xmin``/``EEG.xmax`` floats, written directly (not
            through ``#refs#`` -- these are top-level EEG scalar fields, not
            struct-array elements).
        setname, subject, group_name, condition, session, comments: Top-level
            EEG string metadata fields, written as direct char-code (uint16)
            datasets, same as the ``data_filename`` char array.
        flat: If True, writes the struct's fields directly at the HDF5 root
            with no ``EEG`` group, the flat layout EEGLAB can also save.
        empty_chanlocs, empty_event: If True, write ``EEG.chanlocs = []`` /
            ``EEG.event = []`` the way MATLAB does: not a struct group but a
            ``uint64`` dims dataset flagged ``MATLAB_empty``.
    """
    inner_path = path + ".inner"
    with h5py.File(inner_path, "w") as f:
        refs = f.create_group("#refs#")
        if flat:
            eeg = f
        else:
            eeg = f.create_group("EEG")
            eeg.attrs["MATLAB_class"] = np.bytes_(b"struct")
        eeg.create_dataset("nbchan", data=np.array([[float(header_nbchan or nbchan)]]))
        if srate is not None:
            eeg.create_dataset("srate", data=np.array([[float(srate)]]))
        eeg.create_dataset("pnts", data=np.array([[float(pnts)]]))
        eeg.create_dataset("trials", data=np.array([[float(trials)]]))
        if xmin is not None:
            eeg.create_dataset("xmin", data=np.array([[float(xmin)]]))
        if xmax is not None:
            eeg.create_dataset("xmax", data=np.array([[float(xmax)]]))
        for field_name, value in (
            ("setname", setname),
            ("subject", subject),
            ("group", group_name),
            ("condition", condition),
            ("session", session),
            ("comments", comments),
        ):
            if value is not None:
                eeg.create_dataset(
                    field_name, data=np.array([ord(c) for c in value], dtype=np.uint16)
                )

        ref_counter = [0]

        def new_ref(*, text=None, number=None):
            ref_counter[0] += 1
            name = f"r{ref_counter[0]}"
            if text is not None:
                codes = np.array([ord(c) for c in text], dtype=np.uint16)
                ds = refs.create_dataset(name, data=codes)
                ds.attrs["MATLAB_class"] = np.bytes_(b"char")
            else:
                ds = refs.create_dataset(name, data=np.array([[float(number)]]))
                ds.attrs["MATLAB_class"] = np.bytes_(b"double")
            return ds.ref

        def empty_ref():
            # MATLAB stores an empty `[]` as a uint64 dims dataset flagged
            # MATLAB_empty (real EEGLAB files do this for unset X/Y/Z, and for
            # an event type or channel label left empty).
            ref_counter[0] += 1
            ds = refs.create_dataset(f"e{ref_counter[0]}", data=np.array([0, 0], dtype=np.uint64))
            ds.attrs["MATLAB_class"] = np.bytes_(b"double")
            ds.attrs["MATLAB_empty"] = np.uint8(1)
            return ds.ref

        def text_ref(text):
            return empty_ref() if text is None else new_ref(text=text)

        if data_filename is not None:
            codes = np.array([ord(c) for c in data_filename], dtype=np.uint16)
            eeg.create_dataset("data", data=codes)
        elif data is not None:
            # HDF5/v7.3 stores the matrix TRANSPOSED relative to MATLAB
            # (samples x channels); writing `.T` here reproduces that
            # real-world orientation instead of writing the channel-major
            # shape the rest of biosigio expects. `already_channel_major`
            # opts out, for the one test exercising the defensive
            # "don't re-transpose an already channel-major array" branch.
            on_disk = np.asarray(data) if already_channel_major else np.asarray(data).T
            eeg.create_dataset("data", data=on_disk.astype(np.float32))

        if labels is not None:
            chanlocs = eeg.create_group("chanlocs")
            chanlocs.attrs["MATLAB_class"] = np.bytes_(b"struct")
            chanlocs.create_dataset(
                "labels",
                data=np.array([text_ref(lbl) for lbl in labels], dtype=h5py.ref_dtype),
            )
            if types is not None:
                chanlocs.create_dataset(
                    "type",
                    data=np.array([new_ref(text=t) for t in types], dtype=h5py.ref_dtype),
                )
            if x == "empty":
                chanlocs.create_dataset(
                    "X",
                    data=np.array([empty_ref() for _ in labels], dtype=h5py.ref_dtype),
                )
            elif x is not None:
                chanlocs.create_dataset(
                    "X",
                    data=np.array([new_ref(number=v) for v in x], dtype=h5py.ref_dtype),
                )

        for field_name, empty in (("chanlocs", empty_chanlocs), ("event", empty_event)):
            if empty:
                ds = eeg.create_dataset(field_name, data=np.array([0, 0], dtype=np.uint64))
                ds.attrs["MATLAB_class"] = np.bytes_(b"struct")
                ds.attrs["MATLAB_empty"] = np.uint8(1)

        if events is not None:
            event_grp = eeg.create_group("event")
            event_grp.attrs["MATLAB_class"] = np.bytes_(b"struct")
            event_grp.create_dataset(
                "latency",
                data=np.array([new_ref(number=e["latency"]) for e in events], dtype=h5py.ref_dtype),
            )
            event_grp.create_dataset(
                "type",
                data=np.array([text_ref(e["type"]) for e in events], dtype=h5py.ref_dtype),
            )

    with open(inner_path, "rb") as fh:
        body = fh.read()
    os.remove(inner_path)

    with open(path, "wb") as fh:
        fh.write(_wrap_matlab_v73_header(body))


def test_v73_magic_sniff_detects_v73(tmp_path):
    """`_is_matlab_v73` recognizes a v7.3 file by its leading header text."""
    path = str(tmp_path / "v73.set")
    _write_v73_set(path, nbchan=2, pnts=4, srate=100.0, data=np.zeros((2, 4)))
    assert _is_matlab_v73(path) is True


def test_v73_magic_sniff_does_not_misfire_on_classic_set(tmp_path):
    """A classic (v5/v7) `.set` is never mistaken for v7.3, and still reads.

    Both forms share the `.set` extension, so this is the one thing the sniff
    must never get backwards: a real, healthy classic file must not be routed
    to the HDF5 reader (which would fail to open it outright).
    """
    path = str(tmp_path / "classic.set")
    scipy.io.savemat(
        path,
        {
            "nbchan": np.array([[2]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[4]]),
            "srate": np.array([[100]]),
            "data": np.zeros((2, 4), dtype=np.float32),
        },
    )
    assert _is_matlab_v73(path) is False

    # Regression: dispatch still lands on the unchanged classic (loadmat) path.
    rec = EEGLABImporter().load(path)
    assert rec.signals.shape == (4, 2)


def test_v73_channel_count_and_sampling_rate(tmp_path):
    path = str(tmp_path / "basic.set")
    nbchan, pnts, srate = 5, 40, 250.0
    data = np.zeros((nbchan, pnts), dtype=np.float32)
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=data,
        labels=[f"Ch{i + 1}" for i in range(nbchan)],
    )

    rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)
    assert len(rec.channels) == nbchan
    assert rec.get_metadata("srate") == srate
    assert rec.get_metadata("nbchan") == nbchan
    assert rec.get_metadata("device") == "EEGLAB"
    for info in rec.channels.values():
        assert info["sample_frequency"] == srate
        assert info["physical_dimension"] == "uV"
        assert info["prefilter"] == "n/a"


def test_v73_metadata_fields_extracted(tmp_path):
    """Top-level EEG scalar/string metadata fields load into `rec.metadata`,
    the same fields the classic path's `_extract_metadata` populates."""
    path = str(tmp_path / "metadata.set")
    nbchan, pnts, srate = 2, 10, 100.0
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=np.zeros((nbchan, pnts), dtype=np.float32),
        xmin=0.0,
        xmax=pnts / srate,
        setname="v73_test",
        subject="sub-01",
        group_name="controls",
        condition="rest",
        session="001",
        comments="synthesized fixture",
    )

    rec = EEGLABImporter().load(path)

    assert rec.get_metadata("xmin") == pytest.approx(0.0)
    assert rec.get_metadata("xmax") == pytest.approx(pnts / srate)
    assert rec.get_metadata("setname") == "v73_test"
    assert rec.get_metadata("subject") == "sub-01"
    assert rec.get_metadata("group") == "controls"
    assert rec.get_metadata("condition") == "rest"
    assert rec.get_metadata("session") == "001"
    assert rec.get_metadata("comments") == "synthesized fixture"


def _assert_not_eeglab_error(path):
    """The typed, terminal "not an EEGLAB dataset" error: a FileReadError, but
    never CorruptFileError, since the HDF5 container itself is healthy."""
    with pytest.raises(FileReadError) as info:
        EEGLABImporter().load(path)
    assert not isinstance(info.value, CorruptFileError)
    assert type(info.value) is FileReadError
    assert "is not an EEGLAB v7.3 dataset" in str(info.value)
    assert "corrupt" not in str(info.value).lower()


def test_v73_non_eeg_top_level_raises_not_eeglab_error(tmp_path):
    """A v7.3-magic file that isn't an EEGLAB set (no top-level `EEG` group)
    raises a typed read error, not an unhandled crash and not a corrupt-file
    verdict on a valid HDF5 file."""
    path = str(tmp_path / "not_eeglab.set")
    inner_path = path + ".inner"
    with h5py.File(inner_path, "w") as f:
        f.create_group("SomeOtherMatlabVariable")
    with open(inner_path, "rb") as fh:
        body = fh.read()
    os.remove(inner_path)
    with open(path, "wb") as fh:
        fh.write(_wrap_matlab_v73_header(body))

    assert _is_matlab_v73(path) is True  # magic still says v7.3
    _assert_not_eeglab_error(path)


def test_v73_flat_root_struct_loads(tmp_path):
    """EEGLAB's flat v7.3 layout (fields at the HDF5 root, no `EEG` group)
    is valid and loads, not reported as corrupt."""
    path = str(tmp_path / "flat.set")
    nbchan, pnts, srate = 3, 17, 250.0
    data = np.zeros((nbchan, pnts), dtype=np.float32)
    data[0] = np.arange(pnts)
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=data,
        labels=["Fz", "Cz", "Pz"],
        events=[{"latency": 11, "type": "stim"}],
        flat=True,
    )

    rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)
    assert list(rec.signals.columns) == ["Fz", "Cz", "Pz"]
    assert rec.get_metadata("srate") == srate
    assert rec.get_metadata("nbchan") == nbchan
    np.testing.assert_array_equal(rec.signals["Fz"].to_numpy(), np.arange(pnts))
    assert list(rec.events["description"]) == ["stim"]


def test_v73_flat_root_struct_companion_fdt_resolved(tmp_path):
    """The flat layout still resolves the sibling `.fdt` over a stale
    embedded name, same as the `EEG`-group layout."""
    nbchan, pnts, srate = 2, 30, 100.0
    data = np.arange(nbchan * pnts, dtype=np.float32).reshape(nbchan, pnts)
    data.flatten(order="F").tofile(str(tmp_path / "sub-01_eeg.fdt"))

    set_path = str(tmp_path / "sub-01_eeg.set")
    _write_v73_set(
        set_path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data_filename="original_name_before_bids.fdt",
        flat=True,
    )

    rec = EEGLABImporter().load(set_path)

    assert rec.metadata["eeglab_fdt_recovered"] == {
        "referenced": "original_name_before_bids.fdt",
        "used": "sub-01_eeg.fdt",
    }

    assert rec.signals.shape == (pnts, nbchan)
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_array_equal(rec.signals[label].to_numpy(), data[i])


def test_v73_eeg_group_wins_over_flat_root_fields(tmp_path):
    """With both an `EEG` group and flat root fields present, and disagreeing,
    every value comes from the `EEG` group."""
    path = str(tmp_path / "both_layouts.set")
    inner_path = path + ".inner"
    eeg_data = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    with h5py.File(inner_path, "w") as f:
        for group, (nbchan, srate, pnts, data) in (
            (f.create_group("EEG"), (3, 250.0, 8, eeg_data)),
            (f, (2, 100.0, 5, np.ones((2, 5), dtype=np.float32))),
        ):
            group.create_dataset("nbchan", data=np.array([[float(nbchan)]]))
            group.create_dataset("srate", data=np.array([[srate]]))
            group.create_dataset("pnts", data=np.array([[float(pnts)]]))
            group.create_dataset("trials", data=np.array([[1.0]]))
            group.create_dataset("data", data=data.T)  # v7.3 stores samples x channels
    with open(inner_path, "rb") as fh:
        body = fh.read()
    os.remove(inner_path)
    with open(path, "wb") as fh:
        fh.write(_wrap_matlab_v73_header(body))

    rec = EEGLABImporter().load(path)

    assert rec.get_metadata("srate") == 250.0
    assert rec.get_metadata("nbchan") == 3
    assert rec.get_metadata("pnts") == 8
    assert rec.signals.shape == (8, 3)
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_array_equal(rec.signals[label].to_numpy(), eeg_data[i])


@pytest.mark.parametrize("flat", [False, True])
def test_v73_matlab_empty_chanloc_field_is_no_value(tmp_path, flat):
    """An empty `[]` chanlocs field (MATLAB_empty uint64 marker) is "no value".

    Real EEGLAB files leave X/Y/Z empty when electrode positions are unset;
    the marker used to be decoded as a NUL-character string and then failed
    float conversion, making the whole file unreadable.
    """
    path = str(tmp_path / "empty_xyz.set")
    labels = ["Fz", "Cz", "Pz"]
    _write_v73_set(
        path,
        nbchan=3,
        pnts=10,
        srate=250.0,
        data=np.arange(30, dtype=np.float32).reshape(3, 10),
        labels=labels,
        types=["EEG"] * 3,
        x="empty",
        flat=flat,
    )
    rec = EEGLABImporter().load(path)
    assert list(rec.channels) == labels
    assert rec.signals.shape == (10, 3)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(
    ("empty_chanlocs", "empty_event"), [(True, False), (False, True), (True, True)]
)
def test_v73_empty_top_level_chanlocs_or_event_loads(tmp_path, flat, empty_chanlocs, empty_event):
    """``EEG.chanlocs = []`` / ``EEG.event = []`` is a MATLAB_empty dataset, not a
    struct group; it reads as no channels described / no events, never a crash."""
    path = str(tmp_path / "empty_top.set")
    data = np.arange(20, dtype=np.float32).reshape(2, 10)
    _write_v73_set(
        path,
        nbchan=2,
        pnts=10,
        srate=100.0,
        data=data,
        labels=None if empty_chanlocs else ["Fz", "Cz"],
        events=None if empty_event else [{"latency": 3, "type": "stim"}],
        flat=flat,
        empty_chanlocs=empty_chanlocs,
        empty_event=empty_event,
    )
    with h5py.File(path, "r") as f:
        root = f if flat else f["EEG"]
        for field_name, empty in (("chanlocs", empty_chanlocs), ("event", empty_event)):
            if empty:  # the fixture really has MATLAB's on-disk shape
                assert isinstance(root[field_name], h5py.Dataset)
                assert root[field_name].attrs["MATLAB_empty"] == 1

    rec = EEGLABImporter().load(path)

    expected_labels = ["Channel1", "Channel2"] if empty_chanlocs else ["Fz", "Cz"]
    assert list(rec.channels) == expected_labels
    np.testing.assert_array_equal(rec.signals[expected_labels[1]].to_numpy(), data[1])
    assert list(rec.events["description"]) == ([] if empty_event else ["stim"])


@pytest.mark.parametrize("field_name", ["chanlocs", "event"])
@pytest.mark.parametrize("kind", ["numeric", "char"])
def test_v73_non_empty_non_struct_chanlocs_or_event_is_a_read_error(tmp_path, field_name, kind):
    """A NON-empty numeric or char dataset where EEGLAB writes a struct array
    is not "no channels"/"no events": it is a typed read error naming the
    field, never read silently as none (and never worded as corrupt)."""
    path = str(tmp_path / "odd_field.set")
    _write_v73_set(path, nbchan=2, pnts=10, srate=100.0, data=np.zeros((2, 10)))
    inner_path = path + ".inner"
    with open(path, "rb") as fh:
        body = fh.read()[_HEADER_SIZE:]
    with open(inner_path, "wb") as fh:
        fh.write(body)
    with h5py.File(inner_path, "a") as f:
        if kind == "numeric":
            ds = f["EEG"].create_dataset(field_name, data=np.array([[1.0, 2.0, 3.0]]))
            ds.attrs["MATLAB_class"] = np.bytes_(b"double")
        else:
            ds = f["EEG"].create_dataset(
                field_name, data=np.array([ord(c) for c in "Fz"], dtype=np.uint16)
            )
            ds.attrs["MATLAB_class"] = np.bytes_(b"char")
    with open(inner_path, "rb") as fh:
        body = fh.read()
    os.remove(inner_path)
    with open(path, "wb") as fh:
        fh.write(_wrap_matlab_v73_header(body))

    with pytest.raises(FileReadError, match=f"/EEG/{field_name}") as info:
        EEGLABImporter().load(path)
    assert type(info.value) is FileReadError
    assert "corrupt" not in str(info.value).lower()


def test_v73_flat_root_without_pnts_is_not_accepted(tmp_path):
    """The flat-root guard needs pnts too: nbchan/srate/data alone are not a dataset."""
    path = str(tmp_path / "no_pnts.set")
    inner_path = path + ".inner"
    with h5py.File(inner_path, "w") as f:
        f.create_dataset("nbchan", data=np.array([[2.0]]))
        f.create_dataset("srate", data=np.array([[100.0]]))
        f.create_dataset("data", data=np.zeros((4, 2), dtype=np.float32))
    with open(inner_path, "rb") as fh:
        body = fh.read()
    os.remove(inner_path)
    with open(path, "wb") as fh:
        fh.write(_wrap_matlab_v73_header(body))

    _assert_not_eeglab_error(path)


def test_v73_root_without_core_fields_raises_not_eeglab_error(tmp_path):
    """A root with some struct-like fields but not all of nbchan/srate/pnts/data
    is neither layout, so it raises the typed not-an-EEGLAB-dataset error."""
    path = str(tmp_path / "partial_root.set")
    inner_path = path + ".inner"
    with h5py.File(inner_path, "w") as f:
        f.create_dataset("nbchan", data=np.array([[2.0]]))
        f.create_dataset("srate", data=np.array([[100.0]]))
    with open(inner_path, "rb") as fh:
        body = fh.read()
    os.remove(inner_path)
    with open(path, "wb") as fh:
        fh.write(_wrap_matlab_v73_header(body))

    _assert_not_eeglab_error(path)


def test_v73_sample_values_round_trip_exactly(tmp_path):
    """Every sample value survives the HDF5 round trip exactly (float32)."""
    path = str(tmp_path / "roundtrip.set")
    rng = np.random.default_rng(0)
    nbchan, pnts, srate = 6, 37, 512.0
    data = (rng.standard_normal((nbchan, pnts)) * 100).astype(np.float32)
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=data,
        labels=[f"Ch{i + 1}" for i in range(nbchan)],
    )

    rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_array_equal(rec.signals[label].to_numpy(), data[i])


def test_v73_channel_major_orientation_survives_transpose(tmp_path):
    """A non-square matrix with distinct per-channel waveforms proves the
    transpose is right: a swapped orientation would fail loudly here (either
    a shape mismatch, since n_channels != n_samples, or values crossing
    between channels), not silently "work".
    """
    path = str(tmp_path / "orientation.set")
    nbchan, pnts, srate = 3, 17, 100.0  # deliberately non-square
    data = np.zeros((nbchan, pnts), dtype=np.float32)
    # Each channel gets a distinct, easily-distinguished waveform.
    data[0] = np.arange(pnts)  # ramp
    data[1] = np.full(pnts, 1000.0)  # constant
    data[2] = -np.arange(pnts)  # descending ramp
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=data,
        labels=["ramp", "flat", "negramp"],
    )

    rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)  # samples x channels
    np.testing.assert_array_equal(rec.signals["ramp"].to_numpy(), np.arange(pnts))
    np.testing.assert_array_equal(rec.signals["flat"].to_numpy(), np.full(pnts, 1000.0))
    np.testing.assert_array_equal(rec.signals["negramp"].to_numpy(), -np.arange(pnts))


def test_v73_orientation_cross_check_skips_redundant_transpose(tmp_path):
    """If a file's data were already channel-major, nbchan says so and the
    importer must not transpose it AGAIN into the wrong orientation.

    Real v7.3 files are always sample-major (the case the previous test
    covers); this exercises the defensive nbchan cross-check itself (the
    same one sccn/eegprep's pop_loadset_h5 uses) for the hypothetical case
    where that assumption doesn't hold.
    """
    path = str(tmp_path / "already_channel_major.set")
    nbchan, pnts, srate = 3, 17, 100.0  # non-square, so a stray transpose is loud
    data = np.zeros((nbchan, pnts), dtype=np.float32)
    data[0] = np.arange(pnts)
    data[1] = np.full(pnts, 1000.0)
    data[2] = -np.arange(pnts)
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=data,
        labels=["ramp", "flat", "negramp"],
        already_channel_major=True,
    )

    rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)
    np.testing.assert_array_equal(rec.signals["ramp"].to_numpy(), np.arange(pnts))
    np.testing.assert_array_equal(rec.signals["flat"].to_numpy(), np.full(pnts, 1000.0))
    np.testing.assert_array_equal(rec.signals["negramp"].to_numpy(), -np.arange(pnts))


def test_v73_header_nbchan_mismatch_warns_but_data_matrix_wins(tmp_path):
    """A v7.3 header nbchan that disagrees with the data matrix warns and
    still loads correctly, trusting the data matrix (same rule as the
    classic path's `test_eeglab_header_nbchan_mismatch_warns`)."""
    path = str(tmp_path / "nbchan_mismatch.set")
    nbchan, pnts, srate = 4, 25, 100.0
    data = (np.arange(nbchan * pnts, dtype=np.float32)).reshape(nbchan, pnts)
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=data,
        header_nbchan=99,  # header lies; the (pnts, nbchan)-shaped matrix is authoritative
        labels=[f"C{i + 1}" for i in range(nbchan)],
    )

    with pytest.warns(UserWarning, match="nbchan=99 disagrees"):
        rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_array_equal(rec.signals[label].to_numpy(), data[i])


def test_v73_channel_labels_dereferenced_through_refs(tmp_path):
    """`chanlocs.labels` (an array of #refs# object references) resolves to text."""
    path = str(tmp_path / "labels.set")
    nbchan, pnts, srate = 4, 10, 100.0
    labels = ["Fp1", "Fp2", "Cz", "Oz"]
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=np.zeros((nbchan, pnts), dtype=np.float32),
        labels=labels,
    )

    rec = EEGLABImporter().load(path)
    assert list(rec.signals.columns) == labels
    assert set(rec.channels.keys()) == set(labels)


def test_v73_non_monotonic_labels_preserve_order(tmp_path):
    """Labels out of lexical/numeric order are not silently re-sorted.

    A ref-dereferencing bug that iterates `#refs#` (dict/insertion order of
    the *reference table*) instead of the `chanlocs.labels` array order would
    scramble this; ["ch10", "ch2", "ch1"] catches it because any sort would
    change the order.
    """
    path = str(tmp_path / "nonmonotonic.set")
    nbchan, pnts, srate = 3, 8, 100.0
    labels = ["ch10", "ch2", "ch1"]
    data = np.zeros((nbchan, pnts), dtype=np.float32)
    for i in range(nbchan):
        data[i] = i + 1  # channel i is constant (i+1), so column identity is checkable
    _write_v73_set(path, nbchan=nbchan, pnts=pnts, srate=srate, data=data, labels=labels)

    rec = EEGLABImporter().load(path)

    assert list(rec.signals.columns) == labels
    for i, label in enumerate(labels):
        np.testing.assert_array_equal(rec.signals[label].to_numpy(), np.full(pnts, i + 1))


def test_v73_channel_type_dereferenced(tmp_path):
    """`chanlocs.type` resolves through the same ref mechanism as labels."""
    path = str(tmp_path / "types.set")
    nbchan, pnts, srate = 2, 5, 100.0
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=np.zeros((nbchan, pnts), dtype=np.float32),
        labels=["E1", "E2"],
        types=["EEG", "EEG"],
    )

    rec = EEGLABImporter().load(path)
    for info in rec.channels.values():
        assert info["channel_type"] == "EEG"
        assert info["modality"] == "EEG"


def test_v73_chanlocs_xyz_dereferenced_including_zero(tmp_path):
    """`chanlocs.X` resolves through #refs# for every value, including 0.0.

    `_process_chanlocs_v73`/`_deref_h5_value` gate on `is not None`, not
    truthiness, so a channel legitimately positioned at X=0.0 must not be
    dropped the way a truthiness check (`if value:`) would drop it.
    """
    path = str(tmp_path / "xyz.set")
    nbchan, pnts, srate = 3, 5, 100.0
    x_values = [1.5, -2.25, 0.0]
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=np.zeros((nbchan, pnts), dtype=np.float32),
        labels=["A", "B", "C"],
        x=x_values,
    )

    rec = EEGLABImporter().load(path)

    for label, expected_x in zip(["A", "B", "C"], x_values, strict=True):
        assert rec.channels[label]["X"] == pytest.approx(expected_x)


def test_v73_trials_greater_than_one_raises_not_continuous(tmp_path):
    """Epoched (trials > 1) v7.3 files raise the typed not-continuous error,
    not a silently-flattened continuous stream."""
    path = str(tmp_path / "epoched.set")
    nbchan, pnts, srate = 4, 20, 100.0
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        trials=3,
        data=np.zeros((nbchan, pnts), dtype=np.float32),
    )

    with pytest.raises(NotContinuousRecordingError):
        EEGLABImporter().load(path)


def test_v73_companion_fdt_resolved(tmp_path):
    """EEG.data holding a .fdt filename (char array) loads the sibling .fdt."""
    nbchan, pnts, srate = 3, 50, 200.0
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((nbchan, pnts)) * 5).astype(np.float32)

    fdt_path = str(tmp_path / "sub-01_eeg.fdt")
    data.flatten(order="F").tofile(fdt_path)  # MATLAB column-major, as real EEGLAB writes it

    set_path = str(tmp_path / "sub-01_eeg.set")
    _write_v73_set(
        set_path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data_filename="sub-01_eeg.fdt",
        labels=[f"E{i}" for i in range(nbchan)],
    )

    rec = EEGLABImporter().load(set_path)

    assert rec.signals.shape == (pnts, nbchan)
    for i, label in enumerate(rec.signals.columns):
        np.testing.assert_allclose(rec.signals[label].to_numpy(), data[i], rtol=0, atol=1e-5)


def test_v73_companion_fdt_missing_raises(tmp_path):
    """A v7.3 .set pointing at a .fdt with no resolvable file errors clearly."""
    set_path = str(tmp_path / "sub-02_eeg.set")
    _write_v73_set(
        set_path,
        nbchan=3,
        pnts=20,
        srate=100.0,
        data_filename="nowhere.fdt",
    )

    with pytest.raises(ValueError, match="separate .fdt"):
        EEGLABImporter().load(set_path)


def test_v73_events_loaded(tmp_path):
    """`EEG.event` (latency/type refs) loads into `rec.events` like the classic path."""
    path = str(tmp_path / "events.set")
    nbchan, pnts, srate = 2, 300, 100.0
    events = [{"latency": 101, "type": "stim"}, {"latency": 201, "type": "resp"}]
    _write_v73_set(
        path,
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        data=np.zeros((nbchan, pnts), dtype=np.float32),
        events=events,
    )

    rec = EEGLABImporter().load(path)

    assert len(rec.events) == 2
    assert list(rec.events["description"]) == ["stim", "resp"]
    expected_onsets = [(100) / 100.0, (200) / 100.0]  # (latency - 1) / srate
    assert rec.events["onset"].tolist() == pytest.approx(expected_onsets)


def test_v73_events_with_empty_type_are_dropped_and_counted(tmp_path, caplog):
    """An event whose type is MATLAB's empty ``[]`` cannot be described, so it is
    dropped, and the drop is logged with a count rather than silent."""
    path = str(tmp_path / "empty_type.set")
    events = [
        {"latency": 11, "type": "stim"},
        {"latency": 21, "type": None},
        {"latency": 31, "type": None},
    ]
    _write_v73_set(
        path, nbchan=1, pnts=50, srate=10.0, data=np.zeros((1, 50), np.float32), events=events
    )

    with caplog.at_level("WARNING", logger="biosigio.importers.eeglab"):
        rec = EEGLABImporter().load(path)

    assert list(rec.events["description"]) == ["stim"]
    assert "dropped 2 event(s) with an empty type or latency" in caplog.text
    assert path in caplog.text


def test_v73_empty_channel_label_is_counted(tmp_path, caplog):
    """A chanlocs label that is MATLAB's empty ``[]`` falls back to the default
    ``ChannelN`` name, and the fallback is logged with a count."""
    path = str(tmp_path / "empty_label.set")
    _write_v73_set(
        path,
        nbchan=3,
        pnts=5,
        srate=10.0,
        data=np.zeros((3, 5), np.float32),
        labels=["Fz", None, "Pz"],
    )

    with caplog.at_level("WARNING", logger="biosigio.importers.eeglab"):
        rec = EEGLABImporter().load(path)

    assert list(rec.channels) == ["Fz", "Channel2", "Pz"]
    assert "1 channel(s) have an empty label" in caplog.text


@pytest.mark.parametrize("srate", [None, 0.0])
def test_v73_missing_srate_defaults_with_a_warning(tmp_path, caplog, srate):
    """A header with no usable sampling rate still loads at the 1000 Hz default,
    but the assumption is logged instead of silent."""
    path = str(tmp_path / "no_srate.set")
    _write_v73_set(path, nbchan=1, pnts=4, srate=srate, data=np.zeros((1, 4), np.float32))

    with caplog.at_level("WARNING", logger="biosigio.importers.eeglab"):
        rec = EEGLABImporter().load(path)

    assert rec.get_metadata("srate") == 1000.0
    assert "has no sampling rate" in caplog.text and "assuming 1000 Hz" in caplog.text


def test_v73_present_srate_logs_nothing(tmp_path, caplog):
    path = str(tmp_path / "srate.set")
    _write_v73_set(path, nbchan=1, pnts=4, srate=250.0, data=np.zeros((1, 4), np.float32))
    with caplog.at_level("WARNING", logger="biosigio.importers.eeglab"):
        EEGLABImporter().load(path)
    assert "sampling rate" not in caplog.text


def test_v73_missing_chanlocs_falls_back_to_default_labels(tmp_path):
    """No `chanlocs` at all still loads, with `ChannelN`/OTHER defaults."""
    path = str(tmp_path / "nochanlocs.set")
    nbchan, pnts, srate = 3, 10, 100.0
    _write_v73_set(
        path, nbchan=nbchan, pnts=pnts, srate=srate, data=np.zeros((nbchan, pnts), dtype=np.float32)
    )

    rec = EEGLABImporter().load(path)

    assert rec.signals.shape == (pnts, nbchan)
    assert list(rec.signals.columns) == ["Channel1", "Channel2", "Channel3"]
    for info in rec.channels.values():
        assert info["channel_type"] == "OTHER"


def test_v73_via_recording_from_file(tmp_path):
    """The public `Recording.from_file` entry point also dispatches correctly."""
    from ..core.emg import Recording

    path = str(tmp_path / "public_api.set")
    nbchan, pnts, srate = 2, 8, 100.0
    data = np.zeros((nbchan, pnts), dtype=np.float32)
    data[0] = np.arange(pnts)
    data[1] = np.arange(pnts) * 2
    _write_v73_set(path, nbchan=nbchan, pnts=pnts, srate=srate, data=data, labels=["A", "B"])

    rec = Recording.from_file(path)

    assert rec.signals.shape == (pnts, nbchan)
    np.testing.assert_array_equal(rec.signals["A"].to_numpy(), np.arange(pnts))
    np.testing.assert_array_equal(rec.signals["B"].to_numpy(), np.arange(pnts) * 2)


def test_is_matlab_v73_missing_file_returns_false(tmp_path):
    """A missing/unreadable file is treated as "not v7.3" by the sniff --
    the OSError is swallowed here, and the caller's own open (inside
    `EEGLABImporter._load`/`_load_v73`) surfaces the real error instead."""
    missing = str(tmp_path / "does_not_exist.set")
    assert _is_matlab_v73(missing) is False


def test_deref_h5_value_null_empty_bytes_and_bare_value(tmp_path):
    """Direct exercise of `_deref_h5_value`'s four branches, against real
    h5py objects (a null reference, a reference to an empty target, a
    reference to real char data, and non-reference bare values) rather than
    only indirectly through a full chanlocs/event struct array."""
    path = str(tmp_path / "deref_branches.h5")
    importer = EEGLABImporter()
    with h5py.File(path, "w") as f:
        store = f.create_group("store")
        full = store.create_dataset("full", data=np.array([ord(c) for c in "hi"], dtype=np.uint16))
        full.attrs["MATLAB_class"] = np.bytes_(b"char")
        empty = store.create_dataset("empty", data=np.array([], dtype=np.float64))

        # Null reference -- MATLAB's "no value" for a struct-array element.
        assert importer._deref_h5_value(h5py, f, h5py.Reference()) is None
        # Reference to an empty target (MATLAB's `[]`).
        assert importer._deref_h5_value(h5py, f, empty.ref) is None
        # Reference to real char data resolves through #refs#-style indirection.
        assert importer._deref_h5_value(h5py, f, full.ref) == "hi"
        # Bare bytes value (no indirection) decodes directly.
        assert importer._deref_h5_value(h5py, f, b"raw") == "raw"
        # Bare non-reference, non-bytes value passes through unchanged.
        assert importer._deref_h5_value(h5py, f, 3.5) == 3.5


def test_v73_masked_memory_error_via_h5py_exit_is_still_recognized(tmp_path, monkeypatch):
    """A MemoryError raised inside `with h5py.File(...) as f:` can be MASKED:
    if h5py's own `__exit__` raises a SECOND, unrelated exception while the
    MemoryError is propagating out of the `with` body, Python replaces the
    propagating exception with that second one -- `_load_v73`'s
    `except Exception as e:` binds the OSError from `__exit__`, not the
    MemoryError, with the MemoryError attached as `e.__context__` via
    Python's own automatic implicit chaining (nothing here constructs that
    chain by hand).

    The masking OSError's own message is deliberately NOT exhaustion-flavored
    ("unable to synchronously close file"), so the only way
    `is_resource_exhaustion` can recognize the final exception is by walking
    to `__context__` (see biosigio.exceptions) -- proving the walk does real
    work for this real h5py-context-manager shape, not that a MemoryError
    happened to reach the guard directly (that path is exercised by the
    synthetic/real-data EDF tests in test_edf_memory_error.py).
    """
    path = str(tmp_path / "v73.set")
    _write_v73_set(path, nbchan=2, pnts=4, srate=100.0, data=np.zeros((2, 4)))

    def raise_memory_error(*args, **kwargs):
        raise MemoryError("Unable to allocate 24.8 MiB for an array")

    # _h5_scalar is a @staticmethod; patching it at the class level covers the
    # `self._h5_scalar(...)` call inside the `with` block regardless of which
    # field is read first.
    monkeypatch.setattr(EEGLABImporter, "_h5_scalar", staticmethod(raise_memory_error))

    real_exit = h5py.File.__exit__

    def flaky_exit(self, exc_type, exc_val, exc_tb):
        real_exit(self, exc_type, exc_val, exc_tb)  # real cleanup still runs
        raise OSError("unable to synchronously close file, id_type: 0x1")

    monkeypatch.setattr(h5py.File, "__exit__", flaky_exit)

    with pytest.raises(OSError) as exc_info:
        EEGLABImporter().load(path)

    raised = exc_info.value
    assert not isinstance(raised, FileReadError)
    assert isinstance(raised.__context__, MemoryError)
    assert is_resource_exhaustion(raised) is True
