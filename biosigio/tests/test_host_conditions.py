"""Host I/O errors stay raw OSErrors on every read path -- no mocks.

A file the process may not read (``EACCES``), a failing disk (``EIO``) or a
stale network handle (``ESTALE``) says something about the host, not the
recording: another host, or this one later, reads it fine. A caller that treats
a typed :class:`~biosigio.exceptions.BiosigIOError` as a permanent file failure
must therefore see these as the raw ``OSError`` the operating system raised.

Each test writes a REAL recording, removes read permission with ``chmod 000``,
and reads it through the public importer and, where one exists, the streaming
Zarr source. Skipped for root (which ignores file permissions) and on Windows
(where ``chmod`` cannot remove read permission).
"""

import os
import sys

import numpy as np
import pytest
import scipy.io

from ..core.emg import Recording
from ..exceptions import BiosigIOError
from ..exporters.zarr_stream import _open_stream_source

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not deny reads to root or on Windows",
)


def _unreadable(path) -> None:
    os.chmod(path, 0)
    with pytest.raises(PermissionError):  # the fixture really is unreadable
        open(path, "rb").close()


def _assert_raw_permission_error(read, path) -> None:
    with pytest.raises(PermissionError) as info:
        read(path)
    assert not isinstance(info.value, BiosigIOError)


def _stream(path):
    source = _open_stream_source(str(path), None)
    source.close()


@pytest.fixture(autouse=True)
def _restore_permissions(tmp_path):
    """Give every file back its permissions so pytest can clean tmp_path up."""
    yield
    for root, _dirs, files in os.walk(tmp_path):
        for name in files:
            os.chmod(os.path.join(root, name), 0o600)


# --- EEGLAB ----------------------------------------------------------------------


def _write_classic_set(path) -> None:
    scipy.io.savemat(
        path,
        {
            "nbchan": np.array([[2]]),
            "trials": np.array([[1]]),
            "pnts": np.array([[100]]),
            "srate": np.array([[100.0]]),
            "data": np.zeros((2, 100)),
        },
    )


def test_eeglab_classic_set_unreadable(tmp_path):
    path = str(tmp_path / "sub-01_eeg.set")
    _write_classic_set(path)
    Recording.from_file(path)  # readable first: the fixture itself is valid
    _unreadable(path)
    _assert_raw_permission_error(Recording.from_file, path)


def test_eeglab_v73_unreadable(tmp_path):
    """A MATLAB v7.3 ``.set`` whose data lives in a companion ``.fdt``: with the
    ``.fdt`` unreadable the failure is raised inside the h5py reader's own
    classifying ``except``; with the ``.set`` unreadable, before it."""
    pytest.importorskip("h5py", reason="v7.3 .set reading needs the 'hdf5' extra")
    from .test_eeglab_v73_importer import _write_v73_set

    set_path = str(tmp_path / "sub-01_eeg.set")
    fdt_path = str(tmp_path / "sub-01_eeg.fdt")
    _write_v73_set(set_path, nbchan=2, pnts=100, srate=100.0, data_filename="sub-01_eeg.fdt")
    np.zeros((2, 100), dtype="<f4").flatten(order="F").tofile(fdt_path)
    assert Recording.from_file(set_path).signals.shape == (100, 2)

    _unreadable(fdt_path)
    _assert_raw_permission_error(Recording.from_file, set_path)
    _unreadable(set_path)
    _assert_raw_permission_error(Recording.from_file, set_path)


# --- BrainVision -------------------------------------------------------------------


def _write_brainvision(directory, stem="sub-01_eeg") -> str:
    np.zeros((100, 2), dtype=np.int16).tofile(os.path.join(directory, f"{stem}.eeg"))
    with open(os.path.join(directory, f"{stem}.vmrk"), "w") as f:
        f.write(
            "Brain Vision Data Exchange Marker File, Version 1.0\n"
            f"[Common Infos]\nDataFile={stem}.eeg\n[Marker Infos]\n"
        )
    vhdr = os.path.join(directory, f"{stem}.vhdr")
    with open(vhdr, "w") as f:
        f.write(
            "Brain Vision Data Exchange Header File Version 1.0\n"
            f"[Common Infos]\nDataFile={stem}.eeg\nMarkerFile={stem}.vmrk\n"
            "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\n"
            "SamplingInterval=4000\n[Binary Infos]\nBinaryFormat=INT_16\n"
            "[Channel Infos]\nCh1=C1,,0.1,µV\nCh2=C2,,0.1,µV\n"
        )
    return vhdr


@pytest.mark.parametrize("unreadable", [".vhdr", ".vmrk", ".eeg"])
def test_brainvision_unreadable(tmp_path, unreadable):
    """The header is read by biosigIO's resolver, the markers and data by MNE:
    each is a separate open that must keep the host's error."""
    pytest.importorskip("mne", reason="BrainVision needs the 'meg' extra")
    vhdr = _write_brainvision(str(tmp_path))
    Recording.from_file(vhdr)
    _stream(vhdr)
    _unreadable(os.path.splitext(vhdr)[0] + unreadable)
    _assert_raw_permission_error(Recording.from_file, vhdr)
    if unreadable != ".eeg":  # the streaming source opens the data lazily
        _assert_raw_permission_error(_stream, vhdr)


# --- EDF ---------------------------------------------------------------------------


def _write_edf(path) -> None:
    import pyedflib

    writer = pyedflib.EdfWriter(path, 2, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(
            [
                {
                    "label": f"C{i}",
                    "dimension": "uV",
                    "sample_frequency": 100,
                    "physical_max": 100.0,
                    "physical_min": -100.0,
                    "digital_max": 32767,
                    "digital_min": -32768,
                    "transducer": "",
                    "prefilter": "",
                }
                for i in range(2)
            ]
        )
        writer.writeSamples([np.zeros(100), np.zeros(100)])
    finally:
        writer.close()


def test_edf_unreadable(tmp_path):
    """pyedflib reports a refused open() without its errno; the importer and the
    streaming source still raise the OS's own PermissionError."""
    path = str(tmp_path / "sub-01_eeg.edf")
    _write_edf(path)
    Recording.from_file(path)
    _unreadable(path)
    _assert_raw_permission_error(Recording.from_file, path)
    _assert_raw_permission_error(_stream, path)


# --- WFDB --------------------------------------------------------------------------


@pytest.mark.parametrize("unreadable", [".hea", ".dat"])
def test_wfdb_unreadable(tmp_path, unreadable):
    wfdb = pytest.importorskip("wfdb")
    wfdb.wrsamp(
        "rec",
        fs=100,
        units=["mV", "mV"],
        sig_name=["a", "b"],
        p_signal=np.zeros((100, 2)),
        fmt=["16", "16"],
        write_dir=str(tmp_path),
    )
    header = str(tmp_path / "rec.hea")
    Recording.from_file(header)
    _unreadable(str(tmp_path / f"rec{unreadable}"))
    _assert_raw_permission_error(Recording.from_file, header)
