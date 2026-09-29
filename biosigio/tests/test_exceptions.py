"""Typed read-error classification (biosigio.exceptions).

A caller (the NEMAR Zarr pipeline) needs to know *why* a recording could not be
read so it can surface a specific reason. These tests pin the classifier and the
importers' typed raises on REAL files (no mocks):

* an unknown extension -> UnsupportedFormatError
* a real MNE evoked (``*-ave.fif``: averaged, no continuous raw) -> NotContinuousRecordingError
* a truncated EDF -> CorruptFileError
* mixed-rate EDF -> MixedSamplingRateError (still a ValueError, for back-compat)

The string-matching unit tests cover the reader phrasings we map without needing
each format on hand.
"""

import errno
import importlib.util
import os
import pathlib
import shutil
import tempfile

import numpy as np
import pyedflib
import pytest

from biosigio import Recording
from biosigio.exceptions import (
    REASONS,
    BiosigIOError,
    CorruptFileError,
    FileReadError,
    MixedSamplingRateError,
    NotContinuousRecordingError,
    UnsupportedFormatError,
    classify_read_error,
    is_host_condition,
    is_resource_exhaustion,
)

_HAS_MNE = importlib.util.find_spec("mne") is not None
_REPO = pathlib.Path(__file__).resolve().parents[2]
_MEG_FIF = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"
_CTF_DS = _REPO / "examples/ctf/catch-alp-good-f.ds"


# -- classify_read_error: reader-phrasing -> typed error -----------------------


def test_classify_no_raw_data_is_not_continuous():
    err = classify_read_error(ValueError("No raw data in /x/sub-01_task-a-ave.fif"))
    assert isinstance(err, NotContinuousRecordingError)
    assert err.code == "not_continuous"


def test_classify_not_a_raw_file_is_not_continuous():
    err = classify_read_error(RuntimeError("This is not a raw file"))
    assert isinstance(err, NotContinuousRecordingError)


def test_classify_filesize_is_corrupt():
    err = classify_read_error(OSError("the file is not EDF(+) or BDF(+) compliant (Filesize)"))
    assert isinstance(err, CorruptFileError)
    assert err.code == "corrupt_or_truncated"


def test_classify_incomplete_split_is_corrupt():
    err = classify_read_error(
        ValueError("Split raw file detected but next file ..._split-02_meg.fif does not exist")
    )
    assert isinstance(err, CorruptFileError)


def test_classify_ctf_trial_truncation_is_corrupt():
    err = classify_read_error(
        ValueError("The number of samples is not an even multiple of the trial size")
    )
    assert isinstance(err, CorruptFileError)


def test_classify_unknown_falls_back_to_file_read_error():
    err = classify_read_error(RuntimeError("something unexpected"))
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


def test_classify_passes_through_existing_typed_error():
    original = NotContinuousRecordingError("already typed")
    assert classify_read_error(original) is original


def test_classify_includes_filepath_when_given():
    err = classify_read_error(ValueError("No raw data"), "/data/sub-01_task-a-ave.fif")
    assert "sub-01_task-a-ave.fif" in str(err)


# -- is_resource_exhaustion: resource exhaustion is never a property of the file
# (issue #123) ------------------------------------------------------------------


def test_is_resource_exhaustion_memory_error():
    assert is_resource_exhaustion(MemoryError("out of memory")) is True


def test_is_resource_exhaustion_numpy_array_memory_error():
    """numpy's ``_ArrayMemoryError`` (a MemoryError subclass) is the exact shape
    the on008083 failures logged: "Unable to allocate 24.8 MiB for an array
    with shape (3248000,) and data type float64", raised inside a per-channel
    readSignal loop when the host is saturated. Provoked for real (not
    hand-constructed) so the assertion is about the actual numpy type -- numpy
    decorates it ``@_display_as_base`` so ``str()``/``type().__name__`` both
    read "MemoryError" (by design, for a friendlier traceback), so the type
    is confirmed via its module instead of its display name."""
    with pytest.raises(MemoryError) as exc_info:
        np.zeros(int(1e18), dtype=np.float64)
    exc = exc_info.value
    assert type(exc) is not MemoryError  # genuinely numpy's subclass, not the builtin
    assert type(exc).__module__.startswith("numpy")
    assert is_resource_exhaustion(exc) is True


def test_is_resource_exhaustion_thread_start_failure():
    assert is_resource_exhaustion(RuntimeError("can't start new thread")) is True


def test_is_resource_exhaustion_enomem_oserror():
    assert is_resource_exhaustion(OSError(errno.ENOMEM, "Cannot allocate memory")) is True


def test_is_resource_exhaustion_eagain_oserror():
    """EAGAIN is kept for fork()-under-RLIMIT_NPROC (thread/process exhaustion),
    not because this codebase does nonblocking I/O -- it does not, so a real
    EAGAIN here always means the exhaustion reading. See the docstring on
    _RESOURCE_EXHAUSTION_ERRNOS."""
    assert is_resource_exhaustion(OSError(errno.EAGAIN, "Resource temporarily unavailable")) is True


def test_is_resource_exhaustion_emfile_oserror():
    """Per-process file-descriptor exhaustion (e.g. many parallel EDF/HDF5/Zarr
    readers open at once) is the same "host is saturated" condition as
    MemoryError, just a different resource."""
    assert is_resource_exhaustion(OSError(errno.EMFILE, "Too many open files")) is True


def test_is_resource_exhaustion_enfile_oserror():
    """System-wide (not just per-process) file-descriptor exhaustion."""
    assert is_resource_exhaustion(OSError(errno.ENFILE, "Too many open files in system")) is True


def test_is_resource_exhaustion_cannot_allocate_memory_message_only():
    """No errno set (e.g. re-raised across a process boundary, or constructed
    by a library from a bare message) -- the substring match must still fire."""
    assert is_resource_exhaustion(OSError("cannot allocate memory")) is True


def test_is_resource_exhaustion_too_many_open_files_message_only():
    assert is_resource_exhaustion(OSError("Too many open files")) is True


def test_is_resource_exhaustion_negative_case_genuine_missing_file():
    """A real ENOENT is a file problem, not resource exhaustion -- it must still
    classify as a read error (not be treated as retryable)."""
    assert is_resource_exhaustion(OSError(errno.ENOENT, "No such file or directory")) is False


def test_is_resource_exhaustion_negative_case_different_errno_not_matched():
    """A real, unrelated errno (permission denied) on an OSError whose message
    happens to say nothing exhaustion-flavored must not be matched -- proves
    the errno check is a specific allowlist, not "any OSError with an errno"."""
    assert is_resource_exhaustion(OSError(errno.EACCES, "Permission denied")) is False


def test_is_resource_exhaustion_negative_case_unrelated_error():
    assert is_resource_exhaustion(ValueError("not EDF(+) or BDF(+) compliant (Filesize)")) is False


# -- is_resource_exhaustion walks __cause__/__context__ so cleanup can't mask ---
# resource exhaustion (a finally/__exit__/close() that itself raises while a
# MemoryError is propagating would otherwise hide it from a caller's `except`).


def test_is_resource_exhaustion_walks_explicit_cause_chain():
    inner = MemoryError("Unable to allocate 24.8 MiB for an array")
    outer = RuntimeError("cleanup failed")
    outer.__cause__ = inner  # explicit `raise outer from inner` shape
    assert is_resource_exhaustion(outer) is True


def test_is_resource_exhaustion_walks_implicit_context_chain():
    """The shape Python itself produces when a `finally`/`__exit__` raises while
    another exception is already propagating (no `from` needed -- this is
    automatic, see test_eeglab_v73_masked_memory_error_via_h5py_exit for the
    real h5py-context-manager version of this)."""
    try:
        try:
            raise MemoryError("Unable to allocate 24.8 MiB for an array")
        finally:
            raise OSError("unable to synchronously close file, id_type: 0x1")
    except OSError as outer:
        assert outer.__cause__ is None
        assert isinstance(outer.__context__, MemoryError)
        assert is_resource_exhaustion(outer) is True


def test_is_resource_exhaustion_chain_walk_stops_at_unrelated_root():
    """A chain that never bottoms out in resource exhaustion is correctly False,
    even several links deep -- the walk does not just always return True once
    it starts walking."""
    root = ValueError("not EDF(+) or BDF(+) compliant (Filesize)")
    mid = RuntimeError("re-raised during cleanup")
    mid.__cause__ = root
    outer = OSError("outermost wrapper")
    outer.__cause__ = mid
    assert is_resource_exhaustion(outer) is False


def test_is_resource_exhaustion_chain_depth_is_bounded():
    """A MemoryError buried past _MAX_CHAIN_DEPTH links is NOT found -- the walk
    is bounded, not an unbounded chain traversal (guards against a pathological
    or cyclic __context__ chain hanging the check)."""
    from biosigio.exceptions import _MAX_CHAIN_DEPTH

    memory_error = MemoryError("buried too deep")
    exc: BaseException = memory_error
    # One link past the bound: depth 0 is `outer` itself, so placing the
    # MemoryError at exactly _MAX_CHAIN_DEPTH links away is one step too far.
    for _ in range(_MAX_CHAIN_DEPTH):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = exc
        exc = wrapper
    assert is_resource_exhaustion(exc) is False

    # One link closer (within the bound) IS found.
    exc2: BaseException = memory_error
    for _ in range(_MAX_CHAIN_DEPTH - 1):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = exc2
        exc2 = wrapper
    assert is_resource_exhaustion(exc2) is True


# -- classify_read_error re-raises resource exhaustion, never FileReadError -----


def test_classify_read_error_reraises_memory_error():
    exc = MemoryError("Unable to allocate 24.8 MiB for an array")
    with pytest.raises(MemoryError) as exc_info:
        classify_read_error(exc, "/data/sub-001_ses-01_task-HierPrior_eeg.edf")
    assert exc_info.value is exc


def test_classify_read_error_reraises_numpy_array_memory_error():
    try:
        np.zeros(int(1e18), dtype=np.float64)
    except MemoryError as exc:
        with pytest.raises(MemoryError) as exc_info:
            classify_read_error(exc, "/data/sub-001.edf")
        assert exc_info.value is exc
    else:
        pytest.fail("expected numpy to raise MemoryError for an absurd allocation")


def test_classify_read_error_reraises_thread_exhaustion_runtime_error():
    exc = RuntimeError("can't start new thread")
    with pytest.raises(RuntimeError) as exc_info:
        classify_read_error(exc)
    assert exc_info.value is exc


def test_classify_read_error_reraises_enomem_oserror():
    exc = OSError(errno.ENOMEM, "Cannot allocate memory")
    with pytest.raises(OSError) as exc_info:
        classify_read_error(exc)
    assert exc_info.value is exc


def test_classify_read_error_negative_case_enoent_still_classifies_as_read_error():
    """A genuine "file not found"-flavored OSError is a real read problem, not
    resource exhaustion, and must still fall through to FileReadError."""
    err = classify_read_error(OSError(errno.ENOENT, "No such file or directory"), "/x/missing.edf")
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


# -- host I/O errors are the host, not the file: re-raised, never typed --------

_HOST_IO = [
    errno.EACCES,
    errno.EPERM,
    errno.EIO,
    errno.ENOSPC,
    errno.EROFS,
    errno.ETIMEDOUT,
    *(code for code in (getattr(errno, "EDQUOT", None), getattr(errno, "ESTALE", None)) if code),
]


@pytest.mark.parametrize("code", _HOST_IO, ids=[errno.errorcode[c] for c in _HOST_IO])
def test_classify_read_error_reraises_host_io_errors(code):
    exc = OSError(code, os.strerror(code), "/data/sub-01_eeg.edf")
    assert is_host_condition(exc) is True
    with pytest.raises(OSError) as info:
        classify_read_error(exc, "/data/sub-01_eeg.edf")
    assert info.value is exc


def test_host_condition_is_found_behind_a_wrapper():
    """A reader that wraps the OS error (``raise ValueError(...) from err``) is
    still a host condition, and the wrapper is re-raised as-is."""
    outer = ValueError("could not open recording")
    outer.__cause__ = PermissionError(errno.EACCES, "Permission denied")
    assert is_host_condition(outer) is True
    with pytest.raises(ValueError) as info:
        classify_read_error(outer)
    assert info.value is outer


def test_host_condition_includes_resource_exhaustion():
    assert is_host_condition(MemoryError()) is True


@pytest.mark.parametrize(
    "exc",
    [
        OSError(errno.ENOENT, "No such file or directory"),
        OSError("the file is not EDF(+) or BDF(+) compliant (Filesize)"),  # no errno
        ValueError("header is corrupt"),
    ],
    ids=["enoent", "errno-less-oserror", "valueerror"],
)
def test_file_problems_are_not_host_conditions(exc):
    assert is_host_condition(exc) is False
    assert isinstance(classify_read_error(exc), FileReadError)


# -- typed errors are ValueErrors (back-compat) + have stable codes ------------


def test_typed_errors_are_valueerrors():
    for exc in (
        UnsupportedFormatError,
        FileReadError,
        NotContinuousRecordingError,
        CorruptFileError,
        MixedSamplingRateError,
    ):
        assert issubclass(exc, BiosigIOError)
        assert issubclass(exc, ValueError)


def test_every_code_has_a_reason():
    for exc in (
        BiosigIOError,
        UnsupportedFormatError,
        FileReadError,
        NotContinuousRecordingError,
        CorruptFileError,
        MixedSamplingRateError,
    ):
        assert exc.code in REASONS and REASONS[exc.code]


# -- real files ----------------------------------------------------------------


def test_unknown_extension_raises_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fh:
        with pytest.raises(UnsupportedFormatError):
            Recording.from_file(fh.name)


@pytest.mark.skipif(not _HAS_MNE, reason="MEG path needs the 'meg' extra (mne)")
def test_real_evoked_ave_fif_is_not_continuous():
    """A trial-averaged MNE evoked (``*-ave.fif``) is valid data but carries no
    continuous raw recording -> NotContinuousRecordingError (the on005261 case)."""
    import mne

    info = mne.create_info(["MEG0011", "MEG0021"], sfreq=100.0, ch_types="mag")
    evoked = mne.EvokedArray(np.zeros((2, 50), dtype=float), info, tmin=0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub-01_task-x_ave.fif")  # MNE wants *-ave.fif
        evoked.save(path)
        with pytest.raises(NotContinuousRecordingError):
            Recording.from_file(path)


def test_truncated_edf_is_corrupt():
    """An EDF cut short of its declared length fails pyedflib's Filesize check ->
    CorruptFileError."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.edf")
        rate, n = 100.0, 1000
        w = pyedflib.EdfWriter(path, 1)
        try:
            w.setSignalHeaders(
                [
                    {
                        "label": "EEG1",
                        "dimension": "uV",
                        "sample_frequency": rate,
                        "physical_max": 100.0,
                        "physical_min": -100.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    }
                ]
            )
            w.writeSamples([np.zeros(n)])
        finally:
            w.close()
        # Lop off the back half of the data records -> declared > actual size.
        full = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(full // 2)
        with pytest.raises(CorruptFileError):
            Recording.from_file(path)


def test_mixed_rate_edf_raises_typed_but_valueerror():
    """The mixed-rate policy error is now MixedSamplingRateError yet still a
    ValueError, so existing callers keep catching it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "psg.edf")
        w = pyedflib.EdfWriter(path, 2)
        try:
            w.setSignalHeaders(
                [
                    {
                        "label": "EEG",
                        "dimension": "uV",
                        "sample_frequency": 200.0,
                        "physical_max": 100.0,
                        "physical_min": -100.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    },
                    {
                        "label": "SpO2",
                        "dimension": "%",
                        "sample_frequency": 12.5,
                        "physical_max": 100.0,
                        "physical_min": 0.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    },
                ]
            )
            w.writeSamples([np.zeros(200), np.zeros(13)])
        finally:
            w.close()
        with pytest.raises(MixedSamplingRateError):
            Recording.from_file(path, mixed_rate="error")
        with pytest.raises(ValueError):  # back-compat
            Recording.from_file(path, mixed_rate="error")


@pytest.mark.skipif(not _HAS_MNE or not _MEG_FIF.exists(), reason="needs mne + the FIF fixture")
def test_incomplete_split_fif_chain_is_corrupt():
    """A split FIF whose chain is missing a member (split-01 present, split-02
    gone) -> CorruptFileError, via the real MEG importer (the on005261 split
    failure mode). read_raw_fif(split-01) follows the chain and raises when the
    next file is absent."""
    import mne

    full = mne.io.read_raw_fif(str(_MEG_FIF), preload=True, verbose="ERROR")
    with tempfile.TemporaryDirectory() as d:
        full.save(
            os.path.join(d, "sub-01_task-mouse_meg.fif"),
            split_size="1.5MB",
            split_naming="bids",
            verbose="ERROR",
        )
        first = os.path.join(d, "sub-01_task-mouse_split-01_meg.fif")
        second = os.path.join(d, "sub-01_task-mouse_split-02_meg.fif")
        assert os.path.exists(first) and os.path.exists(second)
        os.remove(second)  # break the chain
        with pytest.raises(CorruptFileError):
            Recording.from_file(first)


@pytest.mark.skipif(not _HAS_MNE or not _CTF_DS.exists(), reason="needs mne + the CTF fixture")
def test_truncated_ctf_meg4_is_corrupt():
    """A CTF .ds whose .meg4 is chopped to a non-multiple of its record size
    (the on004398 truncation) -> CorruptFileError, via the real CTF reader."""
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "catch-alp-good-f.ds")
        shutil.copytree(_CTF_DS, ds)
        meg4 = os.path.join(ds, "catch-alp-good-f.meg4")
        # Cut a few hundred bytes off the tail: the .meg4 is now (size-8) not a
        # clean int32*nchan*nsamp multiple, which read_raw_ctf rejects.
        with open(meg4, "r+b") as fh:
            fh.truncate(os.path.getsize(meg4) - 333)
        with pytest.raises(CorruptFileError):
            Recording.from_file(ds)
