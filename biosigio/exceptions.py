"""Typed exceptions for biosigIO.

A small hierarchy so a caller (e.g. the NEMAR Zarr serving pipeline) can classify
*why* a recording could not be read/converted and surface a specific, stable
reason to users, instead of parsing English error strings out of MNE/pyedflib.

Every exception carries a machine-stable ``code`` (string). Catch :class:`BiosigIOError`
and branch on ``.code`` (or ``isinstance``); the codes are part of the public API
and are safe to map to user-facing copy downstream.

    from biosigio import BiosigIOError
    try:
        rec = Recording.from_file(path)
    except BiosigIOError as e:
        reason = REASONS.get(e.code, "could not be read")
        ...

The reverse-mapping of a low-level reader error to one of these types lives in
:func:`classify_read_error`, kept here (with the format knowledge) so importers
share one classifier rather than each re-inventing string matching.
"""

from __future__ import annotations

import errno
from typing import NoReturn


class BiosigIOError(ValueError):
    """Base for all biosigIO errors. ``code`` is a stable machine-readable tag.

    Subclasses ``ValueError`` so existing ``except ValueError`` / ``raises(ValueError)``
    callers keep working as these typed errors are introduced (they are all "bad
    input/value" conditions about a recording)."""

    code = "biosigio_error"


class UnsupportedFormatError(BiosigIOError):
    """The file's format/extension has no biosigIO reader (e.g. an unknown
    extension, or a format only a missing optional extra could read)."""

    code = "unsupported_format"


class FileReadError(BiosigIOError):
    """A recognized format failed to read. Base for the specific read failures;
    raised directly as the generic fallback when no specific cause is identified."""

    code = "file_read_error"


class NotContinuousRecordingError(FileReadError):
    """The file holds no continuous raw time series -- typically a trial-averaged
    (evoked, ``*-ave.fif``) or epoched (``*-epo.fif``) derivative. The data is
    valid, but the serving/viewer path only handles continuous recordings."""

    code = "not_continuous"


class CorruptFileError(FileReadError):
    """The source is truncated, not format-compliant, or part of an incomplete
    set: an EDF that fails the ``Filesize`` check, an incomplete CTF ``.meg4``,
    or a split recording missing chain members."""

    code = "corrupt_or_truncated"


class EmptyRecordingError(FileReadError):
    """The file read but contains no signal channels to plot."""

    code = "empty_recording"


class MixedSamplingRateError(BiosigIOError):
    """An EDF/BDF carries differing per-channel sampling rates and the caller's
    ``mixed_rate`` policy is ``"error"`` (the default). Not a read failure -- the
    caller can re-read with ``mixed_rate="resample"``. The serving pipeline does
    exactly that, so this rarely surfaces to users."""

    code = "mixed_sampling_rate"


# code -> default human-facing reason. Downstream UIs may override the copy, but
# the codes are stable.
REASONS: dict[str, str] = {
    UnsupportedFormatError.code: "This file format is not yet supported by the viewer.",
    NotContinuousRecordingError.code: (
        "This file is a trial-averaged or epoched derivative, not a continuous "
        "recording, so the time-series viewer is not available."
    ),
    CorruptFileError.code: (
        "This recording's data file appears truncated or corrupt, so the viewer "
        "could not be generated."
    ),
    EmptyRecordingError.code: "This recording contains no signal channels to display.",
    MixedSamplingRateError.code: (
        "This recording mixes per-channel sampling rates; a viewable copy needs "
        "resampling to a common grid."
    ),
    FileReadError.code: "This recording could not be prepared for viewing.",
    BiosigIOError.code: "This recording could not be prepared for viewing.",
}


# Thread/resource-exhaustion messages that surface as OSError/RuntimeError rather
# than MemoryError. Reader-agnostic substrings (CPython's own `_thread` module and
# libc's ENOMEM/EMFILE/ENFILE/EAGAIN strerror text), matched case-insensitively.
_THREAD_OR_RESOURCE_EXHAUSTION_SUBSTRINGS = (
    "can't start new thread",
    "cannot allocate memory",  # ENOMEM strerror text (not every OSError carries .errno)
    "too many open files",  # EMFILE/ENFILE strerror text (fd exhaustion, e.g. many parallel readers)
)
# ENOMEM: out of memory. EMFILE/ENFILE: per-process/system-wide file-descriptor
# exhaustion -- the same "host is saturated" condition as MemoryError, just a
# different resource, and just as likely under many parallel conversions opening
# EDF/HDF5/Zarr stores at once. EAGAIN is kept even though it is normally "call
# again later" for a nonblocking read: on Linux it is also what fork() returns
# when RLIMIT_NPROC is hit (the process table is full), which is thread/process
# exhaustion in the same sense as "can't start new thread" above, not a file
# problem -- and this codebase never uses nonblocking I/O, so a real EAGAIN here
# always means the resource-exhaustion reading, never the "retry the syscall" one.
_RESOURCE_EXHAUSTION_ERRNOS = (errno.ENOMEM, errno.EMFILE, errno.ENFILE, errno.EAGAIN)
# How many links of __cause__/__context__ to follow when the outermost exception
# itself doesn't match (see the docstring below).
_MAX_CHAIN_DEPTH = 5


def is_resource_exhaustion(exc: BaseException) -> bool:
    """Re-raises for resource exhaustion; used by ``classify_read_error`` (below)
    and by importers directly, before it, as their first check.

    True when ``exc`` -- or something it masks -- reflects the host running out
    of memory, threads, or file descriptors, not a problem with the file being
    read.

    Resource exhaustion is never a property of the file: a ``MemoryError`` (or
    numpy's ``_ArrayMemoryError``, a ``MemoryError`` subclass) raised mid-read
    typically means the host was saturated (e.g. many parallel conversions),
    not that the recording is corrupt. Importers must let it -- and the
    thread/allocation/fd-exhaustion ``OSError``/``RuntimeError`` variants below
    -- propagate unchanged instead of reclassifying it as a permanent read
    failure, so a caller (e.g. NEMAR's converter) can retry instead of
    recording a deterministic failure. See :func:`classify_read_error`.

    Resource exhaustion can also be MASKED by cleanup: if a ``MemoryError`` is
    propagating and a ``finally``/context-manager ``__exit__``/``close()``
    during unwinding raises its OWN exception, that second exception -- not the
    ``MemoryError`` -- is what a caller's ``except`` actually binds, with the
    ``MemoryError`` attached as ``__context__`` (implicit chaining) or
    ``__cause__`` (an explicit ``raise ... from``). So this walks the chain --
    the exception itself, then ``__cause__`` or ``__context__``, up to
    :data:`_MAX_CHAIN_DEPTH` links -- rather than only checking the outermost
    exception.
    """
    seen: BaseException | None = exc
    for _ in range(_MAX_CHAIN_DEPTH):
        if seen is None:
            return False
        if isinstance(seen, MemoryError):
            return True
        if isinstance(seen, (OSError, RuntimeError)):
            if getattr(seen, "errno", None) in _RESOURCE_EXHAUSTION_ERRNOS:
                return True
            low = str(seen).lower()
            if any(s in low for s in _THREAD_OR_RESOURCE_EXHAUSTION_SUBSTRINGS):
                return True
        seen = seen.__cause__ or seen.__context__
    return False


# Host I/O conditions: the operating system refused or failed an operation for a
# reason that belongs to this host (its permissions, its disks, its network
# mounts), not to the bytes of the recording. EACCES/EPERM: the process may not
# read the file (a mount or ownership problem). EIO: the device failed the read.
# ENOSPC/EDQUOT/EROFS: a temporary copy could not be written. ETIMEDOUT/ESTALE:
# a network filesystem timed out or its handle went stale. Another host (or the
# same one later) can read the file, so none of these may become a typed error.
_HOST_IO_ERRNOS = frozenset(
    code
    for code in (
        errno.EACCES,
        errno.EPERM,
        errno.EIO,
        errno.ENOSPC,
        errno.EROFS,
        getattr(errno, "EDQUOT", None),  # absent on Windows
        errno.ETIMEDOUT,
        getattr(errno, "ESTALE", None),  # absent on Windows
    )
    if code is not None
)


def is_host_condition(exc: BaseException) -> bool:
    """True when ``exc`` reflects the host, not the file being read.

    A superset of :func:`is_resource_exhaustion`: also true when ``exc``, or an
    exception it chains to (``__cause__``, else ``__context__``, up to
    :data:`_MAX_CHAIN_DEPTH` links), is an ``OSError`` whose ``errno`` is a host
    I/O condition: ``EACCES``, ``EPERM``, ``EIO``, ``ENOSPC``, ``EROFS``,
    ``EDQUOT``, ``ETIMEDOUT`` or ``ESTALE``.

    :func:`classify_read_error` re-raises any such exception unchanged instead of
    typing it, so a caller that treats typed errors as permanent file failures
    retries these instead. An ``OSError`` without an ``errno`` (a reader that
    reports a format problem as ``OSError("...")``) is not a host condition.
    """
    if is_resource_exhaustion(exc):
        return True
    seen: BaseException | None = exc
    for _ in range(_MAX_CHAIN_DEPTH):
        if seen is None:
            return False
        if isinstance(seen, OSError) and seen.errno in _HOST_IO_ERRNOS:
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def classify_read_error(exc: Exception, filepath: str = "") -> BiosigIOError | NoReturn:
    """Re-raises ``exc`` as-is when it is a host condition (see
    :func:`is_host_condition`); otherwise maps a low-level read failure to a
    typed biosigIO error with a stable code.

    Importers call this in their ``except`` and ``raise ... from exc`` the result,
    so the *why* is decided once, where the format context lives. An exception
    that is already a :class:`BiosigIOError` is returned unchanged.

    The host-condition re-raise lives here, not at each call site, so no importer
    can forget it: every path that would type an exception goes through this
    function, and an exception it re-raises keeps its own type, traceback and
    chain. Resource exhaustion is additionally guarded by the importers *before*
    calling this; host I/O errors (``EACCES``, ``EIO``, ``ENOSPC``, ...) rely on
    this check alone.

    The matching is deliberately reader-agnostic (MNE and pyedflib phrase the same
    condition differently); unmatched failures fall through to :class:`FileReadError`
    so a caller always gets a typed error, never a bare ``Exception``.
    """
    if isinstance(exc, BiosigIOError):
        return exc
    if is_host_condition(exc):
        raise exc
    msg = str(exc)
    low = msg.lower()
    where = f" ({filepath})" if filepath else ""

    # Evoked / epoched / averaged derivatives -> no continuous raw time series.
    # MNE: "No raw data in <f>", "... is not a raw file", read_raw on an -ave/-epo.
    if (
        "no raw data" in low
        or "is not a raw" in low
        or "does not contain raw" in low
        or "this is not a raw" in low
    ):
        return NotContinuousRecordingError(f"No continuous raw recording{where}: {msg}")

    # Truncated / not format-compliant / incomplete set.
    # pyedflib: "... not EDF(+) or BDF(+) compliant (Filesize)"; MNE CTF/FIF:
    # incomplete .meg4; split chain: "... split-02 ... does not exist".
    if (
        "filesize" in low
        or "not edf" in low
        or "not bdf" in low
        or "compliant" in low
        or "truncate" in low  # matches "truncated" too
        or "corrupt" in low
        or ("split" in low and "does not exist" in low)
        # CTF .meg4 chopped mid-trial (read_raw_ctf): the data no longer divides
        # into whole trials -- the on004398 truncation signature.
        or "even multiple of the trial" in low
        or "not an even multiple" in low
    ):
        return CorruptFileError(f"Truncated, incomplete, or corrupt file{where}: {msg}")

    # Recognized format, no specific cause -> generic (still typed) read error.
    return FileReadError(f"Could not read recording{where}: {msg}")
