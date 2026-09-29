"""BrainVision headers naming pre-BIDS-rename files (NEMAR on005520, on002158).

BIDS renames the ``.eeg``/``.vmrk`` on disk but not the ``DataFile=``/``MarkerFile=``
references inside the ``.vhdr``; the renamed files sit beside the header with its
stem. Both the in-memory importer and the streaming ``_MneSource`` path must read
such a triplet, leave a correct header alone, and still fail when no sibling exists.
Real tiny triplets are written by hand in each test. NO MOCKS.
"""

import os
import pathlib
import sys

import numpy as np
import pytest

mne = pytest.importorskip(
    "mne", reason="BrainVision import requires the optional 'meg' extra (mne)"
)

from biosigio import Recording  # noqa: E402
from biosigio.exceptions import BiosigIOError, CorruptFileError, FileReadError  # noqa: E402
from biosigio.exporters.zarr_stream import _open_stream_source  # noqa: E402
from biosigio.importers.brainvision import (  # noqa: E402
    HEADER_RECOVERED_KEY,
    BrainVisionHeaderRecoveryError,
    resolved_vhdr,
)

# MNE >= 1.13 recovers a stale MarkerFile= on its own; below that it does not.
_MNE_RECOVERS_MARKER = tuple(int(p) for p in mne.__version__.split(".")[:2]) >= (1, 13)

N_CH, N_SAMPLES, SFREQ = 3, 500, 250.0
STEM = "sub-01_task-MOBAgame_eeg"


def _write_triplet(
    directory,
    data_ref: str,
    marker_ref: str,
    *,
    data_ext: str = ".eeg",
    newline: str = "\n",
    codepage: str | None = "UTF-8",
    encoding: str = "utf-8",
    channel_names: tuple[str, ...] | None = None,
    marker_ext: str = ".vmrk",
) -> tuple[str, np.ndarray]:
    """Write ``<STEM>.vhdr`` referencing ``data_ref``/``marker_ref``, plus the
    correctly named ``<STEM><data_ext>`` and ``<STEM><marker_ext>`` beside it.
    ``codepage=None`` leaves the ``Codepage=`` line out of both files.

    Returns the header path and the int16 data (channels x samples) written.
    """
    codepage_line = [] if codepage is None else [f"Codepage={codepage}"]
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, size=(N_CH, N_SAMPLES), dtype=np.int16)
    data.T.tofile(os.path.join(directory, STEM + data_ext))  # MULTIPLEXED
    vmrk = [
        "Brain Vision Data Exchange Marker File, Version 1.0",
        "[Common Infos]",
        *codepage_line,
        f"DataFile={data_ref}",
        "[Marker Infos]",
        "Mk1=Stimulus,S  1,125,1,0",
        "Mk2=Stimulus,S  2,250,1,0",
    ]
    with open(os.path.join(directory, STEM + marker_ext), "w", encoding=encoding, newline="") as f:
        f.write(newline.join(vmrk) + newline)
    names = channel_names or tuple(f"C{i + 1}é" for i in range(N_CH))
    channels = [f"Ch{i + 1}={name},,0.1,µV" for i, name in enumerate(names)]
    vhdr = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; Écrit à la main",
        "[Common Infos]",
        *codepage_line,
        f"DataFile={data_ref}",
        f"MarkerFile={marker_ref}",
        "DataFormat=BINARY",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={N_CH}",
        f"SamplingInterval={1e6 / SFREQ:g}",
        "[Binary Infos]",
        "BinaryFormat=INT_16",
        "[Channel Infos]",
        *channels,
    ]
    vhdr_path = os.path.join(directory, STEM + ".vhdr")
    with open(vhdr_path, "w", encoding=encoding, newline="") as f:
        f.write(newline.join(vhdr) + newline)
    return vhdr_path, data


def _snapshot(directory) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in pathlib.Path(directory).iterdir()}


def _assert_loaded(rec: Recording, data: np.ndarray) -> None:
    assert rec.signals is not None
    assert rec.signals.shape == (N_SAMPLES, N_CH)
    assert {i["sample_frequency"] for i in rec.channels.values()} == {SFREQ}
    # 0.1 uV resolution -> volts.
    np.testing.assert_allclose(rec.signals.to_numpy().T, data * 0.1e-6)
    assert rec.events is not None and len(rec.events) == 2
    np.testing.assert_allclose(rec.events["onset"].to_numpy(), [124 / SFREQ, 249 / SFREQ])


def _assert_streams(vhdr_path: str, data: np.ndarray) -> None:
    source = _open_stream_source(vhdr_path, None)
    try:
        assert source.sfreq == SFREQ
        assert source.n_samples == N_SAMPLES
        assert len(source.channels) == N_CH
        np.testing.assert_allclose(source.read(list(range(N_CH)), 0, N_SAMPLES), data * 0.1e-6)
        np.testing.assert_allclose(source.read([1], 100, 200), data[1:2, 100:200] * 0.1e-6)
    finally:
        source.close()


def test_matching_header_is_read_unchanged(tmp_path):
    vhdr, data = _write_triplet(tmp_path, f"{STEM}.eeg", f"{STEM}.vmrk")
    with resolved_vhdr(vhdr) as used:
        assert used == vhdr
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


@pytest.mark.parametrize(
    ("data_ref", "marker_ref"),
    [
        ("s13_run2_060717.eeg", "s13_run2_060717.vmrk"),
        ("sub-01_task-MOBA_game_eeg.eeg", "sub-01_task-MOBA_game_eeg.vmrk"),
        # Only the marker is stale. MNE >= 1.13 recovers this itself, so the row
        # would pass with or without resolved_vhdr's marker patch there; it is
        # skipped rather than reported as proof, and runs on MNE < 1.13 (the
        # 1.12.x line NEMAR runs, pinned by a separate CI job).
        pytest.param(
            f"{STEM}.eeg",
            "sub-01_task-MOBA_game_eeg.vmrk",
            marks=pytest.mark.skipif(
                _MNE_RECOVERS_MARKER,
                reason=f"MNE {mne.__version__} recovers a stale MarkerFile= itself, so "
                "this row cannot prove resolved_vhdr's marker patch; it runs on MNE < 1.13",
            ),
            id="marker-only",
        ),
    ],
)
def test_stale_references_resolve_to_same_stem_siblings(tmp_path, data_ref, marker_ref):
    vhdr, data = _write_triplet(tmp_path, data_ref, marker_ref)
    before = _snapshot(tmp_path)
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)
    assert _snapshot(tmp_path) == before  # the dataset files are never touched


def test_legacy_dat_sibling_is_tolerated(tmp_path):
    vhdr, data = _write_triplet(tmp_path, "old_name.dat", "old_name.vmrk", data_ext=".dat")
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


@pytest.mark.parametrize(
    ("newline", "codepage", "encoding"),
    [("\r\n", "ANSI", "cp1252"), ("\n", "UTF-8", "utf-8")],
)
def test_patched_copy_keeps_encoding_and_line_endings(tmp_path, newline, codepage, encoding):
    vhdr, data = _write_triplet(
        tmp_path, "old.eeg", "old.vmrk", newline=newline, codepage=codepage, encoding=encoding
    )
    original = open(vhdr, "rb").read()
    with resolved_vhdr(vhdr) as used:
        assert used != vhdr
        patched = open(used, "rb").read()
    assert not os.path.exists(used)  # the temporary copy is cleaned up
    expected = original.replace(
        b"DataFile=old.eeg", f"DataFile={tmp_path / (STEM + '.eeg')}".encode(encoding)
    ).replace(b"MarkerFile=old.vmrk", f"MarkerFile={tmp_path / (STEM + '.vmrk')}".encode(encoding))
    assert patched == expected
    _assert_loaded(Recording.from_file(vhdr), data)


def test_missing_sibling_still_errors(tmp_path):
    """No sibling to fall back on: both paths raise the same typed read error."""
    vhdr, _ = _write_triplet(tmp_path, "s13_run2_060717.eeg", "s13_run2_060717.vmrk")
    os.remove(tmp_path / f"{STEM}.eeg")
    with pytest.raises(FileReadError, match="s13_run2_060717.eeg") as in_memory:
        Recording.from_file(vhdr)
    with pytest.raises(FileReadError, match="s13_run2_060717.eeg") as streamed:
        _open_stream_source(vhdr, None)
    assert type(streamed.value) is type(in_memory.value)
    assert isinstance(streamed.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("stale", [False, True], ids=["matching", "stale"])
def test_corrupt_header_is_typed_on_both_paths(tmp_path, stale):
    """A header MNE cannot parse is a typed (terminal) read error on the streaming
    path too, not a raw MNE exception a caller would retry forever; and when MNE
    read the patched temp copy, the message names the real header, not the
    deleted ``biosigio-vhdr-*`` copy."""
    refs = ("old.eeg", "old.vmrk") if stale else (f"{STEM}.eeg", f"{STEM}.vmrk")
    vhdr, _ = _write_triplet(tmp_path, *refs)
    text = open(vhdr, encoding="utf-8").read()
    # No SamplingInterval: MNE raises, quoting the header path it was handed.
    open(vhdr, "w", encoding="utf-8").write(text.replace(f"SamplingInterval={1e6 / SFREQ:g}\n", ""))

    errors = []
    with pytest.raises(FileReadError) as in_memory:
        Recording.from_file(vhdr)
    errors.append(in_memory.value)
    with pytest.raises(FileReadError) as streamed:
        _open_stream_source(vhdr, None)
    errors.append(streamed.value)

    for err in errors:
        message = str(err)
        assert "biosigio-vhdr-" not in message
        assert vhdr in message
        if "SamplingInterval from" in message:  # MNE's own wording quotes the path
            assert f"SamplingInterval from {vhdr}" in message
    assert type(errors[0]) is type(errors[1])


@pytest.mark.parametrize(
    "scratch",
    [
        "missing",
        pytest.param(
            "read-only",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="chmod(0o500) does not make a Windows directory unwritable",
            ),
        ),
    ],
)
def test_temp_copy_failure_is_a_host_error_not_a_file_error(tmp_path, monkeypatch, scratch):
    """A temp-copy write that the OS refuses (here ENOENT/EACCES; ENOSPC, EROFS and
    quota take the same path) propagates as the raw OSError, never as a typed
    BiosigIOError that a caller could record as a permanent file failure."""
    import errno
    import tempfile

    from biosigio.exceptions import BiosigIOError

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vhdr, _ = _write_triplet(data_dir, "old.eeg", "old.vmrk")
    tmp_root = tmp_path / "scratch"
    if scratch == "read-only":
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        tmp_root.mkdir()
        tmp_root.chmod(0o500)
        expected = errno.EACCES
    else:
        expected = errno.ENOENT
    # A real temp root the OS cannot write to: no business logic is replaced.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_root))
    try:
        with pytest.raises(OSError) as info:
            Recording.from_file(vhdr)
        assert not isinstance(info.value, BiosigIOError)
        assert info.value.errno == expected
        with pytest.raises(OSError) as info:
            _open_stream_source(vhdr, None)
        assert not isinstance(info.value, BiosigIOError)
    finally:
        if tmp_root.exists():
            tmp_root.chmod(0o700)


def test_nel_inside_a_latin1_channel_name_is_not_a_line_break(tmp_path):
    """``\\x85`` is NEL in Latin-1; ``str.splitlines`` would break on it and could
    see a ``[Common Infos]`` section and a ``DataFile=`` key inside a channel name."""
    tricky = "C1\x85[Common Infos]\x85DataFile=x.eeg"
    vhdr, data = _write_triplet(
        tmp_path,
        "old.eeg",
        "old.vmrk",
        codepage="Latin-1",
        encoding="latin-1",
        channel_names=(tricky, "C2", "C3"),
    )
    original = open(vhdr, "rb").read()
    with resolved_vhdr(vhdr) as used:
        patched = open(used, "rb").read()
    expected = original.replace(
        b"DataFile=old.eeg", f"DataFile={tmp_path / (STEM + '.eeg')}".encode("latin-1")
    ).replace(b"MarkerFile=old.vmrk", f"MarkerFile={tmp_path / (STEM + '.vmrk')}".encode("latin-1"))
    assert patched == expected
    rec = Recording.from_file(vhdr)
    _assert_loaded(rec, data)
    assert tricky in rec.channels


def test_lone_cr_header_is_patched_line_by_line(tmp_path):
    """Old-Mac ``\\r`` line endings still split into lines (as ``splitlines`` did)."""
    vhdr, _ = _write_triplet(tmp_path, "old.eeg", "old.vmrk", newline="\r")
    original = open(vhdr, "rb").read()
    with resolved_vhdr(vhdr) as used:
        patched = open(used, "rb").read()
    expected = original.replace(
        b"DataFile=old.eeg", f"DataFile={tmp_path / (STEM + '.eeg')}".encode()
    ).replace(b"MarkerFile=old.vmrk", f"MarkerFile={tmp_path / (STEM + '.vmrk')}".encode())
    assert patched == expected


def test_unreadable_header_yields_original_without_chained_context(tmp_path, caplog):
    """A header the resolver cannot read is handed to MNE unchanged, and an error
    raised inside the block is not chained to the resolver's own OSError."""
    missing = str(tmp_path / "absent.vhdr")
    with caplog.at_level("WARNING", logger="biosigio.importers.brainvision"):
        with pytest.raises(RuntimeError) as info:
            with resolved_vhdr(missing) as used:
                assert used == missing
                raise RuntimeError("raised by the caller")
    assert info.value.__context__ is None
    # The swallowed OSError is logged with its errno, not dropped.
    assert "ENOENT" in caplog.text and missing in caplog.text
    with pytest.raises(Exception, match="absent"):
        Recording.from_file(missing)


def test_path_outside_the_codepage_is_written_as_utf8(tmp_path):
    """A cp1252 header whose siblings live under a Greek directory name: the patched
    copy is UTF-8, declares ``Codepage=UTF-8`` and keeps CRLF, instead of silently
    falling back to the stale header."""
    directory = tmp_path / "données_Ωμέγα"
    directory.mkdir()
    vhdr, data = _write_triplet(
        directory, "old.eeg", "old.vmrk", newline="\r\n", codepage="ANSI", encoding="cp1252"
    )
    original = open(vhdr, "rb").read().decode("cp1252")
    with resolved_vhdr(vhdr) as used:
        assert used != vhdr
        patched = open(used, "rb").read()
    expected = (
        original.replace("Codepage=ANSI", "Codepage=UTF-8")
        .replace("DataFile=old.eeg", f"DataFile={directory / (STEM + '.eeg')}")
        .replace("MarkerFile=old.vmrk", f"MarkerFile={directory / (STEM + '.vmrk')}")
        .encode("utf-8")
    )
    assert patched == expected
    rec = Recording.from_file(vhdr)
    _assert_loaded(rec, data)
    assert "C1é" in rec.channels
    _assert_streams(vhdr, data)


def test_eeg_sibling_wins_over_dat(tmp_path):
    vhdr, data = _write_triplet(tmp_path, "old.eeg", "old.vmrk")
    decoy = (data[::-1] // 2).astype(np.int16)  # different values, same shape
    decoy.T.tofile(tmp_path / f"{STEM}.dat")
    with resolved_vhdr(vhdr) as used:
        assert f"DataFile={tmp_path / (STEM + '.eeg')}" in open(used, encoding="utf-8").read()
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


def test_header_with_utf8_bom(tmp_path):
    vhdr, data = _write_triplet(tmp_path, "old.eeg", "old.vmrk", encoding="utf-8-sig")
    original = open(vhdr, "rb").read()
    assert original.startswith(b"\xef\xbb\xbf")
    with resolved_vhdr(vhdr) as used:
        patched = open(used, "rb").read()
    expected = original.replace(
        b"DataFile=old.eeg", f"DataFile={tmp_path / (STEM + '.eeg')}".encode()
    ).replace(b"MarkerFile=old.vmrk", f"MarkerFile={tmp_path / (STEM + '.vmrk')}".encode())
    assert patched == expected  # the BOM is kept, not doubled or dropped
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


def test_windows_style_stale_reference(tmp_path):
    vhdr, data = _write_triplet(tmp_path, r"C:\data\old.eeg", r"C:\data\old.vmrk")
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)


def test_concurrent_calls_use_distinct_temp_copies(tmp_path):
    """Two threads resolving the same stale header at once each get their own copy,
    and one leaving does not remove the other's."""
    import threading

    vhdr, data = _write_triplet(tmp_path, "old.eeg", "old.vmrk")
    inside = threading.Barrier(2, timeout=30)
    first_left = threading.Event()
    used: list[str] = []
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            with resolved_vhdr(vhdr) as path:
                used.append(path)
                inside.wait()  # both copies exist at the same time
                if index == 1:
                    assert first_left.wait(timeout=30)
                    # Thread 0 has removed its copy; this one must be untouched.
                    assert os.path.isfile(path)
                    _assert_loaded(Recording.from_file(path), data)
            if index == 0:
                assert not os.path.exists(path)
        except BaseException as e:  # surfaced in the main thread below
            errors.append(e)
        finally:
            # Set however thread 0 ends, so a failure there fails the test at
            # once instead of leaving thread 1 waiting out its timeout.
            if index == 0:
                first_left.set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert len(used) == 2
    assert os.path.dirname(used[0]) != os.path.dirname(used[1])
    assert not any(os.path.exists(p) for p in used)


# --- recovery is recorded, not silent ------------------------------------------


def test_recovery_is_logged_and_recorded_in_metadata(tmp_path, caplog):
    """A successful stale-header read leaves a trace: an info log, and the
    substituted names in the Recording's metadata, the streaming source's
    metadata, and the Zarr store written from either."""
    vhdr, data = _write_triplet(tmp_path, "old.eeg", "old.vmrk")
    expected = {
        "DataFile": {"referenced": "old.eeg", "used": f"{STEM}.eeg"},
        "MarkerFile": {"referenced": "old.vmrk", "used": f"{STEM}.vmrk"},
    }

    with caplog.at_level("INFO", logger="biosigio.importers.brainvision"):
        rec = Recording.from_file(vhdr)
    _assert_loaded(rec, data)
    assert rec.metadata[HEADER_RECOVERED_KEY] == expected
    assert "old.eeg -> " + f"{STEM}.eeg" in caplog.text

    source = _open_stream_source(vhdr, None)
    try:
        assert source.extra_metadata[HEADER_RECOVERED_KEY] == expected
    finally:
        source.close()

    pytest.importorskip("zarr", reason="the store round trip needs the 'zarr' extra")
    from biosigio import stream_to_zarr

    streamed = stream_to_zarr(vhdr, str(tmp_path / "s.zarr"), bids_channels="off")
    in_memory = rec.to_zarr(str(tmp_path / "m.zarr"))
    for store in (streamed, in_memory):
        assert Recording.from_file(store).metadata[HEADER_RECOVERED_KEY] == expected


def test_matching_header_records_nothing(tmp_path):
    vhdr, _ = _write_triplet(tmp_path, f"{STEM}.eeg", f"{STEM}.vmrk")
    substitutions: dict = {}
    with resolved_vhdr(vhdr, substitutions=substitutions) as used:
        assert used == vhdr
    assert substitutions == {}
    assert HEADER_RECOVERED_KEY not in Recording.from_file(vhdr).metadata
    source = _open_stream_source(vhdr, None)
    try:
        assert HEADER_RECOVERED_KEY not in source.extra_metadata
    finally:
        source.close()


# --- case-insensitive sibling lookup -------------------------------------------


def _case_sensitive(directory) -> bool:
    probe = pathlib.Path(directory) / "CaseProbe"
    probe.write_text("")
    try:
        return not (pathlib.Path(directory) / "caseprobe").exists()
    finally:
        probe.unlink()


def test_upper_case_siblings_are_found(tmp_path):
    """``X.EEG``/``X.VMRK`` beside ``X.vhdr`` are found on any filesystem (on a
    case-sensitive one only through the case-insensitive fallback)."""
    vhdr, data = _write_triplet(
        tmp_path, "old.eeg", "old.vmrk", data_ext=".EEG", marker_ext=".VMRK"
    )
    _assert_loaded(Recording.from_file(vhdr), data)
    _assert_streams(vhdr, data)
    rec = Recording.from_file(vhdr)
    recovered = rec.metadata[HEADER_RECOVERED_KEY]
    assert recovered["DataFile"]["used"].lower() == f"{STEM}.eeg".lower()
    assert recovered["MarkerFile"]["used"].lower() == f"{STEM}.vmrk".lower()
    if _case_sensitive(tmp_path):
        assert recovered["DataFile"]["used"] == f"{STEM}.EEG"


def test_ambiguous_case_insensitive_siblings_are_not_guessed(tmp_path):
    """Two files differing only in case: neither is chosen, so the header stays
    unresolved and the read fails on the missing file it names."""
    if not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive filesystem to hold X.EEG and X.Eeg at once")
    vhdr, data = _write_triplet(tmp_path, "old.eeg", f"{STEM}.vmrk", data_ext=".EEG")
    data.T.tofile(tmp_path / f"{STEM}.Eeg")
    substitutions: dict = {}
    with resolved_vhdr(vhdr, substitutions=substitutions) as used:
        assert used == vhdr
    assert substitutions == {}
    with pytest.raises(FileReadError, match="old.eeg"):
        Recording.from_file(vhdr)


# --- encodings the patched copy has to handle ----------------------------------


def test_lone_cr_header_falls_back_to_utf8(tmp_path):
    """Old-Mac ``\\r`` line endings AND a path the cp1252 codepage cannot spell:
    the ``Codepage=`` rewrite finds the version line at the first CR.

    MNE itself cannot parse a lone-CR header (it finds no ``[Common Infos]``
    section even when every reference is correct), so the check is on the
    bytes of the copy; the read then fails exactly as the unpatched header
    would, as a typed error naming the real header rather than the temp copy.
    """
    directory = tmp_path / "Ωμέγα"
    directory.mkdir()
    vhdr, _ = _write_triplet(
        directory, "old.eeg", "old.vmrk", newline="\r", codepage="ANSI", encoding="cp1252"
    )
    original = open(vhdr, "rb").read().decode("cp1252")
    with resolved_vhdr(vhdr) as used:
        patched = open(used, "rb").read()
    expected = (
        original.replace("Codepage=ANSI", "Codepage=UTF-8")
        .replace("DataFile=old.eeg", f"DataFile={directory / (STEM + '.eeg')}")
        .replace("MarkerFile=old.vmrk", f"MarkerFile={directory / (STEM + '.vmrk')}")
        .encode("utf-8")
    )
    assert patched == expected
    assert b"\n" not in patched
    with pytest.raises(FileReadError) as info:
        Recording.from_file(vhdr)
    assert "biosigio-vhdr-" not in str(info.value)


def test_no_codepage_latin1_header_under_a_path_latin1_cannot_spell(tmp_path):
    """No ``Codepage=`` line and bytes that are not UTF-8, so MNE decodes the
    header as Latin-1; the Greek sibling path is not Latin-1, so the copy is
    written as UTF-8. With no ``Codepage=`` to rewrite, MNE decodes that copy as
    UTF-8 (its default), which reads every character back unchanged."""
    directory = tmp_path / "Ωμέγα"
    directory.mkdir()
    vhdr, data = _write_triplet(directory, "old.eeg", "old.vmrk", codepage=None, encoding="latin-1")
    original = open(vhdr, "rb").read()
    with pytest.raises(UnicodeDecodeError):
        original.decode("utf-8")  # the fixture really is not UTF-8
    with resolved_vhdr(vhdr) as used:
        patched = open(used, "rb").read()
    expected = (
        original.decode("latin-1")
        .replace("DataFile=old.eeg", f"DataFile={directory / (STEM + '.eeg')}")
        .replace("MarkerFile=old.vmrk", f"MarkerFile={directory / (STEM + '.vmrk')}")
        .encode("utf-8")
    )
    assert patched == expected
    assert b"Codepage" not in patched
    rec = Recording.from_file(vhdr)
    _assert_loaded(rec, data)
    assert "C1é" in rec.channels
    _assert_streams(vhdr, data)


def _undecodable_dir(tmp_path) -> pathlib.Path:
    """A directory whose name holds bytes that are not UTF-8, so Python spells it
    with surrogate escapes that no codec can encode. Skips where the filesystem
    refuses such a name (macOS APFS, Windows)."""
    directory = tmp_path / os.fsdecode(b"raw-\xff\xfe")
    try:
        directory.mkdir()
    except (OSError, UnicodeEncodeError) as err:
        pytest.skip(f"this filesystem refuses a non-UTF-8 directory name ({err})")
    return directory


@pytest.mark.parametrize("codepage", ["UTF-8", None], ids=["utf8", "no-codepage"])
def test_path_no_encoding_can_spell_raises_a_host_error(tmp_path, codepage):
    """Siblings exist but not even UTF-8 can spell their path: a specific,
    untyped error that says so, instead of handing MNE the stale header and
    letting its FileNotFoundError be recorded as a permanent file failure."""
    directory = _undecodable_dir(tmp_path)
    vhdr, _ = _write_triplet(directory, "old.eeg", "old.vmrk", codepage=codepage)

    with pytest.raises(BrainVisionHeaderRecoveryError, match="names missing files"):
        with resolved_vhdr(vhdr):
            pass
    for read in (Recording.from_file, lambda p: _open_stream_source(p, None)):
        with pytest.raises(BrainVisionHeaderRecoveryError) as info:
            read(vhdr)
        assert not isinstance(info.value, BiosigIOError)
        assert "DataFile=old.eeg" in str(info.value)


def test_header_read_exhaustion_is_raised_not_swallowed(tmp_path):
    """Out of file descriptors while opening the header is the host, not the
    header: it propagates as the raw OSError (retryable) instead of being
    logged and handed to MNE. Real exhaustion, under a lowered fd limit."""
    resource = pytest.importorskip("resource", reason="needs POSIX rlimits")
    import errno

    vhdr, _ = _write_triplet(tmp_path, "old.eeg", "old.vmrk")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    held = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 256), hard))
    try:
        with pytest.raises(OSError) as info:
            while True:  # use up every descriptor the lowered limit allows
                held.append(open(os.devnull, "rb"))
        assert info.value.errno == errno.EMFILE
        with pytest.raises(OSError) as info:
            with resolved_vhdr(vhdr):
                pass
        assert info.value.errno == errno.EMFILE
        assert not isinstance(info.value, BiosigIOError)
    finally:
        for f in held:
            f.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_corrupt_error_type_is_kept_when_the_path_is_rewritten(tmp_path):
    """Rewriting the temp path out of a message keeps the classified type."""
    from biosigio.importers.brainvision import brainvision_read_error

    used = str(tmp_path / "biosigio-vhdr-x" / f"{STEM}.vhdr")
    real = str(tmp_path / f"{STEM}.vhdr")
    err = brainvision_read_error(OSError(f"{used}: file is truncated"), used, real)
    assert type(err) is CorruptFileError
    assert used not in str(err) and real in str(err)
