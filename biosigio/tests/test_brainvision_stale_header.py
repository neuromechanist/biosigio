"""BrainVision headers naming pre-BIDS-rename files (NEMAR on005520, on002158).

BIDS renames the ``.eeg``/``.vmrk`` on disk but not the ``DataFile=``/``MarkerFile=``
references inside the ``.vhdr``; the renamed files sit beside the header with its
stem. Both the in-memory importer and the streaming ``_MneSource`` path must read
such a triplet, leave a correct header alone, and still fail when no sibling exists.
Real tiny triplets are written by hand in each test. NO MOCKS.
"""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")

from biosigio import Recording  # noqa: E402
from biosigio.exporters.zarr_stream import _open_stream_source  # noqa: E402
from biosigio.importers.brainvision import resolved_vhdr  # noqa: E402

N_CH, N_SAMPLES, SFREQ = 3, 500, 250.0
STEM = "sub-01_task-MOBAgame_eeg"


def _write_triplet(
    directory,
    data_ref: str,
    marker_ref: str,
    *,
    data_ext: str = ".eeg",
    newline: str = "\n",
    codepage: str = "UTF-8",
    encoding: str = "utf-8",
    channel_names: tuple[str, ...] | None = None,
) -> tuple[str, np.ndarray]:
    """Write ``<STEM>.vhdr`` referencing ``data_ref``/``marker_ref``, plus the
    correctly named ``<STEM><data_ext>`` and ``<STEM>.vmrk`` beside it.

    Returns the header path and the int16 data (channels x samples) written.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, size=(N_CH, N_SAMPLES), dtype=np.int16)
    data.T.tofile(os.path.join(directory, STEM + data_ext))  # MULTIPLEXED
    vmrk = [
        "Brain Vision Data Exchange Marker File, Version 1.0",
        "[Common Infos]",
        f"Codepage={codepage}",
        f"DataFile={data_ref}",
        "[Marker Infos]",
        "Mk1=Stimulus,S  1,125,1,0",
        "Mk2=Stimulus,S  2,250,1,0",
    ]
    with open(os.path.join(directory, STEM + ".vmrk"), "w", encoding=encoding, newline="") as f:
        f.write(newline.join(vmrk) + newline)
    names = channel_names or tuple(f"C{i + 1}é" for i in range(N_CH))
    channels = [f"Ch{i + 1}={name},,0.1,µV" for i, name in enumerate(names)]
    vhdr = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; Écrit à la main",
        "[Common Infos]",
        f"Codepage={codepage}",
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
        # exercises resolved_vhdr's marker patch only on MNE 1.12.x.
        (f"{STEM}.eeg", "sub-01_task-MOBA_game_eeg.vmrk"),
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
    vhdr, _ = _write_triplet(tmp_path, "s13_run2_060717.eeg", "s13_run2_060717.vmrk")
    os.remove(tmp_path / f"{STEM}.eeg")
    with pytest.raises(Exception, match="s13_run2_060717.eeg"):
        Recording.from_file(vhdr)
    with pytest.raises(FileNotFoundError, match="s13_run2_060717.eeg"):
        _open_stream_source(vhdr, None)


@pytest.mark.parametrize("scratch", ["missing", "read-only"])
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


def test_unreadable_header_yields_original_without_chained_context(tmp_path):
    """A header the resolver cannot read is handed to MNE unchanged, and an error
    raised inside the block is not chained to the resolver's own OSError."""
    missing = str(tmp_path / "absent.vhdr")
    with pytest.raises(RuntimeError) as info:
        with resolved_vhdr(missing) as used:
            assert used == missing
            raise RuntimeError("raised by the caller")
    assert info.value.__context__ is None
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
