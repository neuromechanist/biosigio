"""Repeated channel labels outside EDF/BDF: every channel survives, under a stable name.

A Recording is keyed by channel label, so any path that writes a label twice can
drop a channel (a signal column overwritten) or misattribute one (a sidecar row
for a genuine label landing on a synthesized one). These tests build real files
(XDF, EEGLAB ``.set``, neo, CSV, Parquet) and check that the channel count and
the samples behind each label survive. The EDF/BDF side lives in
``test_edf_duplicate_labels.py``.
"""

import struct

import numpy as np
import pytest

from biosigio import Recording
from biosigio.importers._labels import DEDUPLICATED_LABELS_KEY, suffix_repeated_labels
from biosigio.importers.csv import CSVImporter
from biosigio.importers.trigno import TrignoImporter
from biosigio.importers.xdf import XDFImporter


def test_suffix_keeps_the_first_occurrence():
    labels, renames = suffix_repeated_labels(["Fz", "Cz", "Fz", "Fz"], start=2)
    assert labels == ["Fz", "Cz", "Fz_2", "Fz_3"]
    assert renames == {"Fz_2": "Fz", "Fz_3": "Fz"}


def test_suffix_never_takes_a_genuine_label():
    """A genuine ``Fz_2`` keeps its name even when it comes after the repeat."""
    labels, renames = suffix_repeated_labels(["Fz", "Fz", "Fz_2"], start=2)
    assert labels == ["Fz", "Fz_3", "Fz_2"]
    assert renames == {"Fz_3": "Fz"}


def test_suffix_skips_names_already_synthesized():
    labels, _ = suffix_repeated_labels(["a", "a", "a_1", "a_1"], start=1)
    assert labels == ["a", "a_2", "a_1", "a_1_1"]
    assert len(set(labels)) == 4


def test_suffix_leaves_unique_labels_alone():
    assert suffix_repeated_labels(["x", "y"]) == (["x", "y"], {})


# --- XDF --------------------------------------------------------------------


def _varlen(value: int) -> bytes:
    if value < 256:
        return struct.pack("<BB", 1, value)
    if value < 2**32:
        return struct.pack("<BI", 4, value)
    return struct.pack("<BQ", 8, value)


def _chunk(tag: int, content: bytes, stream_id: int | None = None) -> bytes:
    body = struct.pack("<H", tag)
    if stream_id is not None:
        body += struct.pack("<I", stream_id)
    body += content
    return _varlen(len(body)) + body


def _write_xdf(path, streams):
    """Write a real XDF 1.0 file that pyxdf reads.

    ``streams`` is a list of ``(name, labels, data)`` with ``data`` shaped
    (samples, channels), all sampled at 100 Hz from t = 1 s.
    """
    out = b"XDF:" + _chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    for sid, (name, labels, _data) in enumerate(streams, start=1):
        channels = "".join(
            f"<channel><label>{label}</label><unit>uV</unit></channel>" for label in labels
        )
        header = (
            f'<?xml version="1.0"?><info><name>{name}</name><type>EEG</type>'
            f"<channel_count>{len(labels)}</channel_count><nominal_srate>100</nominal_srate>"
            f"<channel_format>double64</channel_format><source_id>{name}{sid}</source_id>"
            f"<version>1.0</version><desc><channels>{channels}</channels></desc></info>"
        )
        out += _chunk(2, header.encode(), sid)
    for sid, (_name, _labels, data) in enumerate(streams, start=1):
        samples = _varlen(len(data))
        for i, row in enumerate(data):
            samples += struct.pack("<Bd", 8, 1.0 + i / 100.0)
            samples += struct.pack(f"<{len(row)}d", *row)
        out += _chunk(3, samples, sid)
    for sid, (_name, _labels, data) in enumerate(streams, start=1):
        footer = (
            f'<?xml version="1.0"?><info><first_timestamp>1.0</first_timestamp>'
            f"<last_timestamp>{1.0 + (len(data) - 1) / 100.0}</last_timestamp>"
            f"<sample_count>{len(data)}</sample_count></info>"
        )
        out += _chunk(6, footer.encode(), sid)
    path.write_bytes(out)
    return str(path)


def _block(offset: float, n_channels: int, n_samples: int = 200) -> np.ndarray:
    """Samples no other channel shares, so a column can be traced to its source."""
    return offset + np.arange(n_samples * n_channels, dtype=float).reshape(n_samples, n_channels)


def test_xdf_streams_sharing_labels_keep_every_channel(tmp_path):
    a, b = _block(0.0, 2), _block(1e6, 2)
    path = _write_xdf(
        tmp_path / "two.xdf", [("AmpA", ["Ch1", "Ch2"], a), ("AmpB", ["Ch1", "Ch2"], b)]
    )

    rec = XDFImporter().load(path)

    assert list(rec.signals.columns) == ["Ch1", "Ch2", "Ch1_1", "Ch2_1"]
    assert list(rec.channels) == list(rec.signals.columns)
    np.testing.assert_array_equal(rec.signals["Ch1"].to_numpy(), a[:, 0])
    np.testing.assert_array_equal(rec.signals["Ch2_1"].to_numpy(), b[:, 1])
    assert rec.metadata[DEDUPLICATED_LABELS_KEY] == {"Ch1_1": "Ch1", "Ch2_1": "Ch2"}


def test_xdf_suffix_never_takes_a_genuine_label(tmp_path):
    """Stream B's genuine ``Ch1_1`` keeps its name; stream A's repeat gets ``Ch1_2``."""
    a, b = _block(0.0, 2), _block(1e6, 1)
    path = _write_xdf(
        tmp_path / "genuine.xdf", [("AmpA", ["Ch1", "Ch1"], a), ("AmpB", ["Ch1_1"], b)]
    )

    rec = XDFImporter().load(path)

    assert list(rec.signals.columns) == ["Ch1", "Ch1_2", "Ch1_1"]
    np.testing.assert_array_equal(rec.signals["Ch1_1"].to_numpy(), b[:, 0])
    np.testing.assert_array_equal(rec.signals["Ch1_2"].to_numpy(), a[:, 1])
    assert rec.metadata[DEDUPLICATED_LABELS_KEY] == {"Ch1_2": "Ch1"}


def test_xdf_same_named_streams_keep_both_timestamp_channels(tmp_path):
    a, b = _block(0.0, 1), _block(1e6, 1)
    path = _write_xdf(tmp_path / "names.xdf", [("Amp", ["L"], a), ("Amp", ["R"], b)])

    rec = XDFImporter().load(path, include_timestamps=True)

    assert list(rec.signals.columns) == ["L", "R", "Amp_LSL_timestamps", "Amp_LSL_timestamps_1"]
    assert list(rec.channels) == list(rec.signals.columns)
    assert rec.metadata[DEDUPLICATED_LABELS_KEY] == {"Amp_LSL_timestamps_1": "Amp_LSL_timestamps"}


def test_xdf_timestamp_channel_never_overwrites_a_data_channel(tmp_path):
    """A stream that already carries ``<name>_LSL_timestamps`` (a re-written export) keeps it."""
    data = _block(5e5, 2)
    path = _write_xdf(tmp_path / "ts.xdf", [("Amp", ["C3", "Amp_LSL_timestamps"], data)])

    rec = XDFImporter().load(path, include_timestamps=True)

    assert list(rec.signals.columns) == ["C3", "Amp_LSL_timestamps", "Amp_LSL_timestamps_1"]
    np.testing.assert_array_equal(rec.signals["Amp_LSL_timestamps"].to_numpy(), data[:, 1])
    assert rec.channels["Amp_LSL_timestamps"]["physical_dimension"] == "uV"
    assert rec.channels["Amp_LSL_timestamps_1"]["physical_dimension"] == "s"


def test_xdf_unique_labels_record_no_renames(tmp_path):
    path = _write_xdf(tmp_path / "u.xdf", [("Amp", ["C3", "C4"], _block(0.0, 2))])
    rec = XDFImporter().load(path)
    assert list(rec.signals.columns) == ["C3", "C4"]
    assert DEDUPLICATED_LABELS_KEY not in rec.metadata


# --- Recording.select_channels ------------------------------------------------


def test_select_channels_refuses_a_name_listed_twice(tmp_path):
    path = _write_xdf(tmp_path / "sel.xdf", [("Amp", ["C3", "C4"], _block(0.0, 2))])
    rec = XDFImporter().load(path)
    with pytest.raises(ValueError, match=r"more than once: \['C3'\]"):
        rec.select_channels(["C3", "C4", "C3"])
    subset = rec.select_channels(["C4", "C3"])
    assert list(subset.signals.columns) == list(subset.channels) == ["C4", "C3"]


# --- CSV ----------------------------------------------------------------------


def _write_csv(path, header: str) -> str:
    rows = "\n".join(f"{i},{i + 10},{i + 20}" for i in (3, 1, 2))
    path.write_text(f"{header}\n{rows}\n")
    return str(path)


def test_csv_repeated_channel_names_are_refused_by_name(tmp_path):
    path = _write_csv(tmp_path / "x.csv", "a,b,c")
    with pytest.raises(ValueError, match=r"must be unique; repeated: \['A'\]"):
        CSVImporter().load(
            path, force_generic=True, sample_frequency=100, channel_names=["A", "A", "B"]
        )
    with pytest.raises(ValueError, match=r"must be unique; repeated: \['a'\]"):
        CSVImporter().load(path, force_generic=True, sample_frequency=100, columns=["a", "a"])


def test_csv_repeated_header_keeps_every_column(tmp_path):
    """pandas renames a repeated header, so every column still becomes a channel."""
    path = _write_csv(tmp_path / "h.csv", "a,a,a")
    rec = CSVImporter().load(path, force_generic=True, sample_frequency=100)
    assert len(rec.channels) == 3
    assert [rec.signals[c].iloc[0] for c in rec.signals.columns] == [3, 13, 23]


# --- Delsys Trigno --------------------------------------------------------------


def _write_trigno(path, labels: list[str]) -> str:
    """A Trigno export in the layout of ``examples/truncated_trigno_sample.csv``."""
    meta = [
        f"Label: {label} Sampling frequency: 1.000000e+002 Number of points: 3 "
        "start: 0.000000e+000 Unit: V Domain Unit: s"
        for label in labels
    ]
    header = ",".join(f'X[s],"{label}"' for label in labels)
    rows = [",".join(f"{t / 100},{t + 10 * i}" for i in range(len(labels))) for t in range(3)]
    path.write_text("\n".join([*meta, "", header, *rows]) + "\n")
    return str(path)


def test_trigno_distinct_labels_import(tmp_path):
    path = _write_trigno(tmp_path / "ok.csv", ["Sensor 1: EMG 1", "Sensor 2: EMG 2"])
    rec = TrignoImporter().load(path)
    assert list(rec.channels) == ["Sensor 1: EMG 1", "Sensor 2: EMG 2"]


def test_trigno_repeated_label_is_refused_not_dropped(tmp_path):
    path = _write_trigno(tmp_path / "dup.csv", ["Sensor 1: EMG 1", "Sensor 1: EMG 1"])
    with pytest.raises(ValueError, match="repeats the channel label 'Sensor 1: EMG 1'"):
        TrignoImporter().load(path)


# --- Arrow / Feather ------------------------------------------------------------


def test_feather_with_a_repeated_column_is_refused(tmp_path):
    """An edited Feather file naming two columns alike cannot become a Recording."""
    feather = pytest.importorskip("pyarrow.feather")
    pa = pytest.importorskip("pyarrow")
    rec = XDFImporter().load(_write_xdf(tmp_path / "t.xdf", [("A", ["C3", "C4"], _block(0.0, 2))]))
    written = rec.to_arrow(str(tmp_path / "t.feather"))
    table = feather.read_table(written)
    names = ["C3" if name == "C4" else name for name in table.column_names]
    edited = pa.Table.from_arrays(table.columns, names=names, metadata=table.schema.metadata)
    feather.write_feather(edited, str(tmp_path / "edited.feather"))

    with pytest.raises(ValueError, match=r"repeats signal column\(s\) \['C3'\]"):
        Recording.from_file(str(tmp_path / "edited.feather"))
    assert list(Recording.from_file(written).signals.columns) == ["C3", "C4"]
