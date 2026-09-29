"""Tests for the python-neo-backed electrophysiology importer.

NO MOCKS: signals flow through real neo objects and real files. The full load
path is exercised against a real neo file (written with ``NeoMatlabIO``, which
needs only scipy, so no proprietary binary fixture has to be vendored), and the
stream-selection / event-alignment logic is exercised directly against real
in-memory neo ``AnalogSignal``/``Segment`` objects. The proprietary readers
themselves (Intan, Blackrock, ...) are neo's responsibility and are covered by
neo's own test suite; here we verify the neo -> Recording mapping. Skips when
the optional ``neo`` extra is absent.
"""

import numpy as np
import pytest

from biosigio import Recording

neo = pytest.importorskip("neo", reason="neo importer requires the optional 'neo' extra")
import quantities as pq  # noqa: E402  (only importable once neo is present)

from biosigio.importers.neo import NeoImporter, _channel_names  # noqa: E402


def _write_block(path, streams, *, events=None, epochs=None):
    """Write a real neo Block (list of AnalogSignals + optional events) to a .mat.

    ``streams`` is a list of (name, data 2-D array, rate_hz, units, t_start_s).
    NeoMatlabIO round-trips through scipy, so the file is a genuine neo file.
    """
    seg = neo.Segment()
    for name, data, rate, units, t_start in streams:
        sig = neo.AnalogSignal(
            data.astype("float64"),
            units=units,
            sampling_rate=rate * pq.Hz,
            t_start=t_start * pq.s,
            name=name,
        )
        seg.analogsignals.append(sig)
    for ev in events or []:
        seg.events.append(ev)
    for ep in epochs or []:
        seg.epochs.append(ep)
    block = neo.Block()
    block.segments.append(seg)
    neo.io.NeoMatlabIO(filename=str(path)).write_block(block)
    return path


def test_neo_roundtrip_single_stream(tmp_path):
    """A single-stream neo file imports with correct data, rate, and units."""
    t = np.arange(1000)
    data = np.column_stack([np.sin(t / 10.0), np.cos(t / 7.0), t / 100.0])
    path = _write_block(tmp_path / "rec.mat", [("amp", data, 2000.0, "uV", 0.0)])

    rec = NeoImporter().load(str(path))

    assert rec.signals is not None
    assert len(rec.channels) == 3
    assert rec.signals.shape == (1000, 3)
    for ch in rec.channels:
        assert rec.channels[ch]["sample_frequency"] == 2000.0
        assert rec.channels[ch]["physical_dimension"] == "uV"
        assert rec.channels[ch]["channel_type"] == "OTHER"
    assert np.allclose(np.sort(rec.signals.to_numpy(), axis=0), np.sort(data, axis=0))
    assert rec.metadata["t_start_s"] == 0.0


def test_neo_from_file_routing(tmp_path):
    """Recording.from_file dispatches to the neo importer when asked explicitly."""
    data = np.random.randn(200, 2)
    path = _write_block(tmp_path / "rec.mat", [("amp", data, 500.0, "V", 0.0)])
    rec = Recording.from_file(str(path), importer="neo")
    assert isinstance(rec, Recording)
    assert len(rec.channels) == 2


def test_neo_channel_type_labels_whole_recording(tmp_path):
    """channel_type= labels every channel and infers the coarse modality."""
    data = np.random.randn(300, 4)
    path = _write_block(tmp_path / "rec.mat", [("amp", data, 1000.0, "uV", 0.0)])
    rec = NeoImporter().load(str(path), channel_type="SEEG")
    for ch in rec.channels:
        assert rec.channels[ch]["channel_type"] == "SEEG"
        assert rec.channels[ch]["modality"] == "IEEG"


def test_neo_infer_importer_extensions():
    """Proprietary ephys extensions route to the neo importer."""
    for ext in (".rhd", ".rhs", ".ns5", ".smr", ".plx", ".trc", ".ncs"):
        assert Recording._infer_importer(f"rec{ext}") == "neo"


# --- stream selection (real in-memory neo AnalogSignals, no mocks) ---


def _sig(name, n_samples, n_channels, rate, t_start=0.0):
    return neo.AnalogSignal(
        np.zeros((n_samples, n_channels)),
        units="uV",
        sampling_rate=rate * pq.Hz,
        t_start=t_start * pq.s,
        name=name,
    )


def test_select_single_and_same_rate_merge():
    one = [_sig("a", 100, 3, 1000.0)]
    assert NeoImporter._select_streams(one, None, "f") == one

    same_rate = [_sig("a", 100, 3, 1000.0), _sig("b", 100, 2, 1000.0)]
    assert NeoImporter._select_streams(same_rate, None, "f") == same_rate  # merged


def test_select_multirate_requires_selector():
    streams = [_sig("amp", 100, 3, 30000.0), _sig("lfp", 50, 2, 1000.0)]
    with pytest.raises(ValueError, match="different sampling rates"):
        NeoImporter._select_streams(streams, None, "f")

    assert NeoImporter._select_streams(streams, 1, "f") == [streams[1]]  # by index
    assert NeoImporter._select_streams(streams, "lfp", "f") == [streams[1]]  # by name

    with pytest.raises(ValueError, match="out of range"):
        NeoImporter._select_streams(streams, 5, "f")
    with pytest.raises(ValueError, match="No stream named"):
        NeoImporter._select_streams(streams, "missing", "f")


def test_select_ambiguous_stream_name_rejected():
    streams = [_sig("amp", 100, 2, 30000.0), _sig("amp", 50, 1, 1000.0)]
    with pytest.raises(ValueError, match="ambiguous"):
        NeoImporter._select_streams(streams, "amp", "f")


def test_select_same_rate_differing_lengths_rejected():
    streams = [_sig("a", 100, 2, 1000.0), _sig("b", 80, 2, 1000.0)]
    with pytest.raises(ValueError, match="differing lengths"):
        NeoImporter._select_streams(streams, None, "f")


def test_select_same_rate_differing_t_start_rejected():
    streams = [_sig("a", 100, 2, 1000.0, t_start=0.0), _sig("b", 100, 2, 1000.0, t_start=5.0)]
    with pytest.raises(ValueError, match="differing t_start"):
        NeoImporter._select_streams(streams, None, "f")


def test_select_no_signals_rejected():
    with pytest.raises(ValueError, match="No continuous analog signals"):
        NeoImporter._select_streams([], None, "f")


def test_channel_names_from_annotations_else_generated():
    sig = _sig("amp", 10, 3, 1000.0)
    sig.array_annotate(channel_names=np.array(["L1", "L2", "L3"]))
    assert _channel_names(sig, 0, 3) == ["L1", "L2", "L3"]

    bare = _sig("amp", 10, 2, 1000.0)  # no channel_names annotation
    assert _channel_names(bare, 0, 2) == ["amp_0", "amp_1"]


def test_merged_stream_channel_names_are_unique(tmp_path):
    """Two same-rate streams with colliding generated names stay distinct."""
    a = np.random.randn(50, 1)
    b = np.random.randn(50, 1)
    # Both streams unnamed -> generated names would collide on stream0/stream1.
    path = _write_block(
        tmp_path / "rec.mat", [("dup", a, 1000.0, "uV", 0.0), ("dup", b, 1000.0, "uV", 0.0)]
    )
    rec = NeoImporter().load(str(path))
    assert len(rec.channels) == 2
    assert len(set(rec.channels)) == 2  # no duplicate labels


def test_merged_stream_suffix_never_takes_a_genuine_name(tmp_path):
    """A later stream's genuine name is not taken by an earlier stream's suffix.

    Streams ``x``, ``x`` and ``x_0`` (one channel each) generate ``x_0``, ``x_0``
    and ``x_0_0``. The repeat becomes ``x_0_1``; the third stream keeps
    ``x_0_0`` rather than being pushed to ``x_0_0_0``.
    """
    blocks = [np.full((50, 1), float(v)) for v in (1, 2, 3)]
    path = _write_block(
        tmp_path / "rec.mat",
        [(name, b, 1000.0, "uV", 0.0) for name, b in zip(["x", "x", "x_0"], blocks, strict=True)],
    )
    rec = NeoImporter().load(str(path))
    assert list(rec.channels) == ["x_0", "x_0_1", "x_0_0"]
    for label, value in zip(rec.channels, (1, 2, 3), strict=True):
        assert (rec.signals[label].to_numpy() == value).all()
    assert rec.metadata["channel_labels_deduplicated"] == {"x_0_1": "x_0"}


# --- event / epoch alignment (real in-memory neo Segment, no mocks) ---


def test_events_and_epochs_aligned_to_stream_origin():
    """neo Events/Epochs become biosigio events, shifted by the stream's t_start."""
    seg = neo.Segment()
    seg.events.append(
        neo.Event(times=np.array([3.0, 3.7]) * pq.s, labels=np.array(["stim", "resp"]))
    )
    seg.epochs.append(
        neo.Epoch(
            times=np.array([3.2]) * pq.s,
            durations=np.array([0.4]) * pq.s,
            labels=np.array(["burst"]),
        )
    )
    rec = Recording()
    NeoImporter._add_events(rec, seg, t0=2.5)

    events = rec.events.sort_values("onset").reset_index(drop=True)
    assert list(events["description"]) == ["stim", "burst", "resp"]
    assert np.allclose(events["onset"].to_numpy(), [0.5, 0.7, 1.2])
    assert np.allclose(events["duration"].to_numpy(), [0.0, 0.4, 0.0])


def test_events_without_labels_get_generated_descriptions():
    seg = neo.Segment()
    seg.events.append(neo.Event(times=np.array([1.0, 2.0]) * pq.s, name="trig"))
    rec = Recording()
    NeoImporter._add_events(rec, seg, t0=0.0)
    assert list(rec.events["description"]) == ["trig_0", "trig_1"]
