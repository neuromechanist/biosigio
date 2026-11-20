"""Byte-for-byte writer for tiny EDF+/BDF+ files with chosen data-record onsets.

pyedflib refuses to write EDF+D, and no writer produces a file whose records are
deliberately spaced out, so the discontinuity fixtures (``examples/edfd/``) are laid
out here directly from the EDF+ and BDF+ specifications. Nothing in this module
reads a file back through biosigIO, MNE or pyedflib, so it is independent of the
code under test.

One channel, ``EEG Ramp``, is sampled at 10 Hz with 1 s data records unless told
otherwise. Its value in microvolts is the TRUE time in seconds of the sample (the
record's time-keeping onset plus the offset inside the record), so a reader that
places a post-gap sample too early shows it as a value that disagrees with its
position on the time axis. A second signal, the annotations signal, carries the
time-keeping time-stamped annotation list (TAL) of every record, plus optional
extra annotations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

FS = 10  # Hz
SAMPLES_PER_RECORD = 10
ANNOTATION_SAMPLES = 40  # samples of TAL space per record (80 bytes EDF, 120 BDF)

# kind -> (physical range, digital range, digits per microvolt) of the ramp.
# The EDF ramp has 0.01 uV per digit, the BDF ramp 0.001 uV per digit.
_EDF = {"phys": ("-327.68", "327.67"), "dig": ("-32768", "32767"), "scale": 100, "width": 2}
_BDF = {"phys": ("-1000", "1000"), "dig": ("-1000000", "1000000"), "scale": 1000, "width": 3}


def _field(text: str, width: int) -> bytes:
    raw = text.encode("ascii")
    assert len(raw) <= width, (text, width)
    return raw.ljust(width, b" ")


def write_edf_plus(
    path: str | Path,
    onsets: Sequence[float],
    *,
    label: str = "D",
    bdf: bool = False,
    record_duration: float = 1.0,
    extra_annotations: Mapping[int, tuple[float, str]] | None = None,
) -> Path:
    """Write an EDF+ (or BDF+) file whose data records start at ``onsets`` seconds.

    ``label`` is the byte after ``EDF+`` in the reserved field: ``"C"`` for
    continuous, ``"D"`` for discontinuous. It is written as given, so a ``"D"`` file
    whose onsets happen to be contiguous (or a ``"C"`` file whose onsets are not) can
    be made on purpose. ``extra_annotations`` maps a record index to one
    ``(onset, text)`` annotation written after that record's time-keeping TAL.
    """
    spec = _BDF if bdf else _EDF
    width = spec["width"]
    n_records = len(onsets)
    samples = int(round(FS * record_duration))
    ann_label = "BDF Annotations" if bdf else "EDF Annotations"
    plus = "BDF+" if bdf else "EDF+"
    # BDF's first header byte is 0xFF, then "BIOSEMI"; EDF's version is "0".
    version = b"\xffBIOSEMI" if bdf else _field("0", 8)

    header = version + _field("X X X X", 80) + _field("Startdate 01-JAN-2020 X X X", 80)
    header += _field("01.01.20", 8) + _field("12.00.00", 8) + _field(str(256 * 3), 8)
    header += _field(plus + label, 44) + _field(str(n_records), 8)
    header += _field(f"{record_duration:.6g}", 8) + _field("2", 4)
    signals = [
        ("EEG Ramp", "", "uV", *spec["phys"], *spec["dig"], "", str(samples), ""),
        (
            ann_label,
            "",
            "",
            "-1",
            "1",
            spec["dig"][0],
            spec["dig"][1],
            "",
            str(ANNOTATION_SAMPLES),
            "",
        ),
    ]
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    for column in range(10):
        for signal in signals:
            header += _field(signal[column], widths[column])
    assert len(header) == 256 * 3

    body = b""
    for index, onset in enumerate(onsets):
        true_time = onset + np.arange(samples) / FS
        digits = np.round(true_time * spec["scale"]).astype(np.int64)
        if bdf:
            body += b"".join(int(v).to_bytes(3, "little", signed=True) for v in digits)
        else:
            body += digits.astype("<i2").tobytes()
        tal = f"+{onset:.12g}".encode() + b"\x14\x14\x00"  # the time-keeping TAL
        if extra_annotations and index in extra_annotations:
            at, text = extra_annotations[index]
            tal += f"+{at:.12g}".encode() + b"\x14" + text.encode() + b"\x14\x00"
        space = width * ANNOTATION_SAMPLES
        assert len(tal) <= space
        body += tal.ljust(space, b"\x00")

    Path(path).write_bytes(header + body)
    return Path(path)


# The committed fixtures (``examples/edfd/``), name -> keyword arguments, so a test
# can prove the files on disk are exactly what this module writes.
FIXTURES: dict[str, dict] = {
    # EDF+D, records at 0, 1, 10, 11 s: an 8 s gap. "E2" sits in true time at 10.5 s.
    "edfd_gap_8s.edf": {
        "onsets": [0, 1, 10, 11],
        "label": "D",
        "extra": {0: (0.5, "E1"), 2: (10.5, "E2")},
    },
    # EDF+D, records at 0, 1, 3, 4 s: a 1 s gap. "E2" sits in true time at 3.5 s.
    "edfd_gap_1s.edf": {
        "onsets": [0, 1, 3, 4],
        "label": "D",
        "extra": {0: (0.5, "E1"), 2: (3.5, "E2")},
    },
    # BDF+D, records at 0, 1, 10, 11 s.
    "bdfd_gap_8s.bdf": {
        "onsets": [0, 1, 10, 11],
        "label": "D",
        "bdf": True,
        "extra": {0: (0.5, "E1"), 2: (10.5, "E2")},
    },
    # Labeled EDF+D, but the records are contiguous: 0, 1, 2, 3 s.
    "edfd_contiguous.edf": {
        "onsets": [0, 1, 2, 3],
        "label": "D",
        "extra": {0: (0.5, "E1"), 2: (2.5, "E2")},
    },
    # Control: a plain EDF+C file, records at 0, 1, 2, 3 s.
    "edfc_control.edf": {
        "onsets": [0, 1, 2, 3],
        "label": "C",
        "extra": {0: (0.5, "E1"), 2: (2.5, "E2")},
    },
}


def write_fixture(name: str, directory: str | Path) -> Path:
    """Write the committed fixture ``name`` into ``directory``."""
    spec = FIXTURES[name]
    return write_edf_plus(
        Path(directory) / name,
        spec["onsets"],
        label=spec["label"],
        bdf=spec.get("bdf", False),
        extra_annotations=spec["extra"],
    )


if __name__ == "__main__":
    # Regenerate the committed fixtures: ``uv run python -m biosigio.tests.edfd_builder``.
    target = Path(__file__).resolve().parents[2] / "examples" / "edfd"
    target.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        print(write_fixture(fixture, target))
