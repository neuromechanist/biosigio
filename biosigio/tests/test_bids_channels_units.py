"""Tests for BIDS channels.tsv unit adoption (issue #122).

``apply_channels_tsv`` used to set ``physical_dimension`` from the sidecar's
``units`` column while leaving the samples alone, so an MNE-backed importer
(which returns SI volts) had its volts relabelled as microvolts -- a 10^6
label/value mismatch that propagated into the Zarr store's per-channel ``unit``.
Adopting a unit now converts the values with it.

NO MOCKS: fixtures are real BrainVision triples written to disk in the pybv
0.7.6 shape (``IEEE_FLOAT_32``, ``MULTIPLEXED``, per-channel resolution in the
declared unit, values stored as ``value / resolution``) and read back through
the real MNE-backed importer, plus the real EDF iEEG BIDS fixture in
``examples/bids``.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from biosigio import Recording
from biosigio.bids import _rescale, apply_channels_tsv, find_channels_tsv

pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")

_REPO = pathlib.Path(__file__).resolve().parents[2]
IEEG_EDF = (
    _REPO
    / "examples/bids/ieeg/sub-01/ses-postimp/ieeg/sub-01_ses-postimp_task-stim_run-08_ieeg.edf"
)

# A short signal with a distinctive sign pattern and an exact zero, so a
# conversion is visible sample by sample rather than only in a summary statistic.
SAMPLES_IN_MICROVOLTS = np.array([1.0, -2.0, 3.5, 0.0, -4.25, 12.75], dtype=np.float64)


def write_brainvision(
    directory: pathlib.Path,
    stem: str,
    values: np.ndarray,
    *,
    unit: str,
    resolution: float,
    sfreq: float = 250.0,
    names: tuple[str, ...] = ("EEG1",),
) -> pathlib.Path:
    """Write a real BrainVision triple whose samples are ``values`` in ``unit``.

    Mirrors ``pybv.io._write_bveeg_file``: 32-bit float, multiplexed, and the
    on-disk number is ``value / resolution`` so that reading it back and applying
    the header's resolution recovers ``values`` exactly.

    Args:
        directory: Directory to write into.
        stem: BIDS entity stem, e.g. ``"sub-01_task-rest"`` (``_eeg`` is appended).
        values: Samples, ``(n_samples,)`` or ``(n_samples, n_channels)``, in ``unit``.
        unit: The unit the header declares, e.g. ``"µV"`` or ``"V"``.
        resolution: The header's per-channel resolution, in ``unit``.
        sfreq: Sampling frequency in Hz.
        names: Channel names.

    Returns:
        Path to the written ``.vhdr``.
    """
    eeg = directory / f"{stem}_eeg.eeg"
    vhdr = directory / f"{stem}_eeg.vhdr"
    vmrk = directory / f"{stem}_eeg.vmrk"

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    eeg.write_bytes((array / resolution).astype("<f4").tobytes(order="C"))

    channel_lines = "\n".join(
        f"Ch{i + 1}={name},,{resolution},{unit}" for i, name in enumerate(names)
    )
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={eeg.name}\n"
        f"MarkerFile={vmrk.name}\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={len(names)}\n"
        f"SamplingInterval={1e6 / sfreq}\n\n"
        "[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n\n"
        "[Channel Infos]\n"
        f"{channel_lines}\n",
        encoding="utf-8",
    )
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={eeg.name}\n\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,00000000000000000000\n",
        encoding="utf-8",
    )
    return vhdr


def write_channels_tsv(
    directory: pathlib.Path,
    stem: str,
    rows: list[tuple[str, str, str]],
) -> pathlib.Path:
    """Write a sibling ``_channels.tsv`` with ``(name, type, units)`` rows."""
    path = directory / f"{stem}_channels.tsv"
    body = "\n".join("\t".join(row) for row in rows)
    path.write_text(f"name\ttype\tunits\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture
def volts_file(tmp_path):
    """A BrainVision file whose header declares volts, with a µV sidecar.

    MNE returns SI volts either way; declaring volts in the header makes the
    stored numbers and the imported numbers the same, so the 10^6 the sidecar
    asks for is the only scaling in play.
    """
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "uV")])
    return vhdr


def test_channels_tsv_converts_volts_to_microvolts(volts_file):
    """The reported bug: label moved to uV, values must move with it."""
    applied = Recording.from_file(str(volts_file))
    ignored = Recording.from_file(str(volts_file), bids_channels="off")

    assert ignored.channels["EEG1"]["physical_dimension"] == "V"
    assert applied.channels["EEG1"]["physical_dimension"] == "uV"

    raw_volts = ignored.signals["EEG1"].to_numpy()
    assert np.allclose(raw_volts, SAMPLES_IN_MICROVOLTS * 1e-6, rtol=1e-6)
    assert np.allclose(applied.signals["EEG1"].to_numpy(), raw_volts * 1e6, rtol=0, atol=0)
    assert np.allclose(applied.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS, rtol=1e-6)


def test_channels_tsv_recovers_the_files_own_microvolts(tmp_path):
    """The nm000182 shape: header in µV with resolution 0.1, sidecar in uV.

    MNE converts the file's µV to SI volts on read; adopting the sidecar has to
    put the samples back on the scale the file wrote them at.
    """
    stem = "sub-01_task-machinelearning_run-001"
    vhdr = write_brainvision(
        tmp_path, stem, SAMPLES_IN_MICROVOLTS, unit="µV", resolution=0.1, names=("LMacro_01",)
    )
    write_channels_tsv(tmp_path, stem, [("LMacro_01", "SEEG", "uV")])

    rec = Recording.from_file(str(vhdr))
    assert rec.channels["LMacro_01"]["physical_dimension"] == "uV"
    assert rec.channels["LMacro_01"]["channel_type"] == "SEEG"
    assert np.allclose(rec.signals["LMacro_01"].to_numpy(), SAMPLES_IN_MICROVOLTS, rtol=1e-6)


def test_applying_the_sidecar_twice_does_not_double_convert(volts_file):
    """Idempotence: the second pass finds the unit already adopted."""
    rec = Recording.from_file(str(volts_file))
    once = rec.signals["EEG1"].to_numpy().copy()

    sidecar = find_channels_tsv(str(volts_file))
    assert sidecar is not None
    apply_channels_tsv(rec, sidecar)
    apply_channels_tsv(rec, sidecar)

    assert rec.channels["EEG1"]["physical_dimension"] == "uV"
    assert np.array_equal(rec.signals["EEG1"].to_numpy(), once)


def test_spelling_difference_relabels_without_rescaling(tmp_path):
    """uV and µV are the same magnitude: adopt the spelling, leave the numbers."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS, unit="µV", resolution=0.1)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    # Stand in for an importer that already reports microvolts (EEGLAB hardcodes
    # "uV"; the CSV importer emits the micro-sign spelling).
    rec.set_channel("EEG1", physical_dimension="µV")
    rec.signals["EEG1"] = SAMPLES_IN_MICROVOLTS
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "uV")])

    apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv"))

    assert rec.channels["EEG1"]["physical_dimension"] == "uV"
    assert np.array_equal(rec.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS)
    assert "bids_unit" not in rec.channels["EEG1"]


@pytest.mark.parametrize("declared", ["T", "a.u.", "count", "Volt"])
def test_unconvertible_units_keep_the_values_and_record_bids_unit(tmp_path, declared):
    """No silent relabel: the sidecar's claim is recorded, not asserted."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", declared)])

    applied = Recording.from_file(str(vhdr))
    ignored = Recording.from_file(str(vhdr), bids_channels="off")

    assert applied.channels["EEG1"]["physical_dimension"] == "V"
    assert applied.channels["EEG1"]["bids_unit"] == declared
    assert np.array_equal(applied.signals["EEG1"].to_numpy(), ignored.signals["EEG1"].to_numpy())


def test_units_that_already_match_are_left_alone_even_when_unparsable(tmp_path):
    """A unit this module cannot parse is still satisfied when it already matches.

    Without the equality short-circuit an ``a.u.`` channel with an ``a.u.``
    sidecar would take the not-convertible branch and be flagged ``bids_unit``,
    reporting a conflict between a unit and itself.
    """
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    rec.set_channel("EEG1", physical_dimension="a.u.")
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "a.u.")])

    apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv"))

    assert rec.channels["EEG1"]["physical_dimension"] == "a.u."
    assert "bids_unit" not in rec.channels["EEG1"]


@pytest.mark.parametrize(
    ("importer_unit", "declared"),
    [
        ("V", "uV"),  # would need a 1e6 rescale
        ("\u00b5V", "uV"),  # spelling only, ratio exactly 1
    ],
)
def test_a_channel_with_no_samples_is_not_relabelled(tmp_path, importer_unit, declared):
    """No column to scale means no label change either, or the two would diverge.

    The ratio-1 case matters as much as the 1e6 one: there are no numbers for
    the new label to agree with, so the sidecar's claim is recorded rather than
    asserted, even when adopting it would have moved nothing.
    """
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    rec.set_channel("EEG1", physical_dimension=importer_unit)
    rec.signals = rec.signals.drop(columns=["EEG1"])  # channel metadata without data
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", declared)])

    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1

    assert rec.channels["EEG1"]["physical_dimension"] == importer_unit
    assert rec.channels["EEG1"]["bids_unit"] == declared
    assert rec.metadata["channels_tsv_units"]["kept_importer_unit"] == 1


def test_unknown_units_on_the_importer_side_are_not_overwritten(tmp_path):
    """An importer that reports "n/a" makes no numeric claim to convert from."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    rec.set_channel("EEG1", physical_dimension="n/a")
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "uV")])

    apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv"))

    assert rec.channels["EEG1"]["physical_dimension"] == "n/a"
    assert rec.channels["EEG1"]["bids_unit"] == "uV"
    assert np.allclose(rec.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS * 1e-6, rtol=1e-6)


def test_an_unknown_type_no_longer_costs_the_row_its_unit(tmp_path):
    """Type and units are applied independently (they used to share one call)."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    write_channels_tsv(tmp_path, stem, [("EEG1", "NOT_A_BIDS_TYPE", "uV")])

    rec = Recording.from_file(str(vhdr))

    assert rec.channels["EEG1"]["channel_type"] == "EEG"  # importer's guess kept
    assert rec.channels["EEG1"]["physical_dimension"] == "uV"
    assert np.allclose(rec.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS, rtol=1e-6)


@pytest.mark.skipif(not IEEG_EDF.exists(), reason="iEEG BIDS fixture missing")
def test_matching_units_leave_an_edf_recording_untouched():
    """pyedflib already reports the sidecar's unit, so EDF must not move at all.

    This is why issue #57 did not surface the defect, and it is the regression
    that matters most: the fix must be a strict no-op wherever the importer
    already speaks the sidecar's unit.
    """
    applied = Recording.from_file(str(IEEG_EDF))
    ignored = Recording.from_file(str(IEEG_EDF), bids_channels="off")

    assert {i["physical_dimension"] for i in ignored.channels.values()} == {"mV"}
    assert {i["physical_dimension"] for i in applied.channels.values()} == {"mV"}
    assert not any("bids_unit" in i for i in applied.channels.values())
    assert np.array_equal(applied.signals.to_numpy(), ignored.signals.to_numpy())


def test_zarr_export_carries_the_converted_unit_and_values(volts_file, tmp_path):
    """The store's unit/scale/offset describe the converted samples, not the raw ones."""
    zarr = pytest.importorskip("zarr", reason="requires the optional 'zarr' extra")

    rec = Recording.from_file(str(volts_file))
    store_path = rec.to_zarr(str(tmp_path / "store"), dtype="int16")

    group = zarr.open_group(store=zarr.storage.LocalStore(store_path), mode="r")["eeg_250hz"]
    meta = next(c for c in group.attrs["channels"] if c["label"] == "EEG1")
    assert meta["unit"] == "uV"

    reconstructed = group["0"][meta["row_index"], :] * meta["scale"] + meta["offset"]
    step = (SAMPLES_IN_MICROVOLTS.max() - SAMPLES_IN_MICROVOLTS.min()) / 65535.0
    assert np.max(np.abs(reconstructed - SAMPLES_IN_MICROVOLTS)) <= step


def test_discrete_trigger_channels_are_never_rescaled(tmp_path):
    """A TRIG channel holds codes, not a measured quantity (see the MEG test)."""
    stem = "sub-01_task-rest"
    codes = np.array([0.0, 5.0, 3.0, 7.0, 0.0, 5.0])
    vhdr = write_brainvision(tmp_path, stem, codes, unit="V", resolution=1.0, names=("STI014",))
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    rec.set_channel("STI014", channel_type="TRIG")
    write_channels_tsv(tmp_path, stem, [("STI014", "TRIG", "mV")])

    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1

    assert rec.channels["STI014"]["physical_dimension"] == "V"
    assert rec.channels["STI014"]["bids_unit"] == "mV"
    assert np.allclose(rec.signals["STI014"].to_numpy(), codes, rtol=1e-6)
    assert rec.metadata["channels_tsv_units"]["kept_importer_unit"] == 1


def test_rescale_preserves_a_float_columns_precision_and_promotes_integers():
    """The dtype contract of the rescale, asserted on the helper directly.

    Honest note on coverage: pandas treats a scalar operand as *weak* under
    NEP 50, so ``float32_series * 1e6`` is already float32 today and the explicit
    cast back in :func:`~biosigio.bids._rescale` cannot be observed from outside
    on pandas 3 / numpy 2 -- removing it leaves every test here green. It is kept
    as a guard against that promotion rule changing (raw numpy already promotes:
    ``np.float32 array * np.float64`` is float64), which would otherwise silently
    double a large EEGLAB recording's memory the first time a sidecar is applied.
    What *is* pinned here is the contract itself, in both directions.
    """
    float32_column = pd.Series(np.array([1.0, -2.0, 3.0], dtype=np.float32))
    scaled = _rescale(float32_column, 1e6)
    assert scaled.dtype == np.float32
    assert np.allclose(scaled.to_numpy(), np.array([1e6, -2e6, 3e6], dtype=np.float32))

    float64_column = pd.Series(np.array([1.0, -2.0], dtype=np.float64))
    assert _rescale(float64_column, 1e-6).dtype == np.float64

    # An integer column has no precision to keep, and must not stay integral or
    # the fractional part of every conversion would be truncated away.
    integers = _rescale(pd.Series(np.arange(3, dtype=np.int64)), 1e-3)
    assert integers.dtype == np.float64
    assert np.allclose(integers.to_numpy(), [0.0, 1e-3, 2e-3])


def test_adopting_a_unit_drops_a_stale_bids_unit(tmp_path):
    """A recorded conflict must not outlive the disagreement that caused it."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")

    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "T")])
    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1
    assert rec.channels["EEG1"]["bids_unit"] == "T"

    # A later, convertible declaration supersedes it entirely.
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "uV")])
    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1
    assert rec.channels["EEG1"]["physical_dimension"] == "uV"
    assert "bids_unit" not in rec.channels["EEG1"]


def test_a_matching_unit_also_drops_a_stale_bids_unit(tmp_path):
    """Same for the equality short-circuit: agreement clears a recorded conflict."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")

    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "a.u.")])
    apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv"))
    assert rec.channels["EEG1"]["bids_unit"] == "a.u."

    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "V")])  # what the importer says
    apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv"))
    assert rec.channels["EEG1"]["physical_dimension"] == "V"
    assert "bids_unit" not in rec.channels["EEG1"]


def test_float32_columns_stay_float32_through_a_conversion(tmp_path):
    """EEGLAB loads at float32 to halve memory; a sidecar must not double it back."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    rec.signals["EEG1"] = rec.signals["EEG1"].astype(np.float32)
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "uV")])

    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1

    assert rec.signals["EEG1"].dtype == np.float32
    assert np.allclose(rec.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS, rtol=1e-5)


def test_integer_columns_become_float64_when_scaled(tmp_path):
    """An int column has no precision to preserve; float64 is the only right answer."""
    stem = "sub-01_task-rest"
    counts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    vhdr = write_brainvision(tmp_path, stem, counts, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    rec.signals["EEG1"] = np.arange(6, dtype=np.int64)
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "mV")])

    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1

    assert rec.signals["EEG1"].dtype == np.float64
    assert np.array_equal(rec.signals["EEG1"].to_numpy(), np.arange(6) * 1e3)


def test_duplicate_rows_for_one_channel_compose_and_count_once(tmp_path):
    """Two rows naming one channel apply in order; the return counts channels."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "mV"), ("EEG1", "EEG", "uV")])

    # V -> mV (1e3) then mV -> uV (1e3): 1e6 overall, the same place one row
    # declaring uV would have landed.
    assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 1

    assert rec.channels["EEG1"]["physical_dimension"] == "uV"
    assert np.allclose(rec.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS, rtol=1e-6)
    assert rec.metadata["channels_tsv_units"]["converted"] == 2


def test_reapplying_an_unconvertible_sidecar_warns_once(tmp_path, caplog):
    """The kept-unit branch is idempotent too: no re-warn, no re-count."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS * 1e-6, unit="V", resolution=1.0)
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "T")])
    sidecar = str(tmp_path / f"{stem}_channels.tsv")

    with caplog.at_level("WARNING"):
        assert apply_channels_tsv(rec, sidecar) == 1
        first = len([r for r in caplog.records if "bids_unit" in r.getMessage()])
        assert apply_channels_tsv(rec, sidecar) == 1  # the type row still "changes" it
        second = len([r for r in caplog.records if "bids_unit" in r.getMessage()])

    assert first == 1
    assert second == 1  # the second pass added no warning
    assert rec.metadata["channels_tsv_units"]["kept_importer_unit"] == 0


def test_report_distinguishes_no_units_column_from_already_correct(tmp_path, caplog):
    """ "units never attempted" and "units already right" must not look the same."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS, unit="µV", resolution=0.1)

    typed_only = tmp_path / f"{stem}_channels.tsv"
    typed_only.write_text("name\ttype\nEEG1\tEEG\n", encoding="utf-8")
    rec = Recording.from_file(str(vhdr), bids_channels="off")
    with caplog.at_level("WARNING"):
        apply_channels_tsv(rec, str(typed_only))
    assert rec.metadata["channels_tsv_units"] == {
        "converted": 0,
        "relabelled": 0,
        "kept_importer_unit": 0,
        "units_column_present": False,
    }
    assert any("no 'units' column" in r.getMessage() for r in caplog.records)

    # Same counters, but the column was there and the units already agreed.
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "V")])
    already = Recording.from_file(str(vhdr), bids_channels="off")
    apply_channels_tsv(already, str(tmp_path / f"{stem}_channels.tsv"))
    assert already.metadata["channels_tsv_units"]["units_column_present"] is True
    assert already.metadata["channels_tsv_units"]["converted"] == 0


def test_a_blank_units_cell_is_skipped_and_logged_at_debug(tmp_path, caplog):
    """A blank cell declares nothing; it must not be mistaken for a unit."""
    stem = "sub-01_task-rest"
    vhdr = write_brainvision(tmp_path, stem, SAMPLES_IN_MICROVOLTS, unit="µV", resolution=0.1)
    write_channels_tsv(tmp_path, stem, [("EEG1", "EEG", "")])
    rec = Recording.from_file(str(vhdr), bids_channels="off")

    with caplog.at_level("DEBUG"):
        apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv"))

    assert rec.channels["EEG1"]["physical_dimension"] == "V"
    assert "bids_unit" not in rec.channels["EEG1"]
    assert any("declares no unit" in r.getMessage() for r in caplog.records)


def test_channels_missing_from_the_sidecar_are_logged_at_debug(tmp_path, caplog):
    """A name column that matches nothing (not even ignoring case) is findable.

    A name differing only in case used to be the example here; it now matches
    (see test_bids_channels_case.py), so the row differs by more than case.
    """
    stem = "sub-01_task-rest"
    values = np.column_stack([SAMPLES_IN_MICROVOLTS, SAMPLES_IN_MICROVOLTS])
    vhdr = write_brainvision(
        tmp_path, stem, values, unit="µV", resolution=0.1, names=("EEG1", "EEG2")
    )
    write_channels_tsv(tmp_path, stem, [("eeg-1", "EEG", "uV")])  # no such channel
    rec = Recording.from_file(str(vhdr), bids_channels="off")

    with caplog.at_level("DEBUG"):
        assert apply_channels_tsv(rec, str(tmp_path / f"{stem}_channels.tsv")) == 0

    messages = [r.getMessage() for r in caplog.records]
    assert any("names no row for 2 of" in m for m in messages)
    assert any("'EEG1'" in m and "'EEG2'" in m for m in messages)


def test_zarr_reimport_restores_the_converted_unit_and_values(volts_file, tmp_path):
    """Round trip through biosigIO's own Zarr importer, not a raw store inspection."""
    pytest.importorskip("zarr", reason="requires the optional 'zarr' extra")

    rec = Recording.from_file(str(volts_file))
    reloaded = Recording.from_file(rec.to_zarr(str(tmp_path / "store"), dtype="float32"))

    assert reloaded.channels["EEG1"]["physical_dimension"] == "uV"
    assert np.allclose(reloaded.signals["EEG1"].to_numpy(), SAMPLES_IN_MICROVOLTS, rtol=1e-5)
