import logging
from typing import cast

import numpy as np
import pandas as pd
import pyedflib

from ..core.emg import Recording
from ..exceptions import MixedSamplingRateError, classify_read_error, is_resource_exhaustion
from ._edf_tolerant import classify_pyedflib_error, read_edf_tolerant
from ._labels import unique_channel_labels
from .base import BaseImporter

logger = logging.getLogger(__name__)

# Accepted `mixed_rate` policies for a recording whose signals carry differing
# per-channel sampling rates (EDF/BDF allow this; e.g. PSG: EEG ~200 Hz + SpO2
# ~12.5 Hz). "error" (default) refuses to load -- biosigIO stores one uniform grid
# and silently fabricating a common grid would be unfaithful. "resample" opts in to
# upsampling the slower channels to the fastest rate so the whole recording lands on
# one grid; this is a lossy DERIVED view (fine for a serving/plot copy, not the
# authoritative source), and the original per-channel rate is preserved on each
# channel as ``original_sample_frequency`` for provenance.
MIXED_RATE_POLICIES = ("error", "resample")


def _resample_to_length(data: np.ndarray, n_out: int) -> np.ndarray:
    """Polyphase-resample a 1-D channel to exactly ``n_out`` samples.

    Used only by the mixed-rate ``resample`` path to lift a slower channel onto the
    fastest channel's grid. The integer ratio ``n_out/len(data)`` drives a
    Kaiser-windowed ``resample_poly`` (the same anti-alias resampler the Zarr
    exporter uses), then the result is trimmed/edge-padded to the exact length so
    every channel stays length-aligned on the shared grid.
    """
    from math import gcd

    from scipy.signal import resample_poly

    x = np.asarray(data, dtype=float)
    n_in = len(x)
    if n_in == 0 or n_in == n_out:
        return x
    g = gcd(n_out, n_in)
    # padtype="line" extends each edge with a ramp matching the signal's end slope,
    # not zeros. Slow physiological channels (SpO2, respiration) sit on a large DC
    # offset; default zero-padding makes a step at the boundary and the polyphase FIR
    # rings (a 95-unit SpO2 baseline overshot to ~109). "line" removes that transient.
    y = resample_poly(x, n_out // g, n_in // g, padtype="line")
    if len(y) > n_out:
        y = y[:n_out]
    elif len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)), mode="edge")
    return y


class EDFImporter(BaseImporter):
    """Importer for EDF/EDF+/BDF format files."""

    def _determine_channel_type(self, label: str, transducer: str) -> str:
        """
        Determine channel type based on label and transducer info.

        Args:
            label: Channel label from EDF header
            transducer: Transducer type from EDF header

        Returns:
            str: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        label = label.upper()
        transducer = transducer.upper()

        if "EMG" in label or "EMG" in transducer:
            return "EMG"
        elif "EEG" in label or "EEG" in transducer:
            return "EEG"
        elif "ECG" in label or "EKG" in label or "ECG" in transducer:
            return "ECG"
        elif "EOG" in label or "EOG" in transducer:
            return "EOG"
        elif "ACC" in label or "ACCELEROMETER" in transducer:
            return "ACC"
        elif "GYRO" in label or "GYROSCOPE" in transducer:
            return "GYRO"
        elif "TRIG" in label or "TRIGGER" in transducer:
            return "TRIG"
        else:
            return "OTHER"

    def _extract_metadata(self, edf_reader: pyedflib.EdfReader) -> dict:
        """
        Extract metadata from EDF file header.

        Args:
            edf_reader: pyedflib EdfReader instance

        Returns:
            dict: Dictionary containing file metadata
        """
        header = edf_reader.getHeader()
        signals_headers = edf_reader.getSignalHeaders()

        metadata = {
            "recording_info": {
                "startdate": header.get("startdate", None),
                "patientcode": header.get("patientcode", ""),
                "gender": header.get("gender", ""),
                "birthdate": header.get("birthdate", ""),
                "patient_name": header.get("patient_name", ""),
                "patient_additional": header.get("patient_additional", ""),
                "admincode": header.get("admincode", ""),
                "technician": header.get("technician", ""),
                "equipment": header.get("equipment", ""),
                "recording_additional": header.get("recording_additional", ""),
            },
            "file_info": {
                "filetype": header.get("filetype", 0),  # 0: EDF, 1: EDF+, 2: BDF+
                "number_of_signals": len(signals_headers),
                "file_duration": header.get("file_duration", 0),
                "datarecord_duration": header.get("datarecord_duration", 0),
            },
        }

        return metadata

    def _read_signal_data(
        self, edf_reader: pyedflib.EdfReader, signal_idx: int
    ) -> tuple[np.ndarray, dict]:
        """
        Read signal data and header information for a specific channel.

        Args:
            edf_reader: pyedflib EdfReader instance
            signal_idx: Index of the signal to read

        Returns:
            tuple: (signal_data, signal_info)
        """
        # Get signal header
        signal_header = edf_reader.getSignalHeaders()[signal_idx]

        # Read the signal data
        signal_data = edf_reader.readSignal(signal_idx)

        # Extract signal information
        signal_info = {
            "label": cast(str, signal_header["label"]).strip(),
            "transducer": cast(str, signal_header.get("transducer", "")).strip(),
            "physical_dimension": cast(str, signal_header.get("dimension", "")).strip() or "n/a",
            "physical_min": signal_header["physical_min"],
            "physical_max": signal_header["physical_max"],
            "digital_min": signal_header["digital_min"],
            "digital_max": signal_header["digital_max"],
            "prefilter": cast(str, signal_header.get("prefilter", "")).strip() or "n/a",
            "sample_frequency": signal_header["sample_frequency"],
        }

        return signal_data, signal_info

    def _read_annotations(self, edf_reader: pyedflib.EdfReader) -> pd.DataFrame:
        """Read EDF+/BDF+ annotations (events) into an events DataFrame.

        pyedflib parses the dedicated annotations channel and returns parallel
        arrays of onsets (seconds), durations (seconds), and descriptions.
        Plain EDF (no annotations) yields empty arrays. Descriptions may be
        bytes or str depending on the pyedflib version; both are handled, and
        empty placeholder annotations are skipped.

        Returns:
            pd.DataFrame: events with float64 ``onset``/``duration`` and string
            ``description``, sorted by onset (matching :meth:`Recording.add_event`).
        """
        onsets, durations, descriptions = edf_reader.readAnnotations()
        rows = []
        for onset, duration, description in zip(onsets, durations, descriptions, strict=False):
            text = (
                description.decode("utf-8", "replace")
                if isinstance(description, bytes)
                else str(description)
            )
            if text == "":
                continue  # skip empty placeholder annotations
            rows.append((float(onset), float(duration), text))

        events = pd.DataFrame(rows, columns=["onset", "duration", "description"])
        if not events.empty:
            events = events.sort_values(by="onset").reset_index(drop=True)
            events["onset"] = events["onset"].astype("float64")
            events["duration"] = events["duration"].astype("float64")
        return events

    def _collect_via_fallback(
        self, filepath: str, reason: str
    ) -> tuple[list[tuple[dict, np.ndarray]], pd.DataFrame, dict]:
        """Recover a file pyedflib refuses to open (see ``_edf_tolerant``).

        Returns the same ``(signal_info, signal_data)`` pairs the normal path
        builds from ``_read_signal_data``, plus events and a ``file_info`` dict,
        so the rest of :meth:`load` runs unchanged regardless of which path
        produced them.
        """
        fallback = read_edf_tolerant(filepath, reason)
        pairs = [
            (
                {
                    "label": ch.label,
                    "transducer": ch.transducer,
                    "physical_dimension": ch.physical_dimension,
                    "physical_min": ch.physical_min,
                    "physical_max": ch.physical_max,
                    "digital_min": ch.digital_min,
                    "digital_max": ch.digital_max,
                    "prefilter": ch.prefilter,
                    "sample_frequency": ch.sample_frequency,
                    "degenerate_physical_range": ch.degenerate_physical_range,
                },
                ch.data,
            )
            for ch in fallback.channels
        ]
        file_info = {
            "filetype": fallback.filetype,
            "number_of_signals": len(fallback.channels),
            "file_duration": fallback.file_duration,
            "datarecord_duration": fallback.datarecord_duration,
        }
        return pairs, fallback.events, file_info

    def load(self, filepath: str, *, mixed_rate: str = "error") -> Recording:
        """
        Load EMG data from EDF/EDF+/BDF file.

        Three conditions pyedflib's compliance checker rejects outright are
        instead recovered via a tolerant MNE-backed fallback, rescaled back to
        pyedflib-equivalent physical units (see ``_edf_tolerant`` for the full
        rationale and the unit-parity proof): a channel with
        ``physical_min == physical_max`` (issue #109), a numeric header field
        padded with NUL bytes instead of spaces, and a file correctly marked
        EDF+D (discontinuous). A recovered read is flagged in metadata via
        ``edf_tolerant_read`` / ``edf_tolerant_read_reason``. Anything else --
        including a genuinely truncated/corrupt file -- still raises
        :class:`~biosigio.exceptions.CorruptFileError` as before.

        Args:
            filepath: Path to the EDF file
            mixed_rate: Policy when the file's signals carry differing per-channel
                sampling rates (EDF/BDF allow it, e.g. polysomnography). ``"error"``
                (default) raises, since biosigIO stores one uniform grid and a
                fabricated common grid would be unfaithful; ``"resample"`` upsamples
                the slower channels to the fastest rate (a lossy derived view -- the
                original rate is kept on each channel as ``original_sample_frequency``).

        Returns:
            Recording: Recording object containing the loaded data
        """
        if mixed_rate not in MIXED_RATE_POLICIES:
            raise ValueError(f"mixed_rate must be one of {MIXED_RATE_POLICIES}, got {mixed_rate!r}")
        edf_reader: pyedflib.EdfReader | None = None
        try:
            fallback_reason: str | None
            try:
                edf_reader = pyedflib.EdfReader(filepath)
            except Exception as open_exc:
                fallback_reason = classify_pyedflib_error(open_exc)
                if fallback_reason is None:
                    raise
                try:
                    pairs, events, file_info = self._collect_via_fallback(filepath, fallback_reason)
                except ImportError as import_err:
                    # MNE (the `meg` extra) isn't installed -- degrade to the
                    # original pyedflib error rather than leaving this silent,
                    # chained to the ImportError so its install hint survives.
                    logger.warning(
                        "EDF/BDF file %s is recoverable by the tolerant reader, but that "
                        "needs MNE-Python (the 'meg' extra), which is not installed; "
                        "raising the original pyedflib error.",
                        filepath,
                    )
                    raise open_exc from import_err
                recording_info: dict = {}
            else:
                fallback_reason = None
                metadata = self._extract_metadata(edf_reader)
                recording_info = metadata["recording_info"]
                file_info = metadata["file_info"]
                pairs = []
                for i in range(edf_reader.signals_in_file):
                    signal_data, signal_info = self._read_signal_data(edf_reader, i)
                    pairs.append((signal_info, signal_data))
                events = self._read_annotations(edf_reader)

            # Create Recording object
            rec = Recording()

            # Extract and store metadata
            for key, value in recording_info.items():
                if value:  # Only store non-empty values
                    rec.set_metadata(key, value)
            for key, value in file_info.items():
                rec.set_metadata(key, value)

            # Store source file information
            rec.set_metadata("source_file", filepath)

            # A Recording is keyed by label, so a repeated header label would
            # silently overwrite a channel; suffix duplicates as MNE does. The
            # header's own label stays untouched in signal_info.
            labels = unique_channel_labels([info["label"] for info, _ in pairs])

            # Read every signal up front so a mixed per-channel rate is detected
            # before any channel is added: channels of differing native length
            # cannot share the Recording's single time grid, so add_channel would
            # raise mid-load. Each entry: (label, signal_info, signal_data,
            # channel_type).
            collected: list[tuple[str, dict, np.ndarray, str]] = []
            for label, (signal_info, signal_data) in zip(labels, pairs, strict=True):
                channel_type = self._determine_channel_type(label, signal_info["transducer"])
                collected.append((label, signal_info, signal_data, channel_type))

            rates = {info["sample_frequency"] for _, info, _, _ in collected}
            target_rate: float | None = None
            n_out = 0
            if len(rates) > 1:
                if mixed_rate == "error":
                    raise MixedSamplingRateError(
                        "EDF/BDF recording has mixed per-channel sampling rates "
                        f"({sorted(rates)} Hz); biosigIO stores one uniform grid. This "
                        "is a real BIDS montage (e.g. polysomnography: EEG ~200 Hz + "
                        'SpO2/respiration ~12.5 Hz). Pass mixed_rate="resample" to '
                        "upsample the slower channels to the fastest rate for a single-"
                        "grid serving copy (lossy: a derived view, not the faithful "
                        'source), or read channels[ch]["sample_frequency"] per channel '
                        "to handle the rates yourself."
                    )
                # resample: lift every slower channel onto the fastest channel's grid.
                target_rate = max(rates)
                n_out = max(len(data) for _, _, data, _ in collected)

            # Add channels (resampling the slow ones first when mixed_rate="resample").
            for label, signal_info, signal_data, channel_type in collected:
                native = float(signal_info["sample_frequency"])
                freq = native
                resampled = target_rate is not None and native != target_rate
                if resampled:
                    signal_data = _resample_to_length(signal_data, n_out)
                    freq = float(target_rate)  # type: ignore[arg-type]  # non-None when resampled

                rec.add_channel(
                    label=label,
                    data=signal_data,
                    sample_frequency=freq,
                    physical_dimension=signal_info["physical_dimension"],
                    prefilter=signal_info["prefilter"],
                    channel_type=channel_type,
                )

                # Store additional channel-specific metadata
                channel_metadata = {
                    "physical_min": signal_info["physical_min"],
                    "physical_max": signal_info["physical_max"],
                    "digital_min": signal_info["digital_min"],
                    "digital_max": signal_info["digital_max"],
                    "transducer": signal_info["transducer"],
                }
                if resampled:
                    # Preserve the true acquisition rate (metadata loss is data loss).
                    channel_metadata["original_sample_frequency"] = native
                if signal_info.get("degenerate_physical_range"):
                    # physical_min == physical_max: the scale factor is undefined
                    # and the fallback path emitted a constant channel (see
                    # _edf_tolerant.read_edf_tolerant). Flag it the same way
                    # nonfinite_samples flags a suspect channel, so downstream
                    # consumers (e.g. the Zarr exporter) can tell this channel's
                    # data is a physically-forced constant, not a measurement.
                    channel_metadata["degenerate_physical_range"] = True
                rec.channels[label].update(channel_metadata)

            if target_rate is not None:
                # Flag the recording as a resampled derived view so downstream
                # consumers (and humans) know the grid is fabricated, not native.
                rec.set_metadata("mixed_rate_resampled", True)
                rec.set_metadata("mixed_rate_target_hz", float(target_rate))

            if fallback_reason is not None:
                # Flag that this recording only loaded because of the tolerant
                # fallback (see module docstring / _edf_tolerant), so downstream
                # consumers can tell a recovered read from a fully-compliant one.
                rec.set_metadata("edf_tolerant_read", True)
                rec.set_metadata("edf_tolerant_read_reason", fallback_reason)

            # Events (EDF+/BDF+ annotations) survive the import->export->import
            # round-trip (issue #47): assigned directly (not via add_event) for a
            # single sort, since both `_read_annotations` and the fallback's
            # `_events_from_mne_annotations` already return the same schema
            # add_event produces (float64 onset/duration, sorted by onset). Left
            # as the empty __init__ frame when none exist.
            if not events.empty:
                rec.events = events

            return rec

        except Exception as e:
            # Resource exhaustion (MemoryError -- incl. numpy's _ArrayMemoryError
            # from the per-channel readSignal loop above -- or a thread/allocation
            # -exhaustion OSError/RuntimeError) is a host condition, not a file
            # problem: propagate it unchanged rather than reclassifying it as a
            # permanent read failure (issue #123; see
            # biosigio.exceptions.is_resource_exhaustion). This also covers an
            # EdfReader-open failure above that classify_pyedflib_error found no
            # recoverable cause for, which reaches here via a bare re-raise.
            if is_resource_exhaustion(e):
                raise
            # Typed classification: a truncated/non-compliant EDF (pyedflib's
            # "Filesize" check) becomes CorruptFileError so callers can surface a
            # specific reason. A BiosigIOError (e.g. the mixed_rate policy error)
            # passes through unchanged.
            raise classify_read_error(e, filepath) from e

        finally:
            # An exception from the try body (e.g. a MemoryError from the
            # readSignal loop) can be propagating through this finally right
            # now. pyedflib's close() is a C-extension call that can itself
            # fail on a saturated host; unguarded, that failure would REPLACE
            # the propagating exception outright. is_resource_exhaustion's
            # __cause__/__context__ walk (biosigio.exceptions) can recover a
            # masked exception when the replacement is a NEW raised exception,
            # but that is strictly worse than not masking it at all -- so
            # cleanup here never raises, only logs.
            if edf_reader is not None:
                try:
                    edf_reader.close()
                except Exception as close_exc:
                    logger.warning("EDF reader close() failed for %s: %s", filepath, close_exc)
