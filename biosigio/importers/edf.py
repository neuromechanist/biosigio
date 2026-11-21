import logging
from typing import cast

import numpy as np
import pandas as pd
import pyedflib

from ..core.emg import Recording
from ..exceptions import MixedSamplingRateError, classify_read_error, is_resource_exhaustion
from ._edf_tolerant import (
    DISCONTINUOUS_VERIFIED_CONTIGUOUS_KEY,
    LATE_TIMEKEEPING_ONSET,
    classify_pyedflib_error,
    probe_timekeeping_onset,
    raise_host_open_error,
    read_edf_tolerant,
)
from ._labels import DEDUPLICATED_LABELS_KEY, unique_channel_labels
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

    def _events_from_annotations(
        self, onsets: np.ndarray, durations: np.ndarray, descriptions: np.ndarray
    ) -> pd.DataFrame:
        """Turn EDF+/BDF+ annotations (events) into an events DataFrame.

        ``pyedflib.EdfReader.readAnnotations`` parses the dedicated annotations
        channel and returns parallel arrays of onsets (seconds), durations
        (seconds), and descriptions, which is what this takes. Plain EDF (no
        annotations) yields empty arrays. Descriptions may be bytes or str
        depending on the pyedflib version; both are handled, and empty
        placeholder annotations are skipped.

        Returns:
            pd.DataFrame: events with float64 ``onset``/``duration`` and string
            ``description``, sorted by onset (matching :meth:`Recording.add_event`).
        """
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

    def _collect_via_pyedflib(
        self,
        edf_reader: pyedflib.EdfReader,
        annotations: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[list[tuple[dict, np.ndarray]], pd.DataFrame, dict, dict]:
        """Read every signal and header field from an open pyedflib reader.

        ``annotations`` is what ``edf_reader.readAnnotations()`` returns: the
        normal path asks the reader for it, while a file whose annotation walk
        pyedflib refuses (see ``_edf_tolerant.probe_timekeeping_onset``) is opened
        without annotations and passes the arrays the probe validated.

        Returns ``(signal_info, signal_data)`` pairs, events, and the
        ``recording_info`` and ``file_info`` dicts, in the shape :meth:`load` uses
        whichever path produced them.
        """
        metadata = self._extract_metadata(edf_reader)
        pairs = []
        for i in range(edf_reader.signals_in_file):
            signal_data, signal_info = self._read_signal_data(edf_reader, i)
            pairs.append((signal_info, signal_data))
        events = self._events_from_annotations(*annotations)
        return pairs, events, metadata["recording_info"], metadata["file_info"]

    def _collect_via_fallback(
        self, filepath: str, reason: str
    ) -> tuple[list[tuple[dict, np.ndarray]], pd.DataFrame, dict, dict[str, str]]:
        """Recover a file pyedflib refuses to open (see ``_edf_tolerant``).

        Returns the same ``(signal_info, signal_data)`` pairs the normal path
        builds from ``_read_signal_data``, plus events, a ``file_info`` dict and
        the fallback reader's label renames (``{new: header_label}``), so the
        rest of :meth:`load` runs unchanged regardless of which path produced
        them. The pairs already carry the de-duplicated labels, so the renames
        are only known from here.
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
        if fallback.verified_contiguous:
            # Labeled EDF+D/BDF+D, but every record starts where the last ended.
            file_info[DISCONTINUOUS_VERIFIED_CONTIGUOUS_KEY] = True
        return pairs, fallback.events, file_info, fallback.renamed_labels

    def load(self, filepath: str, *, mixed_rate: str = "error") -> Recording:
        """
        Load EMG data from EDF/EDF+/BDF file.

        Three conditions pyedflib's compliance checker rejects outright are
        instead recovered via a tolerant MNE-backed fallback, rescaled back to
        pyedflib-equivalent physical units (see ``_edf_tolerant`` for the full
        rationale and the unit-parity proof): a channel with
        ``physical_min == physical_max`` (issue #109), a numeric header field
        padded with NUL bytes instead of spaces, and a file marked EDF+D
        (discontinuous) whose data records are verified contiguous. An EDF+D/BDF+D
        file with a gap between records is refused with
        :class:`~biosigio.exceptions.DiscontinuousRecordingError` (issue #180).
        A fourth, an EDF+C/BDF+C recording cut out of a
        longer session whose first time-keeping onset is 1 s or more (issue
        #145), is read by pyedflib itself with its annotation walk switched off,
        after a TAL-only copy proves that is the file's only defect: no MNE, and
        the samples are exactly a compliant file's. Its ``startdate`` is the
        header's plus that onset, and ``edf_first_record_onset`` records the
        onset in seconds. A recovered read is flagged in metadata via
        ``edf_tolerant_read`` / ``edf_tolerant_read_reason``. Anything else --
        including a genuinely truncated/corrupt file -- still raises
        :class:`~biosigio.exceptions.CorruptFileError` as before. When one of the
        three MNE-backed conditions needs the fallback and MNE is not installed,
        an ``ImportError`` naming the ``meg`` extra is raised, its ``__cause__``
        the pyedflib error.

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
        # An OSError the late-onset probe raises is the host's (the scratch copy it
        # could not create or write, whatever the errno), never the file's: the
        # handler below lets that exact exception out untyped.
        probe_failure: OSError | None = None
        try:
            fallback_reason: str | None
            first_onset: float | None = None
            recording_info: dict
            try:
                edf_reader = pyedflib.EdfReader(filepath)
            except Exception as open_exc:
                # pyedflib hides an OS refusal (EACCES, EIO) behind its own
                # errno-less message; recover the real OSError so the handler
                # below propagates it as the host condition it is.
                raise_host_open_error(filepath, open_exc)
                fallback_reason = classify_pyedflib_error(open_exc)
                if fallback_reason is None:
                    # EDFlib words a late first time-keeping onset like any other
                    # bad annotation list. When that is provably the only defect,
                    # pyedflib itself reads the file with its annotation walk
                    # switched off: no MNE, and the samples are a normal read's.
                    try:
                        late = probe_timekeeping_onset(filepath, open_exc)
                    except OSError as host_error:
                        probe_failure = host_error
                        raise
                    if late is None:
                        raise
                    fallback_reason = LATE_TIMEKEEPING_ONSET
                    first_onset = late.first_onset
                    edf_reader = pyedflib.EdfReader(
                        filepath, annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS
                    )
                    pairs, events, recording_info, file_info = self._collect_via_pyedflib(
                        edf_reader, late.annotations
                    )
                    fallback_renames = {}
                    # The header's start date is the session's; this recording
                    # starts at its first record, `first_onset` seconds later.
                    recording_info["startdate"] = late.startdate
                else:
                    try:
                        pairs, events, file_info, fallback_renames = self._collect_via_fallback(
                            filepath, fallback_reason
                        )
                    except ImportError as import_err:
                        # MNE (the `meg` extra) isn't installed. That is this
                        # environment, not the file, so it stays an ImportError
                        # (untyped, as the streaming source raises it) rather than
                        # becoming a permanent read failure; the pyedflib error it
                        # would have recovered from is its __cause__.
                        logger.warning(
                            "EDF/BDF file %s is recoverable by the tolerant reader, but that "
                            "needs MNE-Python (the 'meg' extra), which is not installed.",
                            filepath,
                        )
                        raise ImportError(
                            f"EDF/BDF file {filepath} can only be read by the tolerant "
                            "reader, which needs MNE-Python: install the 'meg' extra "
                            f"({import_err})"
                        ) from open_exc
                    recording_info = {}
            else:
                fallback_reason = None
                fallback_renames = {}
                pairs, events, recording_info, file_info = self._collect_via_pyedflib(
                    edf_reader, edf_reader.readAnnotations()
                )

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
            # header's own label stays untouched in signal_info. Every rename is
            # recorded on the Recording (and so in a Zarr store's metadata), not
            # only logged.
            labels, renames = unique_channel_labels(
                [info["label"] for info, _ in pairs], filepath=filepath
            )
            renames = {**fallback_renames, **renames}
            if renames:
                rec.set_metadata(DEDUPLICATED_LABELS_KEY, renames)

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
                # Flag that this recording only loaded because of a tolerant
                # recovery (see module docstring / _edf_tolerant), so downstream
                # consumers can tell a recovered read from a fully-compliant one.
                rec.set_metadata("edf_tolerant_read", True)
                rec.set_metadata("edf_tolerant_read_reason", fallback_reason)
                if first_onset is not None:
                    # Seconds from the session's start to this recording's first
                    # record: the file was cut out of a longer session, and its
                    # `startdate` already includes this offset.
                    rec.set_metadata("edf_first_record_onset", first_onset)

            # Events (EDF+/BDF+ annotations) survive the import->export->import
            # round-trip (issue #47): assigned directly (not via add_event) for a
            # single sort, since both `_events_from_annotations` and the fallback's
            # `_events_from_mne_annotations` already return the same schema
            # add_event produces (float64 onset/duration, sorted by onset). Left
            # as the empty __init__ frame when none exist.
            if not events.empty:
                rec.events = events

            return rec

        except ImportError:
            # Only the tolerant fallback's missing `meg` extra raises this (see
            # above): an environment problem, never typed as a file failure.
            raise
        except Exception as e:
            # Resource exhaustion (MemoryError -- incl. numpy's _ArrayMemoryError
            # from the per-channel readSignal loop above -- or a thread/allocation
            # -exhaustion OSError/RuntimeError) is a host condition, not a file
            # problem: propagate it unchanged rather than reclassifying it as a
            # permanent read failure (issue #123; see
            # biosigio.exceptions.is_resource_exhaustion). This also covers an
            # EdfReader-open failure above that classify_pyedflib_error found no
            # recoverable cause for, which reaches here via a bare re-raise. The
            # late-onset probe's own OSError (see `probe_failure` above) leaves the
            # same way.
            if is_resource_exhaustion(e) or e is probe_failure:
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
