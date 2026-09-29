import os

import wfdb

from ..core.emg import Recording
from ..exceptions import FileReadError, classify_read_error, is_resource_exhaustion

# import numpy as np # Keep commented out until needed
# from typing import List, Dict # Keep commented out until needed
from ._labels import DEDUPLICATED_LABELS_KEY, unique_channel_labels
from .base import BaseImporter


class WFDBImporter(BaseImporter):
    """Importer for WFDB format files."""

    def load(self, filepath: str) -> Recording:
        """
        Load EMG data and annotations from WFDB files.

        Assumes annotation files (.atr) share the same base name as the record file
        and are located in the same directory.

        Args:
            filepath: Path to the WFDB header file (.hea) or just the record name.

        Returns:
            Recording: Recording object containing the loaded data and annotations.
        """
        record_name = os.path.splitext(filepath)[0]

        # Check if header file exists first
        header_filepath = record_name + ".hea"
        if not os.path.exists(header_filepath):
            # Try checking the original filepath in case it was passed without extension
            if not os.path.exists(filepath) and not os.path.exists(filepath + ".hea"):
                raise FileNotFoundError(f"WFDB header file not found for record: {record_name}")
            # If original filepath exists, maybe it *was* the header path
            elif os.path.exists(filepath) and filepath.endswith(".hea"):
                header_filepath = filepath  # Use the provided path
            elif os.path.exists(filepath + ".hea"):
                header_filepath = filepath + ".hea"  # Use the constructed path
            else:  # Fallback if logic is confusing
                raise FileNotFoundError(f"WFDB header file not found for record: {record_name}")
        try:
            # Read record data and header
            # physical=True ensures data is in physical units
            record = wfdb.rdrecord(record_name=record_name, sampfrom=0, sampto=None, physical=True)
            # wfdb's writer refuses a repeated sig_name but its reader accepts one
            # from a hand-edited or foreign header, and a Recording is keyed by
            # label: suffix repeats as the EDF importer does, so every signal
            # survives instead of one replacing another. Inside this try because
            # a repeat that cannot be suffixed is a property of the file too.
            labels, renames = unique_channel_labels(
                list(record.sig_name), source="WFDB", filepath=filepath
            )
        except FileNotFoundError as fnf_e:
            # The header exists (checked above), so rdrecord could not find the
            # signal (.dat) file it names.
            raise FileReadError(
                f"Error reading WFDB record '{record_name}': Data file missing or unreadable ({fnf_e})"
            ) from fnf_e
        except Exception as e:
            # Resource exhaustion is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            # Everything raised while wfdb parses the record is about the file:
            # wfdb reports a malformed header as HeaderSyntaxError (a ValueError),
            # but also as a bare TypeError, IndexError or KeyError, and a
            # truncated signal file as ValueError. Typed like every other importer.
            raise classify_read_error(e, filepath) from e

        try:
            # Create Recording object
            rec = Recording()

            # Store metadata from header
            rec.set_metadata("source_file", filepath)
            rec.set_metadata("record_name", record.record_name)
            rec.set_metadata("sampling_frequency", record.fs)  # Store overall sampling frequency
            if record.base_date and record.base_time:
                rec.set_metadata("startdate", record.base_date)
                rec.set_metadata("starttime", record.base_time)
            if record.comments:
                rec.set_metadata("comments", "\n".join(record.comments))

            # Add channels, under the de-duplicated labels (see above).
            if renames:
                rec.set_metadata(DEDUPLICATED_LABELS_KEY, renames)
            for i, (sig_name, label) in enumerate(zip(record.sig_name, labels, strict=True)):
                rec.add_channel(
                    label=label,
                    data=record.p_signal[:, i],
                    sample_frequency=record.fs,  # Use record's fs, assuming uniform sampling
                    physical_dimension=record.units[i]
                    if record.units and i < len(record.units)
                    else "n/a",
                    # WFDB doesn't store prefilter info directly in standard header fields
                    prefilter="n/a",
                    # Determine channel type based on label/units if possible, default to EMG
                    channel_type="EMG" if "EMG" in sig_name.upper() else "OTHER",
                )
                # Store additional WFDB-specific metadata if needed
                channel_metadata = {
                    "baseline": record.baseline[i]
                    if record.baseline and i < len(record.baseline)
                    else None,
                    "adc_gain": record.adc_gain[i]
                    if record.adc_gain and i < len(record.adc_gain)
                    else None,
                    "adc_zero": record.adc_zero[i]
                    if record.adc_zero and i < len(record.adc_zero)
                    else None,
                    "init_value": record.init_value[i]
                    if record.init_value and i < len(record.init_value)
                    else None,
                    "checksum": record.checksum[i]
                    if record.checksum and i < len(record.checksum)
                    else None,
                }
                # Filter out None values before updating
                filtered_metadata = {k: v for k, v in channel_metadata.items() if v is not None}
                if filtered_metadata:  # Only update if there is metadata to add
                    rec.channels[label].update(filtered_metadata)

            # Read annotations if available (look for .atr file by default)
            # Note: wfdb-python automatically searches common annotation extensions
            try:
                annotation = wfdb.rdann(
                    record_name=record_name, extension="atr", sampfrom=0, sampto=None
                )

                # Add annotations to the Recording object
                # Assuming rec object has an `add_event` or `add_annotation` method
                # The structure (onset, duration, description) is common
                if hasattr(rec, "add_event") and callable(rec.add_event):
                    # Map WFDB annotations to events
                    # Use annotation symbols as descriptions, sample indices as onsets
                    for i, symbol in enumerate(annotation.symbol):
                        onset_sec = annotation.sample[i] / record.fs
                        description = f"WFDB Annotation: {symbol}"
                        # WFDB annotations are typically point events (zero duration)
                        duration_sec = 0
                        rec.add_event(
                            onset=onset_sec, duration=duration_sec, description=description
                        )

                else:
                    # Fallback: Store raw annotation data in metadata if no specific method exists
                    rec.set_metadata(
                        "wfdb_annotations",
                        {
                            "sample": annotation.sample.tolist(),
                            "symbol": annotation.symbol,
                            "subtype": annotation.subtype.tolist(),
                            "chan": annotation.chan.tolist(),
                            "num": annotation.num.tolist(),
                            "aux_note": annotation.aux_note,
                            "fs": annotation.fs,
                        },
                    )

            except FileNotFoundError:
                # This specific FileNotFoundError means the .atr file is missing, which is okay.
                rec.set_metadata("annotation_status", "Annotation file (.atr) not found.")
            except Exception as ann_e:
                # Resource exhaustion here (wfdb.rdann OOMs on a huge annotation
                # file, or a saturated host can't start a thread) is a host
                # condition, not an annotation problem -- swallowing it into a
                # metadata note would return a "successful" Recording and hide
                # the real cause (see biosigio.exceptions.is_resource_exhaustion).
                if is_resource_exhaustion(ann_e):
                    raise
                # Other errors during annotation reading are warnings/metadata entries
                rec.set_metadata("annotation_error", f"Error reading annotations: {str(ann_e)}")

            return rec

        except Exception as e:
            if is_resource_exhaustion(e):
                raise
            # Deliberately NOT classified: wfdb has already parsed the record, so
            # a failure while building the Recording from it is a biosigIO bug,
            # not a property of the file. It stays an untyped ValueError (which a
            # caller may retry) rather than a typed, permanent file failure.
            raise ValueError(f"Error reading WFDB file '{filepath}': {str(e)}") from e
