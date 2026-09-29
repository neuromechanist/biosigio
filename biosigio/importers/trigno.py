import csv
import logging

import pandas as pd

from ..core.emg import Recording
from ._labels import DEDUPLICATED_LABELS_KEY, suffix_repeated_labels
from .base import BaseImporter

logger = logging.getLogger(__name__)


def _channel_type(label: str) -> str:
    """Infer a Trigno channel's type from the words in its label."""
    if "EMG" in label:
        return "EMG"
    if "ACC" in label:
        return "ACC"
    if "GYRO" in label:
        return "GYRO"
    return "OTHER"


class TrignoImporter(BaseImporter):
    """Importer for Delsys Trigno EMG system data."""

    def _analyze_csv_structure(self, csv_path: str) -> tuple[list[str], int, str | None]:
        """
        Analyze the CSV file structure to identify metadata and data sections.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Tuple containing:
                - List of metadata lines
                - Line number where data starts
                - Header line
        """
        metadata_lines = []
        data_start_line = 0
        header_line = None

        with open(csv_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                if "X[s]" in line:  # This is the header line
                    header_line = line
                    data_start_line = i + 1
                    break

                metadata_lines.append(line)

        return metadata_lines, data_start_line, header_line

    def _parse_metadata_entries(self, metadata_lines: list[str]) -> list[tuple[str, dict]]:
        """
        Parse metadata lines into channel entries, in file order.

        Args:
            metadata_lines: List of metadata lines from the file

        Returns:
            ``(label, info)`` for every ``Label:`` line, repeated labels included
        """
        entries = []

        for line in metadata_lines:
            if line.startswith("Label:"):
                # Extract channel name
                name_part = line[line.find("Label:") + 6 : line.find("Sampling")].strip()

                # Extract sampling frequency
                freq_str = line[line.find("frequency:") + 10 :].split()[0]
                sampling_freq = float(freq_str)

                # Extract unit
                unit = line[line.find("Unit:") + 5 : line.find("Domain")].strip()

                entries.append(
                    (name_part, {"sample_frequency": sampling_freq, "physical_dimension": unit})
                )

        return entries

    def _parse_metadata(self, metadata_lines: list[str]) -> dict:
        """
        Parse metadata lines to extract channel information.

        Args:
            metadata_lines: List of metadata lines from the file

        Returns:
            Dictionary containing channel information, keyed by label (a
            repeated label keeps its last entry; :meth:`load` maps repeated
            labels to their columns by order instead)
        """
        return dict(self._parse_metadata_entries(metadata_lines))

    def load(self, filepath: str) -> Recording:
        """
        Load EMG data from Trigno CSV file.

        Args:
            filepath: Path to the Trigno CSV file

        Returns:
            Recording: Recording object containing the loaded data
        """
        # Create Recording object
        rec = Recording()

        # Analyze file structure
        metadata_lines, data_start, header_line = self._analyze_csv_structure(filepath)

        # Parse metadata
        entries = self._parse_metadata_entries(metadata_lines)
        labels = [label for label, _ in entries]

        # Read data section
        df = pd.read_csv(filepath, skiprows=data_start - 1)

        if len(set(labels)) < len(labels):
            self._add_repeated_label_channels(rec, df, entries, header_line, filepath)
        else:
            self._add_channels_by_label(rec, df, dict(entries))

        # Add file metadata
        rec.set_metadata("source_file", filepath)
        rec.set_metadata("device", "Delsys Trigno")

        return rec

    @staticmethod
    def _add_channels_by_label(rec: Recording, df: pd.DataFrame, channel_info: dict) -> None:
        """Add each data column whose label has a ``Label:`` line (labels unique)."""
        # Clean up column names
        df.columns = [col.replace("X[s]", "").strip('"') for col in df.columns]

        # Get valid channel names (excluding time columns and extra columns)
        channel_labels = [col for col in df.columns if col and not col.startswith(".")]

        # Create time index
        time_col = df.columns[0]  # First column is time
        df.set_index(time_col, inplace=True)

        # Add channels to Recording object
        for label in channel_labels:
            if label in channel_info:
                info = channel_info[label]
                rec.add_channel(
                    label=label,
                    data=df[label].values,
                    sample_frequency=info["sample_frequency"],
                    physical_dimension=info["physical_dimension"],
                    channel_type=_channel_type(label),
                )

    @staticmethod
    def _add_repeated_label_channels(
        rec: Recording,
        df: pd.DataFrame,
        entries: list[tuple[str, dict]],
        header_line: str | None,
        filepath: str,
    ) -> None:
        """Add every channel when two ``Label:`` lines name the same channel.

        pandas renames a repeated data column (``EMG 1`` -> ``EMG 1.1``), so a
        repeated label cannot be matched to its column by name. Instead the
        header is read as written and matched by position: the n-th data column
        headed ``L`` takes the n-th ``Label: L`` line. Every occurrence after the
        first is then suffixed (``EMG 1_1``, ...) through
        :func:`suffix_repeated_labels`, never onto a label the file itself uses,
        and the renames are logged and recorded under
        :data:`DEDUPLICATED_LABELS_KEY`.
        """
        header = next(csv.reader([header_line or ""]))
        if len(header) != len(df.columns):
            raise ValueError(
                f"Trigno header in {filepath} has {len(header)} fields but the data "
                f"section has {len(df.columns)} columns; repeated channel labels "
                "cannot be matched to their columns"
            )
        cleaned = [field.replace("X[s]", "").strip('"') for field in header]

        per_label: dict[str, list[dict]] = {}
        for label, info in entries:
            per_label.setdefault(label, []).append(info)

        seen: dict[str, int] = {}
        matched: list[tuple[str, dict, int]] = []
        for position, label in enumerate(cleaned):
            if not label or position == 0:
                continue
            occurrence = seen.get(label, 0)
            seen[label] = occurrence + 1
            infos = per_label.get(label, [])
            if occurrence < len(infos):
                matched.append((label, infos[occurrence], position))

        names, renames = suffix_repeated_labels([label for label, _, _ in matched])
        if renames:
            logger.warning(
                "Trigno channel labels are not unique in %s; renamed %s",
                filepath,
                ", ".join(f"{new!r} ({old!r})" for new, old in renames.items()),
            )

        for name, (label, info, position) in zip(names, matched, strict=True):
            rec.add_channel(
                label=name,
                data=df.iloc[:, position].to_numpy(),
                sample_frequency=info["sample_frequency"],
                physical_dimension=info["physical_dimension"],
                channel_type=_channel_type(label),
            )
        if renames:
            rec.set_metadata(DEDUPLICATED_LABELS_KEY, renames)
