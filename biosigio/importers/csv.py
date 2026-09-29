import logging

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ..exceptions import is_resource_exhaustion
from .base import BaseImporter


class CSVImporter(BaseImporter):
    """
    General purpose CSV importer for EMG data.

    This importer can handle various CSV formats with columnar data, auto-detect
    headers, time columns, and allow for specific column selection.
    """

    def _detect_specialized_format(self, filepath: str) -> str | None:
        """
        Detect if the file matches a known specialized format.

        Args:
            filepath: Path to the CSV file

        Returns:
            Name of the detected specialized format, or None if no specific format is detected
        """
        # Try to read the first few lines to check for format signatures
        try:
            with open(filepath) as f:
                header_lines = [f.readline().strip() for _ in range(20)]
                header_text = "\n".join(header_lines)

                # Check for Trigno format signatures
                if any(marker in header_text for marker in ["Trigno", "Delsys", "Label:", "X[s]"]):
                    return "trigno"

                # Additional format checks can be added here for other importers
                # For example:
                # if 'OTB' in header_text or 'Sessantaquattro' in header_text:
                #     return 'otb'

        except Exception as e:
            # Resource exhaustion must not fall through to the hard-coded parse
            # defaults below (generic delimiter/header guesses that can yield a
            # silently mis-parsed Recording): it needs to reach the guarded
            # pd.read_csv() catch in load() (or the caller), not be absorbed
            # here as "couldn't sniff the format, guess instead" (see
            # biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            # If we can't read the file or encounter an error,
            # don't try to guess the format
            logging.debug("CSV specialized-format sniff failed for %s: %s", filepath, e)

        return None

    def load(self, filepath: str, force_generic: bool = False, **kwargs) -> Recording:
        """
        Load EMG data from a CSV file.

        Args:
            filepath: Path to the CSV file
            force_generic: If True, forces using the generic CSV importer even if a
                          specialized format is detected
            **kwargs: Additional options including:
                - columns: List of column names or indices to include
                - time_column: Name or index of column to use as time index (default: auto-detect)
                - has_header: Whether file has a header row (default: auto-detect)
                - skiprows: Number of rows to skip at the beginning (default: auto-detect)
                - delimiter: Column delimiter (default: auto-detect)
                - sample_frequency: Sampling frequency in Hz (required if no time column)
                - channel_types: Dict mapping column names to channel types ('EMG', 'ACC', etc.)
                - physical_dimensions: Dict mapping column names to physical dimensions
                - metadata: Dict of additional metadata to include

        Returns:
            Recording: Recording object containing the loaded data

        Raises:
            ValueError: If a specialized format is detected and force_generic is False
            FileNotFoundError: If the file does not exist
        """
        # Check if this file matches a specialized format
        if not force_generic:
            format_name = self._detect_specialized_format(filepath)
            if format_name:
                importer_messages = {
                    "trigno": (
                        "This file appears to be a Delsys Trigno CSV export. "
                        "For better metadata extraction and channel detection, use:\n\n"
                        "recording = Recording.from_file(filepath, importer='trigno')\n\n"
                        "If you still want to use the generic CSV importer, set force_generic=True:\n"
                        "importer = CSVImporter()\n"
                        "recording = importer.load(filepath, force_generic=True, **params)"
                    )
                    # Add more format-specific messages here as new importers are developed
                }

                if format_name in importer_messages:
                    raise ValueError(importer_messages[format_name])

        # Extract kwargs with defaults
        columns = kwargs.get("columns", None)
        time_column = kwargs.get("time_column", None)
        has_header = kwargs.get("has_header", None)
        skiprows = kwargs.get("skiprows", None)
        delimiter = kwargs.get("delimiter", None)
        sample_frequency = kwargs.get("sample_frequency", None)
        channel_names = kwargs.get("channel_names", [])
        channel_types = kwargs.get("channel_types", {})
        physical_dimensions = kwargs.get("physical_dimensions", {})
        metadata = kwargs.get("metadata", {})

        # Analyze file structure if parameters not explicitly provided
        try:
            if any(param is None for param in [has_header, skiprows, delimiter]):
                analyzed_params = self._analyze_csv_structure(filepath)

                # Use analyzed parameters for any not explicitly provided
                has_header = has_header if has_header is not None else analyzed_params["has_header"]
                skiprows = skiprows if skiprows is not None else analyzed_params["skiprows"]
                delimiter = delimiter if delimiter is not None else analyzed_params["delimiter"]
        except FileNotFoundError:
            # Pass through file not found errors
            raise

        # Read the CSV file
        try:
            df = pd.read_csv(
                filepath,
                header=0 if has_header else None,
                skiprows=skiprows,
                delimiter=delimiter,
                index_col=None,
            )
        except FileNotFoundError:
            # Pass through file not found errors
            raise
        except Exception as e:
            # Resource exhaustion is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            raise ValueError(f"Failed to read CSV file: {str(e)}") from e

        # If no header, generate column names
        if not has_header:
            df.columns = [f"Channel_{i}" for i in range(len(df.columns))]

        # If channel names are provided, use them.
        # Also, handle the case where the length of channel_names is less than the number of columns.
        if channel_names:
            if len(channel_names) < len(df.columns):
                raise ValueError(
                    "Number of channel names provided is less than the number of columns in the CSV file."
                )
            df.columns = channel_names

        # Filter columns if specified
        if columns is not None:
            if all(isinstance(col, int) for col in columns):
                # Convert numerical indices to column names
                col_names = [df.columns[i] for i in columns]
                # Save original columns for potential renumbering
                df = df[col_names]

                # If using default channel names, renumber them sequentially
                if not has_header and not channel_names:
                    # Check if these are auto-generated channel names
                    if all(col.startswith("Channel_") for col in col_names):
                        # Rename columns to be sequential
                        new_names = [f"Channel_{i}" for i in range(len(col_names))]
                        rename_map = dict(zip(col_names, new_names, strict=True))
                        df = df.rename(columns=rename_map)
            else:
                # Filter by column names
                df = df[columns]

        # A repeated header is renamed by pandas ("EMG", "EMG.1"), but a
        # repeated name in channel_names or columns reaches here intact, and
        # each channel is added by name: fail with the names rather than with
        # pandas' "Data must be 1-dimensional".
        if df.columns.has_duplicates:
            repeated = list(dict.fromkeys(df.columns[df.columns.duplicated()]))
            raise ValueError(f"CSV channel names must be unique; repeated: {repeated}")

        # Handle time column
        if time_column is not None:
            # If time_column is an index, convert to column name
            if isinstance(time_column, int):
                time_column = df.columns[time_column]

            # Set time column as index
            if time_column in df.columns:
                df.set_index(time_column, inplace=True)
            else:
                raise ValueError(f"Time column '{time_column}' not found in data")
        elif has_header:
            # When header exists, try to auto-detect time column only if has_header is True
            time_col = self._detect_time_column(df)
            if time_col:
                df.set_index(time_col, inplace=True)
            elif sample_frequency:
                # Create time index based on provided sampling frequency
                time_index = np.arange(len(df)) / sample_frequency
                df.index = time_index
            else:
                # No time column and no sample frequency provided
                raise ValueError(
                    "No time column detected and no sample_frequency provided. "
                    "Please specify either time_column or sample_frequency."
                )
        else:
            # For headerless data, don't attempt to auto-detect time column
            # to avoid treating the first column as time
            if sample_frequency:
                # Create time index based on provided sampling frequency
                time_index = np.arange(len(df)) / sample_frequency
                df.index = time_index
            else:
                # No sample frequency provided
                raise ValueError(
                    "No sample_frequency provided for headerless data. "
                    "Please specify sample_frequency for proper time indexing."
                )

        # Create recording object
        rec = Recording()

        # Add metadata
        rec.set_metadata("source_file", filepath)
        rec.set_metadata("file_format", "CSV")

        # Add any user-provided metadata
        for key, value in metadata.items():
            rec.set_metadata(key, value)

        # Default sampling frequency if not specified
        default_sample_frequency = 1000.0  # 1 kHz is a common default for EMG
        if hasattr(df.index, "to_series"):
            # Calculate sampling frequency from time index if possible
            try:
                time_diffs = df.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    avg_diff = time_diffs.mean()
                    if avg_diff > 0:
                        calculated_freq = 1.0 / avg_diff
                        default_sample_frequency = calculated_freq
            except Exception:
                # If calculation fails, keep default
                pass

        # Add each column as a channel
        for column in df.columns:
            # Determine channel type
            if column in channel_types:
                ch_type = channel_types[column]
            else:
                # Try to infer channel type from name
                ch_type = self._infer_channel_type(column)

            # Determine physical dimension
            if column in physical_dimensions:
                phys_dim = physical_dimensions[column]
            else:
                # Default based on channel type
                phys_dim = self._default_physical_dimension(ch_type)

            # Add the channel to the recording
            rec.add_channel(
                label=column,
                data=df[column].values,
                sample_frequency=sample_frequency or default_sample_frequency,
                physical_dimension=phys_dim,
                channel_type=ch_type,
            )

        # Encourage user to add metadata if missing essential information
        self._print_metadata_reminder(rec)

        return rec

    def _analyze_csv_structure(self, filepath: str) -> dict:
        """
        Analyze the CSV file structure to detect delimiter, headers, and rows to skip.

        Args:
            filepath: Path to the CSV file

        Returns:
            Dict with detected parameters:
                - delimiter: Detected delimiter character
                - has_header: Whether the file has a header row
                - skiprows: Number of rows to skip
        """
        # Default results
        results = {"delimiter": ",", "has_header": True, "skiprows": 0}

        try:
            # Read the first few lines to analyze structure
            with open(filepath) as f:
                lines = [
                    f.readline().strip() for _ in range(30)
                ]  # Read first 30 lines or until EOF
                lines = [line for line in lines if line]  # Remove empty lines

                # Special case for Trigno CSV format
                data_start = 0
                for i, line in enumerate(lines):
                    if "X[s]" in line:
                        data_start = i
                        results["skiprows"] = data_start
                        results["has_header"] = True
                        break

                if data_start > 0:
                    # Found a header line with X[s], use the line after it as data
                    return results

                # If not a special format, continue with regular analysis
                # Count occurrences of each delimiter and choose the most common one
                delimiters = {",": 0, "\t": 0, ";": 0, "|": 0}

                for line in lines[:5]:  # Check first 5 lines
                    if not line or line.startswith("#"):
                        continue

                    for delim in delimiters:
                        if delim in line:
                            # Count occurrences but also consider how many fields it creates
                            fields = line.split(delim)
                            if len(fields) > 1:  # Must create at least 2 fields to be valid
                                delimiters[delim] += len(fields)

                # Choose the delimiter that creates the most fields
                if any(delimiters.values()):
                    most_common = max(delimiters.items(), key=lambda x: x[1])
                    results["delimiter"] = most_common[0]

                # Infer if file has a header by checking if first row looks different from data rows
                if len(lines) >= 2:
                    possible_header = lines[0]
                    possible_data = lines[1]

                    # If first row contains alphabetic characters and data rows are numeric
                    header_values = possible_header.split(results["delimiter"])
                    data_values = possible_data.split(results["delimiter"])

                    # Check for alpha chars in header
                    has_alpha = any(
                        any(c.isalpha() for c in val) for val in header_values if val.strip()
                    )
                    # Check if data rows are numeric
                    numeric_data = all(self._is_numeric(val) for val in data_values if val.strip())

                    if has_alpha and numeric_data:
                        results["has_header"] = True
                    else:
                        # If no clear distinction, assume no header if all fields look numeric
                        results["has_header"] = not all(
                            self._is_numeric(val) for val in header_values if val.strip()
                        )

        except Exception as e:
            # Resource exhaustion must not fall through to the hard-coded parse
            # defaults below -- same reasoning as the guard in
            # _detect_specialized_format above (see
            # biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            # If analysis fails, return defaults
            logging.debug("CSV structure analysis failed for %s: %s", filepath, e)

        return results

    def _is_numeric(self, value: str) -> bool:
        """Check if a string value is numeric."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _detect_time_column(self, df: pd.DataFrame) -> str | None:
        """
        Try to detect which column represents time.

        Args:
            df: DataFrame with loaded data

        Returns:
            Name of detected time column or None if not found
        """
        time_keywords = ["time", "second", "seconds", "s"]

        # Check column names for time keywords
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                return col

        # Check if first column is monotonically increasing (typical for time)
        first_col = df.columns[0]
        if len(df) > 1 and pd.Series(df[first_col]).is_monotonic_increasing:
            # Check if the values are plausible time values (e.g., not all integers if diff is small)
            if df[first_col].dtype in [np.float64, np.float32]:
                return first_col
            elif (
                df[first_col].diff().dropna().mean() > 1e-9
            ):  # Avoid treating integer indices as time
                return first_col

        return None

    def _infer_channel_type(self, column_name: str) -> str:
        """
        Infer channel type from column name.

        Args:
            column_name: Name of the column

        Returns:
            Inferred channel type
        """
        name_lower = column_name.lower()

        if any(keyword in name_lower for keyword in ["emg", "muscle"]):
            return "EMG"
        elif any(keyword in name_lower for keyword in ["acc", "accel"]):
            return "ACC"
        elif any(keyword in name_lower for keyword in ["gyro"]):
            return "GYRO"
        elif any(keyword in name_lower for keyword in ["time", "second"]):
            # A time column is normally consumed as the index. If one still
            # reaches here it is not a signal channel; classify it as MISC
            # ("TIME" is not a valid BIDS/modality channel type).
            return "MISC"
        else:
            return "OTHER"

    def _default_physical_dimension(self, channel_type: str) -> str:
        """
        Return default physical dimension for a channel type.

        Args:
            channel_type: Type of channel

        Returns:
            Default physical dimension
        """
        dimensions = {"EMG": "µV", "ACC": "g", "GYRO": "deg/s", "TIME": "s", "OTHER": "a.u."}
        return dimensions.get(channel_type, "a.u.")

    def _print_metadata_reminder(self, rec: Recording) -> None:
        """
        Print a reminder to add metadata if essential information is missing.

        Args:
            rec: Recording object to check
        """
        essential_metadata = ["subject", "device", "recording_date"]
        missing = [meta for meta in essential_metadata if meta not in rec.metadata]

        if missing:
            print("[INFO] Reminder: Consider adding essential metadata for better context:")
            for meta in missing:
                print(f"  rec.set_metadata('{meta}', '<Your {meta.replace('_', ' ').title()}>')")
            print("Example: rec.set_metadata('subject', 'S001')")
