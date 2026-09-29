"""XDF (Extensible Data Format) importer for EMG data.

XDF files can contain multiple streams (EMG, EEG, markers, etc.). This module
provides tools to explore XDF contents and selectively import specific streams.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ..core.modality import infer_modality_from_channel_type, validate_channel_type
from ._labels import DEDUPLICATED_LABELS_KEY, suffix_repeated_labels
from .base import BaseImporter

logger = logging.getLogger(__name__)


@dataclass
class XDFStreamInfo:
    """Information about a single XDF stream."""

    stream_id: int
    name: str
    stream_type: str
    channel_count: int
    nominal_srate: float
    effective_srate: float | None
    channel_format: str
    source_id: str
    hostname: str
    sample_count: int
    duration_seconds: float
    channel_labels: list[str]
    channel_types: list[str]
    channel_units: list[str]

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Stream {self.stream_id}: {self.name}",
            f"  Type: {self.stream_type}",
            f"  Channels: {self.channel_count}",
            f"  Nominal srate: {self.nominal_srate} Hz",
        ]
        if self.effective_srate:
            lines.append(f"  Effective srate: {self.effective_srate:.2f} Hz")
        lines.extend(
            [
                f"  Samples: {self.sample_count}",
                f"  Duration: {self.duration_seconds:.2f} s",
                f"  Format: {self.channel_format}",
            ]
        )
        if self.channel_labels:
            labels_preview = ", ".join(self.channel_labels[:5])
            if len(self.channel_labels) > 5:
                labels_preview += f", ... (+{len(self.channel_labels) - 5} more)"
            lines.append(f"  Channel labels: {labels_preview}")
        return "\n".join(lines)


@dataclass
class XDFSummary:
    """Summary of an XDF file's contents."""

    filepath: str
    streams: list[XDFStreamInfo]
    header_info: dict[str, Any]

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"XDF File: {self.filepath}",
            f"Number of streams: {len(self.streams)}",
            "",
        ]
        for stream in self.streams:
            lines.append(str(stream))
            lines.append("")
        return "\n".join(lines)

    def get_streams_by_type(self, stream_type: str) -> list[XDFStreamInfo]:
        """Get all streams of a specific type (case-insensitive)."""
        return [s for s in self.streams if s.stream_type.upper() == stream_type.upper()]

    def get_stream_by_name(self, name: str) -> XDFStreamInfo | None:
        """Get a stream by name (case-insensitive)."""
        for stream in self.streams:
            if stream.name.lower() == name.lower():
                return stream
        return None

    def get_stream_by_id(self, stream_id: int) -> XDFStreamInfo | None:
        """Get a stream by its ID."""
        for stream in self.streams:
            if stream.stream_id == stream_id:
                return stream
        return None


def summarize_xdf(filepath: str | Path) -> XDFSummary:
    """
    Summarize the contents of an XDF file without loading signal data.

    This function parses XDF chunk headers and metadata only, skipping actual
    signal data. This enables memory-efficient exploration of large XDF files
    (even multi-GB files) with minimal RAM usage.

    The function extracts metadata from StreamHeader and StreamFooter chunks,
    which contain all necessary information about streams without requiring
    the actual time series data to be loaded.

    Args:
        filepath: Path to the XDF file

    Returns:
        XDFSummary: Object containing information about all streams in the file

    Example:
        >>> summary = summarize_xdf("recording.xdf")
        >>> print(summary)
        >>> # Find EMG streams
        >>> emg_streams = summary.get_streams_by_type("EMG")
    """
    filepath = str(filepath)
    streams_data, header_info = _parse_xdf_metadata_only(filepath)

    streams = []
    for stream_id, stream_data in streams_data.items():
        header = stream_data.get("header", {})
        footer = stream_data.get("footer", {})

        # Extract basic info from header
        name = header.get("name", "Unknown")
        stream_type = header.get("type", "Unknown")
        channel_count = int(header.get("channel_count", 0))
        nominal_srate = float(header.get("nominal_srate", 0.0))
        channel_format = header.get("channel_format", "unknown")
        source_id = header.get("source_id", "")
        hostname = header.get("hostname", "")

        # Get sample count and timing from footer (if available)
        sample_count = int(footer.get("sample_count", 0))
        first_timestamp = footer.get("first_timestamp")
        last_timestamp = footer.get("last_timestamp")
        measured_srate = footer.get("measured_srate")

        # Calculate effective sample rate
        effective_srate = None
        if measured_srate is not None:
            effective_srate = float(measured_srate)
        elif sample_count > 0 and first_timestamp is not None and last_timestamp is not None:
            duration = float(last_timestamp) - float(first_timestamp)
            if duration > 0:
                effective_srate = sample_count / duration

        # Calculate duration
        if first_timestamp is not None and last_timestamp is not None:
            duration_seconds = float(last_timestamp) - float(first_timestamp)
        elif effective_srate and effective_srate > 0 and sample_count > 0:
            duration_seconds = sample_count / effective_srate
        elif nominal_srate > 0 and sample_count > 0:
            duration_seconds = sample_count / nominal_srate
        else:
            duration_seconds = 0.0

        # Extract channel info from desc
        channel_labels = []
        channel_types = []
        channel_units = []

        desc = header.get("desc", {})
        if isinstance(desc, dict) and "channels" in desc:
            channels_info = desc.get("channels", {})
            if isinstance(channels_info, dict) and "channel" in channels_info:
                channel_list = channels_info["channel"]
                # Handle single channel case (not a list)
                if isinstance(channel_list, dict):
                    channel_list = [channel_list]
                for ch in channel_list:
                    if isinstance(ch, dict):
                        label = ch.get("label", "")
                        ch_type = ch.get("type", "")
                        unit = ch.get("unit", "")
                        channel_labels.append(label)
                        channel_types.append(ch_type)
                        channel_units.append(unit)

        # If no channel info in desc, create default labels
        if not channel_labels:
            channel_labels = [f"Ch{i + 1}" for i in range(channel_count)]
            channel_types = [""] * channel_count
            channel_units = [""] * channel_count

        stream_info = XDFStreamInfo(
            stream_id=stream_id,
            name=name,
            stream_type=stream_type,
            channel_count=channel_count,
            nominal_srate=nominal_srate,
            effective_srate=effective_srate,
            channel_format=channel_format,
            source_id=source_id,
            hostname=hostname,
            sample_count=sample_count,
            duration_seconds=duration_seconds,
            channel_labels=channel_labels,
            channel_types=channel_types,
            channel_units=channel_units,
        )
        streams.append(stream_info)

    return XDFSummary(filepath=filepath, streams=streams, header_info=header_info)


def _parse_xdf_metadata_only(filepath: str) -> tuple[dict, dict]:
    """
    Parse XDF file metadata without loading signal data.

    This function reads only the structural chunks (FileHeader, StreamHeader,
    StreamFooter) and skips over Samples chunks entirely, resulting in minimal
    memory usage even for large files.

    Args:
        filepath: Path to the XDF file

    Returns:
        Tuple of (streams_data, header_info) where:
        - streams_data: dict mapping stream_id to {"header": {...}, "footer": {...}}
        - header_info: dict with file-level header information

    Note:
        Memory estimates for string-type streams (markers) are approximate since
        string lengths vary. The estimate uses 50 bytes per sample as a rough average.
    """
    import gzip
    import struct
    from xml.etree.ElementTree import fromstring

    def read_varlen_int(f):
        """Read a variable-length integer from the file.

        Raises EOFError if the file is truncated.
        """
        length_indicator = f.read(1)
        if not length_indicator:
            raise EOFError("Unexpected end of file while reading variable-length integer.")
        nbytes = length_indicator[0]
        if nbytes == 1:
            data = f.read(1)
            if len(data) != 1:
                raise EOFError("Unexpected end of file while reading 1-byte integer value.")
            return data[0]
        elif nbytes == 4:
            data = f.read(4)
            if len(data) != 4:
                raise EOFError("Unexpected end of file while reading 4-byte integer value.")
            return struct.unpack("<I", data)[0]
        elif nbytes == 8:
            data = f.read(8)
            if len(data) != 8:
                raise EOFError("Unexpected end of file while reading 8-byte integer value.")
            return struct.unpack("<Q", data)[0]
        else:
            raise ValueError(f"Invalid variable-length integer indicator: {nbytes}")

    def xml_to_dict(element):
        """Convert XML element to a dict, similar to pyxdf's _xml2dict."""
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                child_dict = xml_to_dict(child)
                if child.tag in result:
                    # If tag already exists, convert to list
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_dict)
                else:
                    result[child.tag] = child_dict
        return result

    def parse_stream_header_xml(xml_string: str) -> dict:
        """Parse StreamHeader XML into a dict with all metadata including desc."""
        root = fromstring(xml_string)
        result = {}
        for child in root:
            if child.tag == "desc":
                # Parse desc fully for channel info
                result["desc"] = xml_to_dict(child)
            elif len(child) == 0:
                result[child.tag] = child.text
            else:
                result[child.tag] = xml_to_dict(child)
        return result

    def parse_footer_xml(xml_string: str) -> dict:
        """Parse StreamFooter XML into a dict."""
        root = fromstring(xml_string)
        result = {}
        for child in root:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                result[child.tag] = xml_to_dict(child)
        return result

    streams_data: dict = {}
    header_info: dict = {}

    # Initialize file handle to None for safe cleanup
    f = None
    try:
        # Open file (handle both .xdf and .xdfz compressed files)
        if filepath.endswith(".xdfz") or filepath.endswith(".xdf.gz"):
            f = gzip.open(filepath, "rb")
        else:
            f = open(filepath, "rb")

        # Read and verify magic bytes
        magic = f.read(4)
        if magic != b"XDF:":
            raise ValueError(f"Invalid XDF file: expected 'XDF:' magic bytes, got {magic!r}")

        # Process chunks
        while True:
            try:
                chunk_len = read_varlen_int(f)
            except EOFError:
                break

            tag_bytes = f.read(2)
            if len(tag_bytes) < 2:
                break
            tag = struct.unpack("<H", tag_bytes)[0]

            if tag == 1:
                # FileHeader chunk
                xml_bytes = f.read(chunk_len - 2)
                xml_string = xml_bytes.decode("utf-8", errors="replace")
                try:
                    root = fromstring(xml_string)
                    header_info = xml_to_dict(root)
                except Exception as e:
                    # If FileHeader XML is malformed, continue without file-level metadata
                    logger.warning("Failed to parse XDF FileHeader: %s", e)

            elif tag == 2:
                # StreamHeader chunk - parse fully including desc
                stream_id = struct.unpack("<I", f.read(4))[0]
                xml_bytes = f.read(chunk_len - 6)
                xml_string = xml_bytes.decode("utf-8", errors="replace")
                try:
                    header = parse_stream_header_xml(xml_string)
                    if stream_id not in streams_data:
                        streams_data[stream_id] = {"header": {}, "footer": {}}
                    streams_data[stream_id]["header"] = header
                except Exception as e:
                    # If StreamHeader XML is malformed, record the stream with empty header
                    # rather than failing the entire import
                    logger.warning(
                        "Failed to parse StreamHeader for stream %d: %s. "
                        "Channel metadata may be missing.",
                        stream_id,
                        e,
                    )
                    if stream_id not in streams_data:
                        streams_data[stream_id] = {"header": {}, "footer": {}}

            elif tag == 6:
                # StreamFooter chunk - contains sample_count, timestamps, etc.
                stream_id = struct.unpack("<I", f.read(4))[0]
                xml_bytes = f.read(chunk_len - 6)
                xml_string = xml_bytes.decode("utf-8", errors="replace")
                try:
                    footer = parse_footer_xml(xml_string)
                    if stream_id not in streams_data:
                        streams_data[stream_id] = {"header": {}, "footer": {}}
                    streams_data[stream_id]["footer"] = footer
                except Exception as e:
                    # If footer XML is malformed, ignore it and leave this stream
                    # without footer metadata rather than failing the entire import
                    logger.warning(
                        "Failed to parse StreamFooter for stream %d: %s. "
                        "Sample count and duration may be unavailable.",
                        stream_id,
                        e,
                    )

            elif tag in (3, 4, 5):
                # Samples (3), ClockOffset (4), Boundary (5) - skip entirely
                # This is the key optimization: we don't load any signal data
                f.seek(chunk_len - 2, 1)  # Seek relative to current position

            else:
                # Unknown chunk type - skip
                f.seek(chunk_len - 2, 1)

    finally:
        if f is not None:
            f.close()

    return streams_data, header_info


class XDFImporter(BaseImporter):
    """
    Importer for XDF (Extensible Data Format) files.

    XDF files can contain multiple data streams. This importer allows selective
    import of specific streams by name, type, or ID.

    Memory Optimization:
        For large XDF files, use stream selection parameters to load only the
        streams you need. The importer uses pyxdf's native stream selection
        to avoid loading unnecessary data into memory.

    Example:
        >>> # First, explore the file (memory-efficient)
        >>> from biosigio.importers.xdf import summarize_xdf
        >>> summary = summarize_xdf("recording.xdf")
        >>> print(summary)
        >>>
        >>> # Import specific streams (only loads selected streams)
        >>> importer = XDFImporter()
        >>> rec = importer.load("recording.xdf", stream_names=["EMG_stream"])
        >>>
        >>> # Or import by type
        >>> rec = importer.load("recording.xdf", stream_types=["EMG", "EXG"])
    """

    def load(
        self,
        filepath: str,
        stream_names: list[str] | None = None,
        stream_types: list[str] | None = None,
        stream_ids: list[int] | None = None,
        sync_streams: bool = True,
        default_channel_type: str = "OTHER",
        include_timestamps: bool = False,
        reference_stream: str | None = None,
        max_memory_gb: float | None = None,
    ) -> Recording:
        """
        Load EMG data from an XDF file.

        Streams can be selected by name, type, or ID. If multiple selection
        criteria are provided, streams matching ANY criterion are included.
        If no selection criteria are provided, all streams with numeric data
        are loaded.

        Memory Optimization:
            - Stream selection is passed directly to pyxdf, so only requested
              streams are loaded into memory. This significantly reduces RAM
              usage for large files with multiple streams.
            - Use summarize_xdf() first to explore file contents without loading data.
            - The max_memory_gb parameter can warn or raise if estimated memory
              usage exceeds the limit.

        Repeated labels:
            Streams often share channel labels (two amplifiers each with
            ``Ch1``), and a Recording is keyed by label. The first occurrence
            keeps its label; each later one becomes ``<label>_1``, ``<label>_2``,
            ... in stream order, never taking a label some stream really uses.
            The renames are logged with their stream names and recorded as
            ``{new_label: original_label}`` under the recording metadata key
            ``channel_labels_deduplicated``.

        Args:
            filepath: Path to the XDF file
            stream_names: List of stream names to import (case-insensitive)
            stream_types: List of stream types to import (e.g., ["EMG", "EXG"])
            stream_ids: List of stream IDs to import
            sync_streams: If True, synchronize streams to common timestamps.
                         If False, streams are loaded without synchronization.
            default_channel_type: Default channel type for channels without
                                 explicit type info (default: "OTHER"; no silent
                                 EMG assumption)
            include_timestamps: If True, add a timestamp channel for each stream
                               named "{stream_name}_LSL_timestamps" containing
                               the original LSL timestamps. Useful for preserving
                               timing information when exporting to formats like
                               EDF that require regular sampling. A name
                               that is already taken (two streams sharing a
                               name, or a channel of that name) gets a
                               ``_1``, ``_2`` suffix like any repeated label.
            reference_stream: Optional stream name to use as the time base
                             reference. If not specified, the stream with the
                             highest sampling rate is used (recommended to
                             avoid data loss from downsampling).
            max_memory_gb: Optional maximum memory usage in GB. If specified,
                          raises MemoryError if estimated memory exceeds this
                          limit. Use summarize_xdf() to estimate memory needs.

        Returns:
            Recording: Recording object containing the loaded data

        Raises:
            ValueError: If no matching streams found or file cannot be read
            ImportError: If pyxdf is not installed
            MemoryError: If estimated memory exceeds max_memory_gb
        """
        try:
            import pyxdf
        except ImportError as e:
            raise ImportError(
                "pyxdf is required for XDF file support. Install it with: pip install pyxdf"
            ) from e

        filepath = str(filepath)

        # Check memory usage before loading if max_memory_gb is specified
        if max_memory_gb is not None:
            self._check_memory_usage(
                filepath, stream_names, stream_types, stream_ids, max_memory_gb
            )

        # Build pyxdf select_streams parameter for efficient loading
        select_streams = self._build_select_streams(
            filepath, stream_names, stream_types, stream_ids
        )

        # Load only selected streams (pyxdf handles the filtering at load time)
        data, header = pyxdf.load_xdf(filepath, select_streams=select_streams)

        if not data:
            raise ValueError(f"No streams found in XDF file: {filepath}")

        # Filter streams based on selection criteria (for additional filtering)
        selected_streams = self._select_streams(data, stream_names, stream_types, stream_ids)

        if not selected_streams:
            # If no criteria specified, select all streams with numeric data
            if stream_names is None and stream_types is None and stream_ids is None:
                selected_streams = [
                    s
                    for s in data
                    if isinstance(s["time_series"], np.ndarray)
                    and s["time_series"].dtype.kind in "iufc"
                ]
            if not selected_streams:
                raise ValueError(
                    "No matching streams found. Use summarize_xdf() to explore the file."
                )

        # Create Recording object
        rec = Recording()

        # Store metadata
        rec.set_metadata("source_file", filepath)
        rec.set_metadata("device", "XDF")
        rec.set_metadata("stream_count", len(selected_streams))

        if sync_streams and len(selected_streams) > 1:
            self._load_synchronized_streams(
                rec, selected_streams, default_channel_type, include_timestamps, reference_stream
            )
        else:
            # Load streams (uses highest sample rate as reference unless specified)
            self._load_streams(
                rec, selected_streams, default_channel_type, include_timestamps, reference_stream
            )

        return rec

    def _build_select_streams(
        self,
        filepath: str,
        stream_names: list[str] | None,
        stream_types: list[str] | None,
        stream_ids: list[int] | None,
    ) -> list[dict] | list[int] | None:
        """
        Build the value for pyxdf's ``select_streams`` parameter for efficient loading.

        pyxdf accepts ``select_streams`` as either:
        - a list of integer stream IDs (e.g. ``[1, 2, 3]``), or
        - a list of dictionaries with selection criteria (e.g. ``[{"name": "EMG"}]``).

        This method converts our selection criteria into that native format, returning
        either a list of IDs or ``None`` (to load all streams). We resolve names and
        types to IDs using the memory-efficient metadata parser.
        """
        if stream_names is None and stream_types is None and stream_ids is None:
            return None  # Load all streams

        # If stream_ids are specified, use them directly (most efficient)
        if stream_ids is not None:
            return list(stream_ids)  # Return a copy to avoid mutation

        # For stream_names and stream_types, we need to resolve them to IDs
        # using the memory-efficient metadata parser
        if stream_names or stream_types:
            summary = summarize_xdf(filepath)
            # Use a set to avoid duplicate IDs if a stream matches both name and type
            matching_ids: set[int] = set()

            for stream in summary.streams:
                # Check name match (use separate if, not elif, to check all criteria)
                if stream_names and stream.name.lower() in [n.lower() for n in stream_names]:
                    matching_ids.add(stream.stream_id)
                # Check type match
                if stream_types and stream.stream_type.upper() in [t.upper() for t in stream_types]:
                    matching_ids.add(stream.stream_id)

            if matching_ids:
                return list(matching_ids)
            # Criteria were specified but no streams matched: return empty list
            # to avoid loading all streams (which would happen with None)
            return []

        return None

    def _check_memory_usage(
        self,
        filepath: str,
        stream_names: list[str] | None,
        stream_types: list[str] | None,
        stream_ids: list[int] | None,
        max_memory_gb: float,
    ) -> None:
        """
        Estimate memory usage and raise MemoryError if it exceeds the limit.

        Uses the memory-efficient summarize_xdf() to estimate data size.
        """
        # Data type sizes in bytes
        dtype_sizes = {
            "float32": 4,
            "double64": 8,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "string": 50,  # Estimate for strings
        }

        summary = summarize_xdf(filepath)
        total_bytes = 0

        for stream in summary.streams:
            # Check if this stream would be selected (use separate ifs to match
            # _build_select_streams OR logic: include if ANY criterion matches)
            include = False
            if stream_names is None and stream_types is None and stream_ids is None:
                include = True
            if stream_names and stream.name.lower() in [n.lower() for n in stream_names]:
                include = True
            if stream_types and stream.stream_type.upper() in [t.upper() for t in stream_types]:
                include = True
            if stream_ids and stream.stream_id in stream_ids:
                include = True

            if include:
                dtype_size = dtype_sizes.get(stream.channel_format, 8)
                # Estimate: (samples * channels * dtype_size) + (samples * 8 for timestamps)
                stream_bytes = stream.sample_count * stream.channel_count * dtype_size
                stream_bytes += stream.sample_count * 8  # timestamps (float64)
                total_bytes += stream_bytes

        # Add overhead for pandas DataFrame and processing (~50% extra)
        estimated_gb = (total_bytes * 1.5) / (1024**3)

        if estimated_gb > max_memory_gb:
            raise MemoryError(
                f"Estimated memory usage ({estimated_gb:.2f} GB) exceeds limit "
                f"({max_memory_gb:.2f} GB). Consider:\n"
                f"  - Loading fewer streams using stream_names, stream_types, or stream_ids\n"
                f"  - Using summarize_xdf() to identify which streams you need\n"
                f"  - Processing the file in chunks"
            )

    def _select_streams(
        self,
        data: list[dict],
        stream_names: list[str] | None,
        stream_types: list[str] | None,
        stream_ids: list[int] | None,
    ) -> list[dict]:
        """Select streams based on criteria."""
        if stream_names is None and stream_types is None and stream_ids is None:
            return []  # Return empty to trigger "all streams" behavior

        selected = []
        for stream in data:
            info = stream["info"]
            name = info["name"][0] if "name" in info else ""
            stype = info["type"][0] if "type" in info else ""
            sid = info.get("stream_id", 0)

            # Check name match (case-insensitive)
            if stream_names and any(name.lower() == n.lower() for n in stream_names):
                selected.append(stream)
                continue

            # Check type match (case-insensitive)
            if stream_types and any(stype.upper() == t.upper() for t in stream_types):
                selected.append(stream)
                continue

            # Check ID match
            if stream_ids and sid in stream_ids:
                selected.append(stream)
                continue

        return selected

    def _load_streams(
        self,
        rec: Recording,
        streams: list[dict],
        default_channel_type: str,
        include_timestamps: bool = False,
        reference_stream: str | None = None,
    ) -> None:
        """Load streams and resample to a common time base.

        By default, uses the stream with the highest sampling rate as the
        reference to avoid data loss from downsampling. A specific reference
        stream can be specified by name.
        """
        # First pass: collect stream info and find reference stream
        stream_info_list: list[dict[str, Any]] = []
        for stream in streams:
            # pyxdf returns deeply dynamic dict values; cast at this boundary.
            info = cast(dict[str, Any], stream["info"])
            stream_name = info["name"][0] if "name" in info else "Unknown"
            time_series = stream["time_series"]
            timestamps = stream["time_stamps"]

            # Skip non-numpy arrays (e.g., marker streams are lists) or non-numeric data
            if not isinstance(time_series, np.ndarray):
                continue
            if time_series.dtype.kind not in "iufc" or len(time_series) == 0:
                continue

            # Get sampling rate
            srate = stream.get("effective_srate")
            if not srate:
                srate = float(info["nominal_srate"][0]) if "nominal_srate" in info else 0.0

            stream_info_list.append(
                {
                    "stream": stream,
                    "name": stream_name,
                    "info": info,
                    "time_series": time_series,
                    "timestamps": timestamps,
                    "srate": srate,
                }
            )

        if not stream_info_list:
            raise ValueError("No valid data found in selected streams")

        # Determine reference stream: user-specified, or highest sample rate
        ref_stream_info = None
        if reference_stream:
            # Find the user-specified reference stream
            for si in stream_info_list:
                if si["name"].lower() == reference_stream.lower():
                    ref_stream_info = si
                    break
            if ref_stream_info is None:
                raise ValueError(
                    f"Reference stream '{reference_stream}' not found in selected streams. "
                    f"Available: {[si['name'] for si in stream_info_list]}"
                )
        else:
            # Use stream with highest sampling rate (avoids downsampling data loss)
            ref_stream_info = max(stream_info_list, key=lambda x: x["srate"] or 0)

        base_srate = ref_stream_info["srate"]
        base_timestamps = cast(np.ndarray, ref_stream_info["timestamps"])

        # Second pass: collect every channel, in stream order, as a list rather
        # than a dict keyed by label: two streams (or one) can carry the same
        # label, and a dict would keep only the last of them.
        entries: list[dict[str, Any]] = []
        timestamp_entries: list[dict[str, Any]] = []

        for si in stream_info_list:
            stream_name = si["name"]
            time_series = cast(np.ndarray, si["time_series"])
            timestamps = cast(np.ndarray, si["timestamps"])
            srate = si["srate"]
            info = cast(dict[str, Any], si["info"])

            # One timestamp channel per stream, not per stream NAME: two streams
            # sharing a name each keep their own timestamps.
            if include_timestamps:
                timestamp_entries.append(
                    {
                        "label": f"{stream_name}_LSL_timestamps",
                        "stream": stream_name,
                        "timestamps": timestamps,
                        "srate": srate,
                    }
                )

            # Get channel info
            channel_labels, channel_types, channel_units = self._extract_channel_info(
                info, time_series.shape[1] if time_series.ndim > 1 else 1, stream_name
            )

            # Handle 1D data (single channel)
            if time_series.ndim == 1:
                time_series = time_series.reshape(-1, 1)

            for i, label in enumerate(channel_labels):
                if i < time_series.shape[1]:
                    entries.append(
                        {
                            "label": label,
                            "stream": stream_name,
                            "data": time_series[:, i],
                            "timestamps": timestamps,
                            "srate": srate,
                            "unit": channel_units[i] if i < len(channel_units) else "a.u.",
                            "type": channel_types[i]
                            if i < len(channel_types) and channel_types[i]
                            else default_channel_type,
                        }
                    )

        # Labels must be unique: rec.channels and the signal frame are keyed by
        # label. The first occurrence keeps its label and later ones become
        # <label>_1, <label>_2, ..., the scheme this importer has always used,
        # now choosing a suffix no stream uses as a real label. Timestamp
        # channels come last, so a data channel keeps its label if a timestamp
        # channel's name collides with it.
        all_entries = entries + timestamp_entries
        labels, renames = suffix_repeated_labels(
            [e["label"] for e in all_entries], separator="_", start=1
        )
        if renames:
            described = ", ".join(
                f"{new!r} ({renames[new]!r} from stream {e['stream']!r})"
                for new, e in zip(labels, all_entries, strict=True)
                if new in renames
            )
            logger.warning(
                "XDF channel labels are not unique in %s; renamed %s",
                rec.get_metadata("source_file"),
                described,
            )
            rec.set_metadata(DEDUPLICATED_LABELS_KEY, renames)

        # Create time index from reference stream
        # Convert to relative time starting from 0
        if base_timestamps is not None and len(base_timestamps) > 0:
            time_index = base_timestamps - base_timestamps[0]
        else:
            # Fallback: create time index from sample count and rate
            n_samples = len(ref_stream_info["time_series"])
            if base_srate and base_srate > 0:
                time_index = np.arange(n_samples) / base_srate
            else:
                # If no valid sample rate, use sample indices as time
                time_index = np.arange(n_samples, dtype=float)

        # Create DataFrame
        df = pd.DataFrame(index=time_index)

        for label, ch_info in zip(labels[: len(entries)], entries, strict=True):
            # Resample if needed (different stream lengths)
            ch_data = ch_info["data"]
            ch_timestamps = ch_info["timestamps"]
            if len(ch_data) != len(time_index):
                if len(ch_timestamps) > 0:
                    # Interpolate to match base timestamps
                    relative_ch_ts = ch_timestamps - ch_timestamps[0]
                    ch_data = np.interp(time_index, relative_ch_ts, ch_data)
                else:
                    # No timestamps available to resample mismatched data
                    raise ValueError(
                        f"Length mismatch for channel '{label}': "
                        f"{len(ch_data)} samples vs {len(time_index)} time points, "
                        "and no timestamps available for interpolation."
                    )

            df[label] = ch_data

            # Validate the stream-derived type against the modality vocabulary;
            # arbitrary LSL type strings (e.g. "Markers", "Kinematics") map to
            # OTHER rather than being stored unvalidated (which would later fail
            # at export) or silently assumed to be EMG.
            try:
                ch_type = validate_channel_type(ch_info["type"])
            except ValueError:
                ch_type = "OTHER"
            rec.channels[label] = {
                "sample_frequency": ch_info["srate"] if ch_info["srate"] else base_srate,
                "physical_dimension": ch_info["unit"],
                "prefilter": "n/a",
                "channel_type": ch_type,
                "modality": infer_modality_from_channel_type(ch_type),
            }

        # Add timestamp channels if requested
        for ts_label, ts_info in zip(labels[len(entries) :], timestamp_entries, strict=True):
            original_timestamps = ts_info["timestamps"]

            # Resample timestamps to match the common time index
            if len(original_timestamps) == 0:
                # No timestamps available; create a NaN-filled array
                resampled_ts = np.full(len(time_index), np.nan, dtype=float)
            elif len(original_timestamps) != len(time_index):
                relative_ts = original_timestamps - original_timestamps[0]
                resampled_ts = np.interp(time_index, relative_ts, original_timestamps)
            else:
                resampled_ts = original_timestamps

            df[ts_label] = resampled_ts

            rec.channels[ts_label] = {
                "sample_frequency": ts_info["srate"] if ts_info["srate"] else base_srate,
                "physical_dimension": "s",  # seconds
                "prefilter": "n/a",
                "channel_type": "MISC",  # Miscellaneous channel type
                "modality": "MISC",
            }

        rec.signals = df
        rec.set_metadata("srate", base_srate)

    def _load_synchronized_streams(
        self,
        rec: Recording,
        streams: list[dict],
        default_channel_type: str,
        include_timestamps: bool = False,
        reference_stream: str | None = None,
    ) -> None:
        """Load streams with timestamp synchronization.

        This method is an intentional wrapper around _load_streams, serving as
        a dedicated extension point for future synchronization enhancements.
        Currently, pyxdf handles clock synchronization during file loading,
        so this delegates to _load_streams without additional processing.
        """
        self._load_streams(rec, streams, default_channel_type, include_timestamps, reference_stream)

    def _extract_channel_info(
        self,
        info: dict,
        n_channels: int,
        stream_name: str,
    ) -> tuple:
        """Extract channel labels, types, and units from stream info.

        This method safely extracts channel metadata from XDF stream info,
        handling malformed or missing metadata gracefully.
        """
        channel_labels = []
        channel_types = []
        channel_units = []

        try:
            if "desc" in info and info["desc"] and info["desc"][0]:
                desc = info["desc"][0]
                if isinstance(desc, dict) and "channels" in desc and desc["channels"]:
                    channels_info = desc["channels"][0]
                    if isinstance(channels_info, dict) and "channel" in channels_info:
                        for ch in channels_info["channel"]:
                            if isinstance(ch, dict):
                                # Safely extract label
                                label = ""
                                if "label" in ch:
                                    label_val = ch.get("label", [""])
                                    if isinstance(label_val, list) and label_val:
                                        label = str(label_val[0]) if label_val[0] else ""
                                    elif isinstance(label_val, str):
                                        label = label_val

                                # Safely extract type
                                ch_type = ""
                                if "type" in ch:
                                    type_val = ch.get("type", [""])
                                    if isinstance(type_val, list) and type_val:
                                        ch_type = str(type_val[0]) if type_val[0] else ""
                                    elif isinstance(type_val, str):
                                        ch_type = type_val

                                # Safely extract unit
                                unit = ""
                                if "unit" in ch:
                                    unit_val = ch.get("unit", [""])
                                    if isinstance(unit_val, list) and unit_val:
                                        unit = str(unit_val[0]) if unit_val[0] else ""
                                    elif isinstance(unit_val, str):
                                        unit = unit_val

                                channel_labels.append(
                                    label if label else f"{stream_name}_Ch{len(channel_labels) + 1}"
                                )
                                # Infer type from label if not explicitly provided
                                if not ch_type and label:
                                    ch_type = _determine_channel_type_from_label(label)
                                channel_types.append(ch_type)
                                # Default to a.u. (arbitrary units); specific units like uV
                                # should be provided in stream metadata
                                channel_units.append(unit if unit else "a.u.")
        except (KeyError, IndexError, TypeError, AttributeError):
            # If metadata parsing fails, we'll fall back to default labels below
            pass

        # Fill in missing labels
        while len(channel_labels) < n_channels:
            channel_labels.append(f"{stream_name}_Ch{len(channel_labels) + 1}")
            channel_types.append("")
            channel_units.append("a.u.")

        return channel_labels, channel_types, channel_units


def _determine_channel_type_from_label(label: str) -> str:
    """Determine channel type based on label naming conventions."""
    label_upper = label.upper()

    if "EMG" in label_upper or "MUS" in label_upper:
        return "EMG"
    elif "ACC" in label_upper:
        return "ACC"
    elif "GYRO" in label_upper:
        return "GYRO"
    elif "EEG" in label_upper or label_upper in [
        "FP1",
        "FP2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "FZ",
        "CZ",
        "PZ",
        "OZ",
    ]:
        return "EEG"
    elif "ECG" in label_upper or "EKG" in label_upper:
        return "ECG"
    elif "EOG" in label_upper:
        return "EOG"
    elif "TRIG" in label_upper or "MARKER" in label_upper or "EVENT" in label_upper:
        return "TRIG"

    return ""
