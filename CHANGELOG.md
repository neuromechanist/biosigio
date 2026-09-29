# Changelog

All notable changes to biosigIO are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).
Dates are the day the GitHub Release was published (UTC),
which is when a version reaches PyPI.

Entries before 1.2.9 are condensed from the
[GitHub Releases](https://github.com/neuromechanist/biosigio/releases),
which keep the full notes; releases older than 1.1.0 are listed there only.

## [1.2.9] - 2026-09-29

### Breaking

- `Recording.add_channel` raises `ValueError` when the label already exists,
  instead of silently replacing that channel.
  Code that relied on the overwrite should call `set_channel` to change a channel's metadata,
  or assign `rec.signals[label] = new_data` (same length as the recording) to replace its samples.
- EDF/BDF files that repeat a channel label now import with suffixed labels (see Changed),
  so a Zarr store regenerated from such a file has different channel names than one published before.

### Fixed

- EEGLAB: MATLAB v7.3 (HDF5) `.set` files that store the struct's fields flat at the HDF5 root,
  with no `EEG` group, are read instead of being reported as corrupt.
  An `EEG` group still wins when both layouts are present.
- EEGLAB: an empty MATLAB value (`[]`, stored as a `uint64` array flagged `MATLAB_empty`)
  in a v7.3 `chanlocs` field is read as no value,
  instead of failing the float conversion of a coordinate or decoding as characters.
- BrainVision: a `.vhdr` whose `DataFile=` or `MarkerFile=` names a file that no longer exists
  (the usual result of a BIDS rename) is read through a temporary header copy
  that points at the same-stem sibling (`.eeg`, then `.dat`; `.vmrk`).
  The dataset's files are never modified.
  Both the importer and the streaming Zarr export use the new `resolved_vhdr` helper.
- EDF/BDF: a file that repeats a channel label no longer loses all but one of those channels.
  The in-memory importer, the tolerant fallback reader and the streaming Zarr source
  now name the channels of one file identically.
- The tolerant EDF fallback pairs each MNE channel with its header row by position,
  so a file with repeated labels can be recovered too.

### Changed

- Repeated channel labels get MNE-compatible running suffixes
  (`T8-P8` twice becomes `T8-P8-0`, `T8-P8-1`),
  falling through to `-a` .. `-z` when a numbered name would collide with an existing label,
  and a warning names the repeated labels.
  The rule lives in one helper, `biosigio.importers._labels.unique_channel_labels`,
  used by the EDF/BDF importer, the tolerant EDF reader, the streaming EDF source,
  the WFDB importer (repeated signal names) and the Zarr importer
  (stores published before this release that repeat a label).

### Known issues

- The XDF and EEGLAB importers write channels directly rather than through `add_channel`,
  so they are not covered by the new duplicate-label guard
  ([#134](https://github.com/neuromechanist/biosigio/issues/134)).

## [1.2.8] - 2026-09-23

### Changed

- `stream_to_zarr` writes every level-0 shard and view chunk exactly once,
  through scratch memmaps and a new shard-aligned copy pass (#129, #130).
  On a 128-channel, 30-minute EDF this cut wall time from 110 s to 9 s and peak RSS growth from 5109 MiB to 1800 MiB;
  the output stores are identical.
- Scratch space is bounded by one channel group and removed once the group is written.

### Fixed

- A failure inside the streaming export closes the source,
  and a scratch file still mapped on Windows no longer masks the original exception.

## [1.2.7] - 2026-09-02

### Added

- `stream_to_zarr` accepts `bids_channels`, as `Recording.from_file` does,
  and both paths apply a `channels.tsv` through one shared decision table,
  so streamed and in-memory stores agree on units and types (#127, #128).
- `Recording.from_file(bids_channels=...)` accepts an explicit path or DataFrame.
- Both Zarr exporters write `channels[].bids_unit` and a root `channels_tsv_units` report;
  `format_version` stays 2.

### Fixed

- A failing sidecar read no longer leaks the open streaming source.

## [1.2.6] - 2026-09-02

### Added

- Regression tests for BrainVision iEEG channel counts (#120, #121)
  and an opt-in fetcher for real test recordings (`biosigio/tests/real_data.py`).

### Changed

- Zarr view levels are chunked at a constant column count (`view_chunk_columns`, default 1024),
  and each group declares its pyramid in its attributes (#119, #126).
- `bids.apply_channels_tsv` converts sample values when it adopts a `channels.tsv` unit,
  through the new `biosigio.units` module;
  discrete channel types are never rescaled (#122, #125).

### Fixed

- Resource exhaustion (`MemoryError`, ENOMEM and similar) propagates unchanged from every importer
  instead of being re-typed as a permanent read error;
  `is_resource_exhaustion` is public (#123, #124).

## [1.2.5] - 2026-08-23

### Fixed

- CTF `.ds` recordings whose `.hc` names the head coils `Nasion`/`LPA`/`RPA` are read
  instead of failing MNE's coil lookup (#117).
  The shim retires itself once an MNE release carries the upstream fix.

## [1.2.4] - 2026-08-21

### Added

- MATLAB v7.3 (HDF5) EEGLAB `.set` files are read through h5py,
  detected by the file's header rather than its extension (#114, closing #21).
  Requires the new `hdf5` extra.

### Fixed

- Three EDF/BDF conditions pyedflib rejects on intact files are recovered through a tolerant MNE fallback:
  a channel with `physical_min == physical_max`, a numeric header field padded with NUL,
  and discontinuous EDF+D (#116, closing #109).
  Recovered reads are flagged with `edf_tolerant_read` and `edf_tolerant_read_reason`.

## [1.2.3] - 2026-08-21

### Added

- MEF3 (`.mefd`) iEEG import via MNE and pymef, with the new `mef3` extra.
- 4D Neuroimaging/BTi MEG import, detected by directory content.
- Both formats work on the streaming Zarr path.

## [1.2.2] - 2026-08-11

### Fixed

- EEGLAB: column-form `(N, 1)` `chanlocs`/`event` struct arrays load every channel
  instead of being truncated to one (#110, #111).
- EEGLAB: the data matrix decides the channel count,
  and duplicate channel labels are disambiguated with a numeric suffix.

## [1.2.1] - 2026-07-07

### Fixed

- Zarr int16 export zero-fills non-finite samples in a channel and flags it,
  instead of failing the whole recording (#107, #108).

## [1.2.0] - 2026-07-07

### Changed

- Large EDF/BDF recordings stream to Zarr through pyedflib windowed reads,
  matching the in-memory export exactly (#106).

## [1.1.7] - 2026-06-13

### Added

- Typed read errors in `biosigio.exceptions` (`UnsupportedFormatError`, `FileReadError`,
  `NotContinuousRecordingError`, `CorruptFileError`, `EmptyRecordingError`,
  `MixedSamplingRateError`), each with a stable `.code`, and `classify_read_error`.

## [1.1.6] - 2026-06-12

First release since 1.1.3; the internal 1.1.4 and 1.1.5 bumps are folded in.

### Added

- CTF `.ds` and KIT `.con`/`.sqd`/`.kdf` MEG import.
- `stream_to_zarr` for bounded-memory conversion of large recordings.
- `mixed_rate="resample"` for EDF/BDF files that mix sampling rates;
  such files raise by default.

## [1.1.3] - 2026-06-03

### Added

- `CITATION.cff` and Zenodo archiving.

### Fixed

- EEGLAB `.set` files saved as a single `EEG` struct load instead of importing as an empty recording (#100).

## [1.1.2] - 2026-06-03

### Fixed

- EEGLAB importer unwraps the `EEG` struct that `scipy.io.loadmat` returns for real `.set` files.

## [1.1.1] - 2026-06-02

### Fixed

- Bounded peak memory for large Zarr conversions and for EEGLAB `.fdt` loading (#95).

## [1.1.0] - 2026-06-02

### Added

- EEGLAB `.set` files with the signal matrix in a sibling `.fdt` file (#94).
- `bids.apply_events_tsv` to load a BIDS `_events.tsv` into `rec.events`.

[1.2.9]: https://github.com/neuromechanist/biosigio/compare/v1.2.8...v1.2.9
[1.2.8]: https://github.com/neuromechanist/biosigio/compare/v1.2.7...v1.2.8
[1.2.7]: https://github.com/neuromechanist/biosigio/compare/v1.2.6...v1.2.7
[1.2.6]: https://github.com/neuromechanist/biosigio/compare/v1.2.5...v1.2.6
[1.2.5]: https://github.com/neuromechanist/biosigio/compare/v1.2.4...v1.2.5
[1.2.4]: https://github.com/neuromechanist/biosigio/compare/v1.2.3...v1.2.4
[1.2.3]: https://github.com/neuromechanist/biosigio/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/neuromechanist/biosigio/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/neuromechanist/biosigio/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/neuromechanist/biosigio/compare/v1.1.7...v1.2.0
[1.1.7]: https://github.com/neuromechanist/biosigio/compare/v1.1.6...v1.1.7
[1.1.6]: https://github.com/neuromechanist/biosigio/compare/v1.1.3...v1.1.6
[1.1.3]: https://github.com/neuromechanist/biosigio/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/neuromechanist/biosigio/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/neuromechanist/biosigio/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/neuromechanist/biosigio/releases/tag/v1.1.0
