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

### Added

- `is_host_condition(exc)`, exported from `biosigio`: true for resource exhaustion
  and for an `OSError` whose `errno` is `EACCES`, `EPERM`, `EIO`, `ENOSPC`, `EROFS`,
  `EDQUOT`, `ETIMEDOUT` or `ESTALE`, including one chained behind a wrapper.

### Fixed

- Host I/O errors are no longer reported as file failures.
  `classify_read_error` re-raises any exception `is_host_condition` accepts unchanged,
  so every importer and the streaming Zarr export raise a permission, disk or network-filesystem error
  as the operating system's own `OSError`, never as a `FileReadError` that a caller would treat as permanent.
  EDF/BDF: pyedflib reports a refused open without its `errno`,
  so the importer and the streaming source reopen the file to raise the real error.
  EEGLAB: an unreadable classic `.set` is no longer retried as `<name>.set.mat`,
  which turned a permission error into a missing-file error.

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
  Each pairing must be the header label itself or, for a label the header repeats,
  MNE's exact rename of it (`<label>-<number>` or `<label>-<a..z>`), so a reordered pair fails loudly.
- EEGLAB: a v7.3 `.set` whose top-level `chanlocs` or `event` is empty (`[]`)
  loads with default `ChannelN` labels or no events,
  instead of failing because MATLAB stores that empty value as an array rather than a struct group.
  A non-empty numeric or character array in their place raises `FileReadError` naming the field,
  rather than being read as none.
- EEGLAB: a v7.3 file that is valid HDF5 but holds neither EEGLAB layout
  raises `FileReadError` saying it is not an EEGLAB v7.3 dataset,
  instead of `CorruptFileError`.
- Streaming Zarr export of a `.vhdr`: a read failure is typed with `classify_read_error`
  exactly as the BrainVision importer does, so a corrupt header raises a `FileReadError`
  (or subclass) instead of a raw MNE exception.
  On both paths, a message that quoted the temporary header copy names the real `.vhdr` instead.
- BrainVision: the temporary header copy names the files it reads by fixed ASCII names
  (`data<ext>`, `marker.vmrk`) staged beside it as a symbolic link, else a hard link, else a copy,
  so it keeps the header's own codepage and never has to spell the dataset's directory.
  A recording under a directory the codepage cannot spell, even one whose name is not valid UTF-8, is read.
  The streaming Zarr source keeps the staged files until it is closed.
  A read error that quoted a staged file names the dataset file it stood for.
- WFDB: a record `wfdb` cannot parse (malformed header, truncated or missing signal file,
  or a repeated signal name that cannot be suffixed) raises a typed `FileReadError`
  instead of a plain `ValueError`.

### Changed

- Repeated channel labels get MNE-compatible running suffixes
  (`T8-P8` twice becomes `T8-P8-0`, `T8-P8-1`),
  falling through to `-a` .. `-z` when a numbered name would collide with an existing label,
  and a warning names the repeated labels.
  The rule lives in one helper, `biosigio.importers._labels.unique_channel_labels`,
  used by the EDF/BDF importer, the tolerant EDF reader, the streaming EDF source,
  the WFDB importer (repeated signal names) and the Zarr importer
  (stores published before this release that repeat a label).
- The EEGLAB importer keeps its own `_2`, `_3` suffix scheme for repeated labels,
  while EDF/BDF, WFDB and Zarr re-import use the MNE-style `-0`, `-1` suffixes above.
- Renamed channel labels are recorded, not only logged:
  the recording's metadata gets `channel_labels_deduplicated`, a `{new_label: original_label}` map,
  from the EDF/BDF importer, the tolerant EDF reader, the streaming EDF source, the WFDB importer and the Zarr importer.
  Both Zarr export paths write it into the store's `recording_metadata`, so it survives a round trip.
  `unique_channel_labels` now returns `(labels, renames)` and takes a `filepath` that the warning names.
- A recovered read leaves a trace:
  a stale BrainVision header read through its siblings is logged and recorded as
  `brainvision_header_recovered` (`{"DataFile": {"referenced": ..., "used": ...}, ...}`),
  on the importer and the streaming export alike,
  and an EEGLAB `.fdt` read under a name other than the one `EEG.data` gives is recorded as
  `eeglab_fdt_recovered` (`{"referenced": ..., "used": ...}`).
  Each key appears only when it applies; a Zarr store without these keys reads exactly as before.
- BrainVision sibling lookup falls back to a case-insensitive match (`<stem>.EEG`, `<stem>.VMRK`)
  when exactly one file matches; an upper-case marker file is staged under a lower-case name,
  because MNE selects its marker reader by the exact `.vmrk` suffix.
- New warnings replace silent behavior: an EEGLAB v7.3 event dropped for an empty type or latency
  and a channel given the default label for an empty label are counted,
  a missing or zero EEGLAB `srate` says that 1000 Hz is assumed,
  a BrainVision header that cannot be read is logged with its errno,
  and an EDF/BDF file that only the tolerant reader could open logs that the `meg` extra is missing.
- EDF/BDF: a file that only the tolerant reader can open raises `ImportError` when MNE (the `meg` extra) is missing,
  instead of the pyedflib error typed as `CorruptFileError` or `FileReadError`.
  The message names the `meg` extra, and the pyedflib error is its `__cause__`.
  A missing dependency is the environment, not the file, and the streaming Zarr source already raised `ImportError` here.

### Known issues

- The XDF and EEGLAB importers write channels directly rather than through `add_channel`,
  so they are not covered by the new duplicate-label guard
  ([#134](https://github.com/neuromechanist/biosigio/issues/134)).
- `apply_channels_tsv` matches `channels.tsv` names to channels case-sensitively, so a
  sidecar that differs from the source file only in case (for example `Fp1-F7` against an
  EDF header's `FP1-F7`) silently skips that channel's type and unit
  ([#136](https://github.com/neuromechanist/biosigio/issues/136)).

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
