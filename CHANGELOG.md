# Changelog

All notable changes to biosigIO are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).
Dates are the day the GitHub Release was published (UTC),
which is when a version reaches PyPI.

Entries before 1.2.9 are condensed from the
[GitHub Releases](https://github.com/neuromechanist/biosigio/releases),
which keep the full notes; releases older than 1.1.0 are listed there only.

## [1.2.14] - 2026-11-21

This patch release stops reading a gapped EDF+D or BDF+D file onto a wrong time axis (#180).
Such a file is now refused with `DiscontinuousRecordingError`, a subclass of `FileReadError`, naming each gap; a +D file whose records are contiguous is still read and flagged.
Converting a gapped recording waits for a gap-aware design.

### Fixed

- A gapped EDF+D or BDF+D file is refused instead of read onto a wrong time axis (#180).
  The tolerant fallback concatenated the data records through MNE, which ignores the discontinuity, so every sample after a gap landed earlier than it was recorded while annotations kept their true times.
  When the reserved field says EDF+D or BDF+D, biosigIO now reads each record's time-keeping annotation itself and raises `DiscontinuousRecordingError` (code `discontinuous_recording`) naming the file and each gap, from both the importer and `stream_to_zarr`.
  A +D file whose records are contiguous is still read and sets `edf_discontinuous_verified_contiguous`.
  The check allows for the rounding of onsets written with few decimals, and reads `+-0.001` as -0.001 s.
  Converting a gapped file waits for a gap-aware design.

- EDF and BDF export reads back within half a least significant bit (LSB) of the given values, with the LSB taken from the header as stored: `(physical_max - physical_min) / (digital_max - digital_min)` (#170).
  pyedflib's `writeSamples` truncated each value toward zero when it turned it into a digital level, so errors reached a whole LSB.
  The exporter now rounds to the nearest level against the stored bounds, clips to the digital range, and writes the levels with `digital=True`.
  On the four recordings in `examples/` (the wristband recording in mV, the OTB sessantaquattro and mouvi recordings, and the MEG recording), read back with pyedflib and MNE in EDF and BDF, every sample is within half an LSB.
  Before, the wristband recording was up to 1.05 LSB off in EDF and 20.8 LSB in BDF.
- The physical bounds stored in an EDF or BDF header bracket the data window.
  The header keeps each bound as at most 8 characters of fixed-notation text cut off from its binary expansion, so a bound rounded outward to fit could be stored inside the window: `95382.65` was stored as `95382.64` and `-53820.7` as `-53820.6`.
  The extreme samples then saturated and a reader scaled by bounds the writer had not used.
  The exporter now writes the header to a scratch file, reads it back, and moves a bound outward in the last place until the stored bounds bracket the data.
  It then checks the header of the finished file against the bounds its samples were scaled by.
  On the wristband recording in mV, 20 of its 32 channels stored a bound inside their data.

### Changed

- A bound below 1e-4 in magnitude is rounded outward to what its fixed-notation text holds in 8 characters, not to the digits `str()` shows (`1e-05` is written as `0.00001`), and a negative bound keeps one digit fewer than a positive one.
- A window within 1e-8 of zero (a constant zero channel, float residue, or a Tesla-scale MEG channel) is no longer replaced by `[-1e-6, 1e-6]`, whose lower bound `-1e-06` is stored as `-0.00000`: such a channel read back with half the gain and an offset of 5e-7, and a constant zero read back as 5e-7.
  It is fit like any other window, to the bounds the 8-character text can hold: `[0, 1e-6]` for non-negative data, `[-1e-5, 0]` for non-positive data and `[-1e-5, 1e-6]` across zero.
  On the 274 magnetometers of the MEG example (values near 1e-10 T) the read-back now has gain 1 and no offset, and is within half an LSB.
  Their correlation with the source is 0.971 at the lowest, where the old read-back, with its gain and offset, correlated 0.999: a channel far below 1e-5 in its unit keeps few levels, because no smaller range can be stored, so rescale such a channel to a smaller unit (for example T to fT).
- `EDFExporter.export` warns when a channel's data span is above zero and under 4 stored LSB, since its samples then fall on a few digital levels or one.
  The warning names the channel as the recording calls it, not by its EDF label (which is cut to 16 characters and may be numbered), with its span, its span in LSB and its LSB, for up to five channels, then a count.
  The rule is the span alone, with no relative floor: a channel that varies by float residue (values of 0 and 1e-12) is reported in EDF, where it loses its variation, and a constant channel is never reported, since it has none to lose.
  The advice depends on the cause.
  A channel far below 1e-5 in its unit (Tesla-scale MEG) needs a smaller unit, and a small variation on a large offset (410986 mV, give or take 5e-6) needs the offset subtracted before export.
  The warning says which, by checking whether the same span centred on zero would be resolved.
  The MEG example has 305 channels (274 magnetometers, 29 reference magnetometers and 2 trigger channels): in EDF 294 are reported, all 274 magnetometers and 20 of the 29 reference magnetometers, and in BDF none.
  The export still succeeds.
- A constant channel is not mapped onto an exact digital level (#172), so one whose value is under half an LSB reads back as the stored bound at 0, within half an LSB of the source but not equal to it.
- The scratch header used to read back the stored bounds is a hidden `.biosigio-bounds-*` file beside the output, not in the system temporary directory, and is removed on success and on failure.
  If the process is killed outright (SIGKILL), the file is left behind in the output directory and is safe to delete.
- `EDFExporter.export` raises `ValueError` when a physical bound cannot be stored so that it brackets the data, instead of writing a miscalibrated file.
- Every channel still gets its own physical window; a wide EMG channel does not set the scale of an EEG channel in the same file.
- Not changed here: a recording whose length is not a whole number of data records is still padded up to the next whole record (#171), and a constant channel is still not mapped onto an exact digital level (#172).
  The padding reads back as physical 0, or as the nearest bound when 0 lies outside the window, as before this change; it is not the middle of the window, which pyedflib's own padding of digital samples would give.

## [1.2.13] - 2026-10-16

This patch release stops reading a BrainVision data file silently when its header shows that the file was cut short, and types a truncated FIF read through `Recording.from_file` as corrupt.
It adds one opt-in, `allow_partial_frame`, that reads a multiplexed BrainVision file ending in a partial frame, as MNE reads it.
Callers that catch `FileReadError` still catch the new refusals, because `CorruptFileError` is a subclass of it.

### Fixed

- A BrainVision data file that its header shows to be cut short raises `CorruptFileError` instead of reading silently.
  A frame is one sample of every channel (channels times bytes per sample).
  Without a positive `DataPoints` count in the header, a size that is not a whole number of frames is refused: MNE dropped the partial trailing frame, and any marker in it, without a word.
  With a positive `DataPoints` count, a file smaller than that many frames is refused, and a longer file is read as MNE reads it, except that vectorized data must still hold whole frames.
  The in-memory importer and the streaming Zarr export both refuse the file, for a `.vhdr` or an `.ahdr` header alike, naming it by file name.
  An `.ahdr` header counts the extra channel MNE adds.
  Without a `DataPoints` count, a file cut exactly at a frame boundary is not detected: multiplexed data reads as a shorter recording, and vectorized data is corrupted from the second channel on, because each channel is stored whole.
  A vectorized file with whole extra frames beyond its `DataPoints` count is not detected either.
- A zero-byte BrainVision data file raises `CorruptFileError` (code `corrupt_or_truncated`) by default.
  The in-memory reader raised MNE's generic `FileReadError`, and the streaming export raised a bare `ValueError`.
- A `.ahdr` header given to the streaming export takes the `.vhdr` path, so a stale `DataFile=` reference is recovered from its same-stem sibling, as the in-memory importer does.
  Before, the export read the header through MNE's generic reader, which did not recover the reference and did not refuse a partial frame.
  In memory, pass `importer="brainvision"` for an `.ahdr` header: `Recording.from_file` does not infer it from the extension.
- Through `Recording.from_file`, a FIF file whose data is truncated raises `CorruptFileError` (code `corrupt_or_truncated`) instead of a generic `FileReadError`.
  `CorruptFileError` is a `FileReadError`, so existing handlers still catch it.
  The two numpy messages a cut inside the data produces, "buffer size must be a multiple of element size" and "cannot reshape array of size", are now typed as corrupt.
  The match is on the message, so it applies to any reader, not only the FIF reader.
  Beyond FIF, the change applies through `Recording.from_file` to the KIT/Yokogawa `.sqd` data and to the 4D/BTi `config` and `hs_file` files: a truncated one now raises `CorruptFileError`.
  The streaming export does not classify these yet (#168).
  The 4D/BTi data file does not raise either message and keeps its generic `FileReadError`.
  Errors from other readers keep their classification unless they carry one of those two messages.
  Host conditions (memory, permissions, I/O) still propagate unchanged.
  The message of a truncation matched this way names the file by its file name, without its directory; for a 4D/BTi dataset it names the dataset directory.

### Added

- `allow_partial_frame=True` on `Recording.from_file` (as a keyword-only argument), `BrainVisionImporter.load` and `stream_to_zarr` reads the whole frames of a multiplexed BrainVision file that ends in a partial frame and declares no positive `DataPoints`, as MNE reads them, where the default refuses it.
  It emits a `UserWarning` that names the data file and the dropped byte count, and records the count in the metadata as `brainvision_dropped_bytes`; the streaming store carries the key in its `recording_metadata` too.
  Vectorized data, a file smaller than a positive `DataPoints` and a file that holds no whole frame (a zero-byte file included) are refused with the flag on too, and the default is unchanged for every other file.

### Changed

- A BrainVision file whose data is cut short now fails to load with `CorruptFileError`, where it used to read with the tail dropped, and so does a vectorized file that does not hold whole frames even if it is longer than its `DataPoints` declares; callers that catch `FileReadError` still catch it.

### Known issues

- The streaming export raises MNE's bare `ValueError` for a truncated FIF file (#161), and for a truncated KIT `.sqd` file or 4D/BTi `config` or `hs_file` file (#168), where `Recording.from_file` raises `CorruptFileError`.
- A KIT file cut to about 23 to 60 percent of its data reads without an error as an all-zero signal (#164).
  The case is pinned as an expected failure.
- A FIF file cut exactly where a data buffer ends, or within the 15 bytes after it, reads as a shorter recording without an error (#163).
  The case is pinned as an expected failure.

## [1.2.12] - 2026-10-16

This patch release reads one more kind of valid EDF file and types the errors of the streaming Zarr export.
It has one behavior change for callers of the streaming export, listed under Changed.

### Fixed

- A streamed EDF or BDF file that pyedflib refuses raises `CorruptFileError` instead of pyedflib's bare `OSError` ([#149](https://github.com/neuromechanist/biosigio/issues/149)).
  Host conditions (memory, permissions, I/O) still propagate unchanged, so a caller can tell a defective file from a failing host.
- An EDF+C or BDF+C recording cut out of a longer session at a data-record boundary now loads.
  Its first time-keeping onset is 1 s or more (for example +40 s),
  and pyedflib refuses it with the generic "contains format errors" message although every byte is valid.
  `EDFImporter.load` and the streaming Zarr export now read it with pyedflib itself, with annotations switched off,
  after checking that the late onset is the file's only defect:
  a small temporary copy of its annotation lists, with the onsets brought back by whole seconds, must pass pyedflib's full annotation walk.
  No MNE is needed, the samples and header fields equal those of the same file with its onsets brought back,
  and the streaming export keeps its bounded-memory windowed read.
  Any other defect (a gap between records, a malformed later annotation, a bad header field, a truncated file) still raises `CorruptFileError`.
  A first onset that would carry the start date past year 9999, or that a float or an integer cannot hold, is declined the same way on both read paths.
  A temporary directory the operating system refuses (missing, read-only, full) raises its own `OSError` rather than `CorruptFileError`.
  The `mixed_rate` policy is unchanged
  ([#145](https://github.com/neuromechanist/biosigio/issues/145)).

### Added

- A recovered late-onset recording carries `edf_tolerant_read` (`True`), `edf_tolerant_read_reason` (`late_timekeeping_onset`, a fourth value) and `edf_first_record_onset`,
  the first record's time-keeping onset in seconds.
  Its `startdate` is the header's start plus that onset, as datetime arithmetic, so it names the time of the first record;
  event onsets are relative to the first record.
  A fractional onset is the corner that cannot be verified against `pyedflib`, whose `getStartdatetime` (0.1.42) divides EDFlib's 100 ns sub-second by 100.
  The streaming export writes the same three keys to the store's `recording_metadata`.

### Changed

- The streaming export types a missing source, a dangling symlink and an empty source as `FileReadError`, as the in-memory reader already does.
  They used to raise `FileNotFoundError` or a bare `OSError`.
  Code that caught either around `stream_to_zarr` must now catch `FileReadError`, a `BiosigIOError`, which is itself a `ValueError`.
- Every non-host failure in the streaming EDF fallback is typed as `FileReadError` or `CorruptFileError`, so an unparsable header field no longer raises a plain `ValueError`.

## [1.2.11] - 2026-10-06

This patch release adds one opt-in option and changes nothing by default:
a store written without it is the store 1.2.10 writes, apart from its creation time and biosigIO version.
It contains no breaking changes.
It is numbered 1.2.11 rather than 1.3.0 by the maintainer's decision, as 1.2.9 and 1.2.10 were, although it adds public names.

### Added

- `exclude_subject_info` on `Recording.to_zarr`, `ZarrExporter.export` and `stream_to_zarr`.
  A store holds the data, the events, channel names, types and units and technical recording metadata;
  the subject's identity and phenotype belong at dataset scope (`participants.tsv`).
  With `exclude_subject_info=True` the store's `recording_metadata` leaves out every member named in the new
  `biosigio.SUBJECT_INFO_KEYS`:
  the EDF/BDF patient code, `gender`, birth date and additional patient text
  (the EDF importer never copies the patient name; a name arrives only through a caller's `recording_metadata`),
  the technician, administrative code, equipment and additional recording text,
  EEGLAB's `subject`, `group`, `setname`, `filename` and `filepath`,
  WFDB's `record_name`,
  free-text `comments` (EEGLAB's, and WFDB header comments, which carry age, sex and medication in PhysioNet records),
  the read-recovery records `eeglab_fdt_recovered` and `brainvision_header_recovered`,
  and the spellings a caller may pass (pyedflib's `patientname` and `sex`, edfio's identification fields,
  MNE's `subject_info`, `experimenter` and subject fields, and generic ones such as `age` and `handedness`).
  Bare `name` and `description` are not matched, being too generic;
  a caller's free-text member under either is the caller's to leave out.
  It also reduces `source_file` and `bti_pdf_file` to their final path component.
  It keeps the recording start date and time, the events and their label map,
  each channel's label, type, unit and prefilter text, and the source file name.
- Names match after normalization (casefolded, non-alphanumeric characters dropped), exactly, never by substring.
  Matching is by name at any depth, so an older store's `recording_info` and the streaming caller's dict are covered,
  except inside biosigIO's label-keyed maps `channels_tsv_units` and `channel_labels_deduplicated`.
  A technical member spelled like a listed name is removed too,
  and subject information under any other name, or inside a value, is not.
- A store written with the option carries the root attribute `subject_info_excluded: true`,
  which records that the option ran, not that the store was verified clean.
  It is additive, so `format_version` stays at 2,
  and a re-export without the option does not carry it.
- `biosigio.strip_subject_info` applies the same removal to any metadata mapping without modifying it,
  and `biosigio.is_subject_info_key` tests one name.
- The default, `exclude_subject_info=False`, writes the same store as 1.2.10
  (every array and chunk file is byte-identical; only `created_utc` and `biosigio_version` differ).

## [1.2.10] - 2026-09-29

**This patch release contains breaking changes.**
It is numbered 1.2.10 rather than 1.3.0 by the maintainer's decision, as 1.2.9 was;
read Breaking before upgrading.
The breaking changes concern repeated channel labels
(in a file, in a `select_channels` call, or produced by EDF/BDF's 16-character label field)
and `channels.tsv` names that differ from the data file's only in case;
data without either imports and exports as in 1.2.9.
To adapt, de-duplicate names passed to `select_channels` and re-check any store regenerated from a file with repeated labels;
to defer, pin `biosigio==1.2.9`, since a `~=1.2.9` or `<1.3` pin takes 1.2.10 automatically.

### Breaking

- `Recording.select_channels` raises `ValueError` when a name is listed twice,
  instead of returning a recording whose signal frame has the column twice and whose channel table has it once.
  Every iterable of names that selected channels in 1.2.9 (a list, dict keys, numpy array, pandas `Index` or `Series`) still does.
- The Parquet/Arrow importer raises `ValueError` for a table that repeats a signal column name
  instead of building a recording with two columns under one label.
  Parquet cannot read such a file back, so only an edited Feather/Arrow file reaches this.
- For inputs whose repeated labels collide with a label the file itself uses
  (XDF streams `[Ch1, Ch1]` and `[Ch1_1]`, an EEGLAB `Fz, Fz, Fz_2`, neo streams that repeat `x` and `x_0`),
  a label such as `Ch1_1` now points at a different signal than in 1.2.9:
  it is the channel really named `Ch1_1`, where 1.2.9 gave that label to the copy of `Ch1`.
  Likewise, an XDF data channel named like a stream's `{stream_name}_LSL_timestamps` channel now keeps that label,
  where 1.2.9 overwrote it with the timestamps; the timestamps become `{stream_name}_LSL_timestamps_1`.
  A Zarr store regenerated from such a file therefore swaps data under an existing channel name.
  Inputs without such a collision import exactly as before.
- EDF/BDF export writes channels whose labels collide once truncated to 16 characters
  under numbered labels (`Mini sensor 10-0`, `Mini sensor 10-1`; see Fixed),
  and names their `_channels.tsv` rows the same way,
  so such a file exported by 1.2.10 carries different channel labels and sidecar names than one exported by 1.2.9.
- Delsys Trigno files whose `Label:` lines repeat a label (see Fixed):
  the first column under that label now takes its own `Label:` line's rate and unit, where 1.2.9 gave it the last line's,
  and the later columns are imported as new, suffixed channels.
  Such a file is matched to its columns by position, so one whose header line (the line with `X[s]`)
  does not have one field per data column now raises `ValueError`, where 1.2.9 imported the first column under each label.
- `apply_channels_tsv` and `apply_channels_tsv_to_stream` now apply a sidecar row that differs from its channel only in case
  (see Fixed, [#136](https://github.com/neuromechanist/biosigio/issues/136)).
  In 1.2.9 such a row was silently skipped; now its type and unit are applied, and a unit that differs from the data file's CONVERTS the signal,
  so a store regenerated from such a dataset can differ in channel types, units and values from one published before.

### Fixed

- XDF: a channel whose label repeats another's is always kept.
  Two streams that share a name now each keep their `{stream_name}_LSL_timestamps` channel (the second is suffixed `_1`),
  where one of them was dropped,
  and a timestamp channel no longer overwrites a data channel that carries the same name.
  The existing `_1`, `_2` suffixes for repeated labels are kept, so no channel that imported under its own label is renamed,
  but a suffix now never takes a label some stream really uses:
  streams `[Ch1, Ch1]` and `[Ch1_1]` import as `Ch1`, `Ch1_2`, `Ch1_1`, where the genuine `Ch1_1` became `Ch1_1_1`.
  The renames are logged with their stream names
  ([#134](https://github.com/neuromechanist/biosigio/issues/134)).
- EEGLAB (classic and v7.3): a repeated label's `_2`, `_3` suffix never takes a label the file itself uses,
  so `Fz, Fz, Fz_2` imports as `Fz, Fz_3, Fz_2` instead of giving the second `Fz` the genuine `Fz_2`'s label
  and pushing that channel to `Fz_2_2`.
  No channel was dropped before; a `channels.tsv` row for `Fz_2` described the wrong one
  ([#134](https://github.com/neuromechanist/biosigio/issues/134)).
- neo: the same rule for names repeated across merged streams:
  streams naming `x_0`, `x_0` and `x_0_0` import as `x_0`, `x_0_1`, `x_0_0` rather than `x_0`, `x_0_0`, `x_0_0_0`.
- Delsys Trigno: a file whose `Label:` lines repeat a label keeps every channel,
  where 1.2.9 imported only the first column under that label
  and gave it the rate and unit of the LAST `Label:` line with that label.
  Now the n-th column headed with the label takes the n-th `Label:` line's rate and unit;
  later occurrences are suffixed `_1`, `_2`, ... (never onto a label the file uses) and logged.
  Files with unique labels, such as the bundled Delsys export, whose labels carry the sensor number, import exactly as before.
- EDF/BDF export: labels that are no longer unique once truncated to the 16-character field
  (all channels of one Trigno sensor, for example) are numbered within the field (`Mini sensor 10-0`, `Mini sensor 10-1`)
  with a warning, instead of being written under one repeated label.
  In the `_channels.tsv` sidecar, only those numbered channels are named by the label written to the file,
  so their rows match on re-import; a channel that is truncated but stays unique keeps its full label there, as in 1.2.9.
- `apply_channels_tsv` and `apply_channels_tsv_to_stream` fall back to a case-insensitive match
  for a row that matches no channel exactly, when exactly one channel fits, no row names that channel exactly
  and no other row folds to it,
  so a sidecar that writes `Fp1-F7` for an EDF header's `FP1-F7` applies its type and unit.
  Case is compared with full Unicode case folding (`ß` matches `ss`) and without Unicode normalization (NFC and NFD spellings differ).
  The match is logged at info level and listed in the `channels_tsv_units` report as `matched_case_insensitive`,
  `{sidecar_name: channel_label}`, a key present only when such a match happened.
  An ambiguous row is left unapplied with a warning
  ([#136](https://github.com/neuromechanist/biosigio/issues/136)).
- `Recording.select_channels` accepts any iterable of names:
  a tuple or a set, which raised in 1.2.9, now select those channels,
  and a generator, which selected nothing, now selects its names.
  Channels keep the order the iterable yields, so a set gives an arbitrary column order.
- CSV: a repeated name in `channel_names` or `columns` raises `ValueError` naming it,
  instead of pandas' `Data must be 1-dimensional`.

### Changed

- The XDF, EEGLAB, neo and Delsys Trigno importers now record their renames under `channel_labels_deduplicated`
  (`{new_label: original_label}`), as the EDF/BDF, WFDB and Zarr importers already did,
  so a store regenerated from a file with repeated labels carries this metadata key where one published before did not.
  The rule they share is `biosigio.importers._labels.suffix_repeated_labels`:
  the first occurrence keeps its label and later ones are numbered, each format keeping its own separator and first number.

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
- BrainVision sibling lookup falls back to ignoring the extension's case (`<stem>.EEG`, `<stem>.VMRK`)
  when exactly one file matches; the stem must still match exactly,
  up to Unicode normalization (NFC or NFD); an upper-case marker file is staged under a lower-case name,
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

[1.2.14]: https://github.com/neuromechanist/biosigio/compare/v1.2.13...v1.2.14
[1.2.13]: https://github.com/neuromechanist/biosigio/compare/v1.2.12...v1.2.13
[1.2.12]: https://github.com/neuromechanist/biosigio/compare/v1.2.11...v1.2.12
[1.2.11]: https://github.com/neuromechanist/biosigio/compare/v1.2.10...v1.2.11
[1.2.10]: https://github.com/neuromechanist/biosigio/compare/v1.2.9...v1.2.10
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
