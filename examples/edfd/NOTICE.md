# Discontinuous EDF+D and BDF+D fixtures

These are tiny, fully synthetic recordings for the tests that refuse an EDF+D or BDF+D file with a gap between its data records (biosigIO issue #180).
No real recording and no subject data went into them, and they carry no source dataset, DOI or license beyond this repository's.

They are written byte for byte from the EDF+ and BDF+ specifications by `biosigio/tests/edfd_builder.py`, because no writer produces a file whose records are deliberately spaced out.
Regenerate them with `uv run python -m biosigio.tests.edfd_builder`.

Every file has one channel, `EEG Ramp`, at 10 Hz with 1 s data records, and an annotations signal.
The ramp's value in microvolts is the true time in seconds of the sample, so a reader that places a sample after a gap too early shows it as a value that disagrees with its position on the time axis.
Each file carries two annotations: `E1` at true time 0.5 s and `E2` in the third record.

| File | Format | Reserved field | Record onsets (s) | `E2` at (s) |
|---|---|---|---|---|
| `edfd_gap_8s.edf` | EDF+ | `EDF+D` | 0, 1, 10, 11 (an 8 s gap) | 10.5 |
| `edfd_gap_1s.edf` | EDF+ | `EDF+D` | 0, 1, 3, 4 (a 1 s gap) | 3.5 |
| `bdfd_gap_8s.bdf` | BDF+ | `BDF+D` | 0, 1, 10, 11 (an 8 s gap) | 10.5 |
| `edfd_contiguous.edf` | EDF+ | `EDF+D` | 0, 1, 2, 3 (labeled discontinuous, records contiguous) | 2.5 |
| `edfc_control.edf` | EDF+ | `EDF+C` | 0, 1, 2, 3 | 2.5 |
