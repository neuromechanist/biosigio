"""Repeated channel labels outside EDF/BDF: every channel survives, under a stable name.

A Recording is keyed by channel label, so any path that writes a label twice can
drop a channel (a signal column overwritten) or misattribute one (a sidecar row
for a genuine label landing on a synthesized one). These tests build real files
(XDF, EEGLAB ``.set``, neo, CSV, Parquet) and check that the channel count and
the samples behind each label survive. The EDF/BDF side lives in
``test_edf_duplicate_labels.py``.
"""

from biosigio.importers._labels import suffix_repeated_labels


def test_suffix_keeps_the_first_occurrence():
    labels, renames = suffix_repeated_labels(["Fz", "Cz", "Fz", "Fz"], start=2)
    assert labels == ["Fz", "Cz", "Fz_2", "Fz_3"]
    assert renames == {"Fz_2": "Fz", "Fz_3": "Fz"}


def test_suffix_never_takes_a_genuine_label():
    """A genuine ``Fz_2`` keeps its name even when it comes after the repeat."""
    labels, renames = suffix_repeated_labels(["Fz", "Fz", "Fz_2"], start=2)
    assert labels == ["Fz", "Fz_3", "Fz_2"]
    assert renames == {"Fz_3": "Fz"}


def test_suffix_skips_names_already_synthesized():
    labels, _ = suffix_repeated_labels(["a", "a", "a_1", "a_1"], start=1)
    assert labels == ["a", "a_2", "a_1", "a_1_1"]
    assert len(set(labels)) == 4


def test_suffix_leaves_unique_labels_alone():
    assert suffix_repeated_labels(["x", "y"]) == (["x", "y"], {})
