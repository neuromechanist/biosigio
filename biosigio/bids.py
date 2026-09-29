"""Brain Imaging Data Structure (BIDS) sidecar helpers.

When a data file follows the BIDS layout, the authoritative per-channel
metadata lives in a sibling ``*_channels.tsv`` (the ``type`` and ``units``
columns), not in the data file's own headers. These helpers locate that sidecar
and apply it to an :class:`~biosigio.core.emg.Recording` object so imported channels get
their real BIDS types (e.g. ``SEEG``) instead of header/label guesses.

The ``units`` column is a claim about the numbers, not just a label, so adopting
it rescales the samples (see :func:`_decide_unit`). Applying a sidecar therefore
leaves values and unit in agreement, which relabelling alone did not.

**Two callers, one decision table.** :func:`apply_channels_tsv` serves the
in-memory path (a whole :class:`~biosigio.core.emg.Recording` in RAM, whose
columns are rescaled in place), and :func:`apply_channels_tsv_to_stream` serves
the bounded-memory streaming exporter (no Recording exists; the samples arrive
window by window later). Both route every per-channel question through
:func:`_decide_unit`, which decides and returns rather than mutating, so the two
export paths cannot drift on what a sidecar means -- the failure issue #127
reports, where a dataset's small runs served microvolts and its large ones served
volts.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from .core.channel_types import DISCRETE_CHANNEL_TYPES
from .core.modality import infer_modality_from_channel_type, validate_channel_type
from .units import conversion_factor

if TYPE_CHECKING:
    from .core.emg import Recording


def _find_sidecar(data_filepath: str, kind: str) -> str | None:
    """Return the sibling BIDS ``_<kind>.tsv`` path for a data file, or None.

    BIDS names a sidecar with the data file's entities but a different suffix,
    e.g. ``sub-01_task-rest_ieeg.edf`` -> ``sub-01_task-rest_channels.tsv``.

    Args:
        data_filepath: Path to a (BIDS) data file.
        kind: Sidecar kind, e.g. ``"channels"`` or ``"events"``.

    Returns:
        The sidecar path if it exists next to the data file, else ``None``.
    """
    directory, filename = os.path.split(data_filepath)
    stem = os.path.splitext(filename)[0]
    candidates = []
    # BIDS-correct: drop the trailing _<suffix> entity (e.g. _ieeg, _eeg, _meg).
    if "_" in stem:
        candidates.append(f"{stem.rsplit('_', 1)[0]}_{kind}.tsv")
    # Fallback: the full-stem form biosigio's own EDF exporter currently writes,
    # so a to_edf -> from_file round-trip also finds the sidecar.
    candidates.append(f"{stem}_{kind}.tsv")
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def find_channels_tsv(data_filepath: str) -> str | None:
    """Return the sibling BIDS ``_channels.tsv`` path for a data file, or None."""
    return _find_sidecar(data_filepath, "channels")


def find_events_tsv(data_filepath: str) -> str | None:
    """Return the sibling BIDS ``_events.tsv`` path for a data file, or None."""
    return _find_sidecar(data_filepath, "events")


# Per-channel outcomes of adopting a sidecar unit, and the keys of the report
# left in ``rec.metadata["channels_tsv_units"]``.
_UNCHANGED = "unchanged"
_CONVERTED = "converted"
_RELABELLED = "relabelled"
_KEPT = "kept_importer_unit"


class _UnitDecision(NamedTuple):
    """What one sidecar ``units`` cell does to one channel.

    A decision is data, not an edit: it says what the channel's unit label
    becomes, what its samples must be multiplied by to still mean that label,
    and what its recorded ``bids_unit`` should be afterwards. That is what lets
    the in-memory path (which has the samples in a DataFrame and rescales them
    now) and the streaming path (which will see them window by window, later)
    share one decision table instead of two implementations of the same rules.

    Attributes:
        outcome: ``_UNCHANGED``, ``_CONVERTED``, ``_RELABELLED`` or ``_KEPT``;
            the key this channel contributes to the ``channels_tsv_units`` report.
        unit: The unit label the channel ends up with. Only meaningful to write
            when the outcome adopted the sidecar's unit (``_CONVERTED`` /
            ``_RELABELLED``); otherwise it is the unit the channel already had.
        factor: The multiplier the samples need, always exactly ``1.0`` unless
            the outcome is ``_CONVERTED``.
        bids_unit: What the channel's ``bids_unit`` key should hold afterwards --
            the sidecar's declared unit when it was recorded rather than adopted,
            or **None** meaning "there is no unresolved conflict, drop it".
    """

    outcome: str
    unit: str
    factor: float
    bids_unit: str | None


def _keep_importer_unit(
    label: str,
    current: str,
    declared: str,
    reason: str,
    origin: str,
    recorded_bids_unit: str | None,
) -> _UnitDecision:
    """Record the sidecar's unit without asserting it over contradicting values.

    The BIDS metadata is real and worth keeping, so it lands under ``bids_unit``
    rather than being dropped; what it must not do is overwrite a
    ``physical_dimension`` that correctly describes the samples.

    Args:
        label: Channel name, for the warning.
        current: The unit the samples are actually in.
        declared: The sidecar's ``units`` value.
        reason: Why the sidecar's unit was not adopted, for the warning.
        origin: The sidecar's path, or ``<DataFrame>``, for the warning.
        recorded_bids_unit: The channel's existing ``bids_unit``, if any.

    Returns:
        A ``_KEPT`` decision, or an ``_UNCHANGED`` one when this exact
        ``bids_unit`` was already recorded -- so re-applying a sidecar neither
        re-warns nor re-counts.
    """
    if recorded_bids_unit == declared:
        return _UnitDecision(_UNCHANGED, current, 1.0, declared)
    logging.warning(
        "channels.tsv declares units %r for channel %r, whose values are in %r and %s; "
        "keeping the importer's values and unit, and recording the declared unit "
        "as 'bids_unit': %s",
        declared,
        label,
        current,
        reason,
        origin,
    )
    return _UnitDecision(_KEPT, current, 1.0, declared)


def _rescale(column: pd.Series, factor: float) -> pd.Series:
    """Multiply a signal column, preserving a float column's own precision.

    A float column keeps its own dtype: an EEGLAB recording loads at float32
    deliberately, to halve memory, and applying a sidecar must not double it
    back. The cast is defensive rather than currently load-bearing -- pandas
    treats a scalar operand as weak under NEP 50, so ``float32 * factor`` is
    already float32 today -- but raw numpy promotes the same expression to
    float64, so the guarantee is stated in the code rather than inherited from a
    promotion rule that may change.

    An integer column has no precision to preserve and becomes float64, which is
    the only correct result for a non-integral factor.
    """
    scaled = column * factor
    if pd.api.types.is_float_dtype(column.dtype):
        return scaled.astype(column.dtype)
    return scaled


def _decide_unit(
    *,
    label: str,
    current: str,
    declared: str,
    channel_type: str,
    has_samples: bool,
    origin: str,
    recorded_bids_unit: str | None = None,
) -> _UnitDecision:
    """Decide what moving one channel onto the sidecar's unit means.

    **The single decision table for both export paths.** It answers the question
    and returns it (see :class:`_UnitDecision`); the caller performs whatever
    edit that implies -- rescaling a DataFrame column now
    (:func:`apply_channels_tsv`) or carrying a factor into a later windowed read
    (:func:`apply_channels_tsv_to_stream`). Splitting the answer from the edit is
    the whole point: a store built by streaming and a store built in memory must
    disagree about nothing (issue #127).

    A unit label is a claim about the numbers next to it, so the two move
    together or not at all. The sidecar's ``units`` describes the values *as the
    data file stores them*, while the importer's unit describes the values
    biosigIO currently holds -- and those differ whenever the importer rescaled
    on the way in (every MNE-backed importer returns SI volts regardless of the
    file's own µV). Adopting the label without the conversion is what issue #122
    reports: volts relabelled as microvolts, wrong by 10^6.

    The checks run in this order, and the order is load-bearing:

    1. **Already there** (labels equal): nothing at all. This is what makes
       repeated application idempotent, and it comes first so a discrete or
       sample-less channel whose unit already matches is not flagged as a
       conflict with itself.
    2. **Discrete channel type** (``TRIG`` and friends, see
       :data:`~biosigio.core.channel_types.DISCRETE_CHANNEL_TYPES`): never
       rescaled. MNE labels stim channels with the FIFF volts code while they
       hold integer event codes, so a sidecar declaring ``mV`` would turn codes
       5/3/7 into 5000/3000/7000. The declared unit is recorded, not applied.
    3. **No samples**: a channel with metadata but no column cannot be rescaled,
       so it is not relabelled either -- checked before the conversion so even a
       same-magnitude spelling change cannot slip through on a channel whose
       numbers are not there to agree with it. (Always false for a streamed
       channel: every channel the source lists has a row in the transpose memmap.)
    4. **Convertible** (same quantity, e.g. ``V`` -> ``uV``): multiply the samples
       by the ratio and set the label. A ratio of exactly 1 (``uV`` -> ``µV``, a
       spelling difference) sets the label alone, which is not a semantic relabel.
    5. **Not convertible** (unparsable on either side, or different quantities):
       keep the importer's values *and* its label, and record the sidecar's claim
       under ``bids_unit`` so the BIDS metadata is preserved without being
       asserted over numbers that would contradict it.

    Whenever the sidecar's unit *is* adopted, any ``bids_unit`` left by an
    earlier disagreement is dropped, so the two never both describe the channel.

    Args:
        label: Channel name, for warnings.
        current: The unit the channel's values are in now, already stripped.
        declared: The sidecar's ``units`` value, already known to be non-empty
            and not ``n/a``.
        channel_type: The channel's type, for the discrete-code exemption.
        has_samples: Whether there are samples this decision can apply to.
        origin: The sidecar's path, or ``<DataFrame>``, for warnings.
        recorded_bids_unit: The channel's existing ``bids_unit``, if any.

    Returns:
        The :class:`_UnitDecision` for this channel.
    """
    if current == declared:
        return _UnitDecision(_UNCHANGED, current, 1.0, None)

    if str(channel_type or "").upper() in DISCRETE_CHANNEL_TYPES:
        return _keep_importer_unit(
            label,
            current,
            declared,
            "holds discrete codes rather than a measured quantity",
            origin,
            recorded_bids_unit,
        )

    if not has_samples:
        return _keep_importer_unit(
            label,
            current,
            declared,
            "has no samples to rescale",
            origin,
            recorded_bids_unit,
        )

    factor = conversion_factor(current, declared)
    if factor is None:
        return _keep_importer_unit(
            label,
            current,
            declared,
            f"is not convertible to {declared!r}",
            origin,
            recorded_bids_unit,
        )

    return _UnitDecision(_CONVERTED if factor != 1.0 else _RELABELLED, declared, factor, None)


def _adopt_units(rec: Recording, label: str, declared: str, origin: str) -> str:
    """Apply :func:`_decide_unit` to one in-memory channel, rescaling its column.

    Args:
        rec: The Recording object to update in place.
        label: Channel name, already known to exist in ``rec.channels``.
        declared: The sidecar's ``units`` value, already known to be non-empty
            and not ``n/a``.
        origin: The sidecar's path, or ``<DataFrame>``, for warnings.

    Returns:
        One of ``_UNCHANGED``, ``_CONVERTED``, ``_RELABELLED`` or ``_KEPT``.
    """
    info = rec.channels[label]
    signals = rec.signals
    decision = _decide_unit(
        label=label,
        current=str(info.get("physical_dimension") or "").strip(),
        declared=declared,
        channel_type=str(info.get("channel_type") or ""),
        has_samples=signals is not None and label in signals,
        origin=origin,
        recorded_bids_unit=info.get("bids_unit"),
    )
    if decision.outcome in (_CONVERTED, _RELABELLED):
        if decision.factor != 1.0:
            assert signals is not None  # a conversion decision implies has_samples
            signals[label] = _rescale(signals[label], decision.factor)
        info["physical_dimension"] = decision.unit
    if decision.bids_unit is None:
        info.pop("bids_unit", None)
    else:
        info["bids_unit"] = decision.bids_unit
    return decision.outcome


def _read_channels_tsv(channels_tsv: str | os.PathLike | pd.DataFrame) -> pd.DataFrame:
    """Return a ``channels.tsv`` as an all-string frame, from a path or a frame.

    Reading with ``dtype=str, keep_default_na=False`` is what keeps a literal
    ``n/a`` cell (BIDS's own spelling for "not applicable") a string the callers
    can test for, rather than a float NaN. A caller-supplied DataFrame gets the
    equivalent treatment -- stringified, with genuine missing values flattened to
    the empty cell that means "declares nothing" -- so passing a frame and passing
    the file it was read from behave identically.
    """
    if isinstance(channels_tsv, pd.DataFrame):
        return channels_tsv.astype(str).mask(channels_tsv.isna(), "")
    return pd.read_csv(channels_tsv, sep="\t", dtype=str, keep_default_na=False)


def _sidecar_origin(channels_tsv: str | os.PathLike | pd.DataFrame) -> str:
    """What to call the sidecar in a log message: its path, or that it was a frame.

    A DataFrame has no path, and interpolating one into a warning would dump the
    whole table into the log.
    """
    return "<DataFrame>" if isinstance(channels_tsv, pd.DataFrame) else str(channels_tsv)


def resolve_channels_tsv(
    filepath: str, bids_channels: str | os.PathLike | pd.DataFrame | None
) -> str | os.PathLike | pd.DataFrame | None:
    """Turn a ``bids_channels`` argument into the sidecar to apply, or None.

    The one place the ``bids_channels`` vocabulary is interpreted, shared by
    :meth:`~biosigio.core.emg.Recording.from_file` and
    :func:`~biosigio.exporters.zarr_stream.stream_to_zarr` so the two export
    paths cannot come to differ about what an argument means -- which is the
    same reason :func:`_decide_unit` is shared (issue #127).

    - ``"auto"`` (the default both callers use) resolves the sibling
      ``_channels.tsv`` through :func:`find_channels_tsv`, and yields None when
      the recording has none.
    - ``"off"``, and None as its synonym, disable the lookup.
    - Anything else is a path or a DataFrame the caller chose explicitly. NEMAR's
      MaxShield path needs this: it converts a filtered copy written to scratch,
      where the recording's real sidecar is not adjacent.

    The string cases are tested first because a DataFrame compared against
    ``"auto"`` compares elementwise and has no truth value.

    Args:
        filepath: The recording, used only to resolve ``"auto"``.
        bids_channels: ``"auto"``, ``"off"``, None, a path, or a DataFrame.

    Returns:
        A path, a DataFrame, or **None** when no sidecar should be applied.
    """
    if isinstance(bids_channels, str):
        if bids_channels == "auto":
            # rstrip for the directory-valued recordings (CTF .ds, 4D/BTi), so a
            # trailing slash does not leave os.path.split with an empty filename.
            return find_channels_tsv(filepath.rstrip("/\\"))
        return None if bids_channels == "off" else bids_channels
    return bids_channels


# Report key listing the sidecar rows matched to a channel only by ignoring case,
# as ``{sidecar_name: channel_label}``. Present only when there was such a match,
# so a report for a sidecar that matched exactly is unchanged.
_CASE_MATCHED = "matched_case_insensitive"


def _match_sidecar_names(
    row_names: list[str], labels: list[str], origin: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Map each ``channels.tsv`` row name to the channel label it describes.

    A row matches the channel of exactly its name first. A row with no exact
    match falls back to ignoring case (``Fp1-F7`` for an EDF header's ``FP1-F7``,
    as CHB-MIT writes them) only when that is unambiguous: exactly one channel
    label casefolds to the row's name, no row names that channel exactly, and no
    other unmatched row casefolds to it too. Anything else stays unmatched, the
    way every non-exact row used to, and an ambiguous row is warned about.

    Case is compared with :meth:`str.casefold`, full Unicode case folding, so it
    is broader than lower-casing: ``STRASSE`` matches ``Straße`` because ``ß``
    folds to ``ss``. It does not normalize Unicode, so a label spelled with a
    precomposed character (NFC ``é``) and one spelled with a combining accent
    (NFD ``e`` + U+0301) do not match.

    Shared by :func:`apply_channels_tsv` and :func:`apply_channels_tsv_to_stream`
    so the two export paths match the same rows to the same channels.

    Args:
        row_names: The sidecar's ``name`` values, stripped, in file order.
        labels: The recording's channel labels (a streaming source may repeat one).
        origin: The sidecar's path, or ``<DataFrame>``, for log messages.

    Returns:
        ``(matches, case_matches)``: ``{row_name: channel_label}`` for every row
        that describes a channel, and the subset matched only by ignoring case.
    """
    distinct_labels = list(dict.fromkeys(labels))
    label_set = set(distinct_labels)
    rows = list(dict.fromkeys(row_names))
    matches = {name: name for name in rows if name in label_set}
    unmatched = [name for name in rows if name not in matches]

    labels_by_fold: dict[str, list[str]] = {}
    for label in distinct_labels:
        labels_by_fold.setdefault(label.casefold(), []).append(label)
    rows_by_fold: dict[str, list[str]] = {}
    for name in unmatched:
        rows_by_fold.setdefault(name.casefold(), []).append(name)

    matched_labels = set(matches.values())
    case_matches: dict[str, str] = {}
    for name in unmatched:
        candidates = labels_by_fold.get(name.casefold(), [])
        if not candidates:
            continue
        rivals = rows_by_fold[name.casefold()]
        if len(candidates) == 1 and candidates[0] not in matched_labels and len(rivals) == 1:
            matches[name] = case_matches[name] = candidates[0]
            matched_labels.add(candidates[0])
            logging.info(
                "channels.tsv row %r matches channel %r only when case is ignored; "
                "applying it to that channel: %s",
                name,
                candidates[0],
                origin,
            )
        else:
            logging.warning(
                "channels.tsv row %r matches no channel exactly and is ambiguous when "
                "case is ignored (channels %s, rows %s); leaving it unapplied: %s",
                name,
                candidates,
                rivals,
                origin,
            )
    return matches, case_matches


def apply_channels_tsv(rec: Recording, channels_tsv: str | os.PathLike | pd.DataFrame) -> int:
    """Override per-channel ``type``/``units`` in ``rec`` from a ``_channels.tsv``.

    Rows are matched to channels by the ``name`` column, exactly first; a row
    that matches no channel exactly falls back to ignoring case only when that
    is unambiguous (see :func:`_match_sidecar_names`). ``n/a`` and empty
    values are skipped (the importer-inferred value is kept). An unrecognized
    ``type`` is warned about and skipped rather than raising.

    Adopting the sidecar's ``units`` **converts the channel's samples** into that
    unit rather than merely relabelling them (issue #122); see
    :func:`_decide_unit` for the per-channel rule, including the channel types
    and situations that are exempt. Type and units are applied independently, so
    an unrecognized ``type`` no longer costs the row its unit correction.

    A summary of what the ``units`` column did lands in
    ``rec.metadata["channels_tsv_units"]`` as
    ``{"converted", "relabelled", "kept_importer_unit", "units_column_present"}``,
    so a caller can tell "the sidecar declared no units" from "the units were
    already correct" without re-scanning every channel. When a row was matched
    only by ignoring case, the report also carries ``matched_case_insensitive``,
    ``{sidecar_name: channel_label}``; the key is absent otherwise.

    Args:
        rec: The Recording object to update in place.
        channels_tsv: Path to the BIDS ``_channels.tsv``, or an already-loaded
            DataFrame of it (the two are equivalent; see
            :func:`_read_channels_tsv`).

    Returns:
        The number of **distinct channels** whose record changed -- type adopted,
        unit adopted, or ``bids_unit`` recorded. A channel named by two rows, or
        by a sidecar applied twice, counts once; a row naming a channel the
        recording does not have counts not at all.
    """
    df = _read_channels_tsv(channels_tsv)
    origin = _sidecar_origin(channels_tsv)
    if "name" not in df.columns:
        logging.warning("channels.tsv has no 'name' column: %s", origin)
        return 0

    units_present = "units" in df.columns
    if not units_present:
        logging.warning("channels.tsv has no 'units' column: %s", origin)

    report: dict = {
        _CONVERTED: 0,
        _RELABELLED: 0,
        _KEPT: 0,
        "units_column_present": units_present,
    }
    targets, case_matches = _match_sidecar_names(
        [str(name).strip() for name in df["name"]], list(rec.channels), origin
    )
    if case_matches:
        report[_CASE_MATCHED] = case_matches
    changed: set[str] = set()
    matched: set[str] = set()
    for _, row in df.iterrows():
        name = targets.get(str(row["name"]).strip())
        if name is None:
            continue
        matched.add(name)
        ctype = str(row.get("type", "")).strip()
        if ctype and ctype.lower() != "n/a":
            try:
                rec.set_channel(name, channel_type=ctype)
                changed.add(name)
            except ValueError:
                logging.warning(
                    "channels.tsv type %r for channel %r is not a known channel type; "
                    "keeping the importer-inferred type: %s",
                    ctype,
                    name,
                    origin,
                )
        units = str(row.get("units", "")).strip()
        if units and units.lower() != "n/a":
            outcome = _adopt_units(rec, name, units, origin)
            if outcome != _UNCHANGED:
                report[outcome] += 1
                changed.add(name)
        elif units_present:
            logging.debug(
                "channels.tsv declares no unit (%r) for channel %r: %s", units, name, origin
            )

    uncovered = [label for label in rec.channels if label not in matched]
    if uncovered:
        logging.debug(
            "channels.tsv names no row for %d of the recording's channels (e.g. %s): %s",
            len(uncovered),
            ", ".join(repr(label) for label in uncovered[:5]),
            origin,
        )

    rec.set_metadata("channels_tsv_units", report)
    return len(changed)


def apply_channels_tsv_to_stream(
    channels: list[dict],
    channels_tsv: str | os.PathLike | pd.DataFrame,
    *,
    force_modality: str | None = None,
) -> dict | None:
    """Apply a ``_channels.tsv`` to a **streaming** source's channel table.

    The streaming Zarr exporter has no :class:`~biosigio.core.emg.Recording` and
    no samples in memory: it knows each channel's label, type, modality and unit,
    and will read the values window by window afterwards. So this settles every
    per-channel question through the same :func:`_decide_unit` table the
    in-memory path uses and leaves the *arithmetic* for later, as a per-channel
    ``unit_factor`` the exporter multiplies into each channel exactly once. Without
    it, a dataset whose recordings straddle a streaming size threshold serves its
    small runs in the sidecar's unit and its large ones in the importer's native
    unit -- a 10^6 disagreement inside one dataset (issue #127).

    Each entry in ``channels`` is mutated in place: ``channel_type`` (and, unless
    ``force_modality`` pins it, ``modality``) from the sidecar's ``type``;
    ``unit`` and ``unit_factor`` from its ``units``; ``bids_unit`` when a declared
    unit was recorded rather than adopted. Entries the sidecar does not name are
    left exactly as the importer built them.

    Rows are matched by the ``name`` column, with the same exact-then-unambiguous
    case-insensitive rule as :func:`apply_channels_tsv`, and several rows naming
    one channel compose in file order, the same as there. Should a
    source ever list the same label twice, every entry with that label gets the
    same treatment, so duplicate labels cannot end up in different units inside
    one store (the EDF source suffixes repeats MNE-style, so it lists none).

    Args:
        channels: The source's per-channel dicts (``label``, ``channel_type``,
            ``modality``, ``unit``), mutated in place.
        channels_tsv: Path to the sidecar, or an already-loaded DataFrame.
        force_modality: When set, the caller has pinned every channel to one
            modality (the BIDS datatype suffix, say) and the sidecar's ``type``
            must not move it. The ``type`` is still adopted per channel.

    Returns:
        The ``channels_tsv_units`` report, in the same shape
        :func:`apply_channels_tsv` leaves in ``rec.metadata``, or **None** when
        the sidecar has no ``name`` column and nothing could be applied -- so the
        exporter records an attr exactly when the in-memory path would.
    """
    df = _read_channels_tsv(channels_tsv)
    origin = _sidecar_origin(channels_tsv)
    if "name" not in df.columns:
        logging.warning("channels.tsv has no 'name' column: %s", origin)
        return None

    units_present = "units" in df.columns
    if not units_present:
        logging.warning("channels.tsv has no 'units' column: %s", origin)

    report: dict = {
        _CONVERTED: 0,
        _RELABELLED: 0,
        _KEPT: 0,
        "units_column_present": units_present,
    }
    targets, case_matches = _match_sidecar_names(
        [str(name).strip() for name in df["name"]],
        [str(entry["label"]) for entry in channels],
        origin,
    )
    if case_matches:
        report[_CASE_MATCHED] = case_matches
    # Keyed by the channel label each row describes, which is the row's own name
    # unless it matched only by ignoring case.
    rows_by_name: dict[str, list[pd.Series]] = {}
    for _, row in df.iterrows():
        target = targets.get(str(row["name"]).strip())
        if target is not None:
            rows_by_name.setdefault(target, []).append(row)

    uncovered: list[str] = []
    for entry in channels:
        # The label is NOT stripped, and must not be: apply_channels_tsv matches a
        # stripped row name against the Recording's channel keys as they are, so
        # stripping here would match rows on this path that the in-memory path
        # leaves unmatched -- a difference between the two exports, which is the
        # one thing this function exists to prevent.
        label = str(entry["label"])
        rows = rows_by_name.get(label)
        if not rows:
            uncovered.append(label)
            continue
        for row in rows:
            ctype = str(row.get("type", "")).strip()
            if ctype and ctype.lower() != "n/a":
                try:
                    entry["channel_type"] = validate_channel_type(ctype)
                    if force_modality is None:
                        entry["modality"] = infer_modality_from_channel_type(entry["channel_type"])
                except ValueError:
                    logging.warning(
                        "channels.tsv type %r for channel %r is not a known channel type; "
                        "keeping the importer-inferred type: %s",
                        ctype,
                        label,
                        origin,
                    )
            units = str(row.get("units", "")).strip()
            if units and units.lower() != "n/a":
                decision = _decide_unit(
                    label=label,
                    current=str(entry.get("unit") or "").strip(),
                    declared=units,
                    channel_type=str(entry.get("channel_type") or ""),
                    # Every channel the source lists has a row in the exporter's
                    # transpose memmap, so the "no samples to rescale" branch --
                    # which exists for a Recording carrying channel metadata
                    # without a column -- cannot apply here.
                    has_samples=True,
                    origin=origin,
                    recorded_bids_unit=entry.get("bids_unit"),
                )
                if decision.outcome != _UNCHANGED:
                    report[decision.outcome] += 1
                if decision.outcome in (_CONVERTED, _RELABELLED):
                    entry["unit"] = decision.unit
                    entry["unit_factor"] = float(entry.get("unit_factor", 1.0)) * decision.factor
                if decision.bids_unit is None:
                    entry.pop("bids_unit", None)
                else:
                    entry["bids_unit"] = decision.bids_unit
            elif units_present:
                logging.debug(
                    "channels.tsv declares no unit (%r) for channel %r: %s", units, label, origin
                )

    if uncovered:
        logging.debug(
            "channels.tsv names no row for %d of the recording's channels (e.g. %s): %s",
            len(uncovered),
            ", ".join(repr(label) for label in uncovered[:5]),
            origin,
        )
    return report


def read_events_tsv(
    events_tsv_path: str,
    *,
    description_column: str | None = None,
) -> pd.DataFrame | None:
    """Parse a BIDS ``_events.tsv`` into the standard events frame.

    Returns a DataFrame with ``onset``/``duration``/``description`` columns
    (sorted by onset), the same shape :attr:`Recording.events` uses, or **None**
    when the sidecar is unparsable (no ``onset`` column, or a forced
    ``description_column`` is absent) -- distinct from a valid-but-empty table (an
    empty DataFrame). Callers use None to leave any existing events untouched
    rather than wiping them. Used by :func:`apply_events_tsv` and by the streaming
    Zarr exporter (which has no Recording to mutate). See :func:`apply_events_tsv`
    for the column rules.
    """
    df = pd.read_csv(events_tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if "onset" not in df.columns:
        logging.warning("events.tsv has no 'onset' column: %s", events_tsv_path)
        return None

    def _is_na(value: str) -> bool:
        v = value.strip()
        return v == "" or v.lower() == "n/a"

    if description_column is not None:
        if description_column not in df.columns:
            logging.warning("events.tsv has no %r column: %s", description_column, events_tsv_path)
            return None
        desc_columns = [description_column]
    else:
        desc_columns = [c for c in ("trial_type", "value") if c in df.columns]

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []
    skipped = 0
    for _, row in df.iterrows():
        raw_onset = str(row["onset"]).strip()
        try:
            onset = float(raw_onset)
        except ValueError:
            skipped += 1
            continue
        if not math.isfinite(onset):  # NaN or +/-inf
            skipped += 1
            continue
        raw_duration = str(row.get("duration", "")).strip()
        try:
            duration = 0.0 if _is_na(raw_duration) else float(raw_duration)
        except ValueError:
            duration = 0.0
        description = "n/a"
        for col in desc_columns:
            candidate = str(row.get(col, "")).strip()
            if not _is_na(candidate):
                description = candidate
                break
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)

    if skipped:
        logging.warning(
            "events.tsv: skipped %d row(s) with missing/non-numeric onset: %s",
            skipped,
            events_tsv_path,
        )

    return (
        pd.DataFrame({"onset": onsets, "duration": durations, "description": descriptions})
        .sort_values(by="onset")
        .reset_index(drop=True)
    )


def apply_events_tsv(
    rec: Recording,
    events_tsv_path: str,
    *,
    description_column: str | None = None,
) -> int:
    """Replace ``rec.events`` with the authoritative BIDS ``_events.tsv`` list.

    A BIDS ``_events.tsv`` is the curated event table for a recording; it is
    richer and more reliable than the data file's own markers (e.g. a
    BrainVision ``.vmrk`` or an EEGLAB event struct), so when it is present it
    is the source of truth. This loads it into ``rec.events`` as the standard
    ``onset``/``duration``/``description`` frame, overwriting any
    importer-loaded events.

    Columns:
        ``onset`` (required, seconds) and ``duration`` (seconds; ``n/a`` or
        missing -> ``0.0``) follow the BIDS spec. The ``description`` is taken
        from ``description_column`` if given, else per row from the first of
        ``trial_type`` then ``value`` that is present and not ``n/a`` (BIDS
        names the categorical label ``trial_type`` and the raw marker
        ``value``; datasets populate one or both). Rows whose ``onset`` is
        missing or non-numeric are skipped.

    Args:
        rec: The Recording object to update in place.
        events_tsv_path: Path to the BIDS ``_events.tsv``.
        description_column: Force the description to come from this column.

    Returns:
        The number of events loaded into ``rec.events``. An unparsable sidecar
        (no ``onset`` column / missing forced column) loads nothing and leaves any
        importer-loaded events intact, returning 0.
    """
    events = read_events_tsv(events_tsv_path, description_column=description_column)
    if events is None:
        return 0
    rec.events = events
    return len(rec.events)
