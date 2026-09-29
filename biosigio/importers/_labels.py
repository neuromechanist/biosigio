"""Channel-label helpers shared by importers whose formats allow repeated labels."""

import logging

logger = logging.getLogger(__name__)

# Recording-metadata key under which an importer records the renames made by
# :func:`unique_channel_labels`, as ``{new_label: original_label}``. It rides in
# ``Recording.metadata``, so the Zarr exporters (in-memory and streaming) write
# it into the store's ``recording_metadata`` and a re-import restores it.
DEDUPLICATED_LABELS_KEY = "channel_labels_deduplicated"


def suffix_repeated_labels(
    labels: list[str], *, separator: str = "_", start: int = 1
) -> tuple[list[str], dict[str, str]]:
    """Keep the first occurrence of each label and number the later ones.

    This is the scheme the EEGLAB (``Fz``, ``Fz_2``, ``Fz_3``), XDF (``Ch1``,
    ``Ch1_1``) and neo (``ch``, ``ch_0``) importers have always used, so a label
    that imported unchanged before still does; it differs from
    :func:`unique_channel_labels`, which renames every occurrence the way MNE does.

    A suffixed name is never one the source itself uses: every original label is
    reserved before any suffix is chosen, so a genuine channel named ``Fz_2``
    keeps its name and the second ``Fz`` becomes ``Fz_3``. Without that
    reservation the synthesized name would take the genuine channel's label and
    push the genuine channel to ``Fz_2_2``, which is how a ``channels.tsv`` row
    for ``Fz_2`` ends up describing the wrong channel.

    Args:
        labels: Channel labels in source order.
        separator: Placed between the label and its number.
        start: The number the second occurrence gets.

    Returns:
        ``(labels, renames)``: unique labels in input order, and
        ``{new_label: original_label}`` for every label that was changed (empty
        when the labels were already unique). The caller reports the renames in
        its own format's words and records a non-empty mapping under
        :data:`DEDUPLICATED_LABELS_KEY`.
    """
    names = list(labels)
    reserved = set(names)
    taken: set[str] = set()
    out: list[str] = []
    renames: dict[str, str] = {}
    for name in names:
        if name not in taken:
            taken.add(name)
            out.append(name)
            continue
        number = start
        candidate = f"{name}{separator}{number}"
        while candidate in reserved or candidate in taken:
            number += 1
            candidate = f"{name}{separator}{number}"
        taken.add(candidate)
        out.append(candidate)
        renames[candidate] = name
    return out, renames


def unique_channel_labels(
    labels: list[str], *, source: str = "EDF", filepath: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Rename repeated channel labels the way MNE does, so no channel is dropped.

    EDF does not require unique labels (CHB-MIT declares ``T8-P8`` twice and
    uses ``-`` as a placeholder for several unused inputs), but a Recording is
    keyed by label. The same rule serves any other source that can repeat a
    label (a WFDB ``sig_name``, a Zarr store published before this rule
    existed); ``source`` names the format and ``filepath`` the file in the
    warning and error text.
    Every occurrence of a repeated label gets a running suffix
    ``-0``, ``-1``, ...; a suffix that would collide with an existing label
    falls through to ``-a``, ``-b``, ... Unique labels are returned unchanged.

    For a single repeated stem this matches MNE's ``_unique_channel_names``
    name for name, so the names match a ``channels.tsv`` written by MNE-BIDS.
    With several repeated stems whose suffixed names interact (one stem's
    candidate colliding with another stem's label or suffix), the result here
    is deterministic, stems being processed in first-occurrence order, whereas
    MNE iterates a ``set`` of stems, so its result can differ from this one
    and from run to run.

    Returns:
        ``(labels, renames)``: the de-duplicated labels, in input order, and
        ``{new_label: original_label}`` for every label that was changed
        (empty when all labels were already unique). Callers record a
        non-empty mapping under :data:`DEDUPLICATED_LABELS_KEY` so a rename is
        visible downstream rather than only in a log line.

    Raises:
        ValueError: If every candidate suffix (the running number, then
            ``a``..``z``) for an occurrence collides with an existing label.
    """
    names = list(labels)
    dups = [name for name in dict.fromkeys(names) if names.count(name) > 1]
    if not dups:
        return names, {}
    where = f" in {filepath}" if filepath else ""
    logger.warning(
        "%s channel labels are not unique%s, found duplicates for: %s. "
        "Applying running numbers for duplicates.",
        source,
        where,
        dups,
    )
    renames: dict[str, str] = {}
    for stem in dups:
        positions = [i for i, name in enumerate(names) if name == stem]
        for idx, pos in enumerate(positions):
            for suffix in (str(idx), *"abcdefghijklmnopqrstuvwxyz"):
                candidate = f"{stem}-{suffix}"
                if candidate not in names:
                    break
            else:
                raise ValueError(
                    f"Could not de-duplicate {source} channel label {stem!r}{where}: every "
                    "suffixed candidate (the running number, then a..z) collides with an "
                    "existing label"
                )
            names[pos] = candidate
            renames[candidate] = stem
    return names, renames
