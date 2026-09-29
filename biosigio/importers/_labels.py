"""Channel-label helpers shared by importers whose formats allow repeated labels."""

import logging

logger = logging.getLogger(__name__)


def unique_channel_labels(labels: list[str], *, source: str = "EDF") -> list[str]:
    """Rename repeated channel labels the way MNE does, so no channel is dropped.

    EDF does not require unique labels (CHB-MIT declares ``T8-P8`` twice and
    uses ``-`` as a placeholder for several unused inputs), but a Recording is
    keyed by label. The same rule serves any other source that can repeat a
    label (a WFDB ``sig_name``, a Zarr store published before this rule
    existed); ``source`` only names the format in the warning and error text.
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

    Raises:
        ValueError: If every candidate suffix (the running number, then
            ``a``..``z``) for an occurrence collides with an existing label.
    """
    names = list(labels)
    dups = [name for name in dict.fromkeys(names) if names.count(name) > 1]
    if not dups:
        return names
    logger.warning(
        "%s channel labels are not unique, found duplicates for: %s. "
        "Applying running numbers for duplicates.",
        source,
        dups,
    )
    for stem in dups:
        positions = [i for i, name in enumerate(names) if name == stem]
        for idx, pos in enumerate(positions):
            for suffix in (str(idx), *"abcdefghijklmnopqrstuvwxyz"):
                candidate = f"{stem}-{suffix}"
                if candidate not in names:
                    break
            else:
                raise ValueError(
                    f"Could not de-duplicate {source} channel label {stem!r}: every "
                    "suffixed candidate collides with an existing label"
                )
            names[pos] = candidate
    return names
