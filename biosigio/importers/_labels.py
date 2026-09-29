"""Channel-label helpers shared by importers whose formats allow repeated labels."""

import logging


def unique_channel_labels(labels: list[str]) -> list[str]:
    """Rename repeated EDF labels the way MNE does, so no channel is dropped.

    EDF does not require unique labels (CHB-MIT declares ``T8-P8`` twice and
    uses ``-`` as a placeholder for several unused inputs), but a Recording is
    keyed by label. Every occurrence of a repeated label gets a running suffix
    ``-0``, ``-1``, ...; a suffix that would collide with an existing label
    falls through to ``-a``, ``-b``, ... This mirrors MNE's
    ``_unique_channel_names`` exactly, so the names match a ``channels.tsv``
    written by MNE-BIDS. Unique labels are returned unchanged.
    """
    names = list(labels)
    dups = [name for name in dict.fromkeys(names) if names.count(name) > 1]
    if not dups:
        return names
    logging.warning(
        "EDF channel labels are not unique, found duplicates for: %s. "
        "Applying running numbers for duplicates.",
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
                    f"Could not de-duplicate EDF channel label {stem!r}: every "
                    "suffixed candidate collides with an existing label"
                )
            names[pos] = candidate
    return names
