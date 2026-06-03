"""Merkle-tree tamper evidence for audit records.

Periodic Merkle roots (certificate-transparency style) rather than an inline
hash chain — per the architecture's audit design decision (doc §4): aviation /
automotive / industrial-robot audit systems use periodic roots, not chains.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_record(record: dict[str, Any]) -> str:
    """SHA-256 of a record's canonical JSON form (key order independent)."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    return hashlib.sha256((left + right).encode("utf-8")).hexdigest()


def merkle_root(leaf_hashes: list[str]) -> str:
    """Compute the Merkle root of a list of leaf hashes.

    Odd nodes are promoted (duplicated) to the next level, the common
    convention for unbalanced trees.

    Raises:
        ValueError: on an empty leaf list (an empty checkpoint is meaningless).
    """
    if not leaf_hashes:
        raise ValueError("cannot compute a Merkle root over zero records")

    level = list(leaf_hashes)
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                next_level.append(_hash_pair(level[i], level[i + 1]))
            else:
                next_level.append(level[i])  # promote odd node
        level = next_level
    return level[0]


def verify_records(records: list[dict[str, Any]], expected_root: str) -> bool:
    """Recompute the Merkle root of `records` and compare with `expected_root`."""
    try:
        return merkle_root([hash_record(r) for r in records]) == expected_root
    except ValueError:
        return False
