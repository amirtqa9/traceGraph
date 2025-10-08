"""Canonical serialization and hashing.

Everything TraceGraph hashes goes through `canonical_json`: sorted keys, no
insignificant whitespace, UTF-8, and no wall-clock or random values (callers must
not pass them in).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def short_id(prefix: str, obj: Any, length: int = 12) -> str:
    """Stable, human-citable ID such as `ev_3f9a1c0b2d4e`, derived from `obj`."""
    return f"{prefix}_{hashlib.sha256(canonical_json(obj)).hexdigest()[:length]}"
