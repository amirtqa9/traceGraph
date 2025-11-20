"""Record schema versioning.

Every stored record (event, finding, thought) carries `schema_version`. Readers accept
versions up to SCHEMA_VERSION; records missing the field are version 1. A newer record
than this code understands is refused rather than half-read: upgrade the package, or run
a migration (`MIGRATIONS[n]` upgrades a version-n record dict to n+1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    # v1 -> v2: added redactions/cached_from/would_hit/pre_snapshot (events) and
    # read_set (findings). Defaults cover them, so the upgrade only stamps the version.
    1: lambda value: {**value, "schema_version": 2},
}


class SchemaError(RuntimeError):
    pass


def check_schema(value: dict[str, Any], kind: str) -> None:
    """Upgrade `value` in place to SCHEMA_VERSION, or refuse it if it is newer."""
    version = value.get("schema_version", 1)
    if version > SCHEMA_VERSION:
        raise SchemaError(f"{kind} record has schema_version {version}; this tracegraph understands {SCHEMA_VERSION}")
    while version < SCHEMA_VERSION:
        value.update(MIGRATIONS[version](value))
        version = value["schema_version"]
