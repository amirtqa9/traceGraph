"""Action cache: "already done" for read tools, build-system style.

- Lookup key: hash(tool, canonical args). The index maps it to earlier events.
- A hit is the newest earlier event for that key that succeeded, has a non-empty read
  set, and whose read set still fingerprints the same: the inputs are provably unchanged,
  so the output would be too.
- Only `read` tools are ever cached; `write` and `external` tools always run.
- Modes: "off"; "advisory" (run anyway, record `would_hit` so hit correctness and rate
  can be measured); "enforce" (serve the earlier output, record `cached_from`).
- Scoped per project: a cache never matches across unrelated projects.
Fuzzy reuse (similar plans, similar queries) is deliberately not a hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langgraph.store.base import BaseStore

from tracegraph.canonical import short_id
from tracegraph.events import ActionEvent, EventLog
from tracegraph.resources import ResourceRegistry

CacheMode = Literal["off", "advisory", "enforce"]


def action_key(tool: str, args: dict) -> str:
    return short_id("ak", [tool, args], length=16)


class ActionCache:
    def __init__(
        self, store: BaseStore, project_id: str, events: EventLog, registry: ResourceRegistry, mode: CacheMode = "advisory"
    ) -> None:
        if mode not in ("off", "advisory", "enforce"):
            raise ValueError(mode)
        self.store = store
        self.events = events
        self.registry = registry
        self.mode = mode
        self.namespace = ("tracegraph", project_id, "action-cache")

    def record(self, event: ActionEvent, raw_args: dict) -> None:
        if self.mode == "off" or event.kind != "read" or event.status != "success" or not event.read_set:
            return
        key = action_key(event.tool, raw_args)
        item = self.store.get(self.namespace, key)
        ids = list(item.value["event_ids"]) if item else []
        if event.event_id not in ids:
            ids.append(event.event_id)
        self.store.put(self.namespace, key, {"event_ids": ids, "tool": event.tool}, index=False)

    def lookup(self, tool: str, args: dict) -> ActionEvent | None:
        if self.mode == "off":
            return None
        item = self.store.get(self.namespace, action_key(tool, args))
        if item is None:
            return None
        for event_id in reversed(item.value["event_ids"]):
            event = self.events.get(event_id)
            if event is None or not event.read_set:
                continue
            if not self.registry.changed(event.read_set):
                return event
        return None

    def bust(self) -> int:
        """Drop the whole index (events are kept). Returns how many keys were removed."""
        items = self.store.search(self.namespace, limit=1_000_000)
        for item in items:
            self.store.delete(self.namespace, item.key)
        return len(items)


@dataclass(frozen=True)
class CacheStats:
    read_calls: int
    served: int  # enforce-mode hits
    would_hit: int  # advisory-mode hits

    @property
    def hit_rate(self) -> float:
        return (self.served + self.would_hit) / self.read_calls if self.read_calls else 0.0


def cache_stats(events: list[ActionEvent]) -> CacheStats:
    reads = [e for e in events if e.kind == "read"]
    return CacheStats(
        read_calls=len(reads),
        served=sum(e.cached_from is not None for e in reads),
        would_hit=sum(e.would_hit is not None for e in reads),
    )
