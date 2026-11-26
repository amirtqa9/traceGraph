"""Garbage collection.

- `expire(older_than)`: drop events, findings and thoughts older than a cutoff, except
  - anything reachable by citation from a retained finding or thought, followed
    transitively (finding -> finding -> event), so provenance chains stay whole;
  - write and external events, by default: they are what rewind and resume use to tell
    whether a checkpoint is behind side effects, and checkpoints are not expired here.
- `collect_garbage()`: mark-and-sweep over the blob store. The blob store is shared by
  every project in a data directory, so the mark covers every project's events and
  thoughts found in the store, not just the caller's. Blobs younger than `grace` are
  never removed: a writer puts a blob a moment before the event that references it.
Run expire first, then collect. Both support dry runs.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

from langgraph.store.base import BaseStore

from tracegraph.cas import BlobStore
from tracegraph.events import EventLog
from tracegraph.findings import FindingsLedger
from tracegraph.thoughts import ThoughtLog

ROOT = "tracegraph"
DEFAULT_GRACE = dt.timedelta(minutes=15)
_PAGE = 1000


@dataclass(frozen=True)
class GCReport:
    live: int
    removed: int
    bytes_freed: int
    dry_run: bool
    projects: int = 0  # projects whose records were marked
    skipped_recent: int = 0  # unreferenced blobs kept because they are younger than `grace`


def _search_all(store: BaseStore, namespace: tuple[str, ...]):
    """Every item under `namespace`, paged, so the result is complete however large."""
    offset = 0
    while True:
        page = store.search(namespace, limit=_PAGE, offset=offset)
        yield from page
        if len(page) < _PAGE:
            return
        offset += len(page)


def _namespaces(store: BaseStore, kind: str) -> list[tuple[str, ...]]:
    """("tracegraph", <project>, kind) for every project in the store."""
    found: list[tuple[str, ...]] = []
    offset = 0
    while True:
        page = store.list_namespaces(prefix=(ROOT,), max_depth=3, limit=_PAGE, offset=offset)
        found += [ns for ns in page if len(ns) == 3 and ns[2] == kind]
        if len(page) < _PAGE:
            return found
        offset += len(page)


def live_blob_refs(store: BaseStore) -> tuple[set[str], int]:
    """Blob refs referenced by any project's events or thoughts, and how many projects were seen."""
    live: set[str] = set()
    projects: set[str] = set()
    for ns in _namespaces(store, "events"):
        projects.add(ns[1])
        live |= {it.value["output_ref"] for it in _search_all(store, ns) if it.value.get("output_ref")}
    for ns in _namespaces(store, "thoughts"):
        projects.add(ns[1])
        live |= {it.value["opaque_ref"] for it in _search_all(store, ns) if it.value.get("opaque_ref")}
    return live, len(projects)


def collect_garbage(
    blobs: BlobStore, events: EventLog, thoughts: ThoughtLog, *, dry_run: bool = False,
    grace: dt.timedelta = DEFAULT_GRACE,
) -> GCReport:
    """Remove blobs no project references. `events` and `thoughts` supply the store to scan;
    the mark is not limited to their project. Pass `grace=timedelta(0)` only when nothing
    else is writing to this data directory."""
    live, projects = live_blob_refs(events.store)
    # Records for the caller's project are in this same store, but guard against a caller
    # whose logs use a different store than the one listed.
    live |= {e.output_ref for e in events.all()} | {t.opaque_ref for t in thoughts.all() if t.opaque_ref}
    cutoff = time.time() - grace.total_seconds()
    removed = skipped = 0
    freed = 0
    for ref in blobs.refs():
        if ref in live:
            continue
        path = blobs._path(ref)
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        if st.st_mtime > cutoff:
            skipped += 1
            continue
        removed += 1
        freed += st.st_size if dry_run else blobs.delete(ref)
    return GCReport(live=len(live), removed=removed, bytes_freed=freed, dry_run=dry_run,
                    projects=projects, skipped_recent=skipped)


@dataclass(frozen=True)
class ExpireReport:
    events: int
    findings: int
    thoughts: int
    dry_run: bool


def expire(
    events: EventLog, ledger: FindingsLedger, thoughts: ThoughtLog, older_than: dt.timedelta, *,
    dry_run: bool = False, keep_side_effects: bool = True,
) -> ExpireReport:
    cutoff = dt.datetime.now(dt.timezone.utc) - older_than
    store = events.store

    def is_old(it) -> bool:
        return _aware(it.created_at) < cutoff

    finding_items = list(_search_all(store, ledger.namespace))
    event_items = list(_search_all(store, events.namespace))
    thought_items = list(_search_all(store, thoughts.namespace))
    sources = {it.key: list(it.value.get("sources", [])) for it in finding_items}

    # Mark: walk every citation edge from retained roots (young findings, young thoughts).
    reachable: set[str] = set()
    stack = [it.key for it in finding_items if not is_old(it)]
    stack += [s for it in thought_items if not is_old(it) for s in it.value.get("supports", [])]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack += sources.get(node, [])

    drop_findings = [it for it in finding_items if is_old(it) and it.key not in reachable]
    drop_events = [
        it for it in event_items
        if is_old(it) and it.key not in reachable
        and not (keep_side_effects and it.value.get("kind") in ("write", "external"))
    ]
    drop_thoughts = [it for it in thought_items if is_old(it)]
    if not dry_run:
        for ns, items in ((ledger.namespace, drop_findings), (events.namespace, drop_events),
                          (thoughts.namespace, drop_thoughts)):
            for it in items:
                store.delete(ns, it.key)
    return ExpireReport(len(drop_events), len(drop_findings), len(drop_thoughts), dry_run)


def _aware(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
