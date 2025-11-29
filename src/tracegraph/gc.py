"""Garbage collection.

- `expire(older_than)`: drop events, findings and thoughts older than a cutoff, except
  findings marked as pinned by being cited by a retained finding (provenance stays whole).
- `collect_garbage()`: mark-and-sweep over the blob store. Live = every blob referenced
  by a retained event (output) or thought (raw reasoning). Everything else is removed.
Run expire first, then collect. Both support dry runs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from tracegraph.cas import BlobStore
from tracegraph.events import EventLog
from tracegraph.findings import FindingsLedger
from tracegraph.thoughts import ThoughtLog


@dataclass(frozen=True)
class GCReport:
    live: int
    removed: int
    bytes_freed: int
    dry_run: bool


def collect_garbage(blobs: BlobStore, events: EventLog, thoughts: ThoughtLog, *, dry_run: bool = False) -> GCReport:
    live = {e.output_ref for e in events.all()} | {t.opaque_ref for t in thoughts.all() if t.opaque_ref}
    dead = [ref for ref in blobs.refs() if ref not in live]
    freed = 0
    for ref in dead:
        if dry_run:
            freed += blobs._path(ref).stat().st_size
        else:
            freed += blobs.delete(ref)
    return GCReport(live=len(live), removed=len(dead), bytes_freed=freed, dry_run=dry_run)


@dataclass(frozen=True)
class ExpireReport:
    events: int
    findings: int
    thoughts: int
    dry_run: bool


def expire(
    events: EventLog, ledger: FindingsLedger, thoughts: ThoughtLog, older_than: dt.timedelta, *, dry_run: bool = False
) -> ExpireReport:
    cutoff = dt.datetime.now(dt.timezone.utc) - older_than
    store = events.store

    def old(namespace):
        return [it for it in store.search(namespace, limit=1_000_000) if _aware(it.created_at) < cutoff]

    old_findings = old(ledger.namespace)
    old_ids = {it.key for it in old_findings}
    # Keep any old finding still cited by a retained one, so provenance chains stay intact.
    cited = {s for f in ledger.all() if f.finding_id not in old_ids for s in f.sources}
    cited_events = {s for f in ledger.all() if f.finding_id not in old_ids for s in f.sources if s.startswith("ev_")}
    drop_findings = [it for it in old_findings if it.key not in cited]
    drop_events = [it for it in old(events.namespace) if it.key not in cited_events]
    drop_thoughts = old(thoughts.namespace)
    if not dry_run:
        for ns, items in ((ledger.namespace, drop_findings), (events.namespace, drop_events),
                          (thoughts.namespace, drop_thoughts)):
            for it in items:
                store.delete(ns, it.key)
    return ExpireReport(len(drop_events), len(drop_findings), len(drop_thoughts), dry_run)


def _aware(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
