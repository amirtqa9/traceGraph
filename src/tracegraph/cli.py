"""`tracegraph` command line: inspect, validate and maintain a TraceGraph data directory.

    tracegraph --data .tracegraph --project shop ls
    tracegraph --data .tracegraph --project shop show fd_1a2b3c4d5e6f
    tracegraph --data .tracegraph --project shop --workspace . validate [--thread T]
    tracegraph --data .tracegraph --project shop --workspace . brief T
    tracegraph --data .tracegraph --project shop stats
    tracegraph --data .tracegraph --project shop gc [--older-than-days N] [--dry-run]
    tracegraph --data .tracegraph --project shop bust
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from dataclasses import asdict

from tracegraph.cache import cache_stats
from tracegraph.core import TraceGraph
from tracegraph.gc import collect_garbage, expire


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tracegraph", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=".tracegraph", help="data directory (default .tracegraph)")
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", default=None, help="project checkout to validate against")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ls", help="threads with event / finding / thought counts")
    show = sub.add_parser("show", help="one record by ID (ev_, fd_, th_)")
    show.add_argument("id")
    validate = sub.add_parser("validate", help="staleness verdict per finding (needs --workspace)")
    validate.add_argument("--thread")
    brief = sub.add_parser("brief", help="slim resume brief for a thread")
    brief.add_argument("thread")
    brief.add_argument("--reasoning", action="store_true", help="include provider reasoning text")
    sub.add_parser("stats", help="record counts and action-cache hit rate")
    gc = sub.add_parser("gc", help="expire old records, then sweep unreferenced blobs")
    gc.add_argument("--older-than-days", type=float)
    gc.add_argument("--dry-run", action="store_true")
    sub.add_parser("bust", help="drop the action-cache index (records are kept)")
    args = parser.parse_args(argv)

    tg = TraceGraph.open(args.data, args.project, workspace=args.workspace)
    try:
        if args.command == "ls":
            ev = Counter(e.thread_id for e in tg.events.all())
            fd = Counter(f.thread_id for f in tg.ledger.all())
            th = Counter(t.thread_id for t in tg.thoughts.all())
            specs = {s.thread_id: s for s in tg.tasks.all()}
            for thread in sorted(set(ev) | set(fd) | set(th) | set(specs)):
                spec = specs.get(thread)
                parent = f" <- {spec.parent_thread} ({spec.resume_mode})" if spec and spec.parent_thread else ""
                print(f"{thread}{parent}: {ev[thread]} events, {fd[thread]} findings, {th[thread]} thoughts")
        elif args.command == "show":
            prefix = args.id.split("_", 1)[0]
            record = {"ev": tg.events.get, "fd": tg.ledger.get, "th": tg.thoughts.get}.get(prefix, lambda _: None)(args.id)
            if record is None:
                print(f"not found: {args.id}", file=sys.stderr)
                return 1
            _print(asdict(record))
        elif args.command == "validate":
            if tg.workspace is None:
                parser.error("validate needs --workspace")
            for finding_id, verdict in tg.validate(args.thread):
                changed = f"  changed: {', '.join(verdict.changed_locators)}" if verdict.changed else ""
                print(f"{finding_id} {verdict.status}{changed}")
        elif args.command == "brief":
            print(tg.brief(args.thread, provider_reasoning=args.reasoning))
        elif args.command == "stats":
            events = tg.events.all()
            cache = cache_stats(events)
            _print({"events": len(events), "findings": len(tg.ledger.all()), "thoughts": len(tg.thoughts.all()),
                    "blobs": len(tg.blobs.refs()), "cache": {**asdict(cache), "hit_rate": round(cache.hit_rate, 3)}})
        elif args.command == "gc":
            if args.older_than_days is not None:
                _print(asdict(expire(tg.events, tg.ledger, tg.thoughts, dt.timedelta(days=args.older_than_days),
                                     dry_run=args.dry_run)))
            _print(asdict(collect_garbage(tg.blobs, tg.events, tg.thoughts, dry_run=args.dry_run)))
        elif args.command == "bust":
            print(f"removed {tg.cache.bust()} cache keys")
    finally:
        tg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
