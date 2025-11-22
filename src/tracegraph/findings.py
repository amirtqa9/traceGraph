"""Model-authored findings with sources checked against the event log.

A finding is a claim ("files 1-10 checked for the null deref, none found"). It
is trusted only as far as what it cites: tool events (`ev_...`) and earlier findings
(`fd_...`). Cited IDs are checked against the logs, unknown ones are kept as
`rejected_sources`, and a finding with no verified source is "unsourced" and left out
of slim resume.

Its `read_set` is the union of its sources' read sets, transitively: a finding built on
another finding goes stale when anything under the first one changes.

Findings never change. A correction is a new finding that `supersedes` the
old one; nothing is rewritten in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.types import Command

from tracegraph.canonical import short_id
from tracegraph.events import EventLog
from tracegraph.schema import SCHEMA_VERSION, check_schema

FindingKind = Literal["fact", "ruled_out", "hypothesis", "todo", "decision"]


@dataclass(frozen=True)
class Finding:
    finding_id: str
    project_id: str
    thread_id: str
    text: str
    kind: FindingKind
    method: str
    sources: tuple[str, ...]  # verified event and finding IDs
    rejected_sources: tuple[str, ...]  # cited IDs with no matching event
    author: str = "agent"
    supersedes: str | None = None
    promoted_from: str | None = None  # thought ID this finding confirms, if any
    read_set: dict[str, str] = field(default_factory=dict)  # transitive inputs
    schema_version: int = SCHEMA_VERSION

    @property
    def sourced(self) -> bool:
        return bool(self.sources)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["sources"] = list(self.sources)
        value["rejected_sources"] = list(self.rejected_sources)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Finding":
        value = dict(value)
        check_schema(value, "finding")
        value["sources"] = tuple(value["sources"])
        value["rejected_sources"] = tuple(value["rejected_sources"])
        return cls(**value)


class FindingsLedger:
    """Findings stored in LangGraph's Store under ("tracegraph", project, "findings")."""

    def __init__(self, store: BaseStore, project_id: str, events: EventLog) -> None:
        self.store = store
        self.project_id = project_id
        self.events = events
        self.namespace = ("tracegraph", project_id, "findings")

    def add(
        self,
        *,
        thread_id: str,
        text: str,
        kind: FindingKind,
        method: str,
        cited: list[str],
        author: str = "agent",
        supersedes: str | None = None,
        promoted_from: str | None = None,
    ) -> Finding:
        cited = list(dict.fromkeys(cited))  # dedupe, keep order
        read_set: dict[str, str] = {}
        verified_list = []
        for c in cited:
            source = self.events.get(c) if c.startswith("ev_") else self.get(c) if c.startswith("fd_") else None
            if source is not None:
                verified_list.append(c)
                read_set.update(source.read_set)
        verified = tuple(verified_list)
        rejected = tuple(c for c in cited if c not in verified)
        identity = [self.project_id, thread_id, text, kind, method, sorted(verified), supersedes, promoted_from]
        finding = Finding(
            finding_id=short_id("fd", identity),
            project_id=self.project_id,
            thread_id=thread_id,
            text=text,
            kind=kind,
            method=method,
            sources=verified,
            rejected_sources=rejected,
            author=author,
            supersedes=supersedes,
            promoted_from=promoted_from,
            read_set=read_set,
        )
        self.store.put(self.namespace, finding.finding_id, finding.to_dict(), index=False)
        return finding

    def get(self, finding_id: str) -> Finding | None:
        item = self.store.get(self.namespace, finding_id)
        return Finding.from_dict(item.value) if item else None

    def for_threads(self, thread_ids: list[str], limit: int = 10_000) -> list[Finding]:
        """Current findings across `thread_ids`, oldest first, superseded ones removed."""
        items = []
        for thread_id in thread_ids:
            items += self.store.search(self.namespace, filter={"thread_id": thread_id}, limit=limit)
        items.sort(key=lambda it: it.created_at)
        findings = [Finding.from_dict(it.value) for it in items]
        superseded = {f.supersedes for f in findings if f.supersedes}
        return [f for f in findings if f.finding_id not in superseded]


def make_record_finding_tool(ledger: FindingsLedger) -> BaseTool:
    @tool
    def record_finding(
        text: str,
        kind: FindingKind,
        method: str,
        sources: list[str],
        runtime: ToolRuntime,
        supersedes: str | None = None,
        from_thought: str | None = None,
    ) -> Command:
        """Record a conclusion so a later session can continue without redoing work.

        Args:
            text: The conclusion, self-contained (e.g. "pagination.py: paginate() is correct for page 1").
            kind: fact | ruled_out | hypothesis | todo | decision.
            method: How you established it (which tool, what you looked for).
            sources: The `ev_...` event IDs (shown under each tool result) and/or earlier `fd_...`
                finding IDs that support it.
            supersedes: ID of an earlier finding this one corrects, if any.
            from_thought: ID of the working-memory thought (th_...) whose hypothesis this confirms, if any.
        """
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "")
        finding = ledger.add(
            thread_id=thread_id,
            text=text,
            kind=kind,
            method=method,
            cited=sources,
            supersedes=supersedes,
            promoted_from=from_thought,
        )
        reply = f"Recorded {finding.finding_id}."
        if finding.rejected_sources:
            reply += f" Unknown event IDs ignored: {', '.join(finding.rejected_sources)}."
        if not finding.sourced:
            reply += " WARNING: no valid sources, so this finding will not be trusted on resume."
        return Command(
            update={
                "tg_findings": [finding.finding_id],
                "messages": [ToolMessage(reply, tool_call_id=runtime.tool_call_id)],
            }
        )

    return record_finding
