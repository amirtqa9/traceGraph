"""Tool-layer events: what actually ran.

The Recorder sits in ToolNode's `wrap_tool_call` hook, so every call is recorded
by the harness itself rather than reported by the model. Each event gets a short
citable ID (`ev_...`) that is appended to the tool output the model sees; findings
cite these IDs as their sources.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.store.base import BaseStore

from tracegraph.canonical import short_id
from tracegraph.cas import BlobStore

ToolKind = Literal["read", "write", "external"]

EVENT_FOOTER = "[event: {event_id}]"
EVENT_ID_RE = re.compile(r"\[event: (ev_[0-9a-f]{12})\]")


@dataclass(frozen=True)
class ActionEvent:
    event_id: str
    project_id: str
    thread_id: str
    checkpoint_id: str | None
    step: int | None
    tool_call_id: str
    tool: str
    kind: ToolKind
    args: dict[str, Any]
    output_ref: str
    status: str
    # Populated by resource adapters from M1 on; empty in M0.
    read_set: dict[str, str] = field(default_factory=dict)
    write_set: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLog:
    """Events stored in LangGraph's Store under ("tracegraph", project, "events")."""

    def __init__(self, store: BaseStore, project_id: str) -> None:
        self.store = store
        self.project_id = project_id
        self.namespace = ("tracegraph", project_id, "events")

    def put(self, event: ActionEvent) -> None:
        self.store.put(self.namespace, event.event_id, event.to_dict(), index=False)

    def get(self, event_id: str) -> ActionEvent | None:
        item = self.store.get(self.namespace, event_id)
        return ActionEvent(**item.value) if item else None

    def for_thread(self, thread_id: str, limit: int = 10_000) -> list[ActionEvent]:
        items = self.store.search(self.namespace, filter={"thread_id": thread_id}, limit=limit)
        items.sort(key=lambda it: it.created_at)
        return [ActionEvent(**it.value) for it in items]


def message_text(content: Any) -> str:
    """Flatten LangChain message content (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class Recorder:
    """ToolNode `wrap_tool_call` hook that turns every tool call into an ActionEvent.

    `tool_kinds` declares each tool as read/write/external; unknown tools default to
    `external` (never cacheable). Tools in `untracked` (TraceGraph's own bookkeeping
    tools such as `record_finding`) pass through unrecorded.
    """

    def __init__(
        self,
        events: EventLog,
        blobs: BlobStore,
        tool_kinds: dict[str, ToolKind],
        untracked: frozenset[str] = frozenset(),
    ) -> None:
        self.events = events
        self.blobs = blobs
        self.tool_kinds = tool_kinds
        self.untracked = untracked

    def wrap_tool_call(self, request, execute):
        name = request.tool_call["name"]
        if name in self.untracked:
            return execute(request)
        result = execute(request)
        if not isinstance(result, ToolMessage):
            return result

        config = request.runtime.config if request.runtime else {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        tool_call_id = request.tool_call["id"]
        output = message_text(result.content)
        event = ActionEvent(
            event_id=short_id("ev", [self.events.project_id, thread_id, tool_call_id]),
            project_id=self.events.project_id,
            thread_id=thread_id,
            # Inside a node, the root-graph checkpoint this step started from.
            checkpoint_id=configurable.get("checkpoint_map", {}).get("") or configurable.get("checkpoint_id"),
            step=config.get("metadata", {}).get("langgraph_step"),
            tool_call_id=tool_call_id,
            tool=name,
            kind=self.tool_kinds.get(name, "external"),
            args=dict(request.tool_call.get("args", {})),
            output_ref=self.blobs.put(output.encode("utf-8")),
            status=result.status,
        )
        self.events.put(event)
        footer = EVENT_FOOTER.format(event_id=event.event_id)
        return result.model_copy(update={"content": f"{output}\n\n{footer}"})
