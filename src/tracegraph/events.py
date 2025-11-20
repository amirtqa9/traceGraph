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
from tracegraph.redact import redact_text, redact_value
from tracegraph.resources import ResourceRegistry, ToolResources
from tracegraph.schema import SCHEMA_VERSION, check_schema

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
    args: dict[str, Any]  # redacted
    output_ref: str  # blob of the redacted output
    status: str
    read_set: dict[str, str] = field(default_factory=dict)  # resource key -> fingerprint before the call
    write_set: dict[str, str] = field(default_factory=dict)  # resource key -> fingerprint after the call
    redactions: list[str] = field(default_factory=list)  # kinds of secrets removed at ingestion
    cached_from: str | None = None  # served from this earlier event instead of running (enforcing cache)
    would_hit: str | None = None  # an earlier valid event the cache could have served (advisory cache)
    pre_snapshot: str | None = None  # workspace snapshot taken before a write
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionEvent":
        check_schema(value, "event")
        return cls(**value)


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
        return ActionEvent.from_dict(item.value) if item else None

    def for_thread(self, thread_id: str, limit: int = 100_000) -> list[ActionEvent]:
        return self.for_threads([thread_id], limit)

    def for_threads(self, thread_ids: list[str], limit: int = 100_000) -> list[ActionEvent]:
        items = []
        for thread_id in thread_ids:
            items += self.store.search(self.namespace, filter={"thread_id": thread_id}, limit=limit)
        items.sort(key=lambda it: it.created_at)
        return [ActionEvent.from_dict(it.value) for it in items]

    def all(self, limit: int = 1_000_000) -> list[ActionEvent]:
        items = self.store.search(self.namespace, limit=limit)
        items.sort(key=lambda it: it.created_at)
        return [ActionEvent.from_dict(it.value) for it in items]


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

    - `tool_kinds` declares each tool read/write/external; unknown tools are `external`.
    - `resources` + `registry` give each call a read set (fingerprinted before the call)
      and a write set (after it). Tools without a declaration get none.
    - Secrets are redacted from args and output before anything is stored, including the
      output the model sees (it is checkpointed too).
    - `cache` (M4): read tools whose earlier output is still valid are served from the
      cache when its mode is "enforce", or flagged `would_hit` when "advisory".
    - `snapshotter` (M5): a workspace snapshot is taken before every write tool.
    - Tools in `untracked` (TraceGraph's own bookkeeping tools) pass through unrecorded.
    """

    def __init__(
        self,
        events: EventLog,
        blobs: BlobStore,
        tool_kinds: dict[str, ToolKind],
        untracked: frozenset[str] = frozenset(),
        *,
        resources: dict[str, ToolResources] | None = None,
        registry: ResourceRegistry | None = None,
        redact: bool = True,
        cache=None,
        snapshotter=None,
    ) -> None:
        self.events = events
        self.blobs = blobs
        self.tool_kinds = tool_kinds
        self.untracked = untracked
        self.resources = resources or {}
        self.registry = registry
        self.redact = redact
        self.cache = cache
        self.snapshotter = snapshotter

    def _resource_keys(self, name: str, args: dict) -> tuple[list[str], list[str]]:
        declare = self.resources.get(name)
        if declare is None or self.registry is None:
            return [], []
        try:
            return declare(args)
        except (KeyError, ValueError):
            return [], []  # malformed args; the tool call itself will report the error

    def wrap_tool_call(self, request, execute):
        name = request.tool_call["name"]
        if name in self.untracked:
            return execute(request)

        config = request.runtime.config if request.runtime else {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        tool_call_id = request.tool_call["id"]
        kind = self.tool_kinds.get(name, "external")
        args = dict(request.tool_call.get("args", {}))
        event_id = short_id("ev", [self.events.project_id, thread_id, tool_call_id])

        reads, writes = self._resource_keys(name, args)
        read_set = self.registry.snapshot(reads) if reads else {}

        hit = self.cache.lookup(name, args) if self.cache is not None and kind == "read" else None
        served = hit if hit is not None and self.cache.mode == "enforce" else None
        pre_snapshot = (
            self.snapshotter.snapshot(event_id) if self.snapshotter is not None and kind == "write" else None
        )

        if served is not None:
            output = self.blobs.get(served.output_ref).decode("utf-8")
            status = "success"
            read_set = served.read_set
            result = ToolMessage(content=output, tool_call_id=tool_call_id, name=name)
        else:
            result = execute(request)
            if not isinstance(result, ToolMessage):
                return result
            output = message_text(result.content)
            status = result.status

        kinds: set[str] = set()
        if self.redact:
            args, arg_kinds = redact_value(args)
            red = redact_text(output)
            output = red.text
            kinds = set(arg_kinds) | set(red.kinds)

        event = ActionEvent(
            event_id=event_id,
            project_id=self.events.project_id,
            thread_id=thread_id,
            # Inside a node, the root-graph checkpoint this step started from.
            checkpoint_id=configurable.get("checkpoint_map", {}).get("") or configurable.get("checkpoint_id"),
            step=config.get("metadata", {}).get("langgraph_step"),
            tool_call_id=tool_call_id,
            tool=name,
            kind=kind,
            args=args,
            output_ref=self.blobs.put(output.encode("utf-8")),
            status=status,
            read_set=read_set,
            write_set=self.registry.snapshot(writes) if writes else {},
            redactions=sorted(kinds),
            cached_from=served.event_id if served is not None else None,
            would_hit=hit.event_id if hit is not None and served is None else None,
            pre_snapshot=pre_snapshot,
        )
        self.events.put(event)
        if self.cache is not None and served is None:
            self.cache.record(event, dict(request.tool_call.get("args", {})))
        footer = EVENT_FOOTER.format(event_id=event.event_id)
        return result.model_copy(update={"content": f"{output}\n\n{footer}"})
