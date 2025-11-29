"""`TraceGraph`: one object that wires every component for a project.

    tg = TraceGraph.open(".tracegraph", project_id="shop", workspace="path/to/repo")
    graph = builder.compile(checkpointer=tg.checkpointer(), store=tg.store)
    ToolNode(tools + tg.tools(), wrap_tool_call=tg.recorder(tool_kinds, resources).wrap_tool_call,
             handle_tool_errors=True)
    ...
    print(tg.brief(thread_id))           # slim, validated resume brief
    tg.validate(thread_id)               # verdict per finding

`open()` persists to SQLite under `data_dir` (LangGraph's SqliteStore and SqliteSaver)
and blobs to `data_dir/blobs`. Pass any LangGraph `BaseStore` to the constructor
instead for other backends (Postgres, Redis, ...).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from tracegraph.cache import ActionCache, CacheMode
from tracegraph.cas import BlobStore
from tracegraph.events import EventLog, Recorder, ToolKind
from tracegraph.findings import FindingsLedger, make_record_finding_tool
from tracegraph.policy import DecisionLog
from tracegraph.resources import FilesystemAdapter, GitAdapter, ResourceRegistry, ToolResources
from tracegraph.resume import make_recall_tool, slim_brief
from tracegraph.tasks import TaskRegistry
from tracegraph.thoughts import ThoughtLog, make_record_thought_tool
from tracegraph.validator import Validator, Verdict

OWN_TOOLS = frozenset({"record_finding", "record_thought", "recall_event"})


class TraceGraph:
    def __init__(
        self,
        store: BaseStore,
        data_dir: str | os.PathLike[str],
        project_id: str,
        *,
        workspace: str | os.PathLike[str] | None = None,
        cache_mode: CacheMode = "advisory",
    ) -> None:
        self.store = store
        self.data_dir = Path(data_dir)
        self.project_id = project_id
        self.blobs = BlobStore(self.data_dir / "blobs")
        self.events = EventLog(store, project_id)
        self.ledger = FindingsLedger(store, project_id, self.events)
        self.tasks = TaskRegistry(store, project_id)
        self.thoughts = ThoughtLog(store, project_id, self.blobs, self.events)
        self.decisions = DecisionLog(store, project_id)
        self.workspace = Path(workspace).resolve() if workspace else None
        adapters = []
        if self.workspace:
            self.fs = FilesystemAdapter(self.workspace)
            adapters.append(self.fs)
            if (self.workspace / ".git").exists():
                adapters.append(GitAdapter(self.workspace))
        self.registry = ResourceRegistry(*adapters)
        self.validator = Validator(self.registry)
        self.cache = ActionCache(store, project_id, self.events, self.registry, cache_mode)
        self._conns: list[sqlite3.Connection] = []

    @classmethod
    def open(cls, data_dir: str | os.PathLike[str], project_id: str, **kwargs) -> "TraceGraph":
        from langgraph.store.sqlite import SqliteStore

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(data_dir / "store.sqlite", check_same_thread=False, isolation_level=None)
        store = SqliteStore(conn)
        store.setup()
        tg = cls(store, data_dir, project_id, **kwargs)
        tg._conns.append(conn)
        return tg

    def checkpointer(self):
        """A SqliteSaver next to the store (for graphs compiled against this TraceGraph)."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(self.data_dir / "checkpoints.sqlite", check_same_thread=False)
        self._conns.append(conn)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver

    def close(self) -> None:
        for conn in self._conns:
            conn.close()
        self._conns.clear()

    def recorder(
        self, tool_kinds: dict[str, ToolKind], resources: dict[str, ToolResources] | None = None, *, snapshotter=None
    ) -> Recorder:
        return Recorder(
            self.events, self.blobs, tool_kinds, OWN_TOOLS, resources=resources, registry=self.registry,
            cache=self.cache if self.cache.mode != "off" else None, snapshotter=snapshotter,
        )

    def tools(self) -> list[BaseTool]:
        """record_finding, record_thought, recall_event: add these to the agent's tools.

        The graph state must carry `tg_findings` and `tg_thoughts` reducer channels
        (`Annotated[list[str], operator.add]`).
        """
        return [
            make_record_finding_tool(self.ledger),
            make_record_thought_tool(self.thoughts),
            make_recall_tool(self.events, self.blobs),
        ]

    def brief(self, thread_id: str, *, working_memory: bool = True, provider_reasoning: bool = False) -> str:
        return slim_brief(
            self.tasks, self.ledger, thread_id, self.thoughts, working_memory=working_memory,
            provider_reasoning=provider_reasoning, validator=self.validator if self.workspace else None,
            events=self.events,
        )

    def validate(self, thread_id: str | None = None) -> list[tuple[str, Verdict]]:
        findings = self.ledger.for_threads(self.tasks.lineage(thread_id)) if thread_id else self.ledger.all()
        return [(f.finding_id, self.validator.finding(f)) for f in findings]
