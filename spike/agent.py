"""M0 spike agent: a minimal ReAct loop on LangGraph with TraceGraph attached.

Built from StateGraph + ToolNode directly (prebuilt `create_react_agent` is
deprecated in LangGraph 1.x). The Recorder hooks ToolNode's `wrap_tool_call`.
"""

from __future__ import annotations

import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from tracegraph import BlobStore, EventLog, FindingsLedger, Recorder, TaskRegistry
from tracegraph.findings import make_record_finding_tool
from tracegraph.resources import FilesystemAdapter, ResourceRegistry, filesystem_tool_resources
from tracegraph.resume import make_recall_tool
from tracegraph.thoughts import ThoughtLog, make_record_thought_tool

SYSTEM_PROMPT = """You are a careful debugging agent working in a read-only repository.

Each tool result ends with an event ID like [event: ev_1234abcd5678]. Whenever you
establish something worth keeping (a file you've ruled out, a suspect, a confirmed bug),
call record_finding and cite the event IDs that support it. Whenever your plan or your
suspicions change, call record_thought with your current hypotheses (with confidence and
supporting event IDs), plan, open questions and next step. Your session may be cut off
at any time, and a later session will only see your findings and working memory.

When you are done, reply with a final line of the form:
ANSWER: <path>::<function>"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Memory state and thought state, both cumulative and checkpointed at every step,
    # so any checkpoint indexes the findings and thoughts that existed at that point.
    tg_findings: Annotated[list[str], operator.add]
    tg_thoughts: Annotated[list[str], operator.add]


def workspace_tools(root: Path) -> list[BaseTool]:
    root = root.resolve()

    def resolve(path: str) -> Path:
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return target

    @tool
    def list_files(path: str = ".") -> str:
        """List files under a directory of the repository (recursive)."""
        base = resolve(path)
        files = sorted(
            str(p.relative_to(root)) for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
        return "\n".join(files) or "(empty)"

    @tool
    def read_file(path: str) -> str:
        """Read a file from the repository."""
        return resolve(path).read_text()

    @tool
    def grep(pattern: str, path: str = ".") -> str:
        """Search files under `path` for a regular expression. Returns path:line:text."""
        regex = re.compile(pattern)
        hits = []
        for p in sorted(resolve(path).rglob("*.py")):
            for n, line in enumerate(p.read_text().splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{p.relative_to(root)}:{n}:{line}")
        return "\n".join(hits) or "(no matches)"

    return [list_files, read_file, grep]


TOOL_KINDS = {"list_files": "read", "read_file": "read", "grep": "read"}


@dataclass
class TraceGraphComponents:
    events: EventLog
    blobs: BlobStore
    ledger: FindingsLedger
    tasks: TaskRegistry
    thoughts: ThoughtLog


def components(store: BaseStore, project_id: str, data_dir: str | os.PathLike[str]) -> TraceGraphComponents:
    events = EventLog(store, project_id)
    blobs = BlobStore(Path(data_dir) / "blobs")
    return TraceGraphComponents(
        events=events,
        blobs=blobs,
        ledger=FindingsLedger(store, project_id, events),
        tasks=TaskRegistry(store, project_id),
        thoughts=ThoughtLog(store, project_id, blobs, events),
    )


def build_agent(model: BaseChatModel, workspace: Path, tg: TraceGraphComponents, *, checkpointer, store,
                cache=None):
    fs = FilesystemAdapter(workspace)
    recorder = Recorder(
        tg.events,
        tg.blobs,
        tool_kinds=TOOL_KINDS,
        untracked=frozenset({"record_finding", "record_thought", "recall_event"}),
        resources=filesystem_tool_resources(fs),
        registry=ResourceRegistry(fs),
        cache=cache,
    )
    tools = workspace_tools(workspace) + [
        make_record_finding_tool(tg.ledger),
        make_record_thought_tool(tg.thoughts),
        make_recall_tool(tg.events, tg.blobs),
    ]
    bound = model.bind_tools(tools)

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        response = bound.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        thought = tg.thoughts.capture(response, config)
        return {"messages": [response], "tg_thoughts": [thought.thought_id] if thought else []}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools, wrap_tool_call=recorder.wrap_tool_call))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer, store=store)


def claude_model(model_id: str | None = None) -> BaseChatModel:
    """Claude via the Messages API, or via the local `claude -p` shim when TG_BASE_URL is set."""
    from langchain_anthropic import ChatAnthropic

    shim_url = os.environ.get("TG_BASE_URL")
    endpoint = {"base_url": shim_url, "api_key": os.environ.get("ANTHROPIC_API_KEY", "claude-p-shim")} if shim_url else {}
    return ChatAnthropic(
        model=model_id or os.environ.get("TG_MODEL", "claude-opus-5-5"),
        max_tokens=16000,
        **endpoint,
        # "summarized" returns readable thinking summaries for the thought state;
        # the default ("omitted") would leave them empty.
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": os.environ.get("TG_EFFORT", "medium")},
    )
