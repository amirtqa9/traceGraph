"""Building the starting context for the M0 resume arms.

- `slim_brief`: verbatim task + constraints + sourced findings only.
- `mask_observations`: full history with old tool outputs replaced by their event IDs.
- `consistent_prefix`: trims a history cut off mid tool loop.
- `make_recall_tool`: just-in-time fetch of any event's original output.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from tracegraph.cas import BlobStore
from tracegraph.events import EVENT_ID_RE, EventLog, message_text
from tracegraph.findings import FindingsLedger
from tracegraph.tasks import TaskRegistry


def slim_brief(tasks: TaskRegistry, ledger: FindingsLedger, source_thread: str) -> str:
    lineage = tasks.lineage(source_thread)
    spec = tasks.get(lineage[0])
    if spec is None:
        raise KeyError(f"no pinned task for thread {lineage[0]!r}")
    findings = ledger.for_threads(lineage)
    sourced = [f for f in findings if f.sourced]
    unsourced = len(findings) - len(sourced)
    todos = [f for f in sourced if f.kind == "todo"]
    carried = [f for f in sourced if f.kind != "todo"]

    out = [
        "You are resuming a task from an earlier session. You have none of its "
        "conversation, only what is below.",
        "",
        "## Task (verbatim)",
        spec.task.strip(),
    ]
    if spec.constraints:
        out += ["", "## Constraints (verbatim, still in force)"]
        out += [f"- {c}" for c in spec.constraints]
    out += ["", "## Findings from earlier sessions (each backed by recorded tool events)"]
    if carried:
        for f in carried:
            out.append(
                f"- [{f.finding_id}] ({f.kind}) {f.text}\n"
                f"  method: {f.method}; sources: {', '.join(f.sources)}"
            )
    else:
        out.append("- (none)")
    if todos:
        out += ["", "## Open TODOs"] + [f"- [{f.finding_id}] {f.text}" for f in todos]
    if unsourced:
        out += ["", f"({unsourced} unsourced finding(s) were dropped as untrusted.)"]
    out += [
        "",
        "Before relying on a `ruled_out` finding, spot-check one with recall_event(<source id>). "
        "Do not redo work the findings already cover unless a spot-check disagrees.",
    ]
    return "\n".join(out)


def consistent_prefix(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop a trailing AI turn whose tool calls never got all their results."""
    msgs = list(messages)
    while msgs:
        last_ai = max((i for i, m in enumerate(msgs) if isinstance(m, AIMessage)), default=None)
        if last_ai is None or not msgs[last_ai].tool_calls:
            return msgs
        wanted = {tc["id"] for tc in msgs[last_ai].tool_calls}
        answered = {m.tool_call_id for m in msgs[last_ai + 1 :] if isinstance(m, ToolMessage)}
        if wanted <= answered:
            return msgs
        msgs = msgs[:last_ai]
    return msgs


def strip_reasoning(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove thinking blocks from copied AI turns.

    Histories copied into a new thread are edited (masked), and current Claude models
    tie thinking blocks to the exact conversation that produced them. Stripping them
    from every copied arm keeps the arms comparable and the requests valid.
    """
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            kept = [b for b in m.content if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking", "reasoning"))]
            m = m.model_copy(update={"content": kept or ""})
        out.append(m)
    return out


def mask_observations(messages: list[BaseMessage], keep_last: int = 2) -> list[BaseMessage]:
    """Replace all but the last `keep_last` tool outputs with a pointer to their event."""
    tool_positions = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    to_mask = set(tool_positions[: max(0, len(tool_positions) - keep_last)])
    out: list[BaseMessage] = []
    for i, m in enumerate(messages):
        if i in to_mask:
            match = EVENT_ID_RE.search(message_text(m.content))
            pointer = (
                f"[output hidden; call recall_event('{match.group(1)}') to see it]"
                if match
                else "[output hidden]"
            )
            m = m.model_copy(update={"content": pointer})
        out.append(m)
    return out


def resume_message(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def make_recall_tool(events: EventLog, blobs: BlobStore) -> BaseTool:
    @tool
    def recall_event(event_id: str) -> str:
        """Fetch the original output of an earlier tool call by its `ev_...` ID."""
        event = events.get(event_id)
        if event is None:
            return f"No event {event_id!r}."
        output = blobs.get(event.output_ref).decode("utf-8")
        return f"{event.tool}({event.args}) returned:\n{output}"

    return recall_event
