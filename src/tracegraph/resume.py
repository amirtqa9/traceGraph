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
from tracegraph.thoughts import ThoughtLog


def slim_brief(
    tasks: TaskRegistry,
    ledger: FindingsLedger,
    source_thread: str,
    thoughts: ThoughtLog | None = None,
    *,
    working_memory: bool = False,
    provider_reasoning: bool = False,
    max_reasoning: int = 5,
) -> str:
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
    if thoughts is not None and working_memory:
        latest = thoughts.for_threads(lineage, source="agent")[-1:]
        if latest:
            out += [
                "",
                f"## Working memory at cut-off [{latest[0].thought_id}] (your own notes; hypotheses are unconfirmed)",
                latest[0].text,
            ]
    if thoughts is not None and provider_reasoning:
        texts = [t.text for t in thoughts.for_threads(lineage, source="provider") if t.text][-max_reasoning:]
        if texts:
            out += [
                "",
                "## Prior model reasoning (the earlier session's reasoning text; NOT evidence)",
                "Use these only as leads. Verify anything here before relying on it.",
            ]
            out += [f"- {t}" for t in texts]
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


# Generic compaction instruction, modeled on default agent-harness compaction. It
# deliberately does not ask for constraints verbatim: this arm is the baseline.
SELF_SUMMARY_PROMPT = (
    "Your session is about to be reset and a fresh instance of you will continue the work "
    "with no access to this conversation. Write a summary of the session so far: the task, "
    "what you did, what you found, and what to do next. Reply with the summary only."
)


def render_transcript(messages: list[BaseMessage]) -> str:
    """Plain-text transcript, so a summary request needs no tool definitions."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {message_text(m.content)}")
        elif isinstance(m, AIMessage):
            text = message_text(m.content).strip()
            if text:
                lines.append(f"ASSISTANT: {text}")
            for tc in m.tool_calls:
                lines.append(f"ASSISTANT CALLED {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            lines.append(f"TOOL RESULT: {message_text(m.content)}")
    return "\n\n".join(lines)


def self_summary(model, messages: list[BaseMessage]) -> str:
    """The model's own compaction of `messages` (one call, no tools)."""
    transcript = render_transcript(consistent_prefix(messages))
    reply = model.invoke([HumanMessage(f"<transcript>\n{transcript}\n</transcript>\n\n{SELF_SUMMARY_PROMPT}")])
    return message_text(reply.content).strip()
