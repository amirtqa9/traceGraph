"""Building the starting context for a resumed agent.

- `slim_brief`: verbatim task + constraints + sourced findings, optionally validated
  against the current world, plus side effects and thought state.
- `mask_observations`: full history with old tool outputs replaced by their event IDs.
- `consistent_prefix`: trims a history cut off mid tool loop.
- `make_recall_tool`: just-in-time fetch of any event's original output.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from tracegraph.cas import BlobStore
from tracegraph.events import EVENT_ID_RE, PENDING, EventLog, message_text
from tracegraph.findings import FindingsLedger
from tracegraph.tasks import TaskRegistry
from tracegraph.thoughts import ThoughtLog
from tracegraph.validator import Validator


def _render_finding(f, verdict=None) -> str:
    flag = "STALE " if verdict is not None and verdict.status == "stale" else ""
    line = f"- [{f.finding_id}] {flag}({f.kind}) {f.text}\n  method: {f.method}; sources: {', '.join(f.sources)}"
    if verdict is not None and verdict.status == "stale":
        line += f"\n  changed since: {', '.join(verdict.changed_locators)}"
    elif verdict is not None and verdict.status == "unverifiable":
        if verdict.unavailable:
            locs = ", ".join(sorted(k.partition(":")[2] or k for k in verdict.unavailable))
            line += f"\n  (could not re-check here: {locs})"
        else:
            line += "\n  (inputs not tracked; cannot be checked for staleness)"
    return line


def slim_brief(
    tasks: TaskRegistry,
    ledger: FindingsLedger,
    source_thread: str,
    thoughts: ThoughtLog | None = None,
    *,
    working_memory: bool = False,
    provider_reasoning: bool = False,
    max_reasoning: int = 5,
    validator: Validator | None = None,
    events: EventLog | None = None,
) -> str:
    """The seed for a fresh agent: verbatim task + constraints + sourced findings.

    With a `validator`, findings are checked against the current world: stale ones are
    listed separately with what changed, and a startup routine tells the agent to re-check
    them first. With `events`, side effects already performed (writes, external calls)
    are listed so the resumed agent does not repeat or assume them undone.
    """
    lineage = tasks.lineage(source_thread)
    spec = tasks.get(lineage[0])
    if spec is None:
        raise KeyError(f"no pinned task for thread {lineage[0]!r}")
    findings = ledger.for_threads(lineage)
    sourced = [f for f in findings if f.sourced]
    unsourced = len(findings) - len(sourced)
    todos = [f for f in sourced if f.kind == "todo"]
    carried = [f for f in sourced if f.kind != "todo"]
    verdicts = {f.finding_id: validator.finding(f) for f in carried} if validator else {}
    def status(f) -> str | None:
        v = verdicts.get(f.finding_id)
        return v.status if v is not None else None

    stale = [f for f in carried if status(f) == "stale"]
    # With a validator, only findings whose inputs were all re-checked count as current.
    unverified = [f for f in carried if status(f) == "unverifiable"] if validator else []
    current = [f for f in carried if f not in stale and f not in unverified]

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
    heading = "## Findings from earlier sessions (each backed by recorded tool events"
    heading += "; inputs re-checked and unchanged)" if validator else ")"
    out += ["", heading]
    out += [_render_finding(f, verdicts.get(f.finding_id)) for f in current] or ["- (none)"]
    if unverified:
        out += ["", "## Unverified findings (inputs could not be re-checked; confirm before relying on them)"]
        out += [_render_finding(f, verdicts[f.finding_id]) for f in unverified]
    if stale:
        out += ["", "## Stale findings (their inputs changed since; do NOT rely on them)"]
        out += [_render_finding(f, verdicts[f.finding_id]) for f in stale]
    if todos:
        out += ["", "## Open TODOs"] + [f"- [{f.finding_id}] {f.text}" for f in todos]
    if unsourced:
        out += ["", f"({unsourced} unsourced finding(s) were dropped as untrusted.)"]
    if events is not None:
        effects = [e for e in events.for_threads(lineage) if e.kind in ("write", "external")]
        if effects:
            out += ["", "## Side effects already performed (not undone; do not repeat blindly)"]
            out += [
                f"- [{e.event_id}] {e.kind}: {e.tool}({e.args})"
                + (" (outcome unknown: it may or may not have happened)" if e.status == PENDING else "")
                for e in effects
            ]
    if thoughts is not None and working_memory:
        latest = thoughts.for_threads(lineage, source="agent")[-1:]
        if latest:
            note = ""
            if validator is not None and events is not None:
                verdict = validator.thought(latest[0], events)
                if verdict.status == "stale":
                    note = f" (inputs changed since: {', '.join(verdict.changed_locators)})"
            out += [
                "",
                f"## Working memory at cut-off [{latest[0].thought_id}] (your own notes; hypotheses are unconfirmed){note}",
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
    out += ["", "## Startup"]
    if stale:
        out.append("1. Re-check the stale findings first: the files listed changed after they were concluded.")
    out.append(
        f"{2 if stale else 1}. Before relying on a `ruled_out` finding, spot-check one with recall_event(<source id>)."
    )
    out.append(f"{3 if stale else 2}. Do not redo work the current findings already cover unless a check disagrees.")
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


DEFAULT_RECALL_CHARS = 20_000


def _freshness(event, validator: Validator | None) -> str | None:
    """One line on whether this recorded output still matches the world, or None."""
    if event.status == PENDING:
        return "NOTE: this call's outcome is unknown (it never reported back); it may or may not have happened."
    if validator is None or not event.read_set:
        return None
    verdict = validator.event(event)
    if verdict.status == "stale":
        return ("WARNING: this output may be out of date; its inputs changed since: "
                + ", ".join(verdict.changed_locators) + ". Re-run the tool for the current version.")
    if verdict.status == "unverifiable":
        locs = ", ".join(sorted(k.partition(":")[2] or k for k in verdict.unavailable))
        return f"NOTE: could not re-check this output's inputs here ({locs}); it may be out of date."
    return "Inputs re-checked: unchanged since this call."


def make_recall_tool(
    events: EventLog, blobs: BlobStore, *, validator: Validator | None = None,
    max_chars: int = DEFAULT_RECALL_CHARS,
) -> BaseTool:
    """`recall_event`: the recorded output of an earlier tool call, by event ID.

    - It returns the output as recorded (redacted), not a re-run. With a `validator`, a
      line says whether the inputs behind it are unchanged, changed, or uncheckable.
    - Outputs longer than `max_chars` come back in pages; the tool says how to get the next.
    - An event whose output blob is gone gets a clear message instead of an exception.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    @tool
    def recall_event(event_id: str, offset: int = 0) -> str:
        """Fetch the original output of an earlier tool call by its `ev_...` ID.

        Args:
            event_id: The `ev_...` ID shown under a tool result or cited by a finding.
            offset: Character offset to start from, for long outputs returned in pages.
        """
        event = events.get(event_id)
        if event is None:
            return f"No event {event_id!r} (unknown ID, or expired)."
        header = f"{event.tool}({event.args}) returned:"
        note = _freshness(event, validator)
        try:
            output = blobs.get(event.output_ref).decode("utf-8")
        except (FileNotFoundError, ValueError):
            lines = [f"Event {event_id} ({event.tool}) is on record, but its output is no longer stored."]
            return "\n".join(lines + ([note] if note else []))
        start = max(0, offset)
        if start >= len(output) and output:
            return f"{header}\n(offset {start} is past the end; the output is {len(output)} characters)"
        page = output[start:start + max_chars]
        lines = [header] + ([note] if note else []) + [page]
        end = start + len(page)
        if end < len(output):
            lines.append(f"[showing characters {start}-{end} of {len(output)}; "
                         f"call recall_event('{event_id}', offset={end}) for more]")
        return "\n".join(lines)

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
