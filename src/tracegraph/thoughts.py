"""Thought state: what the agent was thinking at each node, kept next to the memory state.

Two sources, one record type (`ThoughtState`), keyed by (thread, checkpoint) like
findings and events:

- **agent** (primary): structured working memory the agent writes itself through
  `record_thought`: hypotheses with confidence and cited events, plan, open questions,
  next step. Portable across models, diffable between nodes, and evidence-linked.
- **provider** (secondary): whatever reasoning the model API returns, normalized
  through LangChain's standard `content_blocks`, which maps Claude thinking blocks,
  OpenAI reasoning items, DeepSeek `reasoning_content` and Gemini thought parts to one
  `reasoning` block type. The readable text is stored as an unverified hypothesis. The
  opaque parts (signatures, encrypted content, redacted blocks) go to the blob store
  for replay and audit only.

Neither is evidence. A thought becomes trusted only when it is promoted to a finding
that cites events (`Finding.promoted_from`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from tracegraph.canonical import canonical_json, short_id
from tracegraph.cas import BlobStore
from tracegraph.events import EventLog

TextKind = Literal["structured", "summary", "visible", "none"]
ReplayScope = Literal["portable", "exact-prefix-same-model", "provider-input", "none"]

# Per provider (LangChain `model_provider`): what the readable text is, and how the
# opaque part may be replayed. Unknown providers fall back to ("visible", "none").
PROVIDER_PROFILES: dict[str, tuple[TextKind, ReplayScope]] = {
    # Thinking blocks carry a summary (display="summarized") and a signature bound to
    # the model and the exact conversation.
    "anthropic": ("summary", "exact-prefix-same-model"),
    # Reasoning items: optional summary; encrypted_content is passed back as input.
    "openai": ("summary", "provider-input"),
    # Thought summaries; thought signatures are passed back on later turns.
    "google_genai": ("summary", "provider-input"),
    "google_vertexai": ("summary", "provider-input"),
    # reasoning_content is visible text and is not meant to be sent back.
    "deepseek": ("visible", "none"),
    # Ollama (local or cloud) returns open-weight models' reasoning as visible text.
    "ollama": ("visible", "none"),
}
OPAQUE_KEYS = ("signature", "encrypted_content", "data", "thought_signature")


@dataclass(frozen=True)
class ThoughtState:
    thought_id: str
    project_id: str
    thread_id: str
    checkpoint_id: str | None  # checkpoint the step started from
    step: int | None
    source: str  # "agent" or a provider name
    model: str | None
    text_kind: TextKind
    text: str  # readable form; a hypothesis, never evidence
    replay_scope: ReplayScope
    opaque_ref: str | None = None  # blob: raw provider reasoning payload
    working_memory: dict[str, Any] | None = None  # agent thoughts only
    supports: tuple[str, ...] = ()  # verified event IDs cited by agent hypotheses
    rejected_supports: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supports"] = list(self.supports)
        value["rejected_supports"] = list(self.rejected_supports)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ThoughtState":
        value = dict(value)
        value["supports"] = tuple(value.get("supports", ()))
        value["rejected_supports"] = tuple(value.get("rejected_supports", ()))
        return cls(**value)


@dataclass(frozen=True)
class NormalizedReasoning:
    source: str
    text_kind: TextKind
    text: str
    replay_scope: ReplayScope
    raw: dict[str, Any]  # everything reasoning-related, as the provider returned it
    has_opaque: bool


def normalize_reasoning(message: AIMessage) -> NormalizedReasoning | None:
    """Provider-neutral view of the reasoning in one model response, or None."""
    blocks = [b for b in message.content_blocks if b.get("type") == "reasoning"]
    redacted = [
        b["value"] for b in message.content_blocks
        if b.get("type") == "non_standard" and b.get("value", {}).get("type") == "redacted_thinking"
    ]
    if not blocks and not redacted:
        return None
    meta = message.response_metadata or {}
    source = meta.get("model_provider") or "unknown"
    text_kind, replay_scope = PROVIDER_PROFILES.get(source, ("visible", "none"))
    text = "\n\n".join(b.get("reasoning", "") for b in blocks if b.get("reasoning")).strip()
    raw_content = message.content if isinstance(message.content, list) else []
    raw = {
        "content_blocks": [
            b for b in raw_content
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking", "reasoning")
        ],
        "additional_kwargs": {
            k: v for k, v in message.additional_kwargs.items() if "reason" in k or "thought" in k
        },
    }
    flat = json.dumps(raw)
    return NormalizedReasoning(
        source=source,
        text_kind=text_kind if text else "none",
        text=text,
        replay_scope=replay_scope,
        raw=raw,
        has_opaque=bool(redacted) or any(f'"{k}"' in flat for k in OPAQUE_KEYS),
    )


def _step_key(config: dict) -> tuple[str, str | None, int | None]:
    configurable = config.get("configurable", {})
    return (
        configurable.get("thread_id", ""),
        configurable.get("checkpoint_map", {}).get("") or configurable.get("checkpoint_id"),
        config.get("metadata", {}).get("langgraph_step"),
    )


class ThoughtLog:
    """Thought states in LangGraph's Store under ("tracegraph", project, "thoughts")."""

    def __init__(self, store: BaseStore, project_id: str, blobs: BlobStore, events: EventLog) -> None:
        self.store = store
        self.project_id = project_id
        self.blobs = blobs
        self.events = events
        self.namespace = ("tracegraph", project_id, "thoughts")

    def _put(self, state: ThoughtState) -> ThoughtState:
        self.store.put(self.namespace, state.thought_id, state.to_dict(), index=False)
        return state

    def capture(self, message: AIMessage, config: dict) -> ThoughtState | None:
        """Record the provider reasoning in one model response. None if there was none."""
        reasoning = normalize_reasoning(message)
        if reasoning is None:
            return None
        thread_id, checkpoint_id, step = _step_key(config)
        opaque_ref = self.blobs.put(canonical_json(reasoning.raw))
        meta = message.response_metadata or {}
        return self._put(ThoughtState(
            thought_id=short_id("th", [self.project_id, thread_id, checkpoint_id, step, opaque_ref]),
            project_id=self.project_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            step=step,
            source=reasoning.source,
            model=meta.get("model") or meta.get("model_name"),
            text_kind=reasoning.text_kind,
            text=reasoning.text,
            replay_scope=reasoning.replay_scope,
            opaque_ref=opaque_ref,
        ))

    def record_working_memory(self, memory: dict[str, Any], config: dict) -> ThoughtState:
        """Record an agent-authored working-memory snapshot."""
        thread_id, checkpoint_id, step = _step_key(config)
        cited = list(dict.fromkeys(e for h in memory.get("hypotheses", []) for e in h.get("supports", [])))
        verified = tuple(c for c in cited if self.events.get(c) is not None)
        return self._put(ThoughtState(
            thought_id=short_id("th", [self.project_id, thread_id, checkpoint_id, step, memory]),
            project_id=self.project_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            step=step,
            source="agent",
            model=None,
            text_kind="structured",
            text=render_working_memory(memory),
            replay_scope="portable",
            working_memory=memory,
            supports=verified,
            rejected_supports=tuple(c for c in cited if c not in verified),
        ))

    def get(self, thought_id: str) -> ThoughtState | None:
        item = self.store.get(self.namespace, thought_id)
        return ThoughtState.from_dict(item.value) if item else None

    def for_threads(self, thread_ids: list[str], source: str | None = None, limit: int = 10_000) -> list[ThoughtState]:
        items = []
        for thread_id in thread_ids:
            items += self.store.search(self.namespace, filter={"thread_id": thread_id}, limit=limit)
        items.sort(key=lambda it: it.created_at)
        states = [ThoughtState.from_dict(it.value) for it in items]
        if source == "agent":
            return [s for s in states if s.source == "agent"]
        if source == "provider":
            return [s for s in states if s.source != "agent"]
        return states

    def all(self, limit: int = 1_000_000) -> list[ThoughtState]:
        items = sorted(self.store.search(self.namespace, limit=limit), key=lambda it: it.created_at)
        return [ThoughtState.from_dict(it.value) for it in items]

    def raw(self, thought: ThoughtState) -> dict[str, Any] | None:
        return json.loads(self.blobs.get(thought.opaque_ref)) if thought.opaque_ref else None


def render_working_memory(memory: dict[str, Any]) -> str:
    lines = []
    for h in memory.get("hypotheses", []):
        cites = f" [supports: {', '.join(h['supports'])}]" if h.get("supports") else " [no supporting events]"
        lines.append(f"- hypothesis ({h.get('confidence', 0):.0%}): {h['text']}{cites}")
    if memory.get("plan"):
        lines.append("- plan: " + " → ".join(memory["plan"]))
    lines += [f"- open question: {q}" for q in memory.get("open_questions", [])]
    if memory.get("next_step"):
        lines.append(f"- next step: {memory['next_step']}")
    return "\n".join(lines)


class Hypothesis(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    supports: list[str] = Field(default_factory=list, description="ev_ IDs that support it")


def make_record_thought_tool(log: ThoughtLog) -> BaseTool:
    @tool
    def record_thought(
        hypotheses: list[Hypothesis],
        plan: list[str],
        runtime: ToolRuntime,
        open_questions: list[str] | None = None,
        next_step: str = "",
    ) -> Command:
        """Save your current working memory so a later session can pick up your line of thought.

        Call it when your plan or hypotheses change. Unlike record_finding, these are
        unconfirmed: hypotheses with a confidence, the plan, open questions and the next step.

        Args:
            hypotheses: What you currently suspect, each with confidence 0-1 and supporting ev_ IDs.
            plan: Remaining steps, in order.
            open_questions: What you don't know yet.
            next_step: What you would do next.
        """
        memory = {
            "hypotheses": [h.model_dump() if isinstance(h, BaseModel) else dict(h) for h in hypotheses],
            "plan": list(plan),
            "open_questions": list(open_questions or []),
            "next_step": next_step,
        }
        state = log.record_working_memory(memory, runtime.config)
        reply = f"Working memory saved as {state.thought_id}."
        if state.rejected_supports:
            reply += f" Unknown event IDs ignored: {', '.join(state.rejected_supports)}."
        return Command(update={
            "tg_thoughts": [state.thought_id],
            "messages": [ToolMessage(reply, tool_call_id=runtime.tool_call_id)],
        })

    return record_thought
