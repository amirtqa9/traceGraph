"""Thought state: what the model was thinking at each node, kept next to the memory state.

The raw chain of thought never leaves the provider. What a client actually gets back
from current Claude models is:

- **signed thinking blocks**: opaque, and bound to the model and the exact conversation
  that produced them. They can be replayed only by continuing that same history,
  unedited, on the same model. Any edit (masking, compaction, a new thread seeded
  with a brief) invalidates them.
- **summaries**: with `thinking.display = "summarized"`, each block's text is a readable
  summary. It is model-authored and unverified, so it counts as a hypothesis, never as evidence.

Both are captured for every agent step and indexed by (thread, checkpoint), the same key
as findings and events. The raw blocks go to the blob store for exact-prefix
replay and audit; the summaries can be offered on a slim resume, clearly labeled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

from tracegraph.canonical import canonical_json, short_id
from tracegraph.cas import BlobStore

THINKING_TYPES = ("thinking", "redacted_thinking")


@dataclass(frozen=True)
class ThoughtState:
    thought_id: str
    project_id: str
    thread_id: str
    checkpoint_id: str | None  # checkpoint the step started from
    step: int | None
    model: str | None
    summary: str  # "" when display is "omitted" or all blocks are redacted
    blocks_ref: str  # blob: the raw blocks, signatures included
    n_blocks: int
    n_redacted: int
    # Signed blocks are only valid when replayed into this exact thread prefix on `model`.
    replay_scope: str = "exact-prefix-same-model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def thinking_blocks(message: AIMessage) -> list[dict]:
    if not isinstance(message.content, list):
        return []
    return [b for b in message.content if isinstance(b, dict) and b.get("type") in THINKING_TYPES]


class ThoughtLog:
    """Thought states in LangGraph's Store under ("tracegraph", project, "thoughts")."""

    def __init__(self, store: BaseStore, project_id: str, blobs: BlobStore) -> None:
        self.store = store
        self.project_id = project_id
        self.blobs = blobs
        self.namespace = ("tracegraph", project_id, "thoughts")

    def capture(self, message: AIMessage, config: dict) -> ThoughtState | None:
        """Record the thinking in one model response. Returns None if there was none."""
        blocks = thinking_blocks(message)
        if not blocks:
            return None
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        checkpoint_id = configurable.get("checkpoint_map", {}).get("") or configurable.get("checkpoint_id")
        step = config.get("metadata", {}).get("langgraph_step")
        blocks_ref = self.blobs.put(canonical_json(blocks))
        meta = message.response_metadata or {}
        state = ThoughtState(
            thought_id=short_id("th", [self.project_id, thread_id, checkpoint_id, step, blocks_ref]),
            project_id=self.project_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            step=step,
            model=meta.get("model") or meta.get("model_name"),
            summary="\n\n".join(b.get("thinking", "") for b in blocks if b.get("thinking")).strip(),
            blocks_ref=blocks_ref,
            n_blocks=len(blocks),
            n_redacted=sum(b.get("type") == "redacted_thinking" for b in blocks),
        )
        self.store.put(self.namespace, state.thought_id, state.to_dict(), index=False)
        return state

    def get(self, thought_id: str) -> ThoughtState | None:
        item = self.store.get(self.namespace, thought_id)
        return ThoughtState(**item.value) if item else None

    def for_threads(self, thread_ids: list[str], limit: int = 10_000) -> list[ThoughtState]:
        items = []
        for thread_id in thread_ids:
            items += self.store.search(self.namespace, filter={"thread_id": thread_id}, limit=limit)
        items.sort(key=lambda it: it.created_at)
        return [ThoughtState(**it.value) for it in items]

    def raw_blocks(self, thought: ThoughtState) -> list[dict]:
        import json

        return json.loads(self.blobs.get(thought.blocks_ref))
