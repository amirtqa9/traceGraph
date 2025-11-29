"""When to resume from findings instead of continuing in-session.

Deliberately simple and explainable (PLAN §2.7):
- In session, the default is to keep going; observation masking is the cheap baseline.
- Past `threshold` of the context window, or when the session has ended, start a fresh
  agent from a slim, validated brief. Context quality degrades well below the window
  limit ("context rot"), so the default threshold is conservative.
- Tool reuse is decided per call by the action cache, not here.
Every decision is logged with its inputs; nothing time-dependent is an input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from langgraph.store.base import BaseStore

from tracegraph.canonical import short_id

Action = Literal["continue", "slim_resume"]


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    context_tokens: int
    window_tokens: int
    threshold: float
    session_ended: bool


def decide(context_tokens: int, window_tokens: int, *, session_ended: bool = False, threshold: float = 0.4) -> Decision:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    used = context_tokens / window_tokens if window_tokens else 1.0
    if session_ended:
        action, reason = "slim_resume", "session ended; nothing to continue in"
    elif used >= threshold:
        action, reason = "slim_resume", f"context at {used:.0%} of the window (threshold {threshold:.0%})"
    else:
        action, reason = "continue", f"context at {used:.0%} of the window, under {threshold:.0%}"
    return Decision(action, reason, context_tokens, window_tokens, threshold, session_ended)


class DecisionLog:
    def __init__(self, store: BaseStore, project_id: str) -> None:
        self.store = store
        self.namespace = ("tracegraph", project_id, "decisions")

    def record(self, thread_id: str, decision: Decision) -> str:
        value = {"thread_id": thread_id, **asdict(decision)}
        key = short_id("dc", [value, len(self.for_thread(thread_id))])
        self.store.put(self.namespace, key, value, index=False)
        return key

    def for_thread(self, thread_id: str) -> list[dict]:
        items = self.store.search(self.namespace, filter={"thread_id": thread_id}, limit=100_000)
        return [it.value for it in sorted(items, key=lambda it: it.created_at)]
