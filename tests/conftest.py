from __future__ import annotations

import itertools
import re
from collections.abc import Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from spike.agent import build_agent, components
from spike.eval import WORKSPACE
from tracegraph.events import EVENT_ID_RE, message_text

_ids = itertools.count()


class ScriptedChatModel(BaseChatModel):
    """Stands in for Claude: `policy(messages) -> AIMessage` decides each turn."""

    policy: Callable[[list[BaseMessage]], AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.policy(messages))])


ANTHROPIC_META = {"model_provider": "anthropic", "model_name": "scripted-claude"}
THOUGHT_ID_RE = re.compile(r"Working memory saved as (th_[0-9a-f]{12})")


def thinking(summary: str) -> dict:
    # Shape of a Claude thinking block with display="summarized": readable summary + opaque signature.
    return {"type": "thinking", "thinking": summary, "signature": f"sig-{next(_ids)}"}


def call(name: str, thought: str = "", **args) -> AIMessage:
    content = [thinking(thought)] if thought else ""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}],
        response_metadata=ANTHROPIC_META,
    )


def debugging_policy(messages: list[BaseMessage]) -> AIMessage:
    """Read pagination.py -> save working memory -> promote it to a finding -> answer.

    Skips steps already covered by visible context (a slim brief or earlier turns),
    so the same policy works for every resume arm.
    """
    text = "\n".join(message_text(m.content) for m in messages)
    read = [m for m in messages if isinstance(m, ToolMessage) and "def paginate" in message_text(m.content)]
    recorded = "Recorded fd_" in text or re.search(r"\[fd_[0-9a-f]{12}\] \(fact\)", text)
    thought = THOUGHT_ID_RE.search(text)
    if recorded:
        return AIMessage(
            content=[
                thinking("Finding confirms the off-by-one; answer."),
                {"type": "text", "text": "The end index is off by one.\nANSWER: shop/pagination.py::paginate"},
            ],
            response_metadata=ANTHROPIC_META,
        )
    if not read:
        return call("read_file", thought="Overlap between pages smells like a slicing bug; check pagination.",
                    path="shop/pagination.py")
    event_id = EVENT_ID_RE.search(message_text(read[-1].content)).group(1)
    if not thought:
        return call(
            "record_thought",
            thought="Suspect the end index; save working memory before confirming.",
            hypotheses=[{"text": "paginate() end index is one too far", "confidence": 0.8,
                         "supports": [event_id, "ev_ffffffffffff"]}],
            plan=["confirm the slice bounds", "record the finding", "answer"],
            open_questions=["does search() add its own offset?"],
            next_step="record the finding",
        )
    return call(
        "record_finding",
        thought="end = start + size + 1 returns one extra item; record it.",
        text="shop/pagination.py::paginate uses end = start + size + 1, so pages overlap by one item",
        kind="fact",
        method="read_file on shop/pagination.py",
        sources=[event_id, "ev_000000000000"],
        from_thought=thought.group(1),
    )


@pytest.fixture
def harness(tmp_path):
    store = InMemoryStore()
    tg = components(store, project_id="shop", data_dir=tmp_path)

    def make(policy=debugging_policy):
        model = ScriptedChatModel(policy=policy)
        return build_agent(model, WORKSPACE, tg, checkpointer=InMemorySaver(), store=store)

    return tg, make
