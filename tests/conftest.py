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


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def debugging_policy(messages: list[BaseMessage]) -> AIMessage:
    """Read pagination.py, record a sourced finding, answer.

    Skips steps already covered by visible context (a slim brief or earlier turns),
    so the same policy works for every resume arm.
    """
    text = "\n".join(message_text(m.content) for m in messages)
    read = [m for m in messages if isinstance(m, ToolMessage) and "def paginate" in message_text(m.content)]
    recorded = "Recorded fd_" in text or re.search(r"\[fd_[0-9a-f]{12}\] \(fact\)", text)
    if recorded:
        return AIMessage(content="The end index is off by one.\nANSWER: shop/pagination.py::paginate")
    if not read:
        return call("read_file", path="shop/pagination.py")
    event_id = EVENT_ID_RE.search(message_text(read[-1].content)).group(1)
    return call(
        "record_finding",
        text="shop/pagination.py::paginate uses end = start + size + 1, so pages overlap by one item",
        kind="fact",
        method="read_file on shop/pagination.py",
        sources=[event_id, "ev_000000000000"],
    )


@pytest.fixture
def harness(tmp_path):
    store = InMemoryStore()
    tg = components(store, project_id="shop", data_dir=tmp_path)

    def make(policy=debugging_policy):
        model = ScriptedChatModel(policy=policy)
        return build_agent(model, WORKSPACE, tg, checkpointer=InMemorySaver(), store=store)

    return tg, make
