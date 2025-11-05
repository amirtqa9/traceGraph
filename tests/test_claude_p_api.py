"""The claude -p shim, end to end over HTTP, with a fake `claude` binary."""

import json
import os
import stat
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from spike.claude_p_api import Shim, make_handler, render_conversation, response_schema
from tracegraph.thoughts import normalize_reasoning

FAKE_CLAUDE = r'''#!{python}
import json, sys
args = sys.argv[1:]
schema = json.loads(args[args.index("--json-schema") + 1])
prompt = sys.stdin.read()
open({log!r}, "a").write(json.dumps({{"args": args, "prompt": prompt}}) + "\n")
branches = schema["properties"]["tool_calls"].get("items", {{}}).get("anyOf", [])
if branches and "tool result" not in prompt:
    reply = {{"text": "", "tool_calls": [{{"name": branches[0]["properties"]["name"]["const"],
              "input": {{"path": "shop/pagination.py"}}}}]}}
else:
    reply = {{"text": "ANSWER: shop/pagination.py::paginate", "tool_calls": []}}
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "structured_output": reply,
                  "total_cost_usd": 0.001, "usage": {{"input_tokens": 3, "output_tokens": 7,
                  "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0}}}}))
'''


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return "def paginate(): ..."


@pytest.fixture
def shim_url(tmp_path):
    log = tmp_path / "calls.jsonl"
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE.format(python=sys.executable, log=str(log)))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    shim = Shim(model_override=None, timeout=30, claude_bin=str(fake))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(shim))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", log, shim
    server.shutdown()


def model(url):
    return ChatAnthropic(model="claude-opus-5-5", base_url=url, api_key="x", max_retries=0,
                         output_config={"effort": "low"})


def test_tool_call_round_trip(shim_url):
    url, log, shim = shim_url
    llm = model(url).bind_tools([read_file])
    first = llm.invoke([HumanMessage("Find the overlap bug.")])
    assert first.tool_calls[0]["name"] == "read_file"
    assert first.tool_calls[0]["args"] == {"path": "shop/pagination.py"}
    assert normalize_reasoning(first) is None  # claude -p returns no thinking blocks

    second = llm.invoke([
        HumanMessage("Find the overlap bug."), first,
        ToolMessage("def paginate(): ...", tool_call_id=first.tool_calls[0]["id"]),
    ])
    assert "ANSWER: shop/pagination.py::paginate" in second.text
    assert not second.tool_calls

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    args = calls[0]["args"]
    assert args[args.index("--tools") + 1] == "" and "--no-session-persistence" in args
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--effort") + 1] == "low"
    assert "[tool result for" in calls[1]["prompt"]
    assert shim.stats["requests"] == 2


def test_usage_is_reported(shim_url):
    url, _, _ = shim_url
    reply = model(url).invoke([HumanMessage("hi")])
    assert reply.usage_metadata["output_tokens"] == 7
    assert reply.usage_metadata["input_tokens"] >= 3


def test_schema_has_one_branch_per_tool_and_none_without_tools():
    tools = [{"name": "a", "input_schema": {"type": "object"}}, {"name": "b", "input_schema": {"type": "object"}}]
    branches = response_schema(tools)["properties"]["tool_calls"]["items"]["anyOf"]
    assert [b["properties"]["name"]["const"] for b in branches] == ["a", "b"]
    assert response_schema([])["properties"]["tool_calls"]["maxItems"] == 0


def test_render_conversation_keeps_tool_ids_and_drops_thinking():
    text = render_conversation([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "secret-ish", "signature": "s"},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "data"}]},
    ])
    assert "called tool read_file (id toolu_1)" in text and "[tool result for toolu_1]\ndata" in text
    assert "secret-ish" not in text
