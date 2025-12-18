"""Local stand-in for the Anthropic Messages API, backed by `claude -p`.

Lets the eval run on a Claude Code login instead of an API key:

    .venv/bin/python -m spike.claude_p_api --port 8787 &
    TG_BASE_URL=http://127.0.0.1:8787 .venv/bin/python -m spike.eval --k 1 --tasks pagination_overlap

How a request is served:
- The conversation (text, tool calls, tool results) is rendered into one prompt; the
  request's system prompt is kept and a response protocol plus the tool list is added.
- `claude -p` runs with its own tools disabled, no session persistence, no MCP, no
  user settings, in an empty working directory.
- `--json-schema` forces a structured reply `{text, tool_calls}`; each tool call is
  validated against that tool's own input schema (one `anyOf` branch per tool).
- The reply is returned as a Messages API response with `text` and `tool_use` blocks.

Simulation limits:
- `claude -p` returns no thinking blocks, so responses carry none. Under the shim the
  provider-reasoning thought state stays empty and the `slim_reasoning` arm matches `slim`.
  Agent working memory (`record_thought`) is unaffected.
- No streaming, no forced tool_choice, no server tools. Usage numbers come from
  `claude -p` and include its prompt-cache accounting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PROTOCOL = """# Response protocol (API simulation)
You are the model behind an API call. You cannot run tools yourself: to use a tool, put the
call in `tool_calls` and stop; the caller runs it and sends the result in the next turn.
Reply only through the structured output:
- `text`: what you say to the user this turn (may be empty when you only call tools).
- `tool_calls`: zero or more tool calls, each `{name, input}` matching that tool's schema."""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def render_conversation(messages: list[dict]) -> str:
    lines = ["<conversation>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            lines.append(f"[{role}]\n{content}")
            continue
        for block in content:
            kind = block.get("type")
            if kind == "text" and block.get("text", "").strip():
                lines.append(f"[{role}]\n{block['text']}")
            elif kind == "tool_use":
                lines.append(
                    f"[assistant called tool {block['name']} (id {block['id']})]\n"
                    f"{json.dumps(block.get('input', {}), sort_keys=True)}"
                )
            elif kind == "tool_result":
                status = " (error)" if block.get("is_error") else ""
                lines.append(f"[tool result for {block['tool_use_id']}{status}]\n{_text_of(block.get('content'))}")
            # thinking / redacted_thinking blocks from earlier turns are not replayed.
    lines.append("</conversation>")
    lines.append("Write the assistant's next turn.")
    return "\n\n".join(lines)


def render_system(system: Any, tools: list[dict]) -> str:
    base = _text_of(system) if not isinstance(system, str) else system
    parts = [base.strip(), PROTOCOL] if base else [PROTOCOL]
    if tools:
        parts.append("# Available tools")
        for t in tools:
            parts.append(
                f"## {t['name']}\n{t.get('description', '').strip()}\n"
                f"input schema: {json.dumps(t.get('input_schema', {}), sort_keys=True)}"
            )
    else:
        parts.append("No tools are available this turn; `tool_calls` must be empty.")
    return "\n\n".join(parts)


def response_schema(tools: list[dict]) -> dict:
    calls: dict[str, Any] = {"type": "array", "maxItems": 0}
    if tools:
        calls = {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "name": {"const": t["name"]},
                            "input": t.get("input_schema") or {"type": "object"},
                        },
                        "required": ["name", "input"],
                    }
                    for t in tools
                ]
            },
        }
    return {
        "type": "object",
        "properties": {"text": {"type": "string"}, "tool_calls": calls},
        "required": ["text", "tool_calls"],
    }


class Shim:
    def __init__(self, *, model_override: str | None, timeout: int, claude_bin: str) -> None:
        self.model_override = model_override
        self.timeout = timeout
        self.claude_bin = claude_bin
        self.workdir = tempfile.mkdtemp(prefix="claude-p-shim-")
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "errors": 0, "cost_usd": 0.0, "seconds": 0.0}

    def complete(self, req: dict) -> dict:
        tools = [] if (req.get("tool_choice") or {}).get("type") == "none" else req.get("tools") or []
        tools = [t for t in tools if "input_schema" in t]  # client tools only
        model = self.model_override or req.get("model")
        cmd = [
            self.claude_bin, "-p",
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "",
            "--system-prompt", render_system(req.get("system", ""), tools),
            "--json-schema", json.dumps(response_schema(tools)),
        ]
        if model:
            cmd += ["--model", model]
        effort = (req.get("output_config") or {}).get("effort")
        if effort:
            cmd += ["--effort", effort]

        started = time.monotonic()
        proc = subprocess.run(
            cmd, input=render_conversation(req["messages"]), capture_output=True, text=True,
            timeout=self.timeout, cwd=self.workdir,
        )
        elapsed = time.monotonic() - started
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude -p exit {proc.returncode}: {proc.stderr.strip()[-500:]}") from exc
        if out.get("is_error") or out.get("subtype") != "success" or "structured_output" not in out:
            raise RuntimeError(f"claude -p failed: {out.get('subtype')} {str(out.get('result'))[:300]}")

        reply = out["structured_output"]
        content: list[dict] = []
        if reply.get("text"):
            content.append({"type": "text", "text": reply["text"]})
        for call in reply.get("tool_calls", []):
            content.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": call["name"], "input": call.get("input", {})})
        if not any(b["type"] in ("text", "tool_use") for b in content):
            content.append({"type": "text", "text": ""})

        usage = out.get("usage", {})
        with self.lock:
            self.stats["requests"] += 1
            self.stats["cost_usd"] += out.get("total_cost_usd") or 0.0
            self.stats["seconds"] += elapsed
        print(
            f"[shim] {model} {elapsed:5.1f}s tools={len(tools)} calls={len(reply.get('tool_calls', []))} "
            f"cost≈${out.get('total_cost_usd') or 0:.4f}",
            flush=True,
        )
        return {
            "id": f"msg_shim_{uuid.uuid4().hex[:20]}",
            "type": "message",
            "role": "assistant",
            "model": model or "claude-p-shim",
            "content": content,
            "stop_reason": "tool_use" if reply.get("tool_calls") else "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            },
        }


def make_handler(shim: Shim):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, kind: str, message: str) -> None:
            self._send(status, {"type": "error", "error": {"type": kind, "message": message}})

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/stats":
                with shim.lock:
                    return self._send(200, dict(shim.stats))
            self._error(404, "not_found_error", self.path)

        def do_POST(self) -> None:
            if self.path.split("?")[0].rstrip("/") != "/v1/messages":
                return self._error(404, "not_found_error", self.path)
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            if req.get("stream"):
                return self._error(400, "invalid_request_error", "streaming is not supported by the claude -p shim")
            try:
                self._send(200, shim.complete(req))
            except subprocess.TimeoutExpired:
                with shim.lock:
                    shim.stats["errors"] += 1
                self._error(504, "timeout_error", f"claude -p exceeded {shim.timeout}s")
            except Exception as exc:  # surfaced to the client as an API error
                with shim.lock:
                    shim.stats["errors"] += 1
                self._error(500, "api_error", str(exc))

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", default=os.environ.get("SHIM_MODEL"),
                        help="force this model for every request (default: the request's model)")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per claude -p call")
    parser.add_argument("--claude-bin", default="claude")
    args = parser.parse_args()
    shim = Shim(model_override=args.model, timeout=args.timeout, claude_bin=args.claude_bin)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shim))
    print(f"claude -p shim on http://{args.host}:{args.port} (workdir {shim.workdir})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
