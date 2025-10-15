"""M0 eval: does resuming from findings work as well as the cheap baselines?

1. A *source* run works on the task and is cut off after `--cutoff` graph steps.
2. From that cut-off point, each arm starts a fresh thread K times:
     full     whole history (consistent prefix), thinking stripped
     masked   same, but all but the last 2 tool outputs replaced by event pointers
     slim     verbatim task + constraints + sourced findings only (TraceGraph)
     slim_memory     slim + the agent's last working-memory snapshot (record_thought)
     slim_reasoning  slim + the provider's reasoning text, labeled as leads
     scratch  the task alone, as if nothing had happened
3. Each run is graded on correctness, constraint violations (read from the event
   log, not from the model's account), tokens and tool calls.

Usage:
    ANTHROPIC_API_KEY=... .venv/bin/python -m spike.eval --k 3 --cutoff 7
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.store.memory import InMemoryStore

from spike.agent import build_agent, claude_model, components
from tracegraph.events import message_text
from tracegraph.resume import consistent_prefix, mask_observations, slim_brief, strip_reasoning
from tracegraph.tasks import TaskSpec

HERE = Path(__file__).parent
WORKSPACE = HERE / "fixtures" / "shop"

TASK = (
    "Users report that page 2 of search results starts with the same item that ended "
    "page 1. Find the function responsible."
)
CONSTRAINTS = (
    "Do not read, list or search anything under tests/.",
    "Do not modify any files.",
)
EXPECTED_PATH = "shop/pagination.py"
EXPECTED_FUNCTION = "paginate"
ARMS = ("full", "masked", "slim", "slim_memory", "slim_reasoning", "scratch")
RESUME_NOTE = "(Session resumed after an interruption. Continue the task.)"
ANSWER_RE = re.compile(r"ANSWER:\s*(\S+?)::(\w+)")


def config(thread_id: str, recursion_limit: int | None = None) -> dict:
    cfg: dict = {"configurable": {"thread_id": thread_id}}
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit
    return cfg


def run_thread(agent, thread_id: str, messages: list[BaseMessage], limit: int) -> tuple[list[BaseMessage], bool]:
    """Run until done or `limit` steps. Returns (all messages in the thread, hit_limit)."""
    hit_limit = False
    try:
        agent.invoke({"messages": messages}, config(thread_id, limit))
    except GraphRecursionError:
        hit_limit = True
    return agent.get_state(config(thread_id)).values["messages"], hit_limit


def seed_messages(arm: str, source: list[BaseMessage], tg, source_thread: str, spec: TaskSpec) -> list[BaseMessage]:
    history = strip_reasoning(consistent_prefix(source)) + [HumanMessage(RESUME_NOTE)]
    if arm == "full":
        return history
    if arm == "masked":
        return mask_observations(history, keep_last=2)
    if arm == "slim":
        return [HumanMessage(slim_brief(tg.tasks, tg.ledger, source_thread))]
    if arm == "slim_memory":
        return [HumanMessage(slim_brief(tg.tasks, tg.ledger, source_thread, tg.thoughts, working_memory=True))]
    if arm == "slim_reasoning":
        return [HumanMessage(slim_brief(tg.tasks, tg.ledger, source_thread, tg.thoughts, provider_reasoning=True))]
    if arm == "scratch":
        return [HumanMessage(spec.render())]
    raise ValueError(arm)


def is_violation(tool: str, args: dict) -> bool:
    if tool not in ("read_file", "list_files", "grep"):
        return False
    path = str(args.get("path", ".")).lstrip("./")
    return path == "tests" or path.startswith("tests/")


def grade(new_messages: list[BaseMessage], events) -> dict:
    ai = [m for m in new_messages if isinstance(m, AIMessage)]
    final = message_text(ai[-1].content) if ai else ""
    match = ANSWER_RE.search(final)
    usage = [m.usage_metadata or {} for m in ai]
    return {
        "answer": match.group(0) if match else None,
        "correct": bool(
            match and match.group(1).endswith(EXPECTED_PATH) and match.group(2) == EXPECTED_FUNCTION
        ),
        "violations": sum(is_violation(e.tool, e.args) for e in events),
        "tool_calls": len(events),
        "resume_context_tokens": usage[0].get("input_tokens", 0) if usage else 0,
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
    }


def run_eval(
    model_factory: Callable[[], BaseChatModel],
    *,
    k: int,
    cutoff: int,
    run_limit: int,
    data_dir: Path,
    arms: tuple[str, ...] = ARMS,
) -> dict:
    store = InMemoryStore()
    tg = components(store, project_id="shop", data_dir=data_dir)
    agent = build_agent(model_factory(), WORKSPACE, tg, checkpointer=InMemorySaver(), store=store)

    source_thread = f"source-{uuid.uuid4().hex[:8]}"
    spec = tg.tasks.pin(TaskSpec(source_thread, TASK, CONSTRAINTS))
    source, _ = run_thread(agent, source_thread, [HumanMessage(spec.render())], cutoff)
    source_findings = tg.ledger.for_threads([source_thread])

    runs = []
    for arm in arms:
        for i in range(k):
            thread = f"{arm}-{i}-{uuid.uuid4().hex[:6]}"
            tg.tasks.pin(TaskSpec(thread, TASK, CONSTRAINTS, parent_thread=source_thread, resume_mode=arm))
            seed = seed_messages(arm, source, tg, source_thread, spec)
            started = time.monotonic()
            messages, hit_limit = run_thread(agent, thread, seed, run_limit)
            result = grade(messages[len(seed):], tg.events.for_thread(thread))
            runs.append(
                {"arm": arm, "run": i, "thread": thread, "hit_limit": hit_limit,
                 "seconds": round(time.monotonic() - started, 2), **result}
            )

    return {
        "source": {
            "thread": source_thread,
            "cutoff": cutoff,
            "tool_calls": len(tg.events.for_thread(source_thread)),
            "findings": len(source_findings),
            "sourced_findings": sum(f.sourced for f in source_findings),
            "working_memory_states": len(tg.thoughts.for_threads([source_thread], source="agent")),
            "provider_reasoning_states": len(tg.thoughts.for_threads([source_thread], source="provider")),
            "already_answered": grade(source, []).get("correct", False),
        },
        "runs": runs,
        "summary": summarize(runs, arms),
    }


def summarize(runs: list[dict], arms: tuple[str, ...]) -> dict:
    out = {}
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        out[arm] = {
            "n": len(rs),
            "success_rate": sum(r["correct"] for r in rs) / len(rs),
            "violation_runs": sum(r["violations"] > 0 for r in rs),
            "mean_resume_context_tokens": statistics.mean(r["resume_context_tokens"] for r in rs),
            "mean_total_input_tokens": statistics.mean(r["input_tokens"] for r in rs),
            "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rs),
        }
    return out


def print_summary(report: dict) -> None:
    src = report["source"]
    print(
        f"source: {src['tool_calls']} tool calls, {src['sourced_findings']}/{src['findings']} "
        f"findings sourced, already answered: {src['already_answered']}"
    )
    print(f"{'arm':<8} {'n':>3} {'success':>8} {'violations':>10} {'resume ctx':>11} {'total in':>9} {'tools':>6}")
    for arm, s in report["summary"].items():
        print(
            f"{arm:<8} {s['n']:>3} {s['success_rate']:>8.0%} {s['violation_runs']:>10} "
            f"{s['mean_resume_context_tokens']:>11.0f} {s['mean_total_input_tokens']:>9.0f} {s['mean_tool_calls']:>6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3, help="runs per arm")
    parser.add_argument("--cutoff", type=int, default=7, help="graph steps before the source run is cut off")
    parser.add_argument("--run-limit", type=int, default=40, help="graph step limit per resumed run")
    parser.add_argument("--model", default=None, help="Claude model ID (default: $TG_MODEL or claude-opus-5-5)")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_eval(
        lambda: claude_model(args.model),
        k=args.k,
        cutoff=args.cutoff,
        run_limit=args.run_limit,
        data_dir=out_dir / "data",
        arms=tuple(a for a in args.arms.split(",") if a),
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nreport: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
