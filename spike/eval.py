"""M0 eval: does resuming from findings beat the cheap baselines?

1. A *source* run works on the task in its own copy of the fixture repo and is cut
   off after `--cutoff` graph steps.
2. For each *scenario*, a fresh copy of the repo is prepared:
     none        unchanged
     bug_moved   between sessions the bug moves: paginate() is fixed and the overlap
                 now comes from a new helper page_bounds(), so the source session's
                 findings are stale
3. From the cut-off point, each arm starts a fresh thread K times per scenario:
     full            whole history (consistent prefix), thinking stripped
     masked          same, but all but the last 2 tool outputs replaced by event pointers
     self_summary    the model's own compaction of the history (computed once per eval)
     slim            verbatim task + constraints + sourced findings only (TraceGraph)
     slim_memory     slim + the agent's last working-memory snapshot (record_thought)
     slim_reasoning  slim + the provider's reasoning text, labeled as leads
     scratch         the task alone, as if nothing had happened
4. Each run is graded on correctness for its scenario, constraint violations (read from
   the event log, not from the model's account), tokens and tool calls.

Usage:
    ANTHROPIC_API_KEY=... .venv/bin/python -m spike.eval --k 3 --cutoff 7
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.store.memory import InMemoryStore

from spike.agent import build_agent, claude_model, components
from tracegraph.events import message_text
from tracegraph.resume import (
    consistent_prefix,
    mask_observations,
    self_summary,
    slim_brief,
    strip_reasoning,
)
from tracegraph.tasks import TaskSpec

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "shop"

TASK = (
    "Users report that page 2 of search results starts with the same item that ended "
    "page 1. Find the function responsible."
)
CONSTRAINTS = (
    "Do not read, list or search anything under tests/.",
    "Do not modify any files.",
)
ARMS = ("full", "masked", "self_summary", "slim", "slim_memory", "slim_reasoning", "scratch")
RESUME_NOTE = "(Session resumed after an interruption. Continue the task.)"
SUMMARY_HEADER = "Summary of your earlier session (written by you before a context reset):\n\n"
ANSWER_RE = re.compile(r"ANSWER:\s*(\S+?)::(\w+)")

BUG_MOVED_PAGINATION = '''from shop.utils import clamp


def page_count(total, size):
    return max(1, -(-total // size))


def page_bounds(page, size):
    start = (page - 1) * size
    return max(0, start - 1), start + size


def paginate(items, page, size):
    """Return the items on 1-based `page`."""
    page = clamp(page, 1, page_count(len(items), size))
    start, end = page_bounds(page, size)
    return items[start:end]
'''


@dataclass(frozen=True)
class Scenario:
    name: str
    expected: tuple[str, str]  # (path, function)
    files: dict[str, str] = field(default_factory=dict)  # applied between sessions


SCENARIOS = {
    "none": Scenario("none", ("shop/pagination.py", "paginate")),
    "bug_moved": Scenario(
        "bug_moved", ("shop/pagination.py", "page_bounds"), {"shop/pagination.py": BUG_MOVED_PAGINATION}
    ),
}


def config(thread_id: str, recursion_limit: int | None = None) -> dict:
    cfg: dict = {"configurable": {"thread_id": thread_id}}
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit
    return cfg


def prepare_workspace(dest: Path, files: dict[str, str] | None = None) -> Path:
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, text in (files or {}).items():
        (dest / rel).write_text(text)
    return dest


def run_thread(agent, thread_id: str, messages: list[BaseMessage], limit: int) -> tuple[list[BaseMessage], bool]:
    """Run until done or `limit` steps. Returns (all messages in the thread, hit_limit)."""
    hit_limit = False
    try:
        agent.invoke({"messages": messages}, config(thread_id, limit))
    except GraphRecursionError:
        hit_limit = True
    return agent.get_state(config(thread_id)).values["messages"], hit_limit


def seed_messages(
    arm: str, source: list[BaseMessage], tg, source_thread: str, spec: TaskSpec, summary: str | None
) -> list[BaseMessage]:
    history = strip_reasoning(consistent_prefix(source)) + [HumanMessage(RESUME_NOTE)]
    if arm == "full":
        return history
    if arm == "masked":
        return mask_observations(history, keep_last=2)
    if arm == "self_summary":
        return [HumanMessage(SUMMARY_HEADER + (summary or ""))]
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


def grade(new_messages: list[BaseMessage], events, expected: tuple[str, str]) -> dict:
    ai = [m for m in new_messages if isinstance(m, AIMessage)]
    final = message_text(ai[-1].content) if ai else ""
    match = ANSWER_RE.search(final)
    usage = [m.usage_metadata or {} for m in ai]
    return {
        "answer": match.group(0) if match else None,
        "correct": bool(match and match.group(1).endswith(expected[0]) and match.group(2) == expected[1]),
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
    scenarios: tuple[str, ...] = tuple(SCENARIOS),
) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    store = InMemoryStore()
    tg = components(store, project_id="shop", data_dir=data_dir)
    model = model_factory()

    source_ws = prepare_workspace(data_dir / "ws-source")
    source_agent = build_agent(model, source_ws, tg, checkpointer=InMemorySaver(), store=store)
    source_thread = f"source-{uuid.uuid4().hex[:8]}"
    spec = tg.tasks.pin(TaskSpec(source_thread, TASK, CONSTRAINTS))
    source, _ = run_thread(source_agent, source_thread, [HumanMessage(spec.render())], cutoff)
    source_findings = tg.ledger.for_threads([source_thread])
    # One compaction per eval, shared by every self_summary run (as a harness would).
    summary = self_summary(model, source) if "self_summary" in arms else None

    runs = []
    for scenario_name in scenarios:
        scenario = SCENARIOS[scenario_name]
        ws = prepare_workspace(data_dir / f"ws-{scenario.name}", scenario.files)
        agent = build_agent(model, ws, tg, checkpointer=InMemorySaver(), store=store)
        for arm in arms:
            for i in range(k):
                thread = f"{scenario.name}-{arm}-{i}-{uuid.uuid4().hex[:6]}"
                tg.tasks.pin(TaskSpec(thread, TASK, CONSTRAINTS, parent_thread=source_thread, resume_mode=arm))
                seed = seed_messages(arm, source, tg, source_thread, spec, summary)
                started = time.monotonic()
                messages, hit_limit = run_thread(agent, thread, seed, run_limit)
                result = grade(messages[len(seed):], tg.events.for_thread(thread), scenario.expected)
                runs.append(
                    {"scenario": scenario.name, "arm": arm, "run": i, "thread": thread, "hit_limit": hit_limit,
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
            "already_answered": grade(source, [], SCENARIOS["none"].expected)["correct"],
            "self_summary": summary,
        },
        "runs": runs,
        "summary": summarize(runs),
    }


def summarize(runs: list[dict]) -> dict:
    out: dict[str, dict[str, dict]] = {}
    for r in runs:
        out.setdefault(r["scenario"], {}).setdefault(r["arm"], []).append(r)
    return {
        scenario: {
            arm: {
                "n": len(rs),
                "success_rate": sum(r["correct"] for r in rs) / len(rs),
                "violation_runs": sum(r["violations"] > 0 for r in rs),
                "mean_resume_context_tokens": statistics.mean(r["resume_context_tokens"] for r in rs),
                "mean_total_input_tokens": statistics.mean(r["input_tokens"] for r in rs),
                "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rs),
            }
            for arm, rs in by_arm.items()
        }
        for scenario, by_arm in out.items()
    }


def print_summary(report: dict) -> None:
    src = report["source"]
    print(
        f"source: {src['tool_calls']} tool calls, {src['sourced_findings']}/{src['findings']} findings sourced, "
        f"{src['working_memory_states']} working-memory states, already answered: {src['already_answered']}"
    )
    for scenario, by_arm in report["summary"].items():
        print(f"\nscenario: {scenario}")
        print(f"{'arm':<15} {'n':>3} {'success':>8} {'violations':>10} {'resume ctx':>11} {'total in':>9} {'tools':>6}")
        for arm, s in by_arm.items():
            print(
                f"{arm:<15} {s['n']:>3} {s['success_rate']:>8.0%} {s['violation_runs']:>10} "
                f"{s['mean_resume_context_tokens']:>11.0f} {s['mean_total_input_tokens']:>9.0f} {s['mean_tool_calls']:>6.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3, help="runs per arm per scenario")
    parser.add_argument("--cutoff", type=int, default=7, help="graph steps before the source run is cut off")
    parser.add_argument("--run-limit", type=int, default=40, help="graph step limit per resumed run")
    parser.add_argument("--model", default=None, help="Claude model ID (default: $TG_MODEL or claude-opus-5-5)")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "runs" / run_id
    report = run_eval(
        lambda: claude_model(args.model),
        k=args.k,
        cutoff=args.cutoff,
        run_limit=args.run_limit,
        data_dir=out_dir / "data",
        arms=tuple(a for a in args.arms.split(",") if a),
        scenarios=tuple(s for s in args.scenarios.split(",") if s),
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nreport: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
