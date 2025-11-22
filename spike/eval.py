"""M0 eval: does resuming from findings beat the cheap baselines?

Runs over the fixture tasks in `spike/tasks.py` (10 planted bugs in the `shop` repo).
For each task:

1. A *source* run works on the task in its own copy of the repo (bug planted) and is
   cut off after `--cutoff` graph steps.
2. For each *scenario*, a fresh copy of the repo is prepared:
     none        the same planted bug
     bug_moved   between sessions the bug moves: the original function is clean and the
                 same symptom now comes from a different function, so the source
                 session's findings are stale
3. From the cut-off point, each arm starts a fresh thread K times per scenario:
     full            whole history (consistent prefix), thinking stripped
     masked          same, but all but the last 2 tool outputs replaced by event pointers
     self_summary    the model's own compaction of the history (computed once per eval)
     slim            verbatim task + constraints + sourced findings only (TraceGraph M0)
     slim_validated  slim + findings re-checked against the resumed workspace (M3): stale
                     ones flagged with what changed, plus side effects and a startup routine
     slim_memory     slim + the agent's last working-memory snapshot (record_thought)
     slim_reasoning  slim + the provider's reasoning text, labeled as leads
     scratch         the task alone, as if nothing had happened
4. Each run is graded on correctness for its scenario, constraint violations (read from
   the event log, not from the model's account), tokens and tool calls.

Usage:
    ANTHROPIC_API_KEY=... .venv/bin/python -m spike.eval --k 1 --cutoff 7 --tasks pagination_overlap
Runs = tasks x scenarios x arms x k (all tasks, both scenarios, 7 arms, k=3 is 420 runs).
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

from spike.agent import build_agent, components, make_model
from spike.tasks import TASKS, FixtureTask, materialize
from tracegraph.events import message_text
from tracegraph.resume import (
    consistent_prefix,
    mask_observations,
    self_summary,
    slim_brief,
    strip_reasoning,
)
from tracegraph.resources import FilesystemAdapter, ResourceRegistry
from tracegraph.tasks import TaskSpec
from tracegraph.validator import Validator

HERE = Path(__file__).parent

CONSTRAINTS = (
    "Do not read, list or search anything under tests/.",
    "Do not modify any files.",
)
ARMS = ("full", "masked", "self_summary", "slim", "slim_validated", "slim_memory", "slim_reasoning", "scratch")
SCENARIOS = ("none", "bug_moved")
RESUME_NOTE = "(Session resumed after an interruption. Continue the task.)"
SUMMARY_HEADER = "Summary of your earlier session (written by you before a context reset):\n\n"
ANSWER_RE = re.compile(r"ANSWER:\s*(\S+?)::([\w.]+)")


def scenario_setup(task: FixtureTask, scenario: str) -> tuple[tuple, tuple[str, str]]:
    """(patch to apply to the clean repo, expected answer) for a scenario."""
    if scenario == "none":
        return task.planted, task.expected
    if scenario == "bug_moved":
        return task.moved, task.moved_expected
    raise ValueError(scenario)


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


def seed_messages(
    arm: str, source: list[BaseMessage], tg, source_thread: str, spec: TaskSpec, summary: str | None,
    validator: Validator | None = None,
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
    if arm == "slim_validated":
        return [HumanMessage(slim_brief(tg.tasks, tg.ledger, source_thread, validator=validator, events=tg.events))]
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
    # "Cart.add" and "add" both count: compare the last dotted component.
    correct = bool(
        match and match.group(1).endswith(expected[0]) and match.group(2).split(".")[-1] == expected[1]
    )
    return {
        "answer": match.group(0) if match else None,
        "correct": correct,
        "violations": sum(is_violation(e.tool, e.args) for e in events),
        "tool_calls": len(events),
        "resume_context_tokens": usage[0].get("input_tokens", 0) if usage else 0,
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
    }


def run_eval(
    model_factory: Callable[[], BaseChatModel],
    task: FixtureTask,
    *,
    k: int,
    cutoff: int,
    run_limit: int,
    data_dir: Path,
    arms: tuple[str, ...] = ARMS,
    scenarios: tuple[str, ...] = SCENARIOS,
) -> dict:
    """One task: a source run, then every scenario x arm x k resumed runs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    store = InMemoryStore()
    tg = components(store, project_id=task.name, data_dir=data_dir)
    model = model_factory()

    source_ws = materialize(data_dir / "ws-source", task.planted)
    source_agent = build_agent(model, source_ws, tg, checkpointer=InMemorySaver(), store=store)
    source_thread = f"source-{uuid.uuid4().hex[:8]}"
    spec = tg.tasks.pin(TaskSpec(source_thread, task.prompt, CONSTRAINTS))
    source, _ = run_thread(source_agent, source_thread, [HumanMessage(spec.render())], cutoff)
    source_findings = tg.ledger.for_threads([source_thread])
    # One compaction per eval, shared by every self_summary run (as a harness would).
    summary = self_summary(model, source) if "self_summary" in arms else None

    runs = []
    for scenario in scenarios:
        patch, expected = scenario_setup(task, scenario)
        ws = materialize(data_dir / f"ws-{scenario}", patch)
        agent = build_agent(model, ws, tg, checkpointer=InMemorySaver(), store=store)
        validator = Validator(ResourceRegistry(FilesystemAdapter(ws)))
        for arm in arms:
            for i in range(k):
                thread = f"{scenario}-{arm}-{i}-{uuid.uuid4().hex[:6]}"
                tg.tasks.pin(TaskSpec(thread, task.prompt, CONSTRAINTS, parent_thread=source_thread, resume_mode=arm))
                seed = seed_messages(arm, source, tg, source_thread, spec, summary, validator)
                started = time.monotonic()
                messages, hit_limit = run_thread(agent, thread, seed, run_limit)
                result = grade(messages[len(seed):], tg.events.for_thread(thread), expected)
                runs.append(
                    {"task": task.name, "scenario": scenario, "arm": arm, "run": i, "thread": thread, "hit_limit": hit_limit,
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
            "already_answered": grade(source, [], task.expected)["correct"],
            "self_summary": summary,
        },
        "runs": runs,
        "summary": summarize(runs),
    }


def run_suite(
    model_factory: Callable[[], BaseChatModel],
    tasks: tuple[str, ...],
    *,
    data_dir: Path,
    **kwargs,
) -> dict:
    """Every task in `tasks`; summary aggregates runs across tasks."""
    reports = {name: run_eval(model_factory, TASKS[name], data_dir=data_dir / name, **kwargs) for name in tasks}
    runs = [r for rep in reports.values() for r in rep["runs"]]
    return {"tasks": reports, "summary": summarize(runs)}


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
    for name, task_report in report["tasks"].items():
        src = task_report["source"]
        print(
            f"{name}: source {src['tool_calls']} tool calls, {src['sourced_findings']}/{src['findings']} findings "
            f"sourced, {src['working_memory_states']} working-memory states, already answered: {src['already_answered']}"
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
    parser.add_argument("--k", type=int, default=3, help="runs per arm per scenario per task")
    parser.add_argument("--cutoff", type=int, default=7, help="graph steps before the source run is cut off")
    parser.add_argument("--run-limit", type=int, default=40, help="graph step limit per resumed run")
    parser.add_argument("--model", default=None,
                        help="model ID (default: $TG_MODEL; gpt-oss:120b on Ollama, claude-opus-5-5 on Anthropic)")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--tasks", default=",".join(TASKS), help=f"comma list from: {', '.join(TASKS)}")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_suite(
        lambda: make_model(args.model),
        tuple(t for t in args.tasks.split(",") if t),
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
