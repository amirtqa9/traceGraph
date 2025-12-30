"""Turn an eval report into Markdown tables with confidence intervals.

    .venv/bin/python -m spike.analyze spike/runs/<id>/report.json > results.md

Success rates get 95% Wilson score intervals (sound for small n and rates near 0 or 1).
Runs that ended in an API/network error are excluded from rates and counted separately.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

from spike.tasks import TASKS


def is_stale_answer(run: dict) -> bool:
    """In bug_moved, did the agent name the bug's *old* (pre-move) location?"""
    task = TASKS.get(run["task"])
    answer = run.get("answer") or ""
    if run["scenario"] != "bug_moved" or task is None or run["correct"] or "::" not in answer:
        return False
    path, function = answer.split("ANSWER:")[-1].strip().split("::", 1)
    return path.endswith(task.expected[0]) and function.split(".")[-1] == task.expected[1]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def arm_table(runs: list[dict], arms: list[str]) -> list[str]:
    lines = [
        "| arm | n | success | 95% CI | stale answers | violations | errors | resume ctx (tok) | total input (tok) | tool calls | cache hits |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        ok = [r for r in rs if not r.get("error")]
        if not rs:
            continue
        wins = sum(r["correct"] for r in ok)
        lo, hi = wilson(wins, len(ok))
        mean = lambda key: statistics.mean(r.get(key, 0) for r in ok) if ok else 0  # noqa: E731
        lines.append(
            f"| {arm} | {len(ok)} | {wins / len(ok) if ok else 0:.0%} | {lo:.0%}–{hi:.0%} | "
            f"{sum(is_stale_answer(r) for r in ok)} | {sum(r['violations'] > 0 for r in ok)} | {len(rs) - len(ok)} | {mean('resume_context_tokens'):.0f} | "
            f"{mean('input_tokens'):.0f} | {mean('tool_calls'):.1f} | {mean('cache_hits'):.1f} |"
        )
    return lines


def task_table(runs: list[dict], arms: list[str], scenario: str) -> list[str]:
    tasks = sorted({r["task"] for r in runs})
    lines = ["| task | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for task in tasks:
        cells = []
        for arm in arms:
            rs = [r for r in runs if r["task"] == task and r["arm"] == arm and r["scenario"] == scenario
                  and not r.get("error")]
            cells.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}" if rs else "–")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return lines


def render(report: dict) -> str:
    runs = [r for t in report["tasks"].values() for r in t["runs"]]
    arms = list(dict.fromkeys(r["arm"] for r in runs))
    scenarios = list(dict.fromkeys(r["scenario"] for r in runs))
    protocol = next(iter(report["tasks"].values())).get("protocol", {})
    out = [f"Runs: {len(runs)} over {len(report['tasks'])} task(s), scenarios {', '.join(scenarios)}; "
           f"protocol: {protocol or 'default prompt'}.", ""]
    src = [t["source"] for t in report["tasks"].values()]
    out += [
        "Source sessions: "
        f"{sum(s['already_answered'] for s in src)}/{len(src)} answered within the cutoff (resumed from just before the answer); "
        f"{sum(s['sourced_findings'] for s in src)} sourced findings, {sum(s['findings'] for s in src)} findings total; "
        f"{sum(s['working_memory_states'] for s in src)} working-memory snapshots; "
        f"{sum(s['provider_reasoning_states'] for s in src)} provider-reasoning states.",
        "",
    ]
    for scenario in scenarios:
        out += [f"### Scenario `{scenario}`", ""]
        out += arm_table([r for r in runs if r["scenario"] == scenario], arms)
        out += ["", f"Per task (`{scenario}`, correct/runs):", ""]
        out += task_table(runs, arms, scenario)
        out.append("")
    by_reason = defaultdict(int)
    for r in runs:
        if r.get("error"):
            by_reason[r["error"].split(":")[0]] += 1
    if by_reason:
        out += ["Errors: " + ", ".join(f"{k} x{v}" for k, v in sorted(by_reason.items())), ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report")
    args = parser.parse_args()
    with open(args.report) as f:
        print(render(json.load(f)))


if __name__ == "__main__":
    main()
