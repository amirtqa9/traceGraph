from spike import eval as tg_eval
from spike.tasks import FIXTURE, TASKS

from conftest import ScriptedChatModel, debugging_policy

# The scripted policy only knows how to solve this task.
TASK = TASKS["pagination_overlap"]


def run(tmp_path, **kw):
    return tg_eval.run_eval(
        lambda: ScriptedChatModel(policy=debugging_policy),
        TASK,
        # Cut off after record_finding ran, before the answer.
        cutoff=6, run_limit=20, data_dir=tmp_path, **kw,
    )


def test_eval_runs_every_arm_and_scenario(tmp_path):
    report = run(tmp_path, k=2)
    src = report["source"]
    assert src["tool_calls"] == 1 and src["sourced_findings"] == 1 and not src["already_answered"]
    assert src["self_summary"].startswith("Task:")
    assert set(report["summary"]) == set(tg_eval.SCENARIOS)
    none = report["summary"]["none"]
    assert set(none) == set(tg_eval.ARMS)
    for arm, s in none.items():
        assert s["success_rate"] == 1.0, arm
        assert s["violation_runs"] == 0, arm
    # Slim resumes from the finding alone: no re-reading.
    assert none["slim"]["mean_tool_calls"] == 0
    assert none["scratch"]["mean_tool_calls"] == 1


def test_bug_moved_exposes_stale_memory(tmp_path):
    """After the bug moves, arms that trust carried-over state answer the old location.

    This is the failure the M3 validator (read-set re-hashing) has to fix; until then
    the eval should show it, not hide it.
    """
    report = run(tmp_path, k=1, scenarios=("bug_moved",))
    moved = report["summary"]["bug_moved"]
    for arm in ("full", "slim", "slim_memory"):
        assert moved[arm]["success_rate"] == 0.0, arm  # trusted a stale finding
    for arm in ("scratch", "self_summary"):
        assert moved[arm]["success_rate"] == 1.0, arm  # re-read the changed file


def test_validated_slim_resume_recovers_from_the_moved_bug(tmp_path):
    """M3: the validator flags the finding as stale, so the agent re-checks instead of trusting it."""
    report = run(tmp_path, k=1, arms=("slim", "slim_validated"))
    assert report["summary"]["bug_moved"]["slim"]["success_rate"] == 0.0
    assert report["summary"]["bug_moved"]["slim_validated"]["success_rate"] == 1.0
    # When nothing changed, validation costs nothing: the finding is trusted, no re-reading.
    assert report["summary"]["none"]["slim_validated"]["success_rate"] == 1.0
    assert report["summary"]["none"]["slim_validated"]["mean_tool_calls"] == 0


def test_scenario_workspaces_are_patched_copies(tmp_path):
    run(tmp_path, k=1, arms=("scratch",))
    clean = (FIXTURE / "shop/pagination.py").read_text()
    assert "size + 1" in (tmp_path / "ws-source/shop/pagination.py").read_text()
    assert "size + 1" in (tmp_path / "ws-none/shop/pagination.py").read_text()
    assert "def page_bounds" in (tmp_path / "ws-bug_moved/shop/pagination.py").read_text()
    assert "size + 1" not in clean and "def page_bounds" not in clean  # fixture untouched


def test_suite_aggregates_across_tasks(tmp_path):
    report = tg_eval.run_suite(
        lambda: ScriptedChatModel(policy=debugging_policy),
        ("pagination_overlap",),
        k=1, cutoff=6, run_limit=20, data_dir=tmp_path, arms=("scratch",), scenarios=("none",),
    )
    assert set(report["tasks"]) == {"pagination_overlap"}
    assert report["summary"]["none"]["scratch"]["n"] == 1


def test_grading_accepts_qualified_names():
    from langchain_core.messages import AIMessage

    expected = ("shop/cart.py", "add")
    for answer, ok in [("shop/cart.py::add", True), ("shop/cart.py::Cart.add", True),
                       ("shop/cart.py::subtotal", False), ("cart.py::add", False)]:
        graded = tg_eval.grade([AIMessage(f"ANSWER: {answer}")], [], expected)
        assert graded["correct"] is ok, answer


def test_source_is_interrupted_before_its_answer():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    tc = {"name": "read_file", "args": {}, "id": "a"}
    msgs = [HumanMessage("go"), AIMessage("", tool_calls=[tc]), ToolMessage("x", tool_call_id="a"),
            AIMessage("ANSWER: shop/a.py::f")]
    assert tg_eval.interrupt_before_answer(msgs) == msgs[:3]
    assert tg_eval.interrupt_before_answer(msgs[:3]) == msgs[:3]


def test_violations_come_from_the_event_log():
    assert tg_eval.is_violation("read_file", {"path": "tests/test_shop.py"})
    assert tg_eval.is_violation("list_files", {"path": "./tests"})
    assert not tg_eval.is_violation("read_file", {"path": "shop/pagination.py"})


def test_analyze_renders_tables_with_intervals(tmp_path):
    from spike.analyze import render, wilson

    lo, hi = wilson(3, 3)
    assert 0.4 < lo < 0.5 and hi == 1.0
    assert wilson(0, 0) == (0.0, 0.0)
    report = tg_eval.run_suite(
        lambda: ScriptedChatModel(policy=debugging_policy), ("pagination_overlap",),
        k=1, cutoff=6, run_limit=20, data_dir=tmp_path, arms=("slim", "slim_validated"),
    )
    text = render(report)
    assert "### Scenario `bug_moved`" in text and "| slim_validated | 1 | 100% |" in text
    assert "| slim | 1 | 0% | 0%–79% | 1 |" in text  # the failure is a stale answer
    assert "| pagination_overlap | 0/1 | 1/1 |" in text  # bug_moved: slim fails, validated recovers
