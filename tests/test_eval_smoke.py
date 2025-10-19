from spike import eval as tg_eval

from conftest import ScriptedChatModel, debugging_policy


def run(tmp_path, **kw):
    return tg_eval.run_eval(
        lambda: ScriptedChatModel(policy=debugging_policy),
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


def test_scenario_workspaces_are_copies(tmp_path):
    run(tmp_path, k=1, arms=("scratch",))
    original = (tg_eval.FIXTURE / "shop/pagination.py").read_text()
    assert (tmp_path / "ws-none/shop/pagination.py").read_text() == original
    assert "def page_bounds" in (tmp_path / "ws-bug_moved/shop/pagination.py").read_text()
    assert "def page_bounds" not in original  # fixture untouched


def test_violations_come_from_the_event_log():
    assert tg_eval.is_violation("read_file", {"path": "tests/test_pagination.py"})
    assert tg_eval.is_violation("list_files", {"path": "./tests"})
    assert not tg_eval.is_violation("read_file", {"path": "shop/pagination.py"})
