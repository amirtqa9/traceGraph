from spike import eval as tg_eval

from conftest import ScriptedChatModel, debugging_policy


def test_eval_runs_every_arm_end_to_end(tmp_path):
    report = tg_eval.run_eval(
        lambda: ScriptedChatModel(policy=debugging_policy),
        # Cut off after record_finding ran, before the answer.
        k=2, cutoff=6, run_limit=20, data_dir=tmp_path,
    )
    assert report["source"]["tool_calls"] == 1
    assert report["source"]["sourced_findings"] == 1
    assert not report["source"]["already_answered"]
    assert set(report["summary"]) == set(tg_eval.ARMS)
    for arm, s in report["summary"].items():
        assert s["success_rate"] == 1.0, arm
        assert s["violation_runs"] == 0, arm
    # The slim arm starts from the brief, so it should not need to re-read the file.
    assert report["summary"]["slim"]["mean_tool_calls"] == 0
    assert report["summary"]["scratch"]["mean_tool_calls"] == 1


def test_violations_come_from_the_event_log():
    assert tg_eval.is_violation("read_file", {"path": "tests/test_pagination.py"})
    assert tg_eval.is_violation("list_files", {"path": "./tests"})
    assert not tg_eval.is_violation("read_file", {"path": "shop/pagination.py"})
