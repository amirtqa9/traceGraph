"""Every fixture task must actually reproduce its symptom, and only where intended."""

import os
import subprocess
import sys

import pytest

from spike.tasks import TASKS, materialize


def shows_bug(workspace, task) -> bool:
    out = subprocess.run(
        [sys.executable, "-c", task.check + "\nprint(bug)"],
        cwd=workspace, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(workspace)},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "True"


@pytest.mark.parametrize("name", list(TASKS))
def test_symptom_only_where_planted(tmp_path, name):
    task = TASKS[name]
    assert not shows_bug(materialize(tmp_path / "base"), task)
    assert shows_bug(materialize(tmp_path / "planted", task.planted), task)
    assert shows_bug(materialize(tmp_path / "moved", task.moved), task)


@pytest.mark.parametrize("name", list(TASKS))
def test_moved_bug_is_in_a_different_function(name):
    task = TASKS[name]
    assert task.moved_expected != task.expected


@pytest.mark.parametrize("name", list(TASKS))
def test_patches_carry_no_comments(name):
    task = TASKS[name]
    for _, _, new in task.planted + task.moved:
        assert "#" not in new, "patched code must not carry comments pointing at the bug"
