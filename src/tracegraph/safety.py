"""Side-effect safety for rewind / fork / resume.

LangGraph can rewind a thread to any checkpoint, but rewinding the conversation does not
rewind the world. Before resuming at checkpoint N, TraceGraph lists what happened after
N that the world still remembers:

- `write` events (files changed): can be undone only if a workspace snapshot was taken
  before them (`GitSnapshotter`, opt-in);
- `external` events (emails, pushes, API calls): can never be undone, only disclosed.

`fork_at` refuses by default to resume behind side effects; the caller must choose to
warn, proceed, or restore the pre-write snapshot.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tracegraph.events import ActionEvent, EventLog

OnSideEffects = Literal["error", "warn", "proceed", "restore"]


class ResumeBehindSideEffectsError(RuntimeError):
    def __init__(self, risk: "ResumeRisk") -> None:
        self.risk = risk
        super().__init__(
            f"resuming at {risk.checkpoint_id} would ignore {len(risk.writes_after)} write(s) and "
            f"{len(risk.external_after)} external call(s) made after it: "
            + ", ".join(f"{e.tool}({e.event_id})" for e in risk.writes_after + risk.external_after)
        )


@dataclass(frozen=True)
class ResumeRisk:
    thread_id: str
    checkpoint_id: str
    step: int
    writes_after: list[ActionEvent] = field(default_factory=list)
    external_after: list[ActionEvent] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.writes_after and not self.external_after

    @property
    def restorable(self) -> bool:
        return not self.external_after and all(e.pre_snapshot for e in self.writes_after)


def resume_risk(graph, thread_id: str, checkpoint_id: str, events: EventLog) -> ResumeRisk:
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}})
    step = snapshot.metadata.get("step", -1)
    after = [e for e in events.for_thread(thread_id) if e.step is not None and e.step > step]
    return ResumeRisk(
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        step=step,
        writes_after=[e for e in after if e.kind == "write"],
        external_after=[e for e in after if e.kind == "external"],
    )


def fork_at(
    graph,
    thread_id: str,
    checkpoint_id: str,
    events: EventLog,
    on_side_effects: OnSideEffects = "error",
    snapshotter: "GitSnapshotter | None" = None,
) -> tuple[dict, ResumeRisk]:
    """Config to resume `thread_id` from `checkpoint_id`, after checking side effects."""
    risk = resume_risk(graph, thread_id, checkpoint_id, events)
    if not risk.safe:
        if on_side_effects == "error":
            raise ResumeBehindSideEffectsError(risk)
        if on_side_effects == "warn":
            warnings.warn(str(ResumeBehindSideEffectsError(risk)), stacklevel=2)
        if on_side_effects == "restore":
            if snapshotter is None or not risk.restorable:
                raise ResumeBehindSideEffectsError(risk)
            snapshotter.restore(risk.writes_after[0].pre_snapshot)
    return {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}, risk


class GitSnapshotter:
    """Workspace snapshots as git commits under refs/tracegraph/, without touching the
    user's index, HEAD or branches (a temporary index is used for both directions)."""

    REF_PREFIX = "refs/tracegraph/"

    def __init__(self, repo: str | os.PathLike[str]) -> None:
        self.repo = Path(repo).resolve()

    def _git(self, *args: str, env: dict | None = None) -> str:
        full_env = {**os.environ, "GIT_AUTHOR_NAME": "tracegraph", "GIT_AUTHOR_EMAIL": "tracegraph@localhost",
                    "GIT_COMMITTER_NAME": "tracegraph", "GIT_COMMITTER_EMAIL": "tracegraph@localhost", **(env or {})}
        return subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True, text=True,
                              env=full_env).stdout.strip()

    def _has_head(self) -> bool:
        try:
            self._git("rev-parse", "--verify", "HEAD")
            return True
        except subprocess.CalledProcessError:
            return False

    def snapshot(self, label: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            if self._has_head():
                self._git("read-tree", "HEAD", env=env)
            self._git("add", "-A", ".", env=env)
            tree = self._git("write-tree", env=env)
        parents = ["-p", "HEAD"] if self._has_head() else []
        commit = self._git("commit-tree", tree, *parents, "-m", f"tracegraph snapshot {label}")
        self._git("update-ref", f"{self.REF_PREFIX}{label}", commit)
        return commit

    def restore(self, commit: str) -> None:
        """Make the working tree match `commit` (tracked + untracked, ignored files kept)."""
        wanted = set(self._git("ls-tree", "-r", "--name-only", commit).splitlines())
        present = set(self._git("ls-files", "--cached", "--others", "--exclude-standard").splitlines())
        for rel in present - wanted:
            (self.repo / rel).unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            self._git("read-tree", commit, env=env)
            self._git("checkout-index", "-a", "-f", env=env)
