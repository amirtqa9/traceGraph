"""The task and its constraints, pinned verbatim.

Constraints ("don't touch tests/") are never folded into model-written findings:
compaction drops how-to-work rules far more often than specs. They are stored
as given and replayed word for word on every resume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langgraph.store.base import BaseStore


@dataclass(frozen=True)
class TaskSpec:
    thread_id: str
    task: str
    constraints: tuple[str, ...] = ()
    parent_thread: str | None = None  # set when this thread resumes another
    resume_mode: str | None = None

    def render(self) -> str:
        lines = [self.task.strip()]
        if self.constraints:
            lines += ["", "Constraints:"] + [f"- {c}" for c in self.constraints]
        return "\n".join(lines)


class TaskRegistry:
    def __init__(self, store: BaseStore, project_id: str) -> None:
        self.store = store
        self.namespace = ("tracegraph", project_id, "tasks")

    def pin(self, spec: TaskSpec) -> TaskSpec:
        value = asdict(spec)
        value["constraints"] = list(spec.constraints)
        self.store.put(self.namespace, spec.thread_id, value, index=False)
        return spec

    def get(self, thread_id: str) -> TaskSpec | None:
        item = self.store.get(self.namespace, thread_id)
        if item is None:
            return None
        value = dict(item.value)
        value["constraints"] = tuple(value["constraints"])
        return TaskSpec(**value)

    def lineage(self, thread_id: str) -> list[str]:
        """This thread and every thread it was resumed from, oldest first."""
        chain: list[str] = []
        current: str | None = thread_id
        while current and current not in chain:
            chain.append(current)
            spec = self.get(current)
            current = spec.parent_thread if spec else None
        return list(reversed(chain))
