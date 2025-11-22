"""Staleness: is what a finding (or event, or thought) rests on still the same?

A subject is valid iff every resource in its read set fingerprints the same now as when
it was recorded. Nothing is deleted or rewritten: a verdict is computed on demand against
the current world (any checkout of the project), and stale subjects are reported with
exactly which inputs changed, so a resumed agent knows what to re-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tracegraph.events import ActionEvent, EventLog
from tracegraph.findings import Finding
from tracegraph.resources import ResourceRegistry
from tracegraph.thoughts import ThoughtState

Status = Literal["valid", "stale", "unverifiable"]


@dataclass(frozen=True)
class Verdict:
    subject_id: str
    status: Status
    changed: dict[str, tuple[str, str]]  # resource key -> (then, now)

    @property
    def changed_locators(self) -> list[str]:
        return sorted(key.partition(":")[2] for key in self.changed)


class Validator:
    def __init__(self, registry: ResourceRegistry) -> None:
        self.registry = registry

    def check(self, subject_id: str, read_set: dict[str, str]) -> Verdict:
        if not read_set:
            return Verdict(subject_id, "unverifiable", {})
        changed = self.registry.changed(read_set)
        return Verdict(subject_id, "stale" if changed else "valid", changed)

    def event(self, event: ActionEvent) -> Verdict:
        return self.check(event.event_id, event.read_set)

    def finding(self, finding: Finding) -> Verdict:
        return self.check(finding.finding_id, finding.read_set)

    def thought(self, thought: ThoughtState, events: EventLog) -> Verdict:
        read_set: dict[str, str] = {}
        for event_id in thought.supports:
            event = events.get(event_id)
            if event is not None:
                read_set.update(event.read_set)
        return self.check(thought.thought_id, read_set)
