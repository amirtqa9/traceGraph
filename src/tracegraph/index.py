"""Per-checkpoint view: memory state and thought state at one node.

LangGraph checkpoints the `tg_findings` and `tg_thoughts` channels at every step,
so the checkpoint is the index. This joins it with the events whose step started
from that checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from tracegraph.events import ActionEvent, EventLog
from tracegraph.findings import Finding, FindingsLedger
from tracegraph.thoughts import ThoughtLog, ThoughtState


@dataclass(frozen=True)
class NodeState:
    thread_id: str
    checkpoint_id: str
    findings: tuple[Finding, ...]  # memory state: cumulative up to this checkpoint
    thoughts: tuple[ThoughtState, ...]  # thought state: cumulative up to this checkpoint
    events: tuple[ActionEvent, ...]  # tool calls whose step started here


def node_state(graph, thread_id: str, checkpoint_id: str, *, ledger: FindingsLedger,
               thoughts: ThoughtLog, events: EventLog) -> NodeState:
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}})
    values = snapshot.values
    return NodeState(
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        findings=tuple(f for fid in values.get("tg_findings", []) if (f := ledger.get(fid))),
        thoughts=tuple(t for tid in values.get("tg_thoughts", []) if (t := thoughts.get(tid))),
        events=tuple(e for e in events.for_thread(thread_id) if e.checkpoint_id == checkpoint_id),
    )
