"""TraceGraph: verified, invalidating agent memory on top of LangGraph.

Two kinds of record are kept strictly apart:

- **Events** (`events.py`) are written by the tool layer. They are facts about
  what ran, what it returned and (from M1) what it read. The model cannot author them.
- **Findings** (`findings.py`) are written by the model. They are claims, and
  each one is trusted only as far as the events it cites.
"""

from tracegraph.canonical import canonical_json, content_hash
from tracegraph.cas import BlobStore
from tracegraph.core import TraceGraph
from tracegraph.events import ActionEvent, EventLog, Recorder
from tracegraph.findings import Finding, FindingsLedger, make_record_finding_tool
from tracegraph.tasks import TaskRegistry, TaskSpec

__all__ = [
    "ActionEvent",
    "BlobStore",
    "EventLog",
    "Finding",
    "FindingsLedger",
    "Recorder",
    "TaskRegistry",
    "TaskSpec",
    "TraceGraph",
    "canonical_json",
    "content_hash",
    "make_record_finding_tool",
]
