"""recall_event: says whether an output is still current, pages long outputs, survives lost blobs."""

import pytest
from langgraph.store.memory import InMemoryStore

from tracegraph.cas import BlobStore
from tracegraph.events import PENDING, ActionEvent, EventLog
from tracegraph.resources import FilesystemAdapter, ResourceRegistry
from tracegraph.resume import make_recall_tool
from tracegraph.validator import Validator


@pytest.fixture
def setup(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("x = 1\n")
    registry = ResourceRegistry(FilesystemAdapter(ws))
    events, blobs = EventLog(InMemoryStore(), "p"), BlobStore(tmp_path / "blobs")

    def record(event_id, output, *, keys=("fs:a.py",), status="success", kind="read"):
        event = ActionEvent(
            event_id=event_id, project_id="p", thread_id="t1", checkpoint_id=None, step=1,
            tool_call_id=event_id, tool="read_file", kind=kind, args={"path": "a.py"},
            output_ref=blobs.put(output.encode()), status=status, read_set=registry.snapshot(keys),
        )
        events.put(event)
        return event

    return ws, registry, events, blobs, record


def recall(tool, event_id, **kw):
    return tool.invoke({"event_id": event_id, **kw})


def test_unchanged_inputs_are_confirmed(setup):
    _, registry, events, blobs, record = setup
    record("ev_000000000001", "x = 1\n")
    out = recall(make_recall_tool(events, blobs, validator=Validator(registry)), "ev_000000000001")
    assert out.startswith("read_file({'path': 'a.py'}) returned:")
    assert "Inputs re-checked: unchanged" in out and out.endswith("x = 1\n")


def test_changed_inputs_are_flagged(setup):
    ws, registry, events, blobs, record = setup
    record("ev_000000000002", "x = 1\n")
    (ws / "a.py").write_text("x = 2\n")
    out = recall(make_recall_tool(events, blobs, validator=Validator(registry)), "ev_000000000002")
    assert "WARNING: this output may be out of date" in out and "a.py" in out
    assert "x = 1" in out  # still the recorded output, not a re-run


def test_uncheckable_inputs_are_flagged(setup):
    _, _, events, blobs, record = setup
    record("ev_000000000003", "x = 1\n")
    out = recall(make_recall_tool(events, blobs, validator=Validator(ResourceRegistry())), "ev_000000000003")
    assert "could not re-check" in out and "a.py" in out


def test_without_validator_no_freshness_claim_is_made(setup):
    _, _, events, blobs, record = setup
    record("ev_000000000004", "x = 1\n")
    out = recall(make_recall_tool(events, blobs), "ev_000000000004")
    assert "re-checked" not in out and "WARNING" not in out


def test_long_outputs_are_paged(setup):
    _, _, events, blobs, record = setup
    text = "".join(f"{i:04d}\n" for i in range(100))  # 500 characters
    record("ev_000000000005", text)
    tool = make_recall_tool(events, blobs, max_chars=200)

    first = recall(tool, "ev_000000000005")
    assert text[:200] in first and text[200:400] not in first
    assert "showing characters 0-200 of 500" in first and "offset=200" in first

    pages = [first]
    offset = 200
    while "offset=" in pages[-1]:
        pages.append(recall(tool, "ev_000000000005", offset=offset))
        offset += 200
    body = "".join(p.split("returned:\n", 1)[1].split("\n[showing")[0] for p in pages)
    assert body == text  # paging loses nothing
    assert "past the end" in recall(tool, "ev_000000000005", offset=999)


def test_missing_blob_is_reported_not_raised(setup):
    _, _, events, blobs, record = setup
    event = record("ev_000000000006", "gone soon\n")
    blobs.delete(event.output_ref)
    out = recall(make_recall_tool(events, blobs), "ev_000000000006")
    assert "on record, but its output is no longer stored" in out


def test_unknown_and_pending_events(setup):
    _, _, events, blobs, record = setup
    tool = make_recall_tool(events, blobs)
    assert "No event 'ev_ffffffffffff'" in recall(tool, "ev_ffffffffffff")
    record("ev_000000000007", "", keys=(), status=PENDING, kind="external")
    assert "outcome is unknown" in recall(tool, "ev_000000000007")
