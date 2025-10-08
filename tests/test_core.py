from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tracegraph import BlobStore, canonical_json, content_hash
from tracegraph.events import EVENT_ID_RE
from tracegraph.resume import consistent_prefix, mask_observations, slim_brief
from tracegraph.tasks import TaskSpec


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": [1, {"y": 2, "x": 1}]}) == canonical_json(
        {"a": [1, {"x": 1, "y": 2}], "b": 1}
    )


def test_blob_store_dedupes_and_round_trips(tmp_path):
    blobs = BlobStore(tmp_path)
    ref = blobs.put(b"hello")
    assert ref == content_hash(b"hello") == blobs.put(b"hello")
    assert blobs.get(ref) == b"hello"
    assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == 1


def test_recorder_writes_event_and_tags_output(harness):
    tg, make = harness
    agent = make()
    out = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t1"}})

    events = tg.events.for_thread("t1")
    assert [e.tool for e in events] == ["read_file"]  # record_finding is untracked
    event = events[0]
    assert event.kind == "read"
    assert event.checkpoint_id and event.step is not None
    assert b"def paginate" in tg.blobs.get(event.output_ref)

    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert EVENT_ID_RE.search(tool_msg.content).group(1) == event.event_id


def test_finding_sources_are_checked_against_events(harness):
    tg, make = harness
    out = make().invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t1"}})

    [finding] = tg.ledger.for_threads(["t1"])
    event_id = tg.events.for_thread("t1")[0].event_id
    assert finding.sources == (event_id,)
    assert finding.rejected_sources == ("ev_000000000000",)
    assert out["tg_findings"] == [finding.finding_id]  # reducer channel in graph state


def test_unsourced_findings_are_dropped_from_slim_brief(harness):
    tg, _ = harness
    tg.tasks.pin(TaskSpec("t1", "Find the bug.", ("Never touch tests/.",)))
    tg.ledger.add(thread_id="t1", text="made-up claim", kind="fact", method="vibes", cited=["ev_nope00000000"])
    brief = slim_brief(tg.tasks, tg.ledger, "t1")
    assert "made-up claim" not in brief
    assert "1 unsourced finding(s) were dropped" in brief
    assert "- Never touch tests/." in brief  # constraints verbatim


def test_superseded_findings_are_hidden_and_lineage_is_followed(harness):
    tg, make = harness
    make().invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t1"}})
    tg.tasks.pin(TaskSpec("t1", "Find the bug."))
    tg.tasks.pin(TaskSpec("t2", "Find the bug.", parent_thread="t1", resume_mode="slim"))
    [old] = tg.ledger.for_threads(["t1"])
    new = tg.ledger.add(
        thread_id="t2", text="corrected claim", kind="fact", method="re-read",
        cited=list(old.sources), supersedes=old.finding_id,
    )
    current = tg.ledger.for_threads(tg.tasks.lineage("t2"))
    assert [f.finding_id for f in current] == [new.finding_id]
    assert tg.ledger.get(old.finding_id) is not None  # history kept, never rewritten


def test_mask_and_consistent_prefix():
    msgs = [
        HumanMessage("go"),
        AIMessage("", tool_calls=[{"name": "read_file", "args": {}, "id": "a"}]),
        ToolMessage("one\n\n[event: ev_aaaaaaaaaaaa]", tool_call_id="a"),
        AIMessage("", tool_calls=[{"name": "read_file", "args": {}, "id": "b"}]),
        ToolMessage("two\n\n[event: ev_bbbbbbbbbbbb]", tool_call_id="b"),
        AIMessage("", tool_calls=[{"name": "read_file", "args": {}, "id": "c"}]),  # cut off here
    ]
    prefix = consistent_prefix(msgs)
    assert len(prefix) == 5
    masked = mask_observations(prefix, keep_last=1)
    assert masked[2].content == "[output hidden; call recall_event('ev_aaaaaaaaaaaa') to see it]"
    assert masked[4].content.startswith("two")
