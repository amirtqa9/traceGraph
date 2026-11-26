"""Regression tests for the P1 fixes: each test fails on 5be22ad and passes after the fix."""

import datetime as dt
import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from tracegraph import TraceGraph
from tracegraph.cache import ActionCache
from tracegraph.cas import BlobStore
from tracegraph.events import PENDING, ActionEvent, EventLog, Recorder
from tracegraph.findings import FindingsLedger
from tracegraph.gc import collect_garbage, expire
from tracegraph.redact import redact_text
from tracegraph.resources import FilesystemAdapter, ResourceRegistry
from tracegraph.resume import slim_brief
from tracegraph.safety import ResumeBehindSideEffectsError, fork_at, resume_risk
from tracegraph.tasks import TaskRegistry, TaskSpec
from tracegraph.thoughts import ThoughtLog
from tracegraph.validator import Validator

from conftest import call
from test_safety import first_checkpoint, git_repo, writing_agent


# ---------------------------------------------------------------- helpers

def put_event(events: EventLog, blobs: BlobStore, event_id: str, output: bytes, *, kind="read",
              read_set=None, thread="t1", step=1) -> ActionEvent:
    event = ActionEvent(
        event_id=event_id, project_id=events.project_id, thread_id=thread, checkpoint_id=None, step=step,
        tool_call_id=event_id, tool="read_file", kind=kind, args={}, output_ref=blobs.put(output),
        status="success", read_set=read_set or {},
    )
    events.put(event)
    return event


def fake_request(name, args, thread="t1", call_id="call_x"):
    config = {"configurable": {"thread_id": thread}, "metadata": {"langgraph_step": 3}}
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": call_id},
                           runtime=SimpleNamespace(config=config))


# ---------------------------------------------------------------- 1. GC across projects

def test_gc_from_one_project_keeps_other_projects_blobs(tmp_path):
    a = TraceGraph.open(tmp_path, "proj-a")
    b = TraceGraph.open(tmp_path, "proj-b")
    ea = put_event(a.events, a.blobs, "ev_aaaaaaaaaaaa", b"only A")
    eb = put_event(b.events, b.blobs, "ev_bbbbbbbbbbbb", b"only B")
    shared_a = put_event(a.events, a.blobs, "ev_aaaaaaaaaaa2", b"both")
    shared_b = put_event(b.events, b.blobs, "ev_bbbbbbbbbbb2", b"both")
    orphan = a.blobs.put(b"nobody")

    report = collect_garbage(a.blobs, a.events, a.thoughts, grace=dt.timedelta(0))

    assert report.projects == 2 and report.removed == 1
    assert not a.blobs.has(orphan)
    for tg, e in ((a, ea), (b, eb), (a, shared_a), (b, shared_b)):
        assert tg.blobs.get(e.output_ref)  # every live output can still be recalled
    a.close(), b.close()


def test_gc_grace_keeps_fresh_unreferenced_blobs(tmp_path):
    tg = TraceGraph.open(tmp_path, "p")
    just_written = tg.blobs.put(b"event not stored yet")  # a writer between put() and events.put()
    report = collect_garbage(tg.blobs, tg.events, tg.thoughts)  # default grace
    assert report.removed == 0 and report.skipped_recent == 1 and tg.blobs.has(just_written)
    tg.close()


# ---------------------------------------------------------------- 2. checkpoints per project

def _echo_graph(tg, label):
    g = StateGraph(MessagesState)
    g.add_node("echo", lambda s: {"messages": [AIMessage(label)]})
    g.add_edge(START, "echo")
    return g.compile(checkpointer=tg.checkpointer())


def test_same_thread_id_in_two_projects_has_separate_history(tmp_path):
    a, b = TraceGraph.open(tmp_path, "proj-a"), TraceGraph.open(tmp_path, "proj/b")
    cfg = {"configurable": {"thread_id": "t1"}}
    _echo_graph(a, "from A").invoke({"messages": [HumanMessage("hi")]}, cfg)
    _echo_graph(b, "from B").invoke({"messages": [HumanMessage("hi")]}, cfg)
    a.close(), b.close()

    a, b = TraceGraph.open(tmp_path, "proj-a"), TraceGraph.open(tmp_path, "proj/b")  # after reopen
    state_a = _echo_graph(a, "x").get_state(cfg).values["messages"]
    state_b = _echo_graph(b, "x").get_state(cfg).values["messages"]
    assert [m.content for m in state_a] == ["hi", "from A"]
    assert [m.content for m in state_b] == ["hi", "from B"]
    assert a.checkpoint_path() != b.checkpoint_path()
    a.close(), b.close()


def test_legacy_shared_checkpoint_file_is_announced(tmp_path):
    (tmp_path / "checkpoints.sqlite").write_bytes(b"")
    tg = TraceGraph.open(tmp_path, "p")
    with pytest.warns(UserWarning, match="shared by every project"):
        tg.checkpointer()
    tg.close()


# ---------------------------------------------------------------- 3. side effects stay visible

def test_command_returning_write_is_recorded_and_blocks_rewind(harness, tmp_path):
    tg, _ = harness
    store, blobs = tg.events.store, tg.blobs
    recorder = Recorder(tg.events, blobs, {"save": "write"})
    request = fake_request("save", {"path": "x.txt"}, call_id="call_cmd")

    def execute(_):
        (tmp_path / "x.txt").write_text("written")
        return Command(update={"messages": [ToolMessage("saved", tool_call_id="call_cmd")], "counter": 1})

    result = recorder.wrap_tool_call(request, execute)

    [event] = tg.events.for_thread("t1")
    assert event.kind == "write" and event.status == "success"
    assert isinstance(result, Command) and result.update["counter"] == 1  # graph update preserved
    [msg] = result.update["messages"]
    assert msg.content.startswith("saved") and f"[event: {event.event_id}]" in msg.content


def test_write_whose_outcome_is_unknown_stays_pending(harness):
    tg, _ = harness
    recorder = Recorder(tg.events, tg.blobs, {"deploy": "external"})

    def crash(_):
        raise RuntimeError("connection reset after the request was sent")

    with pytest.raises(RuntimeError):
        recorder.wrap_tool_call(fake_request("deploy", {"env": "prod"}), crash)
    [event] = tg.events.for_thread("t1")
    assert event.kind == "external" and event.status == PENDING

    tg.tasks.pin(TaskSpec("t1", "Deploy."))
    brief = slim_brief(tg.tasks, tg.ledger, "t1", events=tg.events)
    assert "outcome unknown" in brief


def test_expiry_keeps_write_events_so_rewind_still_refuses(harness, tmp_path):
    tg, _ = harness
    repo = git_repo(tmp_path)
    agent = writing_agent(tg, repo, [call("write_file", path="notes.txt", text="v2\n"), AIMessage("done")])
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "w1"}})
    checkpoint = first_checkpoint(agent, "w1")

    report = expire(tg.events, tg.ledger, tg.thoughts, dt.timedelta(seconds=-1))  # everything is "old"
    assert report.events == 0
    assert not resume_risk(agent, "w1", checkpoint, tg.events).safe
    with pytest.raises(ResumeBehindSideEffectsError):
        fork_at(agent, "w1", checkpoint, tg.events)


# ---------------------------------------------------------------- 4. unavailable != unchanged

def test_missing_adapter_is_unverifiable_and_never_a_cache_hit(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    fs = FilesystemAdapter(tmp_path)
    store = InMemoryStore()
    events, blobs = EventLog(store, "p"), BlobStore(tmp_path / "blobs")
    read_set = ResourceRegistry(fs).snapshot(["fs:a.py"])
    event = put_event(events, blobs, "ev_cccccccccccc", b"x = 1\n", read_set=read_set)

    with_ws = ActionCache(store, "p", events, ResourceRegistry(fs), "enforce")
    with_ws.record(event, {"path": "a.py"})
    assert with_ws.lookup("read_file", {"path": "a.py"}) is not None

    no_ws = ActionCache(store, "p", events, ResourceRegistry(), "enforce")  # reopened without a workspace
    assert no_ws.lookup("read_file", {"path": "a.py"}) is None
    verdict = Validator(ResourceRegistry()).event(event)
    assert verdict.status == "unverifiable" and verdict.unavailable == ("fs:a.py",)


@pytest.mark.parametrize("edit,expected", [
    (False, "unverifiable"),  # one key unchecked, the rest match
    (True, "stale"),          # a confirmed change wins over an unchecked key
])
def test_mixed_read_set(tmp_path, edit, expected):
    (tmp_path / "a.py").write_text("x = 1\n")
    reg = ResourceRegistry(FilesystemAdapter(tmp_path))
    read_set = {**reg.snapshot(["fs:a.py"]), "s3:bucket/key": "etag:1"}
    if edit:
        (tmp_path / "a.py").write_text("x = 2\n")
    assert Validator(reg).check("fd_x", read_set).status == expected


def test_all_matching_inputs_are_still_valid(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    reg = ResourceRegistry(FilesystemAdapter(tmp_path))
    assert Validator(reg).check("fd_x", reg.snapshot(["fs:a.py"])).status == "valid"


def test_brief_does_not_list_unverified_findings_as_rechecked(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    store = InMemoryStore()
    events, blobs = EventLog(store, "p"), BlobStore(tmp_path / "blobs")
    ledger, tasks = FindingsLedger(store, "p", events), TaskRegistry(store, "p")
    read_set = ResourceRegistry(FilesystemAdapter(tmp_path)).snapshot(["fs:a.py"])
    put_event(events, blobs, "ev_dddddddddddd", b"x = 1\n", read_set=read_set)
    ledger.add(thread_id="t1", text="a.py sets x to 1", kind="fact", method="read", cited=["ev_dddddddddddd"])
    tasks.pin(TaskSpec("t1", "Check x."))

    brief = slim_brief(tasks, ledger, "t1", validator=Validator(ResourceRegistry()))
    current = brief.split("## Findings from earlier sessions")[1].split("##")[0]
    assert "a.py sets x to 1" not in current
    assert "## Unverified findings" in brief and "could not re-check here: a.py" in brief


# ---------------------------------------------------------------- 5. redaction

@pytest.mark.parametrize("text", [
    "load risk-assessment-framework-v2",
    "desk-booking-service-handler.py",
    '<div class="sk-spinner-double-bounce-wrapper">',
])
def test_hyphenated_words_are_not_keys(text):
    assert redact_text(text).text == text


@pytest.mark.parametrize("text,kind", [
    ("OPENAI_API_KEY=sk-proj-" + "Ab3" * 10, "openai_key"),
    ('key: "sk-' + "x9" * 12 + '"', "openai_key"),
    ("sk-ant-api03-" + "a" * 40, "anthropic_key"),
])
def test_real_looking_keys_are_still_redacted(text, kind):
    assert kind in redact_text(text).kinds


def test_secret_in_tool_args_never_reaches_event_or_blobs(harness):
    tg, _ = harness
    secret = "ghp_" + "S" * 36
    recorder = Recorder(tg.events, tg.blobs, {"call_api": "external"})
    recorder.wrap_tool_call(fake_request("call_api", {"token": secret}, call_id="call_s"),
                            lambda _: ToolMessage(f"used {secret}", tool_call_id="call_s"))
    [event] = tg.events.for_thread("t1")
    assert secret not in json.dumps(event.to_dict())
    assert all(secret.encode() not in tg.blobs.get(ref) for ref in tg.blobs.refs())


# ---------------------------------------------------------------- 7. provenance chains (same function as #3)

def test_expiry_keeps_whole_chain_under_a_retained_finding(tmp_path):
    store = InMemoryStore()
    events, blobs = EventLog(store, "p"), BlobStore(tmp_path / "blobs")
    ledger = FindingsLedger(store, "p", events)
    thoughts = ThoughtLog(store, "p", blobs, events)
    event = put_event(events, blobs, "ev_eeeeeeeeeeee", b"evidence")
    parent = ledger.add(thread_id="t1", text="parent", kind="fact", method="m", cited=[event.event_id])
    time.sleep(0.05)
    boundary = dt.datetime.now(dt.timezone.utc)
    time.sleep(0.05)
    child = ledger.add(thread_id="t1", text="child", kind="fact", method="m", cited=[parent.finding_id])

    expire(events, ledger, thoughts, dt.datetime.now(dt.timezone.utc) - boundary)
    collect_garbage(blobs, events, thoughts, grace=dt.timedelta(0))

    assert ledger.get(child.finding_id) and ledger.get(parent.finding_id)
    assert events.get(event.event_id) is not None and blobs.get(event.output_ref) == b"evidence"
