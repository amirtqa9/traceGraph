from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from spike.agent import build_agent
from tracegraph.cache import action_key, cache_stats

from conftest import ScriptedChatModel, call


def run(tg, ws, mode, steps, thread="c1"):
    it = iter(steps)
    agent = build_agent(ScriptedChatModel(policy=lambda m: next(it)()), ws, tg,
                        checkpointer=InMemorySaver(), store=tg.events.store, cache_mode=mode)
    out = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": thread}})
    return out, tg.events.for_thread(thread)


def read():
    return call("read_file", path="a.py")


def test_advisory_flags_but_still_runs(harness, tmp_path):
    tg, _ = harness
    (tmp_path / "a.py").write_text("x = 1\n")
    _, events = run(tg, tmp_path, "advisory", [read, read, lambda: AIMessage("done")])
    first, second = events
    assert second.would_hit == first.event_id and second.cached_from is None
    assert cache_stats(events).would_hit == 1


def test_enforce_serves_unchanged_reads(harness, tmp_path):
    tg, _ = harness
    (tmp_path / "a.py").write_text("x = 1\n")
    out, events = run(tg, tmp_path, "enforce", [read, read, lambda: AIMessage("done")])
    first, second = events
    assert second.cached_from == first.event_id
    assert second.output_ref == first.output_ref
    outputs = [m.content.split("\n\n[event:")[0] for m in out["messages"] if isinstance(m, ToolMessage)]
    assert outputs == ["x = 1\n", "x = 1\n"]


def test_changed_input_is_never_a_hit(harness, tmp_path):
    tg, _ = harness
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")

    def edit_then_read():
        path.write_text("x = 2\n")  # the world changes between the two calls
        return read()

    out, events = run(tg, tmp_path, "enforce", [read, edit_then_read, lambda: AIMessage("done")])
    assert events[1].cached_from is None and events[1].would_hit is None
    assert "x = 2" in [m for m in out["messages"] if isinstance(m, ToolMessage)][-1].content


def test_hits_work_across_threads_within_a_project(harness, tmp_path):
    tg, _ = harness
    (tmp_path / "a.py").write_text("x = 1\n")
    _, first = run(tg, tmp_path, "enforce", [read, lambda: AIMessage("done")], thread="s1")
    _, second = run(tg, tmp_path, "enforce", [read, lambda: AIMessage("done")], thread="s2")
    assert second[0].cached_from == first[0].event_id


def test_non_read_tools_and_errors_are_not_cached(harness, tmp_path):
    tg, _ = harness
    missing = lambda: call("read_file", path="missing.py")  # noqa: E731
    _, events = run(tg, tmp_path, "enforce", [missing, missing, lambda: AIMessage("done")])
    assert all(e.status == "error" and e.cached_from is None for e in events)
    assert action_key("read_file", {"path": "a"}) != action_key("read_file", {"path": "b"})
