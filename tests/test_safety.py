import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tracegraph.events import Recorder
from tracegraph.resources import FilesystemAdapter, ResourceRegistry, filesystem_tool_resources
from tracegraph.resume import slim_brief
from tracegraph.safety import GitSnapshotter, ResumeBehindSideEffectsError, fork_at, resume_risk
from tracegraph.tasks import TaskSpec

from conftest import ScriptedChatModel, call


def git_repo(path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "notes.txt").write_text("v1\n")
    return path


def writing_agent(tg, repo, steps, snapshotter=None):
    @tool
    def write_file(path: str, text: str) -> str:
        """Write a file."""
        (repo / path).write_text(text)
        return "ok"

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        return "sent"

    fs = FilesystemAdapter(repo)
    recorder = Recorder(tg.events, tg.blobs, {"write_file": "write", "send_email": "external"},
                        resources=filesystem_tool_resources(fs), registry=ResourceRegistry(fs),
                        snapshotter=snapshotter)
    it = iter(steps)
    model = ScriptedChatModel(policy=lambda m: next(it))
    g = StateGraph(MessagesState)
    g.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]})
    g.add_node("tools", ToolNode([write_file, send_email], wrap_tool_call=recorder.wrap_tool_call))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=InMemorySaver())


def first_checkpoint(agent, thread):
    history = list(agent.get_state_history({"configurable": {"thread_id": thread}}))
    return history[-2].config["configurable"]["checkpoint_id"]  # after the input, before any tool


def test_write_sets_and_resume_risk(harness, tmp_path):
    tg, _ = harness
    repo = git_repo(tmp_path)
    agent = writing_agent(tg, repo, [call("write_file", path="notes.txt", text="v2\n"),
                                     call("send_email", to="ops@example.com"), AIMessage("done")])
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "w1"}})
    write, email = tg.events.for_thread("w1")
    assert set(write.write_set) == {"fs:notes.txt"}

    risk = resume_risk(agent, "w1", first_checkpoint(agent, "w1"), tg.events)
    assert [e.event_id for e in risk.writes_after] == [write.event_id]
    assert [e.event_id for e in risk.external_after] == [email.event_id]
    assert not risk.safe and not risk.restorable  # the email can't be unsent

    with pytest.raises(ResumeBehindSideEffectsError, match="send_email"):
        fork_at(agent, "w1", first_checkpoint(agent, "w1"), tg.events)
    with pytest.warns(UserWarning):
        config, _ = fork_at(agent, "w1", first_checkpoint(agent, "w1"), tg.events, on_side_effects="warn")
    assert config["configurable"]["checkpoint_id"]

    tg.tasks.pin(TaskSpec("w1", "Update notes."))
    brief = slim_brief(tg.tasks, tg.ledger, "w1", events=tg.events)
    assert "## Side effects already performed" in brief and "external: send_email" in brief


def test_snapshot_and_restore_behind_a_write(harness, tmp_path):
    tg, _ = harness
    repo = git_repo(tmp_path)
    (repo / "scratch.txt").write_text("keep me?\n")
    snap = GitSnapshotter(repo)
    agent = writing_agent(tg, repo, [call("write_file", path="notes.txt", text="v2\n"),
                                     call("write_file", path="new.txt", text="created\n"), AIMessage("done")],
                          snapshotter=snap)
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "w2"}})
    first_write = tg.events.for_thread("w2")[0]
    assert first_write.pre_snapshot
    assert (repo / "notes.txt").read_text() == "v2\n" and (repo / "new.txt").exists()

    checkpoint = first_checkpoint(agent, "w2")
    _, risk = fork_at(agent, "w2", checkpoint, tg.events, on_side_effects="restore", snapshotter=snap)
    assert risk.restorable
    assert (repo / "notes.txt").read_text() == "v1\n"
    assert not (repo / "new.txt").exists()
    assert (repo / "scratch.txt").read_text() == "keep me?\n"  # untracked file present at snapshot time
    # The user's own git state is untouched: no commits on a branch, nothing staged.
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.count("A ") == 0
    refs = subprocess.run(["git", "for-each-ref", "refs/tracegraph/"], cwd=repo, capture_output=True, text=True).stdout
    assert refs.count("refs/tracegraph/") == 2


def test_safe_resume_when_nothing_happened_after(harness, tmp_path):
    tg, _ = harness
    agent = writing_agent(tg, git_repo(tmp_path), [AIMessage("nothing to do")])
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "w3"}})
    _, risk = fork_at(agent, "w3", first_checkpoint(agent, "w3"), tg.events)
    assert risk.safe
