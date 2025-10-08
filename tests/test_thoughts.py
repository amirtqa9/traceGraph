from langchain_core.messages import AIMessage, HumanMessage

from spike import eval as tg_eval
from tracegraph.index import node_state
from tracegraph.resume import consistent_prefix, slim_brief, strip_reasoning
from tracegraph.tasks import TaskSpec
from tracegraph.thoughts import thinking_blocks


def run(make, thread="t1"):
    agent = make()
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": thread}})
    return agent


def test_every_thinking_step_gets_a_thought_state(harness):
    tg, make = harness
    run(make)
    thoughts = tg.thoughts.for_threads(["t1"])
    assert len(thoughts) == 3  # read, record, answer
    assert "slicing bug" in thoughts[0].summary
    assert all(t.checkpoint_id and t.replay_scope == "exact-prefix-same-model" for t in thoughts)
    # Raw blocks are kept byte-for-byte, signature included, for exact-prefix replay.
    [block] = tg.thoughts.raw_blocks(thoughts[0])
    assert block["signature"].startswith("sig-")


def test_checkpoint_indexes_memory_and_thought_state_together(harness):
    tg, make = harness
    agent = run(make)
    history = list(agent.get_state_history({"configurable": {"thread_id": "t1"}}))
    final = history[0].config["configurable"]["checkpoint_id"]
    early = history[-2].config["configurable"]["checkpoint_id"]  # right after the input

    at_end = node_state(agent, "t1", final, ledger=tg.ledger, thoughts=tg.thoughts, events=tg.events)
    at_start = node_state(agent, "t1", early, ledger=tg.ledger, thoughts=tg.thoughts, events=tg.events)
    assert len(at_end.findings) == 1 and len(at_end.thoughts) == 3
    assert at_start.findings == () and at_start.thoughts == ()


def test_thought_summaries_are_opt_in_and_labeled(harness):
    tg, make = harness
    run(make)
    tg.tasks.pin(TaskSpec("t1", "Find the bug."))
    plain = slim_brief(tg.tasks, tg.ledger, "t1")
    with_thoughts = slim_brief(tg.tasks, tg.ledger, "t1", thoughts=tg.thoughts)
    assert "slicing bug" not in plain
    assert "slicing bug" in with_thoughts
    assert "NOT evidence" in with_thoughts


def test_copied_histories_carry_no_signed_blocks(harness):
    tg, make = harness
    agent = run(make)
    messages = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert any(thinking_blocks(m) for m in messages if isinstance(m, AIMessage))
    copied = strip_reasoning(consistent_prefix(messages))
    assert not any(thinking_blocks(m) for m in copied if isinstance(m, AIMessage))
    # ...but nothing is lost: the thought log still has every block.
    assert len(tg.thoughts.for_threads(["t1"])) == 3


def test_slim_thoughts_arm_runs(tmp_path):
    from conftest import ScriptedChatModel, debugging_policy

    report = tg_eval.run_eval(
        lambda: ScriptedChatModel(policy=debugging_policy),
        k=1, cutoff=4, run_limit=20, data_dir=tmp_path, arms=("slim_thoughts",),
    )
    assert report["source"]["thought_states"] == 2
    assert report["summary"]["slim_thoughts"]["success_rate"] == 1.0
