from langchain_core.messages import AIMessage, HumanMessage

from spike import eval as tg_eval
from tracegraph.index import node_state
from tracegraph.resume import consistent_prefix, slim_brief, strip_reasoning
from tracegraph.tasks import TaskSpec
from tracegraph.thoughts import normalize_reasoning

from conftest import ScriptedChatModel, debugging_policy


def run(make, thread="t1"):
    agent = make()
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": thread}})
    return agent


# --- provider adapters (shapes as LangChain receives them from each integration) ---

def test_normalize_claude_thinking_and_redacted_blocks():
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "summary text", "signature": "SIG"},
            {"type": "redacted_thinking", "data": "ENC"},
            {"type": "text", "text": "answer"},
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    r = normalize_reasoning(msg)
    assert (r.source, r.text_kind, r.text, r.replay_scope) == (
        "anthropic", "summary", "summary text", "exact-prefix-same-model")
    assert r.has_opaque and len(r.raw["content_blocks"]) == 2


def test_normalize_openai_reasoning_item():
    msg = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "summary text"}],
             "encrypted_content": "ENC"},
            {"type": "text", "text": "answer"},
        ],
        response_metadata={"model_provider": "openai"},
    )
    r = normalize_reasoning(msg)
    assert (r.source, r.text_kind, r.text, r.replay_scope) == ("openai", "summary", "summary text", "provider-input")
    assert r.has_opaque


def test_normalize_deepseek_reasoning_content():
    msg = AIMessage(content="answer", additional_kwargs={"reasoning_content": "visible text"},
                    response_metadata={"model_provider": "deepseek"})
    r = normalize_reasoning(msg)
    assert (r.source, r.text_kind, r.text, r.replay_scope) == ("deepseek", "visible", "visible text", "none")
    assert not r.has_opaque
    assert r.raw["additional_kwargs"] == {"reasoning_content": "visible text"}


def test_normalize_gemini_thought_part():
    msg = AIMessage(content=[{"type": "thinking", "thinking": "summary text"}, {"type": "text", "text": "a"}],
                    response_metadata={"model_provider": "google_genai"})
    r = normalize_reasoning(msg)
    assert (r.source, r.text_kind, r.replay_scope) == ("google_genai", "summary", "provider-input")


def test_no_reasoning_means_no_thought():
    assert normalize_reasoning(AIMessage(content="plain answer")) is None


# --- capture, indexing, promotion ---

def test_provider_reasoning_is_captured_per_step(harness):
    tg, make = harness
    run(make)
    provider = tg.thoughts.for_threads(["t1"], source="provider")
    assert len(provider) == 4  # read, thought, finding, answer
    assert provider[0].source == "anthropic" and "slicing bug" in provider[0].text
    assert all(t.checkpoint_id and t.replay_scope == "exact-prefix-same-model" for t in provider)
    [block] = tg.thoughts.raw(provider[0])["content_blocks"]
    assert block["signature"].startswith("sig-")  # kept byte-for-byte for exact-prefix replay


def test_working_memory_is_structured_and_evidence_checked(harness):
    tg, make = harness
    run(make)
    [wm] = tg.thoughts.for_threads(["t1"], source="agent")
    event_id = tg.events.for_thread("t1")[0].event_id
    assert wm.replay_scope == "portable" and wm.text_kind == "structured"
    assert wm.working_memory["plan"][0] == "confirm the slice bounds"
    assert wm.supports == (event_id,)
    assert wm.rejected_supports == ("ev_ffffffffffff",)
    assert "hypothesis (80%)" in wm.text


def test_thought_promoted_to_finding(harness):
    tg, make = harness
    run(make)
    [wm] = tg.thoughts.for_threads(["t1"], source="agent")
    [finding] = tg.ledger.for_threads(["t1"])
    assert finding.promoted_from == wm.thought_id


def test_checkpoint_indexes_memory_and_thought_state_together(harness):
    tg, make = harness
    agent = run(make)
    history = list(agent.get_state_history({"configurable": {"thread_id": "t1"}}))
    final = history[0].config["configurable"]["checkpoint_id"]
    early = history[-2].config["configurable"]["checkpoint_id"]  # right after the input

    at_end = node_state(agent, "t1", final, ledger=tg.ledger, thoughts=tg.thoughts, events=tg.events)
    at_start = node_state(agent, "t1", early, ledger=tg.ledger, thoughts=tg.thoughts, events=tg.events)
    assert len(at_end.findings) == 1 and len(at_end.thoughts) == 5  # 4 provider + 1 working memory
    assert at_start.findings == () and at_start.thoughts == ()


# --- resume ---

def test_slim_brief_thought_sections_are_opt_in_and_labeled(harness):
    tg, make = harness
    run(make)
    tg.tasks.pin(TaskSpec("t1", "Find the bug."))
    plain = slim_brief(tg.tasks, tg.ledger, "t1", tg.thoughts)
    memory = slim_brief(tg.tasks, tg.ledger, "t1", tg.thoughts, working_memory=True)
    reasoning = slim_brief(tg.tasks, tg.ledger, "t1", tg.thoughts, provider_reasoning=True)
    assert "Working memory" not in plain and "slicing bug" not in plain
    assert "## Working memory at cut-off" in memory and "next step: record the finding" in memory
    assert "slicing bug" in reasoning and "NOT evidence" in reasoning


def test_copied_histories_carry_no_signed_blocks(harness):
    tg, make = harness
    agent = run(make)
    messages = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    copied = strip_reasoning(consistent_prefix(messages))
    assert not any(normalize_reasoning(m) for m in copied if isinstance(m, AIMessage))
    assert len(tg.thoughts.for_threads(["t1"], source="provider")) == 4  # nothing lost


def test_thought_arms_run(tmp_path):
    report = tg_eval.run_eval(
        lambda: ScriptedChatModel(policy=debugging_policy),
        k=1, cutoff=6, run_limit=20, data_dir=tmp_path, arms=("slim_memory", "slim_reasoning"),
    )
    assert report["source"]["working_memory_states"] == 1
    assert report["source"]["provider_reasoning_states"] == 3
    assert all(s["success_rate"] == 1.0 for s in report["summary"].values())
