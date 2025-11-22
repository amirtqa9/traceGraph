from langchain_core.messages import HumanMessage

from spike.agent import build_agent
from tracegraph.resources import FilesystemAdapter, ResourceRegistry
from tracegraph.resume import slim_brief
from tracegraph.tasks import TaskSpec
from tracegraph.validator import Validator

from conftest import ScriptedChatModel, debugging_policy


def run_in(tmp_path, tg, store_agent_ws):
    from langgraph.checkpoint.memory import InMemorySaver

    agent = build_agent(ScriptedChatModel(policy=debugging_policy), store_agent_ws, tg,
                        checkpointer=InMemorySaver(), store=tg.events.store)
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t1"}})
    tg.tasks.pin(TaskSpec("t1", "Find the bug.", ("Never touch tests/.",)))


def workspace(tmp_path):
    from spike.tasks import TASKS, materialize
    return materialize(tmp_path / "ws", TASKS["pagination_overlap"].planted)


def test_finding_inherits_read_set_and_goes_stale(harness, tmp_path):
    tg, _ = harness
    ws = workspace(tmp_path)
    run_in(tmp_path, tg, ws)
    [finding] = tg.ledger.for_threads(["t1"])
    assert set(finding.read_set) == {"fs:shop/pagination.py"}
    validator = Validator(ResourceRegistry(FilesystemAdapter(ws)))
    assert validator.finding(finding).status == "valid"

    (ws / "shop/pagination.py").write_text("# rewritten\n")
    verdict = validator.finding(finding)
    assert verdict.status == "stale" and verdict.changed_locators == ["shop/pagination.py"]


def test_staleness_cascades_through_derived_findings(harness, tmp_path):
    tg, _ = harness
    ws = workspace(tmp_path)
    run_in(tmp_path, tg, ws)
    [base] = tg.ledger.for_threads(["t1"])
    derived = tg.ledger.add(thread_id="t1", text="so search() pages overlap too", kind="fact",
                            method="follows from the paginate finding", cited=[base.finding_id])
    assert derived.sources == (base.finding_id,) and derived.read_set == base.read_set

    validator = Validator(ResourceRegistry(FilesystemAdapter(ws)))
    (ws / "shop/pagination.py").write_text("# rewritten\n")
    assert validator.finding(derived).status == "stale"


def test_unverifiable_without_read_set(harness):
    tg, _ = harness
    validator = Validator(ResourceRegistry())
    finding = tg.ledger.add(thread_id="x", text="t", kind="fact", method="m", cited=[])
    assert validator.finding(finding).status == "unverifiable"


def test_brief_separates_stale_findings_and_adds_startup(harness, tmp_path):
    tg, _ = harness
    ws = workspace(tmp_path)
    run_in(tmp_path, tg, ws)
    validator = Validator(ResourceRegistry(FilesystemAdapter(ws)))
    fresh = slim_brief(tg.tasks, tg.ledger, "t1", validator=validator, events=tg.events)
    assert "## Stale findings" not in fresh and "inputs re-checked and unchanged" in fresh

    (ws / "shop/pagination.py").write_text("# rewritten\n")
    brief = slim_brief(tg.tasks, tg.ledger, "t1", tg.thoughts, working_memory=True,
                       validator=validator, events=tg.events)
    stale_section = brief.split("## Stale findings")[1]
    assert "STALE (fact)" in stale_section and "changed since: shop/pagination.py" in stale_section
    assert "1. Re-check the stale findings first" in brief
    assert "(inputs changed since: shop/pagination.py)" in brief  # working memory flagged too
