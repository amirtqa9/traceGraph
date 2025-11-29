import datetime as dt

from langchain_core.messages import HumanMessage

from spike.agent import build_agent
from spike.tasks import TASKS, materialize
from tracegraph import TraceGraph
from tracegraph.cli import main
from tracegraph.gc import collect_garbage, expire
from tracegraph.policy import DecisionLog, decide
from tracegraph.tasks import TaskSpec

from conftest import ScriptedChatModel, debugging_policy


def populate(tmp_path):
    ws = materialize(tmp_path / "ws", TASKS["pagination_overlap"].planted)
    data = tmp_path / "data"
    tg = TraceGraph.open(data, "shop", workspace=ws)
    tg.tasks.pin(TaskSpec("t1", "Find the overlap bug.", ("Do not read tests/.",)))
    agent = build_agent(ScriptedChatModel(policy=debugging_policy), ws, tg,
                        checkpointer=tg.checkpointer(), store=tg.store)
    agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t1"}})
    tg.close()
    return ws, data


def test_records_persist_across_reopen(tmp_path):
    ws, data = populate(tmp_path)
    tg = TraceGraph.open(data, "shop", workspace=ws)
    assert len(tg.events.for_thread("t1")) == 1
    [finding] = tg.ledger.for_threads(["t1"])
    assert finding.read_set and tg.thoughts.for_threads(["t1"], source="agent")
    assert [v.status for _, v in tg.validate("t1")] == ["valid"]
    assert "## Working memory at cut-off" in tg.brief("t1")
    other = TraceGraph.open(data, "other-project")
    assert other.events.all() == []  # projects are isolated namespaces
    tg.close(), other.close()


def test_cli_commands(tmp_path, capsys):
    ws, data = populate(tmp_path)
    base = ["--data", str(data), "--project", "shop", "--workspace", str(ws)]

    assert main(base + ["ls"]) == 0
    assert "t1: 1 events, 1 findings, 5 thoughts" in capsys.readouterr().out

    tg = TraceGraph.open(data, "shop")
    finding_id = tg.ledger.all()[0].finding_id
    tg.close()
    assert main(base + ["show", finding_id]) == 0
    assert '"kind": "fact"' in capsys.readouterr().out
    assert main(base + ["show", "fd_nope"]) == 1

    assert main(base + ["validate", "--thread", "t1"]) == 0
    assert f"{finding_id} valid" in capsys.readouterr().out
    (ws / "shop/pagination.py").write_text("# changed\n")
    main(base + ["validate"])
    assert "stale  changed: shop/pagination.py" in capsys.readouterr().out

    main(base + ["brief", "t1"])
    assert "## Stale findings" in capsys.readouterr().out

    main(base + ["stats"])
    out = capsys.readouterr().out
    assert '"events": 1' in out and '"read_calls": 1' in out

    main(base + ["bust"])
    assert "cache keys" in capsys.readouterr().out


def test_gc_sweeps_orphans_and_expire_keeps_provenance(tmp_path):
    ws, data = populate(tmp_path)
    tg = TraceGraph.open(data, "shop", workspace=ws)
    orphan = tg.blobs.put(b"nobody references me")
    dry = collect_garbage(tg.blobs, tg.events, tg.thoughts, dry_run=True)
    assert dry.removed == 1 and tg.blobs.has(orphan)
    real = collect_garbage(tg.blobs, tg.events, tg.thoughts)
    assert real.removed == 1 and not tg.blobs.has(orphan)
    assert all(tg.blobs.has(e.output_ref) for e in tg.events.all())

    report = expire(tg.events, tg.ledger, tg.thoughts, dt.timedelta(seconds=-1))
    assert report.findings == 1 and report.thoughts == 5
    assert tg.ledger.all() == [] and tg.events.all() == []
    tg.close()


def test_expire_keeps_events_cited_by_retained_findings(tmp_path):
    ws, data = populate(tmp_path)
    tg = TraceGraph.open(data, "shop", workspace=ws)
    event = tg.events.all()[0]
    # Expire nothing that is younger than an hour: the cited event survives with its finding.
    report = expire(tg.events, tg.ledger, tg.thoughts, dt.timedelta(hours=1))
    assert report.events == 0 and tg.events.get(event.event_id) is not None
    tg.close()


def test_policy_decisions_are_explained_and_logged(tmp_path):
    assert decide(10_000, 200_000).action == "continue"
    d = decide(90_000, 200_000)
    assert d.action == "slim_resume" and "45%" in d.reason
    assert decide(0, 200_000, session_ended=True).action == "slim_resume"
    tg = TraceGraph.open(tmp_path / "d", "p")
    log = DecisionLog(tg.store, "p")
    log.record("t1", d)
    log.record("t1", decide(1, 10))
    assert [x["action"] for x in log.for_thread("t1")] == ["slim_resume", "continue"]
    tg.close()
