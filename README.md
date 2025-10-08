# TraceGraph

Verified, invalidating agent memory on top of LangGraph. LangGraph saves and
restores *everything*; TraceGraph records *what was learned*, *what it rests on*,
and (from M1 on) *whether it is still true*, so a fresh agent can resume from a
small, trusted state instead of the full history.

The current design is **[PLAN.md](PLAN.md)**. The original 12 design docs are kept
unchanged, for reference only, in [`archive/deepseek-v1/`](archive/deepseek-v1/);
they are superseded and are not the spec.

## Status: M0 spike (go/no-go)

M0 asks one question before anything else is built: does resuming from findings
beat the cheap baselines? See PLAN.md §5.

| Built | Not yet |
|---|---|
| Recorder on ToolNode `wrap_tool_call` → events with citable `ev_` IDs | Read/write sets (M1) |
| Content-addressed blob store for tool outputs | Staleness validation (M3) |
| `record_finding` tool; cited sources checked against the event log | Self-summary arm; cross-session mutation arm |
| Verbatim task + constraints pinned per thread, with resume lineage | Action cache (M4), side-effect safety (M5) |
| Thought state per agent step (thinking summaries + signed blocks), indexed by checkpoint next to findings | Persistent backends (all in-memory for now) |
| Eval arms: full / masked / slim / slim_thoughts / scratch, graded from the event log | |

Two kinds of record are kept apart throughout: **events** are written by the
tool layer and the model cannot author them; **findings** are written by the model and
are trusted only as far as the events they cite.

## Layout

```
src/tracegraph/      library: canonical hashing, CAS, events/Recorder, findings, thoughts, tasks, resume, index
spike/agent.py       minimal ReAct loop on StateGraph + ToolNode with TraceGraph attached
spike/eval.py        M0 eval harness
spike/fixtures/shop  fixture repo with one planted bug (shop/pagination.py::paginate, end index off by one)
tests/               offline tests with a scripted model; no API key needed
archive/deepseek-v1/ original design docs (superseded)
```

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[spike,dev]"
.venv/bin/python -m pytest -q                      # offline
ANTHROPIC_API_KEY=... .venv/bin/python -m spike.eval --k 3 --cutoff 7
```

The eval defaults to `claude-opus-5-5` at effort `medium` (`TG_MODEL`, `TG_EFFORT`
or `--model` to change). Reports go to `spike/runs/<timestamp>/report.json`.

Eval notes:
- The fixture's `tests/` directory gives the answer away. The task forbids reading it,
  and violations are counted from recorded events, not from what the model says.
- Reasoning stays on the server. Signed thinking blocks are bound to the model and the exact
  conversation that produced them, so copied histories (full / masked arms) have them stripped.
  They are not lost: every agent step's blocks and readable summaries are kept as a thought
  state indexed by checkpoint (PLAN.md §2.8). The `slim_thoughts` arm measures whether
  offering those summaries on resume helps.
