# TraceGraph (beta)

Verified, invalidating agent memory on top of LangGraph.

LangGraph saves and restores *everything*. TraceGraph records **what an agent learned,
what each conclusion rests on, and whether it is still true**. A fresh agent can then
resume from a small, validated state instead of the full history, and an "already
done" check can reuse earlier work only when its inputs are unchanged.

- **Spec:** [PLAN.md](PLAN.md)
- **Findings:** [docs/paper.md](docs/paper.md)
- **Session log and next steps:** [HANDOFF.md](HANDOFF.md)
- **Original design docs** (superseded, kept for reference): [`archive/deepseek-v1/`](archive/deepseek-v1/)

## What it does

| Record | Written by | Trusted as |
|---|---|---|
| **Event** (`ev_…`) | the tool layer (Recorder on `ToolNode.wrap_tool_call`) | fact: what ran, its redacted output, its **read set** and **write set** |
| **Finding** (`fd_…`) | the model (`record_finding`) | claim, only as far as the events and findings it cites; inherits their read sets transitively |
| **Thought** (`th_…`) | the model (`record_thought`: working memory) or the provider (reasoning, normalized) | hypothesis, never evidence |
| **Task** | the caller | verbatim task and constraints, replayed on every resume |

On top of these records:
- **Validator:** a finding is valid only while every resource in its read set still fingerprints the same. Stale findings are reported with exactly what changed.
- **Slim resume** (`TraceGraph.brief`): verbatim task and constraints, then current findings, then stale findings and what changed, side effects already performed, working memory, and a startup routine.
- **Action cache:** a read tool's earlier output is reused only on the same tool and arguments *and* an unchanged read set. Advisory (measure) or enforcing (serve).
- **Side-effect safety:** `fork_at` refuses to rewind past writes or external calls unless told to warn, proceed, or restore an opt-in git snapshot taken before each write.
- **Hygiene:** secrets are redacted before anything is stored; schema versioning; age-based expiry that keeps provenance chains; blob garbage collection; projects isolated by namespace.

## Use

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tracegraph import TraceGraph
from tracegraph.resources import filesystem_tool_resources

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tg_findings: Annotated[list[str], operator.add]   # memory state
    tg_thoughts: Annotated[list[str], operator.add]   # thought state

tg = TraceGraph.open(".tracegraph", project_id="my-repo", workspace=".")
recorder = tg.recorder({"read_file": "read", "grep": "read", "write_file": "write"},
                       filesystem_tool_resources(tg.fs))
tools = my_tools + tg.tools()          # + record_finding, record_thought, recall_event
graph = StateGraph(State)
# ... agent node calls model.bind_tools(tools) ...
graph.add_node("tools", ToolNode(tools, wrap_tool_call=recorder.wrap_tool_call, handle_tool_errors=True))
app = graph.compile(checkpointer=tg.checkpointer(), store=tg.store)

# Later, in a new session: a fresh agent starts from the validated brief.
seed = tg.brief("thread-1")
```

`spike/agent.py` is a complete, small example (a ReAct loop with TraceGraph attached).

### CLI

```bash
tracegraph --data .tracegraph --project my-repo ls
tracegraph --data .tracegraph --project my-repo --workspace . validate
tracegraph --data .tracegraph --project my-repo --workspace . brief thread-1
tracegraph --data .tracegraph --project my-repo stats
tracegraph --data .tracegraph --project my-repo gc --older-than-days 30
```

## Install and test

```bash
# This repo sits in iCloud-synced ~/Desktop: keep the venv in a *.nosync folder (see HANDOFF §3)
python3 -m venv .venv.nosync && ln -s .venv.nosync .venv
.venv/bin/pip install -e ".[spike,dev]"
.venv/bin/python -m pytest -q          # offline, scripted model, no keys needed
```

## Evaluation

`spike/eval.py` interrupts an agent mid-task and resumes it eight ways (full history,
masked, self-summary, slim, slim + validation, slim + working memory, slim + provider
reasoning, scratch). It runs over 10 fixture tasks (`spike/tasks.py`), each validated by an
executable symptom check, in two scenarios: `none`, and `bug_moved`, where the bug moves
between sessions so carried-over memory goes stale.

```bash
# Model backends (make_model): Ollama (OLLAMA_BASE_URL + OLLAMA_API_KEY in .env, default gpt-oss:120b)
# or Anthropic (ANTHROPIC_API_KEY, or the local claude -p shim via TG_BASE_URL).
.venv/bin/python -m spike.eval --k 3                     # full suite: 10 tasks x 2 scenarios x 8 arms x 3
.venv/bin/python -m spike.analyze spike/runs/<id>/report.json
```

Results and their limits are in [docs/paper.md](docs/paper.md).

## Layout

```
src/tracegraph/   canonical, cas, events (Recorder), resources, redact, findings, thoughts,
                  tasks, validator, cache, safety, resume, index, policy, gc, schema, core, cli
spike/            agent.py, eval.py, analyze.py, tasks.py, claude_p_api.py, fixtures/shop
tests/            offline suite (scripted model, fake claude binary, temp git repos)
docs/paper.md     design, evaluation and findings
```
