# TraceGraph (beta)

Verified, invalidating agent memory on top of LangGraph.

LangGraph saves and restores *everything*. TraceGraph records **what an agent learned,
what each conclusion rests on, and whether it is still true**. A fresh agent can then
resume from a small, validated state instead of the full history, and an "already
done" check can reuse earlier work only when its inputs are unchanged.

Design, evaluation and findings: [docs/paper.md](docs/paper.md).

## What it does

| Record | Written by | Trusted as |
|---|---|---|
| **Event** (`ev_…`) | the tool layer (Recorder on `ToolNode.wrap_tool_call`) | fact: what ran, its redacted output, its **read set** and **write set** |
| **Finding** (`fd_…`) | the model (`record_finding`) | claim, only as far as the events and findings it cites; inherits their read sets transitively |
| **Thought** (`th_…`) | the model (`record_thought`: working memory) or the provider (reasoning, normalized) | hypothesis, never evidence |
| **Task** | the caller | verbatim task and constraints, replayed on every resume |

On top of these records:
- **Validator:** a finding is valid only while every resource in its read set can be re-checked and still fingerprints the same. A confirmed change makes it stale, reported with exactly what changed. A resource that cannot be checked here (no adapter, no workspace) makes it unverifiable: missing evidence is never treated as a match.
- **Slim resume** (`TraceGraph.brief`): verbatim task and constraints, then current findings, then findings whose inputs could not be re-checked, then stale findings and what changed, side effects already performed (pending writes and external calls are flagged as outcome unknown), working memory, and a startup routine.
- **Action cache:** a read tool's earlier output is reused only on the same tool and arguments *and* a read set whose every entry could be re-checked and still fingerprints the same. An uncheckable key is a miss, never a hit. Advisory (measure) or enforcing (serve).
- **Recall** (`recall_event`): the recorded output of an earlier tool call, not a re-run, with a line saying whether its inputs are unchanged, changed, or uncheckable. Long outputs are paged; a missing blob is reported instead of raised.
- **Side-effect safety:** write and external tools are recorded as pending before they run, so a crash still leaves a record that the side effect may have happened; rewind and resume treat that record as if it did. `fork_at` refuses to rewind past writes or external calls unless told to warn, proceed, or restore an opt-in git snapshot taken before each write.
- **Hygiene:** secrets matching known patterns are redacted from event records, stored outputs and the tool results the model sees; schema versioning; age-based expiry that keeps provenance chains and side-effect records; blob garbage collection that marks every project sharing the data directory and keeps blobs younger than a grace period (default 15 minutes); projects isolated by namespace and by checkpoint database.

> **Redaction boundary.** Redaction is pattern-based and covers what the Recorder stores. It
> does not touch the model's own message: a secret the model puts in a tool call's arguments
> stays in the checkpointed history (LangGraph needs the original call to resume it). Keep
> secrets out of prompts and tool arguments where you can (resolve them inside the tool), and
> protect checkpoints at rest with `tg.checkpointer(serde=EncryptedSerializer.from_pycryptodome_aes())`
> (from `langgraph.checkpoint.serde.encrypted`; needs `pycryptodome` and `LANGGRAPH_AES_KEY`).

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
tracegraph --data .tracegraph --project my-repo gc --older-than-days 30 --grace-minutes 15
```

## Install and test

```bash
# On an iCloud-synced folder (e.g. ~/Desktop), keep the venv in a *.nosync folder: iCloud sets the
# macOS hidden flag on the editable-install .pth file, and Python 3.12 then skips it.
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
