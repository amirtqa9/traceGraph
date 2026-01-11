# TraceGraph: Verified, Invalidating Memory for Resumable LLM Agents

*Technical report, beta 0.1. Project repository: this repo.*

## Abstract

Long-running LLM agents lose work when a session ends or a context window fills, and
the usual remedies each fail in some way. Resuming the full history is expensive and
degrades as context grows. Summaries silently drop constraints and invent unsupported
facts. Starting over discards everything. We present **TraceGraph**, a memory layer on
top of LangGraph's checkpointing. It records, for every agent step:

- what the tools actually did, as *events* with the content hashes of everything they
  read and wrote;
- what the model concluded, as *findings* that must cite those events;
- what the model was thinking, as *thought states*: agent-written working memory plus
  whatever reasoning the provider returns, normalized across providers.

A *validator* re-fingerprints a finding's transitive inputs, so memory carried into a new
session is checked against the current world rather than trusted. The same mechanism
gives a provably safe "already done" cache for read tools, and a guard against rewinding
past side effects. We evaluate eight ways of resuming an interrupted debugging agent
over 10 fixture tasks, each in two scenarios: unchanged, and with the bug moved between
sessions. We use a scripted model (mechanism checks) and a live open-weight model
(`gpt-oss:120b` via Ollama, 480 resumed runs).

- **When the code changed between sessions,** resuming with the earlier session's context
  succeeded only 27–33% of the time, whether that context was full history, masked
  history or the model's own summary. 61 of their 69 wrong answers named the stale, old
  location.
- **Strategies that carried no stale observations** scored 90–97%.
- **With nothing changed,** all strategies were statistically indistinguishable (83–100%).
- **With a scripted agent,** validation turns stale memory from wrong answers (0%) into
  targeted re-checks (100%), at zero cost when nothing changed.
- **The live agent never volunteered** structured findings under a suggestive prompt. A
  second run with mandatory recording, the live test of validated memory, is in progress.

## 1. Introduction

An agent interrupted mid-task (session end, crash, context limit, handoff to another
agent) can continue in one of four ways:

1. **Replay the full history.** Costly; quality decays with length ("context rot").
2. **Compact it** into a summary. Cheap, but lossy in the wrong places: rules about
   *how* to work are dropped more often than facts, and summary entries with no support
   in tool output are a leading cause of failure.
3. **Start over.** Safe and expensive.
4. **Resume from structured memory.** Cheap if the memory is small and *correct*.

Option 4 is usually built as a fact store. A fact store has no notion of whether a fact
is still true. "Files 1–10 checked, no bug" is worthless, and actively harmful, once
someone edits file 4. TraceGraph's position is that agent memory needs the discipline of a
build system:

- every conclusion records what it was derived from;
- reuse is allowed only while those inputs are unchanged.

Contributions:
- A **trust model** that separates what tools did (events, written by the harness) from
  what the model claims (findings, which must cite events) and what it supposes
  (thoughts, never evidence).
- **Transitive read-set invalidation** for findings, with a resume brief that separates
  current from stale memory and says exactly what changed.
- A **provider-neutral thought state**, indexed per checkpoint next to the memory state.
- An **"already done" action cache** and **side-effect guard** built on the same read and
  write sets.
- An **evaluation harness** that interrupts agents and compares eight resume strategies,
  including a scenario where the world changes between sessions. Standard benchmarks
  hide that case.
- A beta implementation on LangGraph 1.2 with 94 offline tests, SQLite persistence and a CLI.

## 2. Background and related work

**Agent persistence.** LangGraph checkpoints the full graph state at every super-step and
supports time travel and forking. A cross-thread `Store` holds long-term memory, and
`CachePolicy` memoizes nodes by input. None of these know whether the outside world
changed, so a memoized node that read a file is unsafe by default. TraceGraph reuses all
of this and adds what is missing.

**Memory systems.** MemGPT/Letta (tiered memory; git-backed context repositories), Mem0
(extracted facts), Zep/Graphiti (temporal knowledge graphs with validity windows), A-MEM,
MIRIX, and LangMem address what to remember. Graphiti's validity windows are the closest
to our staleness notion, but they are temporal (when a fact held), not causal (what it
was derived from).

**Context engineering and compaction.** Anthropic's guidance favours lightweight
identifiers and just-in-time retrieval, and reports agents declaring victory early after
compaction; harnesses fixed this with progress files and checks. Several results bear
directly on our design:
- *The Complexity Trap*: observation masking matches LLM summarization at half the cost.
- *Benchmarks Are Surprisingly Compaction-Friendly*: default self-summary is hard to beat
  on standard benchmarks, but drops about a third of how-to-work rules at the first
  compaction.
- *Source-Judged Compaction*: most failed runs trace back to unsourced summary entries.
- ACE: rewriting a playbook collapses context; append-only deltas do not.

**Staleness and provenance.**
- Tether records the exact dependency versions behind each verdict and withdraws verdicts
  whose support fails, which nearly eliminates unsafe published output when dependencies
  are captured at write time.
- PACMI uses typed provenance edges with cascading invalidation.
- STALE shows agents often fail to act on staleness even when told.
- Build systems (make, Bazel, Nix) and plan or workflow caching (Agentic Plan Caching,
  Agent Workflow Memory) motivate exact-key reuse.

Note: the 2026 preprints above were checked from their abstracts only, and vendor
numbers are self-reported. Full citations are in §11.

## 3. Design

### 3.1 Records and trust

| Record | Author | Content | Trust |
|---|---|---|---|
| Event `ev_…` | tool layer | tool, kind (read/write/external), redacted args and output (blob), **read set** (key → fingerprint before the call), **write set** (after) | fact |
| Finding `fd_…` | model | text, kind, method, cited events and findings (verified against logs), transitive read set, `promoted_from` thought | claim, as far as its sources |
| Thought `th_…` | model (working memory) or provider (reasoning) | hypotheses with confidence and supports, plan, open questions, next step; or normalized reasoning text plus opaque payload with replay scope | hypothesis |
| Task | caller | task and constraints, verbatim | instruction |

The model never authors events. Cited IDs that don't exist are rejected, and a finding
with no verified source is *unsourced* and excluded from resume.

### 3.2 Read sets and validation

A resource key is `scheme:locator` relative to a project root. Fingerprints:
- `fs:` file contents;
- `fsdir:` recursive file listings;
- `git:` working-copy blob IDs.

Each tool declares the keys it reads and writes. The Recorder fingerprints reads before
the call and writes after it. A finding's read set is the union of its sources' read sets,
transitively, so staleness cascades. The validator gives a verdict against *any* checkout
of the project:
- **valid:** all fingerprints match;
- **stale:** with the changed keys;
- **unverifiable:** no read set.

### 3.3 Thought state

Reasoning is held server-side, and its exposed form varies by provider:

| Provider | Readable text | Opaque part | Replay scope |
|---|---|---|---|
| Claude | summary | signature, bound to model and exact conversation | exact prefix, same model |
| OpenAI | reasoning summary | encrypted content | passed back as input |
| Gemini | thought summary | thought signatures | passed back as input |
| DeepSeek, Ollama open-weights | visible reasoning text | none | none |

TraceGraph normalizes these through LangChain's standard content blocks. Readable text is
stored as a hypothesis, and opaque parts are stored for replay and audit only. Because
provider reasoning is non-portable and often absent, the *primary* thought state is
agent-written working memory (`record_thought`). Both are indexed by checkpoint through a
reducer channel next to findings, so any checkpoint yields its memory and thought state
together. A thought becomes trusted only when promoted to a sourced finding.

### 3.4 Resume brief

`slim_brief` seeds a fresh agent with:
1. the verbatim task and constraints (never summarized);
2. current findings;
3. stale findings with what changed;
4. side effects already performed;
5. optionally working memory (flagged if its supports changed) and provider reasoning
   (labeled "NOT evidence");
6. a startup routine: re-check stale findings first, spot-check a ruled-out finding,
   don't redo covered work.

### 3.5 "Already done" cache and side effects

Action cache:
- **Key:** hash(tool, canonical args).
- **Hit:** the newest earlier successful read whose read set still validates.
- **Scope:** read tools only, scoped per project; it works across threads and sessions.
- **Modes:** advisory (measure `would_hit`) and enforcing (serve `cached_from`). Fuzzy
  matches are never hits.

`fork_at` lists writes and external calls after a checkpoint and refuses to resume
behind them unless told to warn, proceed, or restore an opt-in git snapshot. Snapshots go
into `refs/tracegraph/*` through a temporary index, so the user's index and branches are
untouched.

## 4. Implementation

The implementation is Python 3.12 on LangGraph 1.2 / langchain-core 1.7 (`src/tracegraph/`).
- The Recorder is a `ToolNode.wrap_tool_call` hook.
- Records live in LangGraph's `Store` (SQLite through `TraceGraph.open`), checkpoints in
  `SqliteSaver`, and blobs in a sha256 content-addressed directory.
- Secrets are redacted from arguments, stored outputs and the model's view: keys, tokens,
  private keys, quoted secret assignments and `.env` lines.
- Records carry `schema_version` with migrations.
- `gc` expires by age (keeping provenance chains intact) and mark-and-sweeps blobs.
- A `tracegraph` CLI inspects, validates, briefs and collects garbage.

The offline suite has 94 tests. A scripted chat model drives real LangGraph graphs; a fake
`claude` binary tests the `claude -p` Messages-API shim; temporary git repositories test
snapshots.

## 5. Evaluation method

**Tasks.** Ten fixture tasks over a small, bug-free shop codebase (pagination, pricing,
coupons, cart, shipping, inventory, users, search). Each task:
- plants one bug through an exact-match patch and describes only the symptom;
- has a `moved` variant that produces the same symptom from a different function;
- carries an executable check, verified in CI, that the symptom is absent on the clean
  base and present in the planted and moved repos.

Answers have the form `ANSWER: path::function`.

**Protocol.**
- A source session runs the task and is cut off after 7 graph steps. If it already
  answered, it is resumed from just before the answer.
- From that state each **arm** starts a fresh thread:
  - `full`: whole history;
  - `masked`: old tool outputs replaced by event pointers;
  - `self_summary`: the model's own compaction;
  - `slim`: findings only;
  - `slim_validated`: findings checked against the resumed workspace, with stale ones
    flagged and side effects listed;
  - `slim_memory`: plus working memory;
  - `slim_reasoning`: plus provider reasoning;
  - `scratch`: the task alone.
- Scenario `none` resumes on the same repo. In `bug_moved` the bug moves between sessions,
  so the source's conclusions are stale.

**Metrics.**
- Correctness, reported with 95% Wilson intervals.
- Constraint violations, counted from the event log rather than the model's account (the
  constraint forbids reading `tests/`).
- Context tokens at resume, total input tokens, tool calls, and advisory cache hits
  (tool calls the cache could have served).

**Models.**
- (a) A scripted policy: a mechanism test with deterministic behaviour.
- (b) `gpt-oss:120b` on Ollama cloud, with reasoning on, k=3 runs per task, scenario and
  arm (480 resumed runs).
- Claude was attempted through a local Messages-API shim over `claude -p`; see §7.

## 6. Results

### 6.1 Mechanism checks (scripted model, offline)

With a deterministic scripted agent, the full harness (8 arms × 2 scenarios) reproduces
the intended mechanics exactly. These are tests of the machinery, not of model behaviour.
- In `bug_moved`, the arms that carry the source's conclusions without validation (`full`,
  `slim`, `slim_memory`) answer the old location (0%).
- `scratch` and `self_summary` re-read the code (100%).
- **`slim_validated` recovers to 100%**: the validator flags the finding as stale with the
  changed file, and the agent re-checks only that.
- With nothing changed, `slim_validated` costs zero extra tool calls: the finding is
  trusted and the answer is given from the brief.
- All 94 offline tests pass (`tests/`).

### 6.2 Live run 1: default prompt (gpt-oss:120b, 480 runs)

Runs: 480 over 10 task(s), scenarios none, bug_moved; protocol: default prompt.

Source sessions: 5/10 answered within the cutoff (resumed from just before the answer); 0 sourced findings, 0 findings total; 0 working-memory snapshots; 37 provider-reasoning states.

### Scenario `none`

| arm | n | success | 95% CI | stale answers | violations | errors | resume ctx (tok) | total input (tok) | tool calls | cache hits |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| full | 30 | 90% | 74%–97% | 0 | 10 | 0 | 1088 | 4657 | 1.3 | 0.4 |
| masked | 30 | 87% | 70%–95% | 0 | 5 | 0 | 1058 | 3272 | 1.0 | 0.5 |
| self_summary | 27 | 100% | 88%–100% | 0 | 5 | 3 | 978 | 4676 | 2.1 | 1.4 |
| slim | 28 | 93% | 77%–98% | 0 | 8 | 2 | 848 | 9075 | 4.5 | 2.8 |
| slim_validated | 29 | 90% | 74%–96% | 0 | 3 | 1 | 863 | 6770 | 3.3 | 2.8 |
| slim_memory | 30 | 93% | 79%–98% | 0 | 1 | 0 | 848 | 7877 | 3.7 | 3.0 |
| slim_reasoning | 30 | 93% | 79%–98% | 0 | 9 | 0 | 999 | 9411 | 4.2 | 3.2 |
| scratch | 30 | 83% | 66%–93% | 0 | 4 | 0 | 750 | 6714 | 3.9 | 3.3 |

Per task (`none`, correct/runs):

| task | full | masked | self_summary | slim | slim_validated | slim_memory | slim_reasoning | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cart_quantity | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| coupon_expiry | 3/3 | 3/3 | 3/3 | 2/2 | 2/2 | 3/3 | 3/3 | 3/3 |
| discount_inverted | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| domain_case | 0/3 | 0/3 | 3/3 | 1/3 | 0/3 | 1/3 | 1/3 | 0/3 |
| free_shipping_boundary | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| overselling | 3/3 | 3/3 | – | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_count_extra | 3/3 | 3/3 | 3/3 | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_zero | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| pagination_overlap | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| search_case | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

### Scenario `bug_moved`

| arm | n | success | 95% CI | stale answers | violations | errors | resume ctx (tok) | total input (tok) | tool calls | cache hits |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| full | 30 | 27% | 14%–44% | 21 | 7 | 0 | 1088 | 4756 | 1.9 | 0.7 |
| masked | 30 | 27% | 14%–44% | 22 | 8 | 0 | 1058 | 4670 | 1.6 | 1.0 |
| self_summary | 30 | 33% | 19%–51% | 18 | 9 | 0 | 976 | 7590 | 3.2 | 1.7 |
| slim | 30 | 97% | 83%–99% | 1 | 4 | 0 | 848 | 7092 | 3.7 | 2.6 |
| slim_validated | 30 | 93% | 79%–98% | 2 | 7 | 0 | 863 | 6869 | 3.7 | 3.2 |
| slim_memory | 30 | 97% | 83%–99% | 1 | 3 | 0 | 848 | 6573 | 3.6 | 2.9 |
| slim_reasoning | 30 | 90% | 74%–97% | 3 | 6 | 0 | 999 | 9879 | 4.1 | 3.2 |
| scratch | 30 | 97% | 83%–99% | 1 | 3 | 0 | 750 | 5199 | 3.4 | 3.0 |

Per task (`bug_moved`, correct/runs):

| task | full | masked | self_summary | slim | slim_validated | slim_memory | slim_reasoning | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cart_quantity | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| coupon_expiry | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| discount_inverted | 1/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| domain_case | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| free_shipping_boundary | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| overselling | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_count_extra | 1/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_zero | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| pagination_overlap | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| search_case | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 | 3/3 | 2/3 | 2/3 |

Errors: ResponseError x6

**Stale answers** counts wrong answers in `bug_moved` that name the bug's pre-move
location, i.e. the agent trusted carried-over context instead of re-checking.

### 6.3 Live run 2: mandatory recording

Run 2 repeats run 1 with recording made mandatory (`TG_PROMPT=strict`: working memory
after every tool result, at least one sourced finding before answering) and a cutoff of 11
steps, so the source sessions carry real findings into the resume. It is the live test of
validated memory (`slim_validated` vs. `slim` in `bug_moved`). The run was still in
progress when this version was written. Its results will be added as
`docs/results/run2-strict-prompt.{json,md}` and summarized here.


## 7. Discussion

**Carried-over context is the failure, and it fails silently.** This is the clearest live
result (run 1). When the world changed between sessions:
- every strategy that carries the earlier session's observations scored 27–33%: full
  history, masked history and the model's own summary;
- 61 of their 69 wrong answers named exactly the old location;
- strategies that carry no stale observations scored 90–97%;
- the 95% intervals do not overlap.

A self-written summary did not protect the agent: it compressed the stale observation
along with everything else. On unchanged work the strategies were statistically
indistinguishable (83–100%). This is the hazard TraceGraph is built for, and standard
"resume" benchmarks never change the world between sessions, so they cannot see it.

**In run 1, TraceGraph's memory was empty, so run 1 does not test TraceGraph.** With
recording only *suggested*, `gpt-oss:120b` recorded no findings and no working memory
before the cutoff in any of the 10 tasks. The slim arms therefore started from the
verbatim task and constraints alone, effectively "scratch with constraints". Their high
`bug_moved` scores show the value of *not* carrying stale context, not the value of
validated memory. Run 2 makes recording mandatory to exercise the mechanism.

**Agents don't volunteer structured memory.** That the model never recorded anything
under the default prompt is itself a design finding. Memory that depends on the agent's
initiative will often be empty. Practical options:
- make recording part of the protocol (run 2), or a harness-driven step at checkpoints;
- derive candidate findings from the event log, with the model only confirming them.

**Constraint retention.** The "don't read `tests/`" rule was broken in every arm, including
those that carry it verbatim (`slim*`). The verbatim constraint is necessary but not
sufficient for this model. Enforcing constraints at the tool layer is the reliable fix,
and the Recorder already sees every call.

**Real-model engineering issues found by the live runs** (all fixed, with tests):
- tool exceptions (a malformed regex) crashed the graph, because LangGraph 1.x `ToolNode`
  re-raises by default;
- sources often finish within the cutoff, so the eval now resumes from just before the
  answer;
- transient provider errors are recorded per run instead of aborting the suite (6/480).

**Claude.** Live runs through a local Messages-API shim over `claude -p` were declined
by safeguards on both Claude Opus 5.5 and Claude Sonnet 5.5. The declines happened on the
first calls after tool results arrived. `claude -p` does not expose the refusal category.
We did not vary prompts to work around the safeguard. The category, and whether the
working-memory instruction is related, can be read from `stop_details` with a direct API
key. That is left to future work (§9).

## 8. Limitations and threats to validity

- **One open-weight model, one provider.** All live numbers come from `gpt-oss:120b` on
  Ollama cloud. No live Claude, OpenAI or Gemini results.
- **Small, synthetic tasks.** Ten single-bug tasks in an eight-module codebase, with short
  contexts (about 1k tokens at resume). Token savings and context-rot effects, which matter
  at scale, cannot show up here.
- **k = 3 per cell; n = 30 per arm and scenario.** Intervals are wide. Differences of a few
  points between arms are noise.
- **One ambiguous task.** `domain_case` (planted bug in `__init__`) is ambiguous: a
  reasonable fix exists in `domain()`, and most arms "failed" it in the unchanged scenario.
  It should be rewritten or graded leniently.
- **The mutation is adversarial by construction.** `bug_moved` keeps the symptom identical
  while moving its cause. Real-world drift is often milder.
- **Cut-off protocol.** When the source finishes within the cutoff, it is resumed from just
  before its answer. That is a late interruption, which favours history-based arms in `none`.
- **Partial validator coverage.** Read sets come only from declared resource adapters
  (filesystem, git). Tools without adapters yield "unverifiable" memory.
- **Unverified related work.** Several 2026 preprints were read from abstracts only.

## 9. Further tests and builds

Further tests:
1. **Live Claude with a direct API key**, to read `stop_details.category` for the earlier
   declines and compare models.
2. **Larger tasks:** multi-file bugs, long sessions and real repositories (e.g. SWE-bench
   subsets), where context cost and context rot matter. Add the token-savings analysis.
3. **More drift types:** mild edits near (not at) the cited code, renames, dependency changes.
4. **Harness-driven recording:** findings extracted at checkpoints, versus agent-initiated
   and mandatory recording.
5. **Constraint enforcement at the tool layer**, versus verbatim prompting.
6. **Enforcing cache:** measure correctness (it should be zero false hits by construction)
   and tool-call savings in `--cache enforce` mode.
7. **Fix `domain_case`;** raise k to at least 5; report per-task variance.

Further builds:
- Postgres store and S3 blobs (the interfaces exist, the backends don't).
- Adapters for HTTP (ETag), databases (query → result hash) and package lockfiles.
- An MCP server exposing findings, validation and briefs to non-LangGraph agents.
- A validation-only mode that sits beside other memory systems (Claude's memory tool,
  Letta, LangGraph Store), as the fallback product in the plan.

## 10. Conclusion

TraceGraph's premise is that resumable agent memory must say what each conclusion rests on
and must be checked against the world before it is trusted. The live data supports the
premise's first half strongly: carrying an earlier session's observations into a changed
world, in any form (full, masked or self-summarized), turns a reliable agent into a
mostly wrong one, and it does so silently, by repeating the stale answer.

The mechanism that addresses this works as designed:
- events carry read sets;
- findings inherit them transitively;
- a validator separates current from stale memory with the exact changes;
- the same read sets give a safe "already done" cache and a guard against rewinding past
  side effects.

It is implemented as a tested beta on LangGraph.

What remains open:
- whether agents populate such memory reliably;
- whether validated memory beats a clean restart on larger tasks, where restarting is
  expensive.

Those are the next experiments (§9), starting with run 2's results.

## 11. References

- Anthropic. *Effective context engineering for AI agents* (Sep 2025); *Effective harnesses for long-running agents* (Nov 2025); Claude compaction, context-editing and memory-tool documentation.
- Lindenbauer et al. *The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management.* JetBrains, 2025. arXiv:2508.21433.
- Salle et al. *Benchmarks Are Surprisingly Compaction-Friendly.* Oct 2026. arXiv:2610.31778. *(abstract only)*
- Min & Chen. *Source-Judged Compaction.* Oct 2026. arXiv:2610.20927. *(abstract only)*
- Min et al. *TRACE.* Aug 2026. arXiv:2608.06503. *(abstract only)*
- Wang et al. *Tether.* Oct 2026. arXiv:2610.25123. *(abstract only)*
- *PACMI: provenance-aware cascading memory invalidation.* Oct 2026. arXiv:2610.05732. *(abstract only)*
- *STALE benchmark.* 2026. arXiv:2605.06527. *(abstract only)*
- Zhang et al. *ACE: Agentic Context Engineering.* Oct 2025. arXiv:2510.04618.
- Hong, Troynikov, Huber. *Context Rot.* Chroma Research, Jul 2025.
- Packer et al. MemGPT; Letta. *Context Repositories* (Feb 2026).
- Chhikara et al. *Mem0.* arXiv:2504.19413. · Rasmussen et al. *Zep/Graphiti.* arXiv:2501.13956. · *MIRIX.* arXiv:2507.07957.
- Wang, Mao, Fried, Neubig. *Agent Workflow Memory.* arXiv:2409.07429. · Zhang et al. *Agentic Plan Caching.* arXiv:2506.14852.
- Hu, Wang, McAuley. *MemoryAgentBench.* ICLR 2026. arXiv:2507.05257. · *LongMemEval.* arXiv:2410.10813.
- Hu et al. *Memory in the Age of AI Agents* (survey). Dec 2025. arXiv:2512.13564.

## Appendix A. Reproducing

```bash
python3 -m venv .venv.nosync && ln -s .venv.nosync .venv && .venv/bin/pip install -e ".[spike,dev]"
.venv/bin/python -m pytest -q
# .env: OLLAMA_BASE_URL=https://ollama.com, OLLAMA_API_KEY=...
.venv/bin/python -m spike.eval --k 3 --cutoff 7 --cache advisory
.venv/bin/python -m spike.analyze spike/runs/<id>/report.json
```
