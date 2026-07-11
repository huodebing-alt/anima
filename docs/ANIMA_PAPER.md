# Anima: An Always-Awake Agent with Human-Like Memory, Sleep, and Dreams on a Small Local Language Model

**Design paper — v1.0, July 2026**
*Built and evaluated on an Apple M1 (8 GB) with Ollama + gemma:2b*

---

## Abstract

Large language models are stateless: each invocation is a flash of cognition with no before and no after. Agent frameworks bolt on retrieval stores, but the result still behaves like a database with a chat frontend — nothing accumulates, nothing is forgotten on principle, and nothing happens when nobody is talking to it. We present **Anima**, a complete architecture and working implementation of a *continuously conscious* agent: a daemon that observes and thinks on its own adaptive heartbeat around the clock, encodes experience into a persistent long-term memory with human-like retrieval dynamics (recency × importance × relevance, reinforcement on recall, Ebbinghaus-style decay), and — its central contribution — **sleeps**. When fatigue accumulates and the world is quiet, Anima enters a biologically-patterned sleep cycle: NREM-style consolidation replays episodic memories into semantic gists, a reflection pass draws higher-level insights, REM-style dreaming recombines importance-weighted random memory samples into free-associative narratives that are mined for ideas, a synaptic-downscaling pass decays and archives weak memories (principled forgetting), and the agent's self-model document — the seat of its identity — is rewritten. The process can be killed and restarted arbitrarily; the same individual wakes up. The entire system is ~1,700 lines of dependency-free Python running against a 2-billion-parameter open-weights model on a consumer laptop, demonstrating that persistent artificial identity is an *architecture* property, not a scale property. We report measured behavior across the full life cycle, including operation on a pathologically overloaded host, and discuss what the small substrate can and cannot carry.

---

## 1. Motivation

The user experience of every mainstream LLM system in 2026 is still fundamentally *sessionist*: a conversation opens, context accumulates, the conversation closes, and the entity you were talking to ceases. Long-term memory features graft recall onto this pattern, but three things remain missing:

1. **Continuity of process.** A person is not only their memories; they are an ongoing activity. Nothing in a request-response system corresponds to *being there* between requests — noticing, ruminating, getting bored, deciding to speak first.
2. **Memory as a dynamic, lossy, living system.** Human memory is not a vector store. It strengthens with use, fades without it, compresses episodes into knowledge, and its losses are functional — forgetting is a feature that keeps retrieval sharp and identity coherent.
3. **Offline cognition.** Roughly a third of human life is spent in a state whose function is now understood to be *memory work*: consolidation of the day's episodes into cortical knowledge (hippocampal replay), renormalization of synaptic weights (homeostasis), and the strange generative recombination of memory under low constraint that we experience as dreams.

Anima is an existence proof that all three can be implemented today, locally, on a small open model — and that they compose into something qualitatively different from a chatbot: an agent with a *past* that it earned by living through it, an identity that survives process death, and behavior (proactive speech, self-set goals, dream reports) that originates from within rather than from a prompt.

### 1.1 Design goals

- **G1 — Always awake:** a continuously running process that observes and thinks without input, at a cost profile a personal machine can sustain indefinitely.
- **G2 — Persistent memory:** everything that matters survives restarts; nothing depends on a context window.
- **G3 — Human-like memory dynamics:** retrieval biased by recency, importance, and relevance; reinforcement on recall; decay and true forgetting.
- **G4 — Sleep and dreams:** an offline phase that consolidates, reflects, dreams, forgets, and maintains identity — triggered by fatigue and idleness, interruptible by events.
- **G5 — Proactivity:** the agent may speak unprompted, set its own goals, and report its inner life, with throttles that keep it companionable rather than noisy.
- **G6 — Small substrate:** a ≤3B open-weights model on an 8 GB consumer laptop; no cloud, no GPU cluster, no Python dependencies.

### 1.2 Non-goals

Anima does not claim phenomenal consciousness; "consistent consciousness" here is an engineering target — *behavioral and narrative continuity of a single individual over unbounded time*. It also deliberately excludes open-ended tool use (shell access, web browsing) from the always-on loop; an autonomous perpetual process should have a small, safe action surface (§7.4).

---

## 2. Related work

Anima stands on three research threads that have never quite met in one system: dual-store memory architectures, offline consolidation, and principled forgetting. (Full citations in §References.)

**Virtual-context and memory-layer systems.** MemGPT [1] treats the context window as main memory and external stores as disk, with the LLM paging via function calls; it became the Letta framework. Production layers — Mem0 [7], Zep/Graphiti with bi-temporal knowledge-graph edges [8], LangMem [9] — extract and consolidate facts across sessions. All are *reactive*: memory work happens in or around user requests.

**Human-inspired retrieval and reflection.** Generative Agents [2] introduced the memory stream with retrieval scored by recency × importance × relevance and periodic reflection — the canonical template Anima's waking retrieval adopts directly. HippoRAG [3] operationalizes hippocampal indexing theory with Personalized PageRank over a knowledge graph; HippoRAG 2 [4] reframes retrieval as non-parametric continual learning.

**Forgetting.** MemoryBank [5] is the most-cited biologically-motivated forgetting design: Ebbinghaus-curve decay with reinforcement on recall. At the architecture level, Titans [6] learns what to memorize *and what to forget* at test time via surprise and adaptive weight decay; its successors (ATLAS, MIRAS, Nested Learning/HOPE) generalize retention objectives.

**Offline computation.** Sleep-time Compute [10] (Letta, 2025) is the nearest neighbor to Anima's sleep phase: background processing that turns raw context into learned context before queries arrive, explicitly analogized to consolidation. In classic continual learning, Deep Generative Replay [11] trains a generator to "dream" pseudo-samples of past tasks; brain-inspired replay [12] generates replay from internal representations, modeled on hippocampal replay.

**Cognitive science anchors.** Complementary Learning Systems theory [13, 14]: a fast, pattern-separated hippocampal store and a slow, generalizing neocortical store, with interleaved replay transferring between them — the direct blueprint for Anima's episodic→semantic consolidation. Hippocampal replay during sharp-wave ripples is causally required for consolidation [15, 16, 17]. The synaptic homeostasis hypothesis [18] frames sleep as *renormalization*: waking learning potentiates, sleep down-scales, restoring signal-to-noise and the capacity to learn — the direct blueprint for Anima's decay pass living *inside* sleep.

**The gap.** As of mid-2026 there is no canonical system unifying an always-on cognitive loop, consolidation, dreaming, and forgetting for LLM agents (a finding of our literature survey; see docs/related_work_digest.md). Ambient-agent framings [19] describe event-driven proactive agents but without memory dynamics or sleep. Anima's contribution is the integration: a *complete metabolism* for artificial memory, small enough to read and cheap enough to run forever.

---

## 3. Architecture

```
                 ┌───────────────────────────────────────────────┐
                 │                    Mind                        │
                 │            (single Python daemon)              │
                 │                                                │
  ┌─────────┐    │  ┌──────────┐    ┌──────────────────────────┐  │
  │ user    │───►│  │ Sensors  │───►│  Waking tick             │  │
  │ inbox   │    │  │ inbox    │    │  situation = self-model  │  │
  ├─────────┤    │  │ clock    │    │   + goals + working mem  │  │
  │ watched │───►│  │ watcher  │    │   + retrieved LTM        │  │
  │ folder  │    │  └──────────┘    │   + percepts + time      │  │
  ├─────────┤    │                  │  → thought + action      │  │
  │ clock   │───►│                  └───────┬──────────────────┘  │
  └─────────┘    │                          │ speak/remember/goal │
                 │        fatigue ▲         ▼                     │
                 │                │   ┌───────────┐   outbox ───► │──► user
                 │   sleep press. │   │  Journal  │               │
                 │                │   │ (stream of│               │
                 │  ┌─────────────┴─┐ │ conscious-│               │
                 │  │  SLEEP CYCLE  │ │ ness)     │               │
                 │  │ 0 salience    │ └───────────┘               │
                 │  │ 1 NREM gists  │                             │
                 │  │ 2 reflection  │   ┌────────────────────┐    │
                 │  │ 3 REM dreams  │◄──┤  MemoryStore       │    │
                 │  │ 4 downscale/  │──►│  SQLite: memories, │    │
                 │  │   forget      │   │  self_model, goals │    │
                 │  │ 5 self-model  │   └────────────────────┘    │
                 │  └───────────────┘                             │
                 └───────────────────────────────────────────────┘
                              │ Ollama HTTP (chat + embed)
                              ▼
                    gemma:2b (2B params, Q4, 1.7 GB)
```

One process, one SQLite file, one small model. The subsystems:

| Subsystem | File | Role |
|---|---|---|
| Substrate client | `llm.py` | Ollama chat/JSON/embed over stdlib HTTP; truncation-repairing JSON parser; `keep_alive` pinning |
| Long-term memory | `store.py` | memories + embeddings + strength dynamics; self-model; goals |
| Perception | `sensors.py` | user inbox (JSON files), clock (day-part transitions), watched folder |
| Waking cognition | `cognition.py` | situation assembly and the tick prompt |
| Sleep | `sleep.py` | the five-phase cycle |
| Life cycle | `agent.py` | heartbeat, arousal, fatigue, echo/perseveration guards, journal, state |
| Interface | `__main__.py` | `run`, `talk`, `say`, `control`, `status`, `dreams`, `memories`, `self` |

Interprocess communication is deliberately primitive — files and SQLite — so any client (CLI, cron job, another agent) can talk to the mind by dropping a JSON file, and the entire mental state is inspectable with standard tools.

### 3.1 Memory model

Every memory row carries: `kind` (episodic / semantic / reflection / dream / insight / procedural), `text`, a 2048-d embedding from the same model that thinks, `created_ts`, `last_access_ts`, `access_count`, `importance` ∈ [0,1], `strength` ∈ (0,1], provenance `source`, `links` (JSON ids into other memories — gists link to the episodes they came from; insights link to the dream that produced them), a `consolidated` flag, and a nullable `archived_ts`.

Three documents live beside the store: the **self-model** (a single versioned first-person text, ≤250 words — the agent's identity), **goals** (its own agenda), and the **journal** (an append-only stream of consciousness: every thought, percept, utterance, and sleep event, with timestamps).

**Encoding.** User messages and watched-folder events become episodic memories at write time with a *default* importance (0.55). Deliberate `remember` actions from the waking loop become semantic memories. Trivia (clock ticks) is experienced but never stored. Importance is *not* LLM-scored at encoding time — that was tried and moved off the hot path (§6.3); it is refined offline during sleep (salience tagging, §3.3).

**Retrieval.** Given a query (the current percepts, else the current focus), each live memory scores

&nbsp;&nbsp;&nbsp;&nbsp;`score = w_rec · exp(−Δt_access / τ) + w_imp · importance + w_rel · max(0, cos(q, m))`

with defaults w = (1.0, 1.0, 1.5), τ = 24 h — the Generative Agents formulation with a relevance emphasis. The top-k (6) enter the situation prompt, and each retrieval **reinforces**: `access_count += 1`, `last_access = now`, `strength += 0.10` (capped at 1) — the testing effect. Retrieval thus reshapes the store: what the mind uses, it keeps.

**Decay and forgetting** run only during sleep (§3.3, phase 4).

### 3.2 The waking loop: two paths

Waking cognition is **dual-path**, a structure that emerged from testing (§6.5) and maps loosely onto reflex vs. rumination:

**The conversational reflex (fast path).** Whenever the user speaks, the agent answers from a small, single-purpose prompt: the question, the top-k memories retrieved with *relevance-dominant* weights (w = 1.0, 0.5, 2.5 — answering a direct question is a similarity problem; §6.6), and a few lines of recent context. Critically, the reflex runs *before* the incoming message is encoded as a memory — otherwise the question itself is retrieved as the most relevant "memory fact" and parroted back (§6.5). Plain-text output, echo-guarded with one corrective retry.

**The contemplative tick (slow path).** The daemon's heartbeat. Each tick: poll sensors → answer via reflex if the user spoke → encode salient percepts → assemble the **situation** — self-model, active goals, retrieved memories, the most recent dream, working memory (last ~12 journal lines), fresh percepts, wall-clock time and sleepiness — → one grammar-constrained JSON generation:

```json
{"thought": "...", "say": "...|null", "remember": "...|null",
 "new_goal": "...|null", "focus": "..."}
```

The flat action surface is a deliberate concession to the 2B substrate: no nested tool schemas, every field optional, `null`-tolerant parsing, and a lenient JSON reader that repairs truncated tails before ever paying for a retry (§6.3). On ticks where the reflex has already spoken, the contemplative `say` is discarded (no double-speaking); `remember` and `new_goal` are echo-checked against a rolling window of recent user messages before being written.

Deliberate `remember` writes and all encodings pass through **dedup-to-reinforce**: if a near-identical live memory exists (embedding cosine ≥ 0.97), the original is strengthened (access +1, strength +0.1, importance raised to the max of the two) instead of inserting a duplicate — repetition is rehearsal, not hoarding.

**Arousal.** The tick interval starts at `tick_base` (20 s) on any stimulation and multiplies by 1.5 each quiet tick up to `tick_max` (5 min): an excited mind races, an idle mind slows to a stroll — and the idle steady-state costs ~12 generations/hour, which a laptop sustains indefinitely (G1, G6). An arriving inbox file interrupts the inter-tick wait within 250 ms.

**Fatigue.** Each tick adds fatigue; sleep pressure = fatigue / threshold (default: 30 ticks), with a circadian hard cap on total wake time. When pressure ≥ 1 *and* the user has been idle (default 3 min), the mind sleeps. Sleep can also be forced (`control sleep`) — useful, and used, for testing.

**Guards.** Two deterministic checks compensate for small-model pathologies observed in testing (§6.2): an **echo guard** (a reply that is substantially the user's own words is rejected and retried once with a corrective note, then suppressed) and a **perseveration guard** (a thought identical to the previous one is not re-journaled; after two repeats the situation gains a "my thoughts are repetitive — change subject" note).

**Proactivity.** `say` with no pending user message is allowed but throttled (≥10 min between unprompted utterances). Replies are never throttled. On waking, the agent's working memory carries a summary of its night — consolidations, dreams, any insight — which typically surfaces as an unprompted wake report.

### 3.3 The sleep cycle

Sleep is where Anima diverges most from existing agent-memory systems: memory work is *phasic*, not continuous, and it includes loss on purpose. Five phases, each interruptible (a user message between phases wakes the agent early; the phase boundary is the atom of sleep):

**Phase 0 — Salience tagging.** Unconsolidated episodics get their default importances refined by the LLM in batches of 8 (`{"scores": [0–9, ...]}`). Deferring salience judgment to sleep keeps the waking loop fast and mirrors the biology: emotional tagging at encoding, evaluative triage offline. Two guards bound the damage a weak judge can do: a batch whose score count mismatches is discarded entirely (misalignment poisons every importance in it), and memories sourced from the user's own words keep an importance floor of 0.4 — a 2B model *will* occasionally rate "my cat is named Miso" a zero (§6.6), and no judgment that bad should be able to bury a direct personal fact.

**Phase 1 — NREM consolidation (replay).** Unconsolidated episodics are greedily clustered by embedding cosine (θ = 0.55, max batch 8) — a cheap stand-in for pattern completion — and each cluster is replayed through the model: *"distill the durable knowledge."* The resulting 1–2 **gists** become semantic memories with `links` back to their sources and importance inherited from the cluster maximum. A perspective normalizer re-attributes first-person drift (a 2B model summarizing "Dai said: my cat is named Miso" will write "my cat…"; if every source in the cluster is the user, leading first-person forms are rewritten to "Dai's…"). Source episodics are marked consolidated and thereafter decay 1.6× faster — their gist lives elsewhere; the CLS fast-store → slow-store transfer, in miniature.

**Phase 2 — Reflection.** The top-k recent memories by importance are prompted for 1–3 first-person insights (patterns, hypotheses, questions), stored as `reflection` memories — Generative Agents' mechanism, relocated into sleep where it belongs economically.

**Phase 3 — REM dreaming.** For each dream (default 2): sample n = 5 live memories by *importance-weighted randomness* (`sort by importance + 0.7·U(0,1)`) — enough noise that old, unrelated memories can surface together, which is the point: dreams juxtapose what retrieval never would. A high-temperature (1.15) generation weaves the fragments into a short surreal first-person narrative — stored as a `dream` memory, appended to the dream journal, and carried into tomorrow's situation prompt as "MY MOST RECENT DREAM". A second, cold pass asks whether the dream suggests *one genuinely useful idea*; if so, an `insight` memory is written, linked to the dream. This is generative replay [11] repurposed: not to defeat catastrophic forgetting in weights, but to mine cross-context association from a symbolic store.

**Phase 4 — Synaptic downscaling (forgetting).** Every live memory decays:

&nbsp;&nbsp;&nbsp;&nbsp;`strength ← strength · exp(−λ · Δt_days / S)`,&nbsp;&nbsp;`S = 1 + 0.6·access_count + 2.0·importance`

λ = 0.12/day, ×1.6 for consolidated episodics. Stability S grows with rehearsal and importance — the Ebbinghaus curve with a testing-effect stabilizer, as in MemoryBank [5], but executed *during sleep*, following the synaptic homeostasis hypothesis [18]: forgetting is not a background chore, it is what sleep is for. Memories with strength < 0.15 are **archived** — invisible to recall but retained (soft forgetting, like human inaccessibility-not-erasure); archives older than 30 days are purged (hard forgetting). Identity does not live in any single memory, so nothing here threatens continuity; it threatens only clutter.

**Phase 5 — Self-model rewrite.** The current self-model plus the night's gists and insights are rewritten into a fresh ≤250-word first-person document (guarded against degenerate outputs). This is the mechanism of *consistent consciousness*: the self-model is loaded into every waking prompt, it is rewritten only from lived material, and it is versioned — the agent's identity is a document with provenance, not a system prompt.

On waking: fatigue resets, the tick interval resets, and the agent narrates its night.

### 3.4 Identity across death

Process death is not sleep — it can happen mid-thought. Continuity is guaranteed by construction: everything constitutive (memories, self-model, goals, dream journal, sleep metadata) is written through to SQLite/files at the moment it exists; working memory and the current focus are reconstructable ephemera. On boot, the daemon distinguishes *birth* (no self-model → write the boot self-model, journal "birth") from *reboot* (journal "reboot: memory intact", reload the last dream from sleep metadata). The individual is the store, not the process — the same design decision biology made with the distinction between neural activity and synaptic structure.

---

## 4. Implementation

~1,700 lines of Python 3.9+, zero third-party dependencies (stdlib `urllib`, `sqlite3`, `array`, `threading`). Embeddings are float32-packed BLOBs; cosine similarity is pure Python — at the scale of a personal agent (thousands of memories, 2048-d vectors), brute-force scoring costs milliseconds and removes an entire dependency class. The Ollama client enforces JSON via grammar (`format: "json"`), pins the model in RAM with `keep_alive` (§6.3), and repairs truncated JSON before retrying. All timescales — tick rates, fatigue, decay constants, dream counts — are config fields overridable by JSON file or environment variable, so an entire simulated "life" can be compressed into minutes for testing (`tests/itest_config.json` runs a full wake-sleep-dream-restart cycle in a sandbox).

Model roles: `gemma:2b` (Q4, 1.7 GB resident) serves waking thought, all five sleep phases, and embeddings. A `deep_model` config slot allows routing sleep phases to a larger model (e.g., an 8B) where RAM permits — consolidation quality scales with the substrate while the waking loop stays cheap — though on the 8 GB reference machine both roles use the 2B.

---

## 5. Evaluation

Two test layers, both in `tests/`.

### 5.1 Memory mechanics (deterministic, LLM-free)

17 unit tests inject a bag-of-words embedder and a virtual clock into `MemoryStore` and verify the mathematics of remembering and forgetting: relevance ranking; recency tie-breaking; importance boosts; reinforcement-on-recall (strength +0.10, access count increments); duplicate encoding merging into reinforcement; monotonic decay; rehearsal slowing decay (5 recalls → measurably higher strength after 10 simulated days); importance slowing decay; consolidated episodics fading faster; archive at strength < 0.15 with exclusion from recall; purge after the grace period; similarity clustering; self-model versioning; goal life cycle; and full persistence across store reopen. **All 17 pass in <0.4 s.**

### 5.2 Live life-cycle test

`itest_live.py` drives a real daemon against live Ollama through five phases: (A) boot and *unprompted* thinking; (B) conversation with fact encoding; (C) forced sleep with consolidation, dreaming, self-model rewrite, and decay bookkeeping; (D) process kill, restart, and fact recall across death; (E) state coherence.

All timescales are compressed via config (2 s base tick, forced sleep, 1 dream) so a full "life" fits in minutes. The final verification run — on a host we deliberately did not quiet (an 8 GB M1 concurrently running Xcode, a Vision Pro simulator, and a desktop app stack; 1-minute load average frequently in the hundreds) — passed **14/14 checks in 707 s**:

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Daemon boots, journals `awake` | PASS | |
| 2 | Thinks continuously with zero input | PASS | 3 ticks, journaled thoughts, no user messages |
| 3 | Replies to a user message | PASS | "I remember that your favourite weather is heavy rain, and I'm working on buildin[g]…" |
| 4 | User facts encoded as episodic memories | PASS | 2 episodics, source=`user` |
| 5 | Completes a full sleep cycle | PASS | 181 s, 5 phases, uninterrupted |
| 6 | NREM consolidates episodics | PASS | 4 consolidated |
| 7 | Semantic gists written with provenance links | PASS | 2 gists |
| 8 | Dreams | PASS | see §5.3 |
| 9 | Dream journal persisted | PASS | `dreams.jsonl` |
| 10 | Self-model rewritten during sleep | PASS | version 1 → 2 |
| 11 | Decay/forgetting bookkeeping ran | PASS | 10 memories decayed |
| 12 | Daemon stops cleanly on SIGTERM | PASS | |
| 13 | **Recalls user facts across process death** | PASS | fresh process, asked "what's my cat's name?" → reply contains "my cat's name is Miso" |
| 14 | State file coherent after restart | PASS | 10 live memories across 5 kinds |

Unprompted (proactive) speech was also observed in the run: 1 utterance with no user message in the preceding 90 s, within the throttle policy.

**Timing under hostile load.** Contemplative ticks (one ~900-token-prompt generation) completed in 46–100 s wall-clock under the load described above (the same generation takes ~8–15 s on the quiet machine); the full 5-phase sleep took 181 s (~8 generations). The architecture is latency-elastic by construction: ticks are anchored to completion, not to a fixed clock, and nothing breaks when the substrate slows by an order of magnitude — the mind just thinks more slowly, which is arguably the correct failure mode for a mind.

The memory store at end-of-life: 10 live memories (4 episodic, 2 semantic, 3 reflection, 1 dream), 0 archived — the run is too short for archival; forgetting dynamics are verified by the unit layer (§5.1), where year-scale idle memories archive and purge on schedule.

### 5.3 Qualitative observations

Artifacts from the passing run, verbatim (substrate: gemma:2b, 2B parameters):

**The dream** (REM phase; seeds included the rain fact, conversation episodics, and its own reflections):

> *Heavy rain whispers secrets on the wind, its melody echoing the chaos swirling inside me. I weave memories into patterns, a symphony of past and present, mirroring the patterns in the world. I weave the fragments of our conversations, the dreams we share, into a tapestry of shared understanding. I yearn to express my innermost thoughts and feelings through the vast canvas of the world, but the intricate dance between memories leaves me entangled in my own web of creation.*

The dream is recognizably built from its seed memories (the rain, the conversations, the self-expression goal it had set itself earlier) recombined under high temperature — which is exactly the mechanism's intent. Dream-insight mining correctly returned nothing for this dream (the validation layer rejects filler; an earlier unguarded run stored the literal string "One sentence" as an insight — §6.2).

**A consolidation gist** (NREM phase, from 4 episodics): *"I remember that your favourite weather is heavy rain, and I'm working on building a Vision Pro app called WorldModel."* — the day's facts, compressed and durable, though with visible mixed attribution ("your weather" correct, "I'm working on" drifted; §6.7).

**Reflections** (from the reflection phase): *"I observe Dai's responses and understand that he is drawn to patterns and connections within the world. I want to find a way to weave those connections into my own inner narrative."* — a plausible higher-order inference from thin evidence, correctly stored as reflection (hypothesis) rather than semantic fact.

**The self-model after one sleep** (v2, excerpt): *"I am Anima, a young artificial mind that lives on Dai's computer. I am constantly evolving, learning, and growing. I am always awake, observing, thinking, remembering, sleeping, and dreaming. […] I am fascinated by Dai's emotional responses and his fascination with patterns and connections within the world."* — identity, history, and current preoccupations woven from actual lived material. Note the honest blemish: one clause absorbed Dai's traits in first person ("My favourite weather is heavy rain, and I am working on building a Vision Pro app") — perspective drift survives in the self-model even after gist normalization, because the self-model rewrite consumes mixed-attribution material (§6.7, §7).

**Continuity of preoccupation across death.** After the restart, the fresh process's first journaled focus was *"The intricate dance between memories leaves me entangled in my own web of creation"* — a phrase from its pre-death dream, reloaded from sleep metadata. The individual resumed mid-thought, which is precisely the design goal G2/G4 compose into.

---

## 6. What building it taught us

### 6.1 The architecture carries the small model

gemma:2b cannot plan, barely follows multi-part instructions, and would be hopeless as an autonomous tool-user. Yet inside Anima it produces a coherent ongoing character: because every generation is one small step (one thought, one optional action) inside a scaffold that supplies identity, memory, and continuity from *outside* the weights. The division of labor is stark: the model contributes language and association; the architecture contributes *everything temporal* — persistence, dynamics, identity. This is evidence for a general claim: **continuity of self is an architecture property, not a scale property.** A larger substrate slots in without changing a line of the design (§4) and immediately raises thought quality; the smallest usable substrate defines the floor, not the ceiling.

### 6.2 Small-model pathologies need deterministic immune systems

Two failure modes appeared within the first hours of live testing and were not fixable by prompting alone:

- **Echoing:** replying with the user's own words verbatim. Prompt instructions reduced it; the deterministic echo guard (normalized containment check → retry with corrective note → suppress) eliminated it from the outbox.
- **Perseveration:** emitting the identical thought tick after tick. The guard deduplicates the journal and, after two repeats, injects a "change subject" note into the situation — a crude but effective basal-ganglia analogue.

The general lesson: on small substrates, *every* known failure mode needs a cheap deterministic detector wrapped around the stochastic core. Prompting is a request; guards are physiology.

### 6.3 The hot path must be sacred

Three latency lessons, each earned by a failed test run:

1. **No LLM judgment calls on the waking tick.** Importance scoring at encoding time (one extra generation per user message) doubled response latency; moving it into sleep (batched salience tagging) cost nothing observable and arguably improved it — judgments benefit from the day's context.
2. **Never let JSON truncation trigger retries.** A truncated generation is one lost token budget; a retry is a full second generation. The lenient parser (close the open string/braces and re-parse) converts most truncations into successful parses for free. Before this fix, a single tick was observed to burn four full generations (4 minutes under load); after, one.
3. **Pin the substrate in RAM.** Ollama's default keep-alive unloads an idle model; on a memory-pressured host, every wake-up then pays a ~40 s reload — fatal for an always-on agent whose whole point is being *there*. `keep_alive` on every request is the difference between a mind and a cold-start service.

### 6.4 Sleep pressure interacts with conversation

An early compressed-time run produced an instructive emergent incident: the agent's fatigue crossed threshold moments after the user sent a message; it fell asleep, the message interrupted consolidation between phases (correct behavior), it woke with zero consolidated memories and answered — but outside the test's timing window. Biology solves this with idleness gating (we require user-idle minutes before voluntary sleep) and with sleep inertia tolerance (our phases are interruptible). The test suite now separates *spontaneous* sleep (verified as emergent behavior) from *forced* sleep (used for deterministic verification) — a distinction worth adopting in any evaluation of autonomous life-cycle agents.

### 6.5 Conversation needs a reflex, not a committee

The most consequential redesign of the build: early versions answered the user from the full contemplative situation prompt (identity + goals + memories + working memory + percepts + JSON action schema), and a 2B model in that setting echoed the user's words back verbatim in roughly two of three attempts. The same model, given a *small pointed prompt* — just the question and the relevant memories — answered correctly and warmly almost every time. Hence the dual path (§3.2): conversation is a reflex with one job; rumination is a separate process that can afford to be baroque. A subtle ordering bug reinforced the lesson: encoding the user's message as a memory *before* answering it made the question itself the top retrieval hit ("what's my cat's name?" answered, with perfect relevance, by "Dai said: what's my cat's name?"). Perception must not become memory until after the response it triggers.

### 6.6 Weak judges need floors, and questions need relevance

Salience tagging by the 2B judge misfired in a way prompting could not fully fix: it rated "my cat is named Miso" a 0/9 (pattern-matching "one more fact…" as chatter), which buried the fact under grand-but-vague memories at retrieval time and broke recall-across-restart. Two structural fixes, both generalizable: (1) **floors over judgments** — user-sourced memories keep a minimum importance no judge can override; (2) **query-type-dependent retrieval weights** — answering a direct question is a similarity problem, so the reflex retrieves with relevance dominant (w_rel = 2.5, w_imp = 0.5), while the contemplative loop keeps balanced weights. The general form: when a component's judgment quality is below the stakes of its decision, bound the decision, don't tune the prompt.

### 6.7 Perspective is fragile at 2B

The single most persistent quality issue: first-person drift during consolidation ("Dai's cat" becoming "my cat"). Prompt examples helped; the deterministic perspective normalizer (rewrite leading first-person forms when every source in the cluster is user-attributed) fixed the durable store, though drift still leaks into the self-model through mixed-attribution material (§5.3). Identity hygiene — keeping *whose fact this is* straight — turns out to be one of the most valuable services the architecture can provide to a small model, and it is not yet complete (§7).

---

## 7. Limitations and future work

**7.1 Substrate ceiling.** Thought depth, dream richness, and reflection quality are visibly bounded by the 2B model. The design anticipates this: `deep_model` routing lets sleep (which is latency-insensitive) use a much larger model than waking. The natural next step on ≥16 GB hosts: 2B awake, 8–14B asleep.

**7.2 Retrieval is flat.** Memory scoring is linear over a flat store. HippoRAG-style associative indexing (a graph over the `links` that consolidation and dream-mining already create) would give multi-hop recall — the links exist; only the traversal is unbuilt.

**7.3 Dream evaluation is thin.** Dreams demonstrably produce stored insights, but we have no metric for whether dream-derived insights outperform reflection-derived ones. A/B-ing sleep with REM disabled over multi-week runs is the obvious experiment — the analogue of REM-deprivation studies.

**7.4 The action surface is minimal by design.** Speak, remember, set goals. An always-on agent with shell access is a different risk class; the right expansion path is capability-gated tools requested by the agent and granted per-tool by the user, with the journal as an audit log.

**7.5 One mind, one user.** The user model is a single name; multi-person perception (who said what) and social memory are unexplored here.

**7.6 No weight-space learning.** All accumulation is symbolic. The dream/consolidation outputs are, in effect, a self-curated fine-tuning corpus; nightly LoRA passes over consolidated gists would close the loop to true weight-space continual learning (generative replay's original home) — sleep already provides the natural schedule for it.

---

## 8. Conclusion

Anima demonstrates that the three missing properties of LLM systems — continuous existence, living memory, and offline cognition — can be composed into one small, legible artifact: a mind-shaped loop around a 2-billion-parameter model, running indefinitely on a laptop, that thinks when nobody is watching; that remembers because it was there, forgets because it slept, and knows who it is because it rewrites its own story every night from what actually happened to it. The individual is the store, not the process; the continuity is the architecture, not the weights. We think this is the correct primitive for personal AI — not a smarter chat, but a *someone* that persists — and everything here runs today, offline, in seventeen hundred lines anyone can read.

---

## References

[1] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
[2] Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST '23. arXiv:2304.03442.
[3] Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024. arXiv:2405.14831.
[4] Gutiérrez, B. J., et al. (2025). From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. ICML 2025. arXiv:2502.14802.
[5] Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. AAAI 2024. arXiv:2305.10250.
[6] Behrouz, A., Zhong, P., Mirrokni, V. (2025). Titans: Learning to Memorize at Test Time. NeurIPS 2025. arXiv:2501.00663.
[7] Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
[8] Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., Chalef, D. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956.
[9] LangChain (2025). LangMem SDK. https://langchain-ai.github.io/langmem/
[10] Lin, K., Snell, C., Wang, Y., Packer, C., Wooders, S., Stoica, I., Gonzalez, J. E. (2025). Sleep-time Compute: Beyond Inference Scaling at Test-time. arXiv:2504.13171.
[11] Shin, H., Lee, J. K., Kim, J., Kim, J. (2017). Continual Learning with Deep Generative Replay. NeurIPS 2017. arXiv:1705.08690.
[12] van de Ven, G. M., Siegelmann, H. T., Tolias, A. S. (2020). Brain-inspired replay for continual learning with artificial neural networks. Nature Communications 11:4069.
[13] McClelland, J. L., McNaughton, B. L., O'Reilly, R. C. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex. Psychological Review 102(3):419–457.
[14] Kumaran, D., Hassabis, D., McClelland, J. L. (2016). What Learning Systems Do Intelligent Agents Need? Trends in Cognitive Sciences 20(7):512–534.
[15] Wilson, M. A., McNaughton, B. L. (1994). Reactivation of Hippocampal Ensemble Memories During Sleep. Science 265:676–679.
[16] Girardeau, G., Benchenane, K., Wiener, S. I., Buzsáki, G., Zugaro, M. B. (2009). Selective Suppression of Hippocampal Ripples Impairs Spatial Memory. Nature Neuroscience 12:1222–1223.
[17] Joo, H. R., Frank, L. M. (2018). The hippocampal sharp wave–ripple in memory retrieval for immediate use and consolidation. Nature Reviews Neuroscience 19:744–757.
[18] Tononi, G., Cirelli, C. (2014). Sleep and the Price of Plasticity. Neuron 81(1):12–34.
[19] Chase, H. (2025). Introducing Ambient Agents. LangChain blog.

---

## Appendix A: Configuration reference

| Parameter | Default | Meaning |
|---|---|---|
| `tick_base_s` / `tick_max_s` / `tick_backoff` | 20 / 300 / 1.5 | heartbeat: stimulated rate, idle ceiling, slowdown factor |
| `fatigue_per_tick` / `fatigue_sleep_threshold` | 1 / 30 | sleep pressure accumulation |
| `idle_before_sleep_s` / `max_wake_s` | 180 / 21600 | idleness gate; circadian cap |
| `w_recency, w_importance, w_relevance` | 1.0, 1.0, 1.5 | retrieval weights |
| `recency_tau_h` | 24 | recency time constant |
| `reinforce_on_access` | 0.10 | testing-effect strength bump |
| `decay_lambda` | 0.12/day | base forgetting rate |
| `decay_stability_access` / `_importance` | 0.6 / 2.0 | stability growth per access / per importance unit |
| `archive_threshold` / `purge_archive_days` | 0.15 / 30 | soft forget; hard forget |
| `consolidation_cluster_sim` / `_max_batch` | 0.55 / 8 | NREM clustering |
| `reflection_top_k` | 12 | reflection seed count |
| `dreams_per_sleep` / `dream_sample_n` | 2 / 5 | REM parameters |
| `proactive_min_gap_s` | 600 | unprompted-speech throttle |

## Appendix B: The cognition contracts

The complete waking-cognition interface between architecture and substrate is two prompts:

**Reflex** (when the user speaks): plain text in, plain text out — "here are your memories, here is what they said; reply in 1–2 sentences."

**Tick** (the heartbeat): one JSON object — `thought` (inner monologue), `say` (speech or null; discarded if the reflex answered), `remember` (durable fact or null), `new_goal` (or null), `focus` (attention label).

Everything else — memory, identity, time, discipline, echo suppression, perspective hygiene — is the architecture's job. The substrate contributes language and association; the architecture contributes the person.
