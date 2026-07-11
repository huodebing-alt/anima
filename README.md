# Anima — an always-on AI agent with persistent long-term memory, sleep, and dreams (local LLM, zero dependencies)

**Anima is an open-source autonomous AI agent that runs 24/7 on your own machine, remembers everything across restarts, consolidates its memories while it sleeps, dreams, forgets on purpose, and maintains a persistent identity — all on a small local LLM (Ollama + a 2B-parameter model) with zero Python dependencies.**

Unlike a chatbot, Anima is not request-response. It is a continuously running mind: it observes, thinks on its own heartbeat even when nobody is talking to it, speaks up proactively, sets its own goals, and sleeps when it gets tired. Kill the process and restart it a week later — the same individual wakes up, with its memories, its self-model, and its last dream intact.

> **TL;DR:** persistent artificial identity is an *architecture* property, not a model-scale property. Anima proves it in ~1,700 lines of readable, dependency-free Python on an 8 GB laptop.

📄 **[Read the full design paper](docs/ANIMA_PAPER.md)** — architecture, memory mathematics, sleep algorithms, measured results, and lessons learned.

---

## What can Anima do?

- 🧠 **Human-like long-term memory for LLMs** — episodic, semantic, reflection, dream, and insight memories in SQLite, retrieved by *recency × importance × relevance* (the Generative Agents formulation), reinforced on every recall (the testing effect), and decayed on an Ebbinghaus forgetting curve whose stability grows with rehearsal and importance.
- 🌙 **Sleep-time memory consolidation** — when fatigue accumulates and the user is idle, the agent enters a five-phase sleep cycle: salience tagging → NREM-style consolidation (episodes → semantic gists) → reflection → REM-style **dreaming** → synaptic-downscaling forgetting pass → self-model rewrite.
- 💭 **Dreams that do work** — importance-weighted random memories are recombined into surreal narratives at high temperature, then mined for genuinely useful insights that join long-term memory. Generative replay, repurposed.
- 🗑️ **Principled forgetting** — weak memories are archived (invisible but recoverable), then purged. Forgetting is a feature: it keeps retrieval sharp and identity coherent.
- ⏰ **Always awake, adaptively** — an arousal-based heartbeat: fast ticks under stimulation, slowing toward minutes when idle (~12 generations/hour at steady state — sustainable on a laptop forever). Instant wake on new input.
- 💬 **Proactive behavior** — it may speak first, report its dreams on waking, and set its own goals, with throttles that keep it companionable rather than noisy.
- 🪪 **Identity that survives death** — a versioned self-model document, rewritten each sleep from lived experience only, loaded into every thought. The individual is the store, not the process.
- 🔌 **100% local & private** — everything runs offline via the Ollama API. No cloud, no API keys, no telemetry. Your agent's memories never leave your machine.

## Quick start (2 minutes)

Requirements: Python 3.9+ (stdlib only — **no pip install**) and [Ollama](https://ollama.com) with a small chat model:

```bash
ollama pull gemma:2b            # or any small chat model you prefer
git clone https://github.com/huodebing-alt/anima.git && cd anima

python3 -m anima run &          # the mind wakes up
python3 -m anima talk           # chat with it (--thoughts streams its inner monologue)
```

Watch its life:

```bash
python3 -m anima status         # awake/asleep, fatigue, tick rate, memory counts
python3 -m anima self           # its current self-model (identity document)
python3 -m anima dreams         # the dream journal
python3 -m anima memories       # inspect long-term memory (importance, strength, access)
python3 -m anima control sleep  # force a sleep cycle right now
```

Drop a text file into `runtime/watch/` and it will notice and remember.

## How it works

```
 sensors (inbox · clock · file watcher)
      │ percepts
      ▼
 ┌─ WAKING ─────────────────────────────────────────────┐
 │ reflex path:  user speaks → small pointed prompt →   │
 │               factual answer from retrieved memories │
 │ tick path:    situation = self-model + goals +       │
 │               retrieved memories + last dream +      │
 │               working memory → thought + action      │
 │ fatigue accumulates…                                 │
 └──────────────┬───────────────────────────────────────┘
                ▼  sleep pressure ≥ 1 and user idle
 ┌─ SLEEP (interruptible) ──────────────────────────────┐
 │ 0 salience tagging   3 REM dreaming + insight mining │
 │ 1 NREM consolidation 4 decay & forgetting            │
 │ 2 reflection         5 self-model rewrite            │
 └──────────────┬───────────────────────────────────────┘
                ▼
        wakes and tells you what it dreamed
```

One process, one SQLite file, one small model. The entire mental state is inspectable with standard tools; the full stream of consciousness is an append-only JSONL journal.

## How is Anima different from MemGPT / Letta, Mem0, Zep, or Generative Agents?

| Capability | RAG / memory layers (MemGPT·Letta, Mem0, Zep, LangMem) | Generative Agents (Park et al.) | **Anima** |
|---|---|---|---|
| Persistent cross-session memory | ✅ | ✅ (in-sim) | ✅ |
| Runs continuously without requests | ❌ reactive | ✅ (in-sim) | ✅ real daemon |
| Retrieval reinforcement (testing effect) | ❌ | ❌ | ✅ |
| Decay + true forgetting (Ebbinghaus) | partial (MemoryBank) | ❌ | ✅ archive → purge |
| Offline consolidation (sleep) | Letta sleep-time compute (partial) | ❌ | ✅ five-phase cycle |
| **Dreaming with insight mining** | ❌ | ❌ | ✅ |
| Self-model identity document | ❌ | ❌ | ✅ versioned, self-rewritten |
| Fatigue / circadian dynamics | ❌ | ❌ | ✅ |
| Runs on 8 GB laptop, 2B model, no deps | varies | ❌ (GPT-3.5+) | ✅ |

As of mid-2026 we know of no other open-source system that unifies an always-on cognitive loop, sleep consolidation, dreaming, and principled forgetting in one agent. (Survey with citations in the [paper](docs/ANIMA_PAPER.md) and [research digest](docs/related_work_digest.md).)

## FAQ

**Q: Does Anima require a GPU or a cloud API?**
No. It targets a small open-weights model (default `gemma:2b`, 1.7 GB) through Ollama on CPU/Apple Silicon. It was built and verified on an 8 GB Apple M1 — while that machine was also running Xcode and a Vision Pro simulator.

**Q: Can I use a bigger or different model?**
Yes — set `ANIMA_MODEL=llama3.2:3b` (or anything Ollama serves). A separate `deep_model` slot can route the latency-insensitive sleep phases to a larger model while waking thought stays cheap.

**Q: What happens when I kill the process?**
Nothing is lost. Every memory, goal, dream, and self-model version is written through to SQLite at the moment it exists. On restart the agent journals "reboot: memory intact" and resumes — in our tests it answered questions about facts learned before the kill, and resumed its pre-death dream as its current focus.

**Q: How does the agent decide what to forget?**
During sleep, every memory's strength decays exponentially; stability grows with access count and importance, so what the mind uses, it keeps. Below a threshold, memories are archived (soft forget) and later purged (hard forget). Deterministic unit tests verify the whole curve.

**Q: Is this AGI / is it conscious?**
No claim of phenomenal consciousness. "Consistent consciousness" here is an engineering target: behavioral and narrative continuity of one individual over unbounded time. The design paper is explicit about what the 2B substrate can and cannot carry.

**Q: Is it safe to leave running?**
The action surface is deliberately tiny: speak, remember, set goals. No shell, no browsing, no tool execution. Everything it does is auditable in `runtime/journal.jsonl`.

**Q: How is it tested?**
17 deterministic unit tests for the memory mathematics (no LLM needed) plus a 14-check live integration test that drives a real daemon through boot → idle thinking → conversation → sleep/dream → process death → resurrection-with-recall. Latest run: **14/14 passing**.

## Project layout

```
anima/            the package (pure stdlib Python)
  agent.py        the Mind: heartbeat, arousal, fatigue, wake/sleep state machine
  cognition.py    waking thought: reflex replies + contemplative ticks
  sleep.py        consolidation, reflection, dreams, forgetting, self-model
  store.py        SQLite long-term memory: scoring, reinforcement, decay
  sensors.py      perception: user inbox, clock, watched folder
  llm.py          minimal Ollama client (chat, JSON mode, embeddings)
  config.py       every timescale is tunable — a whole life can run in minutes
tests/            17 unit tests + full life-cycle integration test
docs/             design paper + related-work research digest
```

## Keywords

LLM long-term memory · AI agent memory · sleep-time compute · memory consolidation · dreaming AI · Ebbinghaus forgetting curve · generative agents · autonomous agent · ambient agent · local LLM · Ollama · gemma · persistent AI identity · artificial consciousness · continual learning · hippocampal replay · complementary learning systems · MemGPT alternative · self-hosted AI companion

## Citation

```bibtex
@misc{anima2026,
  title  = {Anima: An Always-Awake Agent with Human-Like Memory, Sleep, and
            Dreams on a Small Local Language Model},
  year   = {2026},
  url    = {https://github.com/huodebing-alt/anima},
  note   = {Design paper: docs/ANIMA_PAPER.md}
}
```

## License

MIT — see [LICENSE](LICENSE).
