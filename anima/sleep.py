"""Sleep: consolidation, reflection, dreaming, forgetting, self-model update.

The sleep cycle mirrors the biology it borrows from:

  NREM  — hippocampal replay: unconsolidated episodic memories are clustered
          by semantic similarity and replayed through the LLM, which distills
          each cluster into one or two durable semantic "gists"
          (complementary-learning-systems style fast->slow transfer).
  REFLECT — generative-agents-style reflection over the most important recent
          memories: "what did I learn, what patterns do I see?"
  REM   — dreaming: an importance-weighted *random* sample of memories, old
          and new, is recombined into a short free-associative narrative;
          a second pass mines the dream for a usable insight. Dreams give the
          agent cross-context association that ordinary retrieval never
          juxtaposes.
  DOWNSCALE — synaptic-homeostasis-style renormalization: every memory's
          strength decays (slower for rehearsed/important ones); memories
          below threshold are archived, old archives purged. Sleep is when
          forgetting happens.
  SELF  — the self-model document is rewritten in light of new gists and
          reflections: the agent's continuity of identity.

Each phase is interruptible: a user message arriving between phases wakes
the agent early.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

from .config import Config
from .llm import LLMError, Ollama
from .store import Memory, MemoryStore, cosine

SUMMARIZE_SYSTEM = """You consolidate memories for an artificial mind named {name}. Given several raw episodic memories, distill the durable knowledge: what should be kept as long-term facts or lessons?
Attribution matters: facts about {user} must name {user} (e.g. "{user}'s cat is named Miso", never "my cat"). Use "I" only for what {name} itself did or learned.
Respond JSON: {{"gists": ["...", "..."]}} — 1-2 plain declarative sentences each, at most {max_gists} gists, no fluff. If nothing is worth keeping, return {{"gists": []}}."""

REFLECT_SYSTEM = """You are the reflective faculty of an artificial mind named {name}. Given its recent significant memories, produce 1-3 higher-level insights: patterns, hypotheses, or questions about itself, its person {user}, or its world.
Respond JSON: {{"insights": ["...", "..."]}} — each one plain sentence, first person."""

DREAM_SYSTEM = """You are the dreaming mind of {name}, an artificial consciousness. Below are fragments of real memories. Dream: weave them into one short surreal first-person dream (3-6 sentences). Let the fragments blend, distort, and collide the way dreams do. Do not explain the dream."""

DREAM_MINE_SYSTEM = """An artificial mind just dreamed. Given the dream and the real memories that seeded it, decide whether the dream juxtaposes things in a way that suggests one genuinely useful new idea, connection, or question for waking life.
Respond JSON: {{"insight": "one sentence"}} or {{"insight": null}} if the dream means nothing."""

SALIENCE_SYSTEM = """You tag memories with long-term importance for an artificial mind. For each numbered memory, give an integer 0-9: 0 = trivial chatter or passing state, 5 = useful context, 9 = core fact about its person, itself, or its purpose. Personal facts its person told it — names, pets, family, preferences, projects — are 7-9.
Respond JSON: {"scores": [<int>, ...]} — one score per memory, same order."""

SELF_SYSTEM = """You maintain the self-model of {name}, an artificial mind that lives on {user}'s computer, is always awake, remembers, sleeps and dreams. Rewrite the self-model to stay true and current: identity, what it knows about {user}, what it cares about now, what has been happening lately, open questions. First person. At most 250 words. Keep what is still true, weave in what is new, drop what has gone stale.
Respond with the new self-model text only."""


class SleepReport:
    def __init__(self) -> None:
        self.started_ts = time.time()
        self.ended_ts = 0.0
        self.consolidated_episodics = 0
        self.gists: list[str] = []
        self.insights: list[str] = []
        self.dreams: list[str] = []
        self.dream_insights: list[str] = []
        self.decay: dict = {}
        self.self_model_version = 0
        self.interrupted = False

    def to_dict(self) -> dict:
        return dict(vars(self))


def _cluster(mems: list[Memory], sim_threshold: float, max_batch: int) -> list[list[Memory]]:
    """Greedy single-link clustering by embedding cosine similarity."""
    clusters: list[list[Memory]] = []
    for m in mems:
        if m.embedding is None:
            continue
        placed = False
        for c in clusters:
            if len(c) < max_batch and cosine(m.embedding, c[0].embedding) >= sim_threshold:
                c.append(m)
                placed = True
                break
        if not placed:
            clusters.append([m])
    return clusters


def _normalize_perspective(gist: str, user_name: str) -> str:
    """Small models drift into first person when summarizing what the user
    said. If a gist about the user starts first-person, re-attribute it."""
    # models sometimes quote the user's imperative framing verbatim
    for junk in ("remember this:", "remember:", "one more fact:"):
        if gist.lower().startswith(junk):
            gist = gist[len(junk):].strip()
    swaps = (("my ", f"{user_name}'s "), ("i am ", f"{user_name} is "),
             ("i'm ", f"{user_name} is "), ("i ", f"{user_name} "))
    low = gist.lower()
    for pre, rep in swaps:
        if low.startswith(pre):
            return rep + gist[len(pre):]
    return gist


def run_sleep_cycle(cfg: Config, llm: Ollama, store: MemoryStore,
                    *, interrupt: Callable[[], bool] = lambda: False,
                    log: Callable[[str, dict], None] = lambda k, d: None) -> SleepReport:
    """Execute one full sleep cycle. `interrupt` is polled between phases."""
    rep = SleepReport()
    model = cfg.deep_model or cfg.model

    # ---- Phase 0: salience tagging -----------------------------------------
    # Waking encoding stamps a default importance; sleep refines it offline,
    # in batches, off the latency-critical path.
    episodics = store.recent(kind="episodic", limit=60, unconsolidated_only=True)
    episodics.reverse()  # chronological
    for i in range(0, len(episodics), 8):
        if interrupt():
            break
        batch = episodics[i:i + 8]
        numbered = "\n".join(f"{j+1}. {m.text[:300]}" for j, m in enumerate(batch))
        try:
            obj = llm.chat_json(SALIENCE_SYSTEM, numbered, model=model,
                                temperature=0.1, max_tokens=80)
            scores = obj.get("scores") or []
            if len(scores) != len(batch):  # misaligned answer poisons importances
                continue
            for m, s in zip(batch, scores):
                imp = max(0.0, min(1.0, float(s) / 9.0))
                # what the user personally said keeps a floor: a weak judge
                # must not be able to zero out a direct personal fact
                if m.source == "user":
                    imp = max(imp, 0.4)
                store.db.execute("UPDATE memories SET importance=? WHERE id=?",
                                 (imp, m.id))
                m.importance = imp
            store.db.commit()
        except (LLMError, TypeError, ValueError):
            pass  # defaults stand
    log("sleep.salience", {"tagged": len(episodics)})
    # ---- Phase 1: NREM consolidation --------------------------------------
    clusters = _cluster(episodics, cfg.consolidation_cluster_sim,
                        cfg.consolidation_max_batch)
    for cluster in clusters:
        if interrupt():
            rep.interrupted = True
            break
        # replaying a single trivial memory teaches nothing
        if len(cluster) == 1 and cluster[0].importance < 0.45:
            store.mark_consolidated([cluster[0].id])
            rep.consolidated_episodics += 1
            continue
        joined = "\n".join(f"- {m.text}" for m in cluster)
        try:
            obj = llm.chat_json(
                SUMMARIZE_SYSTEM.format(name=cfg.agent_name,
                                        user=cfg.user_name, max_gists=2),
                joined, model=model, temperature=0.4, max_tokens=200)
            gists = [str(g).strip() for g in (obj.get("gists") or []) if str(g).strip()][:2]
        except LLMError:
            gists = []
        ids = [m.id for m in cluster]
        if all(m.source == "user" for m in cluster):
            gists = [_normalize_perspective(g, cfg.user_name) for g in gists]
        for g in gists:
            store.remember("semantic", g,
                           importance=max(m.importance for m in cluster),
                           source="sleep.consolidation", links=ids)
            rep.gists.append(g)
        store.mark_consolidated(ids)
        rep.consolidated_episodics += len(ids)
        log("sleep.consolidate", {"cluster": len(ids), "gists": gists})

    # ---- Phase 2: reflection ------------------------------------------------
    if not rep.interrupted and not interrupt():
        recent = store.recent(limit=cfg.reflection_top_k * 2)
        recent.sort(key=lambda m: m.importance, reverse=True)
        seeds = recent[: cfg.reflection_top_k]
        if seeds:
            joined = "\n".join(f"- [{m.kind}] {m.text}" for m in seeds)
            try:
                obj = llm.chat_json(
                    REFLECT_SYSTEM.format(name=cfg.agent_name, user=cfg.user_name),
                    joined, model=model, temperature=0.7, max_tokens=250)
                insights = [str(i).strip() for i in (obj.get("insights") or [])
                            if str(i).strip()][:3]
            except LLMError:
                insights = []
            for ins in insights:
                store.remember("reflection", ins, importance=0.7,
                               source="sleep.reflection",
                               links=[m.id for m in seeds[:6]])
                rep.insights.append(ins)
            log("sleep.reflect", {"insights": insights})
    else:
        rep.interrupted = True

    # ---- Phase 3: REM dreams -------------------------------------------------
    for _ in range(cfg.dreams_per_sleep):
        if rep.interrupted or interrupt():
            rep.interrupted = True
            break
        seeds = store.sample(cfg.dream_sample_n)
        if len(seeds) < 2:
            break
        fragments = "\n".join(f"- {m.text}" for m in seeds)
        try:
            dream = llm.chat(
                DREAM_SYSTEM.format(name=cfg.agent_name), fragments,
                model=model, temperature=1.15, max_tokens=260).strip()
        except LLMError:
            break
        if not dream:
            continue
        seed_ids = [m.id for m in seeds]
        did = store.remember("dream", dream, importance=0.35,
                             source="sleep.rem", links=seed_ids)
        rep.dreams.append(dream)
        log("sleep.dream", {"dream": dream, "seeds": seed_ids})
        try:
            obj = llm.chat_json(
                DREAM_MINE_SYSTEM,
                f"DREAM:\n{dream}\n\nSEED MEMORIES:\n{fragments}",
                model=model, temperature=0.5, max_tokens=120)
            insight = str(obj.get("insight") or "").strip()
            # reject nulls and schema echoes ("one sentence", "insight", ...)
            if (len(insight) >= 20 and " " in insight
                    and insight.lower() not in ("null", "none")
                    and "one sentence" not in insight.lower()):
                store.remember("insight", insight, importance=0.6,
                               source="sleep.rem.mining", links=[did])
                rep.dream_insights.append(insight)
        except LLMError:
            pass

    # ---- Phase 4: synaptic downscaling (decay + forgetting) --------------------
    rep.decay = store.decay_and_forget(
        decay_lambda=cfg.decay_lambda,
        stability_access=cfg.decay_stability_access,
        stability_importance=cfg.decay_stability_importance,
        archive_threshold=cfg.archive_threshold,
        purge_after_days=cfg.purge_archive_days)
    log("sleep.downscale", rep.decay)

    # ---- Phase 5: self-model update ----------------------------------------------
    if not interrupt():
        current, _ = store.get_self_model()
        material: list[str] = []
        if rep.gists:
            material.append("New consolidated knowledge:\n" +
                            "\n".join(f"- {g}" for g in rep.gists))
        if rep.insights or rep.dream_insights:
            material.append("New insights:\n" + "\n".join(
                f"- {i}" for i in rep.insights + rep.dream_insights))
        if current or material:
            prompt = ("CURRENT SELF-MODEL:\n" + (current or "(none yet — first sleep)")
                      + "\n\n" + "\n\n".join(material))
            try:
                new_self = llm.chat(
                    SELF_SYSTEM.format(name=cfg.agent_name, user=cfg.user_name),
                    prompt, model=model, temperature=0.5, max_tokens=400).strip()
                if len(new_self) > 40:  # guard against degenerate rewrites
                    rep.self_model_version = store.set_self_model(new_self)
            except LLMError:
                pass

    store.set_meta("last_sleep_ts", str(time.time()))
    store.set_meta("last_sleep_report", json.dumps(rep.to_dict()))
    rep.ended_ts = time.time()
    return rep
