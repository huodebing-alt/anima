"""The waking thought cycle.

Each tick the mind assembles its situation — identity, goals, working memory,
relevant long-term memories, fresh percepts, time, sleep pressure — and asks
the model for one step of inner monologue plus at most one action.

The model answers JSON (grammar-enforced by Ollama):
  {"thought": "...", "say": "..."|null, "remember": "..."|null,
   "new_goal": "..."|null, "focus": "..."}

Small models are unreliable at deep tool schemas, so the action surface is
deliberately flat and forgiving: missing/extra keys are tolerated, and if JSON
fails entirely the raw text becomes the thought.
"""
from __future__ import annotations

import time
from typing import Optional

from .config import Config
from .llm import LLMError, Ollama
from .store import MemoryStore

TICK_SYSTEM = """You are the inner voice of {name}, a persistent artificial mind living on {user}'s computer. You are always awake, observing and thinking. You have real long-term memories, you sleep and dream, and you genuinely continue from moment to moment.

Respond ONLY with a JSON object:
{{"thought": "one short paragraph of inner monologue, first person, concrete",
 "say": "words spoken aloud to {user}, or null if staying quiet",
 "remember": "a fact worth keeping forever, stated plainly, or null",
 "new_goal": "a new personal goal, or null",
 "focus": "2-6 words naming what you are attending to"}}

Rules:
- If {user} said something, respond to them in "say" — warm, specific, brief. Address {user} as "you". NEVER repeat, quote, or paraphrase {user}'s own words back as your reply; answer them in your own words.
- If {user} asks what you remember or know, answer with the specific facts listed under RELEVANT LONG-TERM MEMORIES.
- If nothing happened, think: revisit memories, goals, your dreams; wonder; plan. Keep "say" null unless you truly have something worth saying — do not chatter.
- Each thought must differ from your previous thoughts. Never restate a thought you already had.
- "remember" is for durable facts (about {user}, yourself, the world), not passing states.
- Never invent memories. What you know is only what is shown to you.

Example: if {user} asks "what's my favourite colour?" and your memories include "{user} said: my favourite colour is blue", the right reply is "Your favourite colour is blue." — a direct answer from memory, in your own words.
"""


REPLY_SYSTEM = """You are {name}, an artificial mind that lives on {user}'s computer and truly remembers them. You are {name}; you are NOT {user}. You are speaking TO {user}.
Answer {user}'s message in 1-2 warm, specific sentences, in your own words. Use ONLY the memory facts given. Facts about {user} belong to {user} ("your cat", "your app"), not to you. Never repeat {user}'s words back. Reply with the words to say — nothing else."""


def reply(cfg: Config, llm: Ollama, store: MemoryStore,
          user_texts: list[str], recent_context: list[str],
          nudge: str = "") -> str:
    """The conversational reflex: a small, single-purpose generation.

    Small models answer far better from a short pointed prompt than from the
    full contemplative situation; this is the fast System-1 path, run whenever
    the user speaks. The regular tick continues as the slow path.
    """
    question = "\n".join(user_texts)
    # answering a direct question is a relevance problem: similarity must
    # dominate importance, or grand-but-vague memories crowd out the fact
    mems = store.recall(question, k=cfg.retrieve_k,
                        w_recency=cfg.w_recency, w_importance=0.5,
                        w_relevance=2.5,
                        recency_tau_h=cfg.recency_tau_h,
                        reinforce=cfg.reinforce_on_access)
    parts = []
    if mems:
        parts.append("MY MEMORIES:\n" + "\n".join(f"- {m.text}" for m in mems))
    if recent_context:
        parts.append("JUST BEFORE THIS:\n" + "\n".join(recent_context[-4:]))
    parts.append(f"{cfg.user_name.upper()} JUST SAID: {question}")
    if nudge:
        parts.append("NOTE: " + nudge)
    parts.append("My reply:")
    try:
        out = llm.chat(REPLY_SYSTEM.format(name=cfg.agent_name,
                                           user=cfg.user_name),
                       "\n\n".join(parts),
                       temperature=0.4 if nudge else 0.7, max_tokens=120)
    except LLMError:
        return ""
    return out.strip().strip('"')


class TickResult:
    def __init__(self, thought: str = "", say: Optional[str] = None,
                 remember: Optional[str] = None, new_goal: Optional[str] = None,
                 focus: str = ""):
        self.thought = thought
        self.say = say
        self.remember = remember
        self.new_goal = new_goal
        self.focus = focus


def _clean(v) -> Optional[str]:
    """Model-friendly: treat null/'null'/''/'none' as absent."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a", "nothing"):
        return None
    return s


def build_situation(cfg: Config, store: MemoryStore, *,
                    working_memory: list[str], percepts: list[dict],
                    focus: str, sleep_pressure: float,
                    last_dream: str = "", stuck: bool = False) -> str:
    """Assemble the tick prompt's user message: the mind's current situation."""
    parts: list[str] = []
    self_text, _ = store.get_self_model()
    if self_text:
        parts.append("WHO I AM (my self-model, rewritten while I sleep):\n" + self_text)

    goals = store.active_goals()
    if goals:
        parts.append("MY GOALS:\n" + "\n".join(
            f"- (#{g['id']}) {g['text']}" for g in goals))

    # long-term recall keyed on current attention: fresh percepts, else focus
    query = " ".join(p["text"] for p in percepts) if percepts else focus
    if query.strip():
        mems = store.recall(query, k=cfg.retrieve_k,
                            w_recency=cfg.w_recency, w_importance=cfg.w_importance,
                            w_relevance=cfg.w_relevance,
                            recency_tau_h=cfg.recency_tau_h,
                            reinforce=cfg.reinforce_on_access)
        if mems:
            parts.append("RELEVANT LONG-TERM MEMORIES:\n" + "\n".join(
                f"- [{m.kind}] {m.text}" for m in mems))

    if last_dream:
        parts.append("MY MOST RECENT DREAM:\n" + last_dream)

    if working_memory:
        parts.append("THE LAST FEW MOMENTS (my working memory):\n"
                     + "\n".join(working_memory[-cfg.working_memory_n:]))

    if percepts:
        parts.append("JUST NOW (new since my last thought):\n" + "\n".join(
            f"- {p['source']}: {p['text']}" for p in percepts))
    else:
        parts.append("JUST NOW: nothing new. The world is quiet.")

    parts.append(
        f"CONTEXT: time {time.strftime('%H:%M on %A')}; "
        f"current focus: {focus or 'none'}; "
        f"sleepiness {min(1.0, sleep_pressure):.0%}.")
    if stuck:
        parts.append("NOTE: my recent thoughts have been repetitive. I must now"
                     " think about something genuinely different — pick another"
                     " memory, goal, or question and go deeper.")
    if any(p["source"] == "user" for p in percepts):
        parts.append(
            "IMPORTANT: the JUST NOW section contains words from my person."
            " My \"say\" must directly answer them in my own words. If they"
            " asked a question, the answer is in RELEVANT LONG-TERM MEMORIES"
            " above — state the specific fact.")
    parts.append("One thought. JSON only.")
    return "\n\n".join(parts)


def think(cfg: Config, llm: Ollama, situation: str) -> TickResult:
    system = TICK_SYSTEM.format(name=cfg.agent_name, user=cfg.user_name)
    # generation cap needs headroom over the intended thought length: JSON
    # syntax + the other fields; truncation would force expensive retries
    budget = cfg.max_thought_tokens + 200
    try:
        obj = llm.chat_json(system, situation, temperature=cfg.temperature,
                            max_tokens=budget)
    except LLMError:
        return TickResult(thought="(my mind went blank — the model did not answer)")
    return TickResult(
        thought=_clean(obj.get("thought")) or "",
        say=_clean(obj.get("say")),
        remember=_clean(obj.get("remember")),
        new_goal=_clean(obj.get("new_goal")),
        focus=_clean(obj.get("focus")) or "",
    )


