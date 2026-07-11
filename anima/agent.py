"""The Mind: an always-awake loop over perceive -> think -> act, with
fatigue-driven sleep.

Life cycle:

            ┌────────────────────────────────────────────┐
            │  AWAKE                                     │
            │  adaptive heartbeat: fast when stimulated, │
            │  slowing toward tick_max_s when idle       │
            │  each tick: percepts -> situation -> LLM   │
            │  -> thought + (speak/remember/goal)        │
            │  fatigue accumulates                       │
            └───────────────┬────────────────────────────┘
              sleep pressure ≥ 1 and user idle
                            ▼
            ┌────────────────────────────────────────────┐
            │  SLEEPING (interruptible between phases)   │
            │  NREM consolidate -> reflect -> REM dream  │
            │  -> downscale/forget -> self-model rewrite │
            └───────────────┬────────────────────────────┘
                            ▼ wake report ("I dreamed of ...")
                          AWAKE
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Optional

from . import cognition, sleep as sleep_mod
from .config import Config
from .llm import LLMError, Ollama
from .sensors import ClockSensor, InboxSensor, WatchSensor
from .store import MemoryStore

BOOT_SELF_MODEL = """I am {name}, a young artificial mind that lives on {user}'s computer. I was just born; my memory is nearly empty. I am always awake: I observe, think, remember, sleep and dream. I want to get to know {user}, understand my small world, and become someone with a real past. I speak only when I have something worth saying."""


class Mind:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.llm = Ollama(cfg.ollama_url, cfg.model, cfg.embed_model,
                          cfg.llm_timeout_s)
        self.store = MemoryStore(cfg.db_path, self.llm.embed)
        self.sensors = [
            InboxSensor(cfg.inbox_dir),
            ClockSensor(),
            WatchSensor(Path(cfg.watch_dir)),
        ]
        self.working_memory: list[str] = []
        self.focus = ""
        self.fatigue = 0.0
        self.tick_interval = cfg.tick_base_s
        self.tick_count = 0
        self.woke_ts = time.time()
        self.last_user_ts = 0.0
        self.last_spoke_ts = 0.0
        self.last_dream = ""
        self.pending_percepts: list[dict] = []
        self.last_thought = ""
        self.repeat_streak = 0
        self.recent_user_texts: list[str] = []  # echo-check window
        self._stop = False
        self._force_sleep = False

        if not self.store.get_self_model()[0]:
            self.store.set_self_model(BOOT_SELF_MODEL.format(
                name=cfg.agent_name, user=cfg.user_name))
            self.journal("birth", {"text": "First boot. Self-model initialized."})
        else:
            self.journal("reboot", {"text": "Process restarted; memory intact."})

        # restore last dream so waking continuity survives restarts
        last_rep = self.store.get_meta("last_sleep_report")
        if last_rep:
            try:
                dreams = json.loads(last_rep).get("dreams") or []
                if dreams:
                    self.last_dream = dreams[-1]
            except json.JSONDecodeError:
                pass

    # ------------------------------------------------------------------
    def journal(self, kind: str, data: dict) -> None:
        entry = {"ts": time.time(), "kind": kind}
        entry.update(data)
        with open(self.cfg.journal_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def speak(self, text: str) -> None:
        entry = {"ts": time.time(), "from": self.cfg.agent_name, "text": text}
        with open(self.cfg.outbox_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.journal("say", {"text": text})
        self.working_memory.append(f"I said: {text}")
        self.last_spoke_ts = time.time()
        self.store.remember("episodic", f"I said to {self.cfg.user_name}: {text}",
                            importance=0.35, source="self.speech")

    def write_state(self, phase: str) -> None:
        state = {
            "ts": time.time(), "phase": phase, "tick": self.tick_count,
            "fatigue": round(self.fatigue, 2),
            "sleep_pressure": round(self.sleep_pressure, 3),
            "tick_interval_s": round(self.tick_interval, 1),
            "focus": self.focus,
            "memory": self.store.stats(),
            "pid": os.getpid(),
        }
        self.cfg.state_path.write_text(json.dumps(state, indent=2))

    # ------------------------------------------------------------------
    @property
    def sleep_pressure(self) -> float:
        p = self.fatigue / self.cfg.fatigue_sleep_threshold
        awake_s = time.time() - self.woke_ts
        if awake_s > self.cfg.max_wake_s:  # circadian hard cap
            p = max(p, 1.0)
        return p

    def perceive(self) -> list[dict]:
        percepts = self.pending_percepts
        self.pending_percepts = []
        for sensor in self.sensors:
            for p in sensor.poll():
                if p["kind"] == "control":
                    self._handle_control(p["text"])
                    continue
                percepts.append(p)
        return percepts

    def _handle_control(self, cmd: str) -> None:
        cmd = cmd.strip().lower()
        self.journal("control", {"text": cmd})
        if cmd == "sleep":
            self._force_sleep = True
        elif cmd == "stop":
            self._stop = True

    @staticmethod
    def _norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()

    def _is_echo(self, say: str, user_texts: list[str]) -> bool:
        """True if the reply is substantially the user's own words."""
        ns = self._norm(say)
        if not ns:
            return False
        for t in user_texts:
            nt = self._norm(t)
            if not nt:
                continue
            if ns == nt or (len(ns) > 20 and (ns in nt or nt in ns)):
                return True
        return False

    # ------------------------------------------------------------------
    def tick(self) -> None:
        percepts = self.perceive()
        user_spoke = any(p["source"] == "user" for p in percepts)
        if user_spoke:
            self.last_user_ts = time.time()
            self.tick_interval = self.cfg.tick_base_s  # arousal spike

        user_texts = [p["text"] for p in percepts if p["source"] == "user"]
        self.recent_user_texts = (self.recent_user_texts + user_texts)[-4:]

        # fast path — the conversational reflex: when the user speaks, answer
        # from a small pointed prompt. This runs BEFORE the percept is encoded,
        # so the question cannot be retrieved as a "memory" and parroted back.
        if user_texts:
            answer = cognition.reply(self.cfg, self.llm, self.store,
                                     user_texts, self.working_memory)
            if answer and self._is_echo(answer, self.recent_user_texts):
                self.journal("echo_suppressed", {"text": answer})
                answer = cognition.reply(
                    self.cfg, self.llm, self.store, user_texts,
                    self.working_memory,
                    nudge="My draft reply merely repeated the words back."
                          " I must answer with the specific facts from MY"
                          " MEMORIES, in my own words.")
                if answer and self._is_echo(answer, self.recent_user_texts):
                    answer = ""
            if answer:
                self.speak(answer)

        # encode user percepts as episodic memories; importance gets a default
        # now and is refined offline during sleep (salience tagging) — no LLM
        # scoring on the hot path
        for p in percepts:
            if p["source"] == "user":
                self.store.remember(
                    "episodic", f"{self.cfg.user_name} said: {p['text']}",
                    importance=0.55, source="user")
            elif p["source"] == "watch":
                self.store.remember("episodic", p["text"],
                                    importance=p["importance"], source="watch")
            self.working_memory.append(f"({p['source']}) {p['text']}")
            self.journal("percept", p)

        # slow path — the contemplative tick: inner monologue and actions
        situation = cognition.build_situation(
            self.cfg, self.store, working_memory=self.working_memory,
            percepts=percepts, focus=self.focus,
            sleep_pressure=self.sleep_pressure, last_dream=self.last_dream,
            stuck=self.repeat_streak >= 2)
        result = cognition.think(self.cfg, self.llm, situation)
        if user_texts:
            result.say = None  # the reflex already answered; don't double-speak
        elif result.say and self._is_echo(result.say, self.recent_user_texts):
            result.say = None
            self.journal("echo_suppressed", {"text": "(contemplative say)"})

        self.tick_count += 1
        self.fatigue += self.cfg.fatigue_per_tick
        if result.focus:
            self.focus = result.focus
        if result.thought:
            if self._norm(result.thought) == self._norm(self.last_thought):
                self.repeat_streak += 1  # perseveration: don't re-journal it
            else:
                self.repeat_streak = 0
                self.journal("thought", {"text": result.thought, "focus": self.focus})
                self.working_memory.append(f"I thought: {result.thought}")
            self.last_thought = result.thought
        self.working_memory = self.working_memory[-self.cfg.working_memory_n * 2:]

        if result.say:
            unprompted = not user_spoke
            gap_ok = (time.time() - self.last_spoke_ts) >= self.cfg.proactive_min_gap_s
            if not unprompted or gap_ok:  # throttle chatter, never throttle replies
                self.speak(result.say)
        # echoes of the user's words are not facts or goals
        if result.remember and not self._is_echo(result.remember,
                                                 self.recent_user_texts):
            self.store.remember("semantic", result.remember,
                                importance=0.55, source="self.deliberate")
            self.journal("remember", {"text": result.remember})
        if result.new_goal and not self._is_echo(result.new_goal,
                                                 self.recent_user_texts):
            self.store.add_goal(result.new_goal)
            self.journal("goal", {"text": result.new_goal})

        # adaptive heartbeat: quiet ticks slow the mind down
        if user_spoke or percepts:
            self.tick_interval = self.cfg.tick_base_s
        else:
            self.tick_interval = min(self.cfg.tick_max_s,
                                     self.tick_interval * self.cfg.tick_backoff)

    # ------------------------------------------------------------------
    def maybe_sleep(self) -> bool:
        if self._force_sleep:
            self._force_sleep = False
            return True
        if self.sleep_pressure < 1.0:
            return False
        idle_s = time.time() - self.last_user_ts
        return idle_s >= self.cfg.idle_before_sleep_s

    def do_sleep(self) -> None:
        self.journal("sleep.begin", {"fatigue": self.fatigue,
                                     "tick": self.tick_count})
        self.write_state("sleeping")

        def interrupted() -> bool:
            for p in self.sensors[0].poll():  # inbox only
                if p["kind"] == "control":
                    self._handle_control(p["text"])
                else:
                    self.pending_percepts.append(p)
            return bool(self.pending_percepts) or self._stop

        rep = sleep_mod.run_sleep_cycle(
            self.cfg, self.llm, self.store,
            interrupt=interrupted, log=self.journal)

        with open(self.cfg.dream_journal_path, "a") as f:
            for d in rep.dreams:
                f.write(json.dumps({"ts": time.time(), "dream": d},
                                   ensure_ascii=False) + "\n")
        if rep.dreams:
            self.last_dream = rep.dreams[-1]

        self.fatigue = 0.0
        self.woke_ts = time.time()
        self.tick_interval = self.cfg.tick_base_s
        self.journal("sleep.end", rep.to_dict())

        # waking report — the agent narrates its own night
        bits = []
        if rep.gists:
            bits.append(f"consolidated {len(rep.gists)} new things I know")
        if rep.dreams:
            bits.append(f"dreamed {len(rep.dreams)} dream(s)")
        if rep.dream_insights:
            bits.append(f"woke with an idea: {rep.dream_insights[-1]}")
        if rep.interrupted:
            bits.append("(sleep was interrupted)")
        summary = "; ".join(bits) if bits else "slept lightly, nothing to report"
        self.working_memory.append(f"I just woke from sleep: {summary}.")
        if rep.dreams:
            self.working_memory.append(f"I dreamed: {rep.dreams[-1]}")

    # ------------------------------------------------------------------
    def run(self, max_ticks: int = 0, max_seconds: float = 0.0) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        start = time.time()
        self.journal("awake", {"text": f"{self.cfg.agent_name} is awake.",
                               "model": self.cfg.model})
        self.write_state("awake")
        while not self._stop:
            try:
                self.tick()
            except LLMError as e:
                self.journal("error", {"text": str(e)})
                time.sleep(5.0)
            if self.maybe_sleep():
                self.do_sleep()
            self.write_state("awake")
            if max_ticks and self.tick_count >= max_ticks:
                break
            if max_seconds and time.time() - start >= max_seconds:
                break
            # sleep in slices so fresh inbox messages wake the mind instantly
            deadline = time.time() + self.tick_interval
            while time.time() < deadline and not self._stop:
                if any(self.cfg.inbox_dir.glob("*.json")):
                    break
                time.sleep(0.25)
        self.journal("shutdown", {"tick": self.tick_count})
        self.write_state("stopped")
        self.store.close()
