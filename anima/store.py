"""Long-term memory store.

A single SQLite database holds every memory the agent has, each with an
embedding, an importance score, and a *strength* that models human memory
dynamics:

- retrieval score  = w_rec * recency + w_imp * importance + w_rel * relevance
  (the Generative Agents formulation, Park et al. 2023)
- accessing a memory reinforces it (testing effect)
- during sleep, strength decays exponentially, but decay slows with
  access_count and importance (Ebbinghaus-style forgetting with growing
  stability)
- memories that fall below a strength threshold are archived (soft forget),
  then purged after a grace period (hard forget)

Memory kinds:
  episodic    raw experience (a message, an observation, something that happened)
  semantic    consolidated gist/fact distilled from episodics during sleep
  reflection  higher-level insight the agent drew about itself/its world
  dream       a dream narrative from REM sleep
  insight     useful association mined from a dream
  procedural  how-to knowledge
"""
from __future__ import annotations

import array
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

KINDS = ("episodic", "semantic", "reflection", "dream", "insight", "procedural")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    created_ts REAL NOT NULL,
    last_access_ts REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.3,
    strength REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT '',
    links TEXT NOT NULL DEFAULT '[]',
    consolidated INTEGER NOT NULL DEFAULT 0,
    archived_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_mem_arch ON memories(archived_ts);

CREATE TABLE IF NOT EXISTS self_model (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    text TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    updated_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    priority REAL NOT NULL DEFAULT 0.5,
    created_ts REAL NOT NULL,
    done_ts REAL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class Memory:
    id: int
    kind: str
    text: str
    embedding: Optional[list[float]]
    created_ts: float
    last_access_ts: float
    access_count: int
    importance: float
    strength: float
    source: str = ""
    links: list[int] = field(default_factory=list)
    consolidated: bool = False
    score: float = 0.0  # transient: last retrieval score


def _pack(vec: Optional[Iterable[float]]) -> Optional[bytes]:
    if vec is None:
        return None
    return array.array("f", vec).tobytes()


def _unpack(blob: Optional[bytes]) -> Optional[list[float]]:
    if blob is None:
        return None
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


def cosine(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


class MemoryStore:
    """All persistence. `embed_fn` is injected so tests need no LLM server."""

    def __init__(self, db_path: str | Path,
                 embed_fn: Callable[[str], list[float]],
                 now_fn: Callable[[], float] = time.time):
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.embed_fn = embed_fn
        self.now = now_fn

    def close(self) -> None:
        self.db.close()

    # -- rows <-> objects ----------------------------------------------
    @staticmethod
    def _to_mem(r: sqlite3.Row) -> Memory:
        return Memory(
            id=r["id"], kind=r["kind"], text=r["text"],
            embedding=_unpack(r["embedding"]),
            created_ts=r["created_ts"], last_access_ts=r["last_access_ts"],
            access_count=r["access_count"], importance=r["importance"],
            strength=r["strength"], source=r["source"],
            links=json.loads(r["links"]), consolidated=bool(r["consolidated"]),
        )

    # -- encode ----------------------------------------------------------
    def remember(self, kind: str, text: str, *, importance: float = 0.3,
                 source: str = "", links: Optional[list[int]] = None,
                 embed: bool = True, dedup_sim: float = 0.97) -> int:
        """Insert a memory — unless a near-identical one already exists, in
        which case repetition reinforces the original instead of duplicating
        it (rehearsal, not hoarding)."""
        assert kind in KINDS, kind
        now = self.now()
        raw_vec = self.embed_fn(text) if embed else None
        if raw_vec is not None and dedup_sim < 1.0:
            for r in self.db.execute(
                    "SELECT id, embedding, importance FROM memories"
                    " WHERE kind=? AND archived_ts IS NULL", (kind,)):
                old = _unpack(r["embedding"])
                if old and cosine(raw_vec, old) >= dedup_sim:
                    self.db.execute(
                        "UPDATE memories SET last_access_ts=?,"
                        " access_count=access_count+1,"
                        " strength=MIN(1.0, strength+0.1), importance=MAX(importance,?)"
                        " WHERE id=?", (now, importance, r["id"]))
                    self.db.commit()
                    return int(r["id"])
        vec = _pack(raw_vec) if raw_vec is not None else None
        cur = self.db.execute(
            "INSERT INTO memories (kind, text, embedding, created_ts,"
            " last_access_ts, importance, strength, source, links)"
            " VALUES (?,?,?,?,?,?,1.0,?,?)",
            (kind, text, vec, now, now, max(0.0, min(1.0, importance)),
             source, json.dumps(links or [])))
        self.db.commit()
        return int(cur.lastrowid)

    # -- retrieve ---------------------------------------------------------
    def recall(self, query: str, k: int = 6, *,
               w_recency: float = 1.0, w_importance: float = 1.0,
               w_relevance: float = 1.5, recency_tau_h: float = 24.0,
               reinforce: float = 0.10,
               kinds: Optional[tuple[str, ...]] = None) -> list[Memory]:
        """Top-k live memories by recency*importance*relevance; reinforces hits."""
        qvec = self.embed_fn(query)
        now = self.now()
        rows = self.db.execute(
            "SELECT * FROM memories WHERE archived_ts IS NULL"
            + (" AND kind IN (%s)" % ",".join("?" * len(kinds)) if kinds else ""),
            tuple(kinds) if kinds else ()).fetchall()
        scored: list[Memory] = []
        for r in rows:
            m = self._to_mem(r)
            if m.embedding is None:
                continue
            rec = math.exp(-max(0.0, now - m.last_access_ts) / (recency_tau_h * 3600.0))
            rel = max(0.0, cosine(qvec, m.embedding))  # clamp: negatives = irrelevant
            m.score = w_recency * rec + w_importance * m.importance + w_relevance * rel
            scored.append(m)
        scored.sort(key=lambda m: m.score, reverse=True)
        top = scored[:k]
        for m in top:  # the testing effect: retrieval strengthens
            self.db.execute(
                "UPDATE memories SET last_access_ts=?, access_count=access_count+1,"
                " strength=MIN(1.0, strength+?) WHERE id=?",
                (now, reinforce, m.id))
        self.db.commit()
        return top

    def recent(self, kind: Optional[str] = None, limit: int = 20,
               unconsolidated_only: bool = False) -> list[Memory]:
        q = "SELECT * FROM memories WHERE archived_ts IS NULL"
        args: list = []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if unconsolidated_only:
            q += " AND consolidated=0"
        q += " ORDER BY created_ts DESC LIMIT ?"
        args.append(limit)
        return [self._to_mem(r) for r in self.db.execute(q, args).fetchall()]

    def get(self, mem_id: int) -> Optional[Memory]:
        r = self.db.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
        return self._to_mem(r) if r else None

    def sample(self, n: int, *, exclude_kinds: tuple[str, ...] = ("dream",),
               rand: Callable[[], float] = None) -> list[Memory]:
        """Importance-weighted random sample of live memories (dream fodder)."""
        import random as _random
        rnd = rand or _random.random
        rows = self.db.execute(
            "SELECT * FROM memories WHERE archived_ts IS NULL AND kind NOT IN (%s)"
            % ",".join("?" * len(exclude_kinds)), exclude_kinds).fetchall()
        mems = [self._to_mem(r) for r in rows]
        if len(mems) <= n:
            return mems
        # weighted sampling without replacement (importance + noise)
        keyed = sorted(mems, key=lambda m: -(m.importance + 0.7 * rnd()))
        return keyed[:n]

    # -- dynamics: decay & forgetting (run during sleep) -------------------
    def decay_and_forget(self, *, decay_lambda: float = 0.12,
                         stability_access: float = 0.6,
                         stability_importance: float = 2.0,
                         archive_threshold: float = 0.15,
                         purge_after_days: float = 30.0) -> dict:
        """Ebbinghaus-style decay: strength *= exp(-lambda*days / stability).

        Stability grows with rehearsal (access_count) and importance, so
        well-used or important memories become durable. Weak memories are
        archived (invisible to recall); old archives are purged for good.
        """
        now = self.now()
        decayed = archived = 0
        for r in self.db.execute(
                "SELECT id, last_access_ts, access_count, importance, strength,"
                " kind, consolidated FROM memories WHERE archived_ts IS NULL"):
            days = max(0.0, now - r["last_access_ts"]) / 86400.0
            stability = 1.0 + stability_access * r["access_count"] \
                + stability_importance * r["importance"]
            # consolidated episodics fade faster: their gist lives elsewhere
            lam = decay_lambda * (1.6 if (r["kind"] == "episodic" and r["consolidated"]) else 1.0)
            s = r["strength"] * math.exp(-lam * days / stability)
            if s < archive_threshold:
                self.db.execute("UPDATE memories SET strength=?, archived_ts=? WHERE id=?",
                                (s, now, r["id"]))
                archived += 1
            else:
                self.db.execute("UPDATE memories SET strength=? WHERE id=?", (s, r["id"]))
                decayed += 1
        purged = self.db.execute(
            "DELETE FROM memories WHERE archived_ts IS NOT NULL AND archived_ts < ?",
            (now - purge_after_days * 86400.0,)).rowcount
        self.db.commit()
        return {"decayed": decayed, "archived": archived, "purged": purged}

    def mark_consolidated(self, ids: list[int]) -> None:
        self.db.executemany("UPDATE memories SET consolidated=1 WHERE id=?",
                            [(i,) for i in ids])
        self.db.commit()

    # -- self model ---------------------------------------------------------
    def get_self_model(self) -> tuple[str, int]:
        r = self.db.execute("SELECT text, version FROM self_model WHERE id=1").fetchone()
        return (r["text"], r["version"]) if r else ("", 0)

    def set_self_model(self, text: str) -> int:
        _, v = self.get_self_model()
        self.db.execute(
            "INSERT INTO self_model (id, text, version, updated_ts) VALUES (1,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET text=?, version=?, updated_ts=?",
            (text, v + 1, self.now(), text, v + 1, self.now()))
        self.db.commit()
        return v + 1

    # -- goals ---------------------------------------------------------------
    def add_goal(self, text: str, priority: float = 0.5) -> int:
        cur = self.db.execute(
            "INSERT INTO goals (text, priority, created_ts) VALUES (?,?,?)",
            (text, priority, self.now()))
        self.db.commit()
        return int(cur.lastrowid)

    def complete_goal(self, goal_id: int) -> None:
        self.db.execute("UPDATE goals SET status='done', done_ts=? WHERE id=?",
                        (self.now(), goal_id))
        self.db.commit()

    def active_goals(self, limit: int = 5) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM goals WHERE status='active'"
            " ORDER BY priority DESC, id LIMIT ?", (limit,)).fetchall()

    # -- meta ------------------------------------------------------------------
    def get_meta(self, key: str, default: str = "") -> str:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=?", (key, value, value))
        self.db.commit()

    # -- stats -------------------------------------------------------------------
    def stats(self) -> dict:
        out = {"live": {}, "archived": 0, "total": 0}
        for r in self.db.execute(
                "SELECT kind, COUNT(*) n FROM memories"
                " WHERE archived_ts IS NULL GROUP BY kind"):
            out["live"][r["kind"]] = r["n"]
        out["archived"] = self.db.execute(
            "SELECT COUNT(*) n FROM memories WHERE archived_ts IS NOT NULL").fetchone()["n"]
        out["total"] = self.db.execute("SELECT COUNT(*) n FROM memories").fetchone()["n"]
        return out
