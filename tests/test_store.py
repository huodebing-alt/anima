"""Deterministic unit tests for memory mechanics — no LLM server needed.

Fake embeddings: bag-of-words over a tiny vocabulary, so cosine similarity
behaves intuitively. Fake clock: controllable virtual time.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anima.store import MemoryStore, cosine  # noqa: E402
from anima.sleep import _cluster  # noqa: E402
from anima.store import Memory  # noqa: E402

VOCAB = ["cat", "dog", "moon", "code", "python", "dream", "coffee", "music",
         "dai", "walk", "sea", "rain"]


def fake_embed(text: str) -> list[float]:
    words = text.lower().replace(",", " ").replace(".", " ").split()
    v = [float(sum(1 for w in words if t in w)) for t in VOCAB]
    return v if any(v) else [0.001] * len(VOCAB)


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.store = MemoryStore(Path(self.tmp.name) / "t.db", fake_embed,
                                 now_fn=self.clock)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    # -- retrieval ---------------------------------------------------------
    def test_recall_ranks_by_relevance(self):
        self.store.remember("episodic", "the cat sat by the window")
        self.store.remember("episodic", "python code needs a refactor")
        self.store.remember("episodic", "walking in the rain by the sea")
        top = self.store.recall("my cat is hungry", k=1)
        self.assertIn("cat", top[0].text)

    def test_recall_recency_breaks_ties(self):
        # texts must differ in fake-vocab space or dedup merges them
        a = self.store.remember("episodic", "dog walk in the morning")
        self.clock.advance(3600 * 48)
        b = self.store.remember("episodic", "dog music at the window")
        top = self.store.recall("dog", k=2)
        self.assertEqual(top[0].id, b, "fresher memory should outrank older twin")
        self.assertEqual(top[1].id, a)

    def test_importance_boosts_rank(self):
        self.store.remember("episodic", "coffee at 9", importance=0.05)
        vip = self.store.remember("episodic", "coffee with dai, big news",
                                  importance=0.95)
        top = self.store.recall("coffee", k=1)
        self.assertEqual(top[0].id, vip)

    def test_recall_reinforces(self):
        mid = self.store.remember("episodic", "music on the radio")
        before = self.store.get(mid)
        # decay it a bit first so the bump is visible below the 1.0 cap
        self.clock.advance(86400 * 5)
        self.store.decay_and_forget()
        decayed = self.store.get(mid).strength
        self.assertLess(decayed, before.strength)
        self.store.recall("music", k=1)
        after = self.store.get(mid)
        self.assertEqual(after.access_count, before.access_count + 1)
        self.assertAlmostEqual(after.strength, decayed + 0.10, places=5)

    def test_recall_excludes_archived(self):
        mid = self.store.remember("episodic", "cat on the roof", importance=0.0)
        self.clock.advance(86400 * 365)  # a year untouched
        self.store.decay_and_forget()
        self.assertIsNotNone(self.store.get(mid))  # still exists (archived)
        top = self.store.recall("cat", k=5)
        self.assertNotIn(mid, [m.id for m in top])

    # -- decay & forgetting ---------------------------------------------------
    def test_decay_monotonic(self):
        mid = self.store.remember("episodic", "rain all day", importance=0.3)
        s0 = self.store.get(mid).strength
        self.clock.advance(86400 * 3)
        self.store.decay_and_forget()
        s1 = self.store.get(mid).strength
        self.assertLess(s1, s0)

    def test_rehearsal_slows_decay(self):
        weak = self.store.remember("episodic", "walk by the sea", importance=0.3)
        strong = self.store.remember("episodic", "walk with the dog", importance=0.3)
        for _ in range(5):  # rehearse one of them
            hits = self.store.recall("dog", k=1)
            self.assertEqual(hits[0].id, strong)
        # reset strengths equal so only stability differs
        self.store.db.execute("UPDATE memories SET strength=1.0")
        self.store.db.commit()
        self.clock.advance(86400 * 10)
        self.store.decay_and_forget()
        self.assertGreater(self.store.get(strong).strength,
                           self.store.get(weak).strength)

    def test_importance_slows_decay(self):
        triv = self.store.remember("episodic", "walk to the shop", importance=0.0)
        core = self.store.remember("semantic", "dai is my person", importance=1.0)
        self.clock.advance(86400 * 10)
        self.store.decay_and_forget()
        self.assertGreater(self.store.get(core).strength,
                           self.store.get(triv).strength)

    def test_consolidated_episodics_fade_faster(self):
        kept = self.store.remember("episodic", "moon rise over hills")
        gone = self.store.remember("episodic", "sea mist at dawn")
        self.store.mark_consolidated([gone])
        self.clock.advance(86400 * 6)
        self.store.decay_and_forget()
        self.assertGreater(self.store.get(kept).strength,
                           self.store.get(gone).strength)

    def test_archive_then_purge(self):
        mid = self.store.remember("episodic", "dust mote", importance=0.0)
        self.clock.advance(86400 * 400)
        r = self.store.decay_and_forget(purge_after_days=30.0)
        self.assertEqual(r["archived"], 1)
        self.clock.advance(86400 * 40)
        r = self.store.decay_and_forget(purge_after_days=30.0)
        self.assertEqual(r["purged"], 1)
        self.assertIsNone(self.store.get(mid))

    def test_duplicate_memory_reinforces_not_duplicates(self):
        a = self.store.remember("episodic", "the cat sat on the mat")
        b = self.store.remember("episodic", "the cat sat on the mat")
        self.assertEqual(a, b, "identical memory should merge into the original")
        m = self.store.get(a)
        self.assertEqual(m.access_count, 1)
        rows = self.store.db.execute(
            "SELECT COUNT(*) n FROM memories").fetchone()
        self.assertEqual(rows["n"], 1)

    # -- consolidation clustering ------------------------------------------------
    def test_cluster_groups_similar(self):
        mems = [
            Memory(1, "episodic", "cat cat cat", fake_embed("cat cat cat"),
                   0, 0, 0, 0.3, 1.0),
            Memory(2, "episodic", "cat and cat", fake_embed("cat and cat"),
                   0, 0, 0, 0.3, 1.0),
            Memory(3, "episodic", "python code", fake_embed("python code"),
                   0, 0, 0, 0.3, 1.0),
        ]
        clusters = _cluster(mems, sim_threshold=0.55, max_batch=8)
        self.assertEqual(len(clusters), 2)
        self.assertEqual({m.id for m in clusters[0]}, {1, 2})

    # -- identity & persistence ----------------------------------------------------
    def test_self_model_versioning(self):
        self.assertEqual(self.store.get_self_model(), ("", 0))
        v1 = self.store.set_self_model("I am new.")
        v2 = self.store.set_self_model("I am growing.")
        self.assertEqual((v1, v2), (1, 2))
        self.assertEqual(self.store.get_self_model(), ("I am growing.", 2))

    def test_goals(self):
        gid = self.store.add_goal("learn about dai", priority=0.9)
        self.assertEqual(len(self.store.active_goals()), 1)
        self.store.complete_goal(gid)
        self.assertEqual(len(self.store.active_goals()), 0)

    def test_persistence_across_reopen(self):
        self.store.remember("semantic", "dai likes rain and python",
                            importance=0.8)
        self.store.set_self_model("I persist.")
        path = Path(self.tmp.name) / "t.db"
        self.store.close()
        store2 = MemoryStore(path, fake_embed, now_fn=self.clock)
        top = store2.recall("what does dai like? rain python", k=1)
        self.assertIn("rain", top[0].text)
        self.assertEqual(store2.get_self_model()[0], "I persist.")
        store2.close()
        self.store = MemoryStore(path, fake_embed, now_fn=self.clock)

    def test_stats(self):
        self.store.remember("episodic", "cat")
        self.store.remember("semantic", "dog fact")
        s = self.store.stats()
        self.assertEqual(s["live"], {"episodic": 1, "semantic": 1})
        self.assertEqual(s["total"], 2)

    def test_cosine_basics(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine([1, 1], [1, 1]),
                               1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
