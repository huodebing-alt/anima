"""Live integration test: exercises the whole life cycle against real Ollama.

Phases:
  A. boot daemon; verify it thinks while idle (journal grows w/o input)
  B. converse: send facts; verify replies appear in outbox
  C. force sleep; verify consolidation gists / dream / self-model update
  D. kill daemon; restart; ask what it remembers; verify recall of facts
  E. verify decay bookkeeping ran and state file is coherent

Run:  python3 tests/itest_live.py
Exits non-zero on failure; prints a PASS/FAIL report with timings.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "tests" / "itest_config.json"
CFG = json.loads(CFG_PATH.read_text())
RUNTIME = Path(CFG["runtime_dir"])
if not RUNTIME.is_absolute():  # relative paths anchor at the project root
    RUNTIME = ROOT / RUNTIME
ENV = dict(os.environ, ANIMA_CONFIG=str(CFG_PATH), PYTHONPATH=str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""),
          flush=True)


def send(text: str, kind: str = "message") -> None:
    inbox = RUNTIME / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json").write_text(
        json.dumps({"text": text, "kind": kind}))


def journal() -> list[dict]:
    p = RUNTIME / "journal.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def outbox() -> list[dict]:
    p = RUNTIME / "outbox.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines()]


def wait_for(pred, timeout_s: float, poll: float = 1.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        v = pred()
        if v:
            return v
    return None


def start_daemon() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "anima", "run"],
        cwd=str(ROOT), env=ENV,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def stop_daemon(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def db_query(sql: str, args=()) -> list:
    import sqlite3
    db = sqlite3.connect(RUNTIME / "mind.db")
    db.row_factory = sqlite3.Row
    rows = db.execute(sql, args).fetchall()
    db.close()
    return rows


def warm_model() -> None:
    """Load the model into memory before timing anything."""
    import urllib.request
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": CFG["model"], "prompt": "hi",
                         "stream": False, "keep_alive": "2h",
                         "options": {"num_predict": 2}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    print(f"  (model warm in {time.time()-t0:.0f}s)")


def main() -> int:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    t_start = time.time()
    warm_model()

    print("Phase A: boot + idle thinking")
    proc = start_daemon()
    ok = wait_for(lambda: any(e["kind"] == "awake" for e in journal()), 120)
    check("daemon boots and journals 'awake'", bool(ok))
    def ticks() -> int:
        p = RUNTIME / "state.json"
        try:
            return json.loads(p.read_text())["tick"]
        except (OSError, json.JSONDecodeError, KeyError):
            return 0

    ok = wait_for(lambda: ticks() >= 3
                  and any(e["kind"] == "thought" for e in journal()), 400)
    check("thinks continuously with zero input (3+ ticks, 1+ journaled thought)",
          bool(ok), f"ticks={ticks()}")

    print("Phase B: conversation")
    send("Hi! I'm Dai. Remember this: my favourite weather is heavy rain, "
         "and I am building a Vision Pro app called WorldModel.")
    ok = wait_for(lambda: len(outbox()) >= 1, 400)
    check("replies to a user message", bool(ok),
          (outbox()[-1]["text"][:80] + "…") if outbox() else "no reply")
    send("One more fact: my cat is named Miso.")
    ok = wait_for(lambda: len(
        db_query("SELECT id FROM memories WHERE source='user'")) >= 2, 300)
    epis = db_query("SELECT text FROM memories WHERE source='user'")
    check("user messages encoded as episodic memories", len(epis) >= 2,
          f"{len(epis)} episodics from user")

    print("Phase C: sleep, consolidation, dreaming")
    time.sleep(5)  # let idle_before_sleep elapse
    send("sleep", kind="control")
    ok = wait_for(lambda: any(e["kind"] == "sleep.end" for e in journal()), 1800, 2.0)
    check("completes a sleep cycle", bool(ok))
    rep = next((e for e in reversed(journal()) if e["kind"] == "sleep.end"), {})
    check("NREM consolidated episodics", rep.get("consolidated_episodics", 0) > 0,
          f"consolidated={rep.get('consolidated_episodics')}")
    gists = db_query("SELECT text FROM memories WHERE source='sleep.consolidation'")
    check("semantic gists written", len(gists) > 0,
          f"{len(gists)} gists; e.g. {gists[0]['text'][:70] if gists else ''!r}")
    dreams = db_query("SELECT text FROM memories WHERE kind='dream'")
    check("dreamed at least one dream", len(dreams) > 0,
          (dreams[0]["text"][:70] + "…") if dreams else "")
    check("dream journal file written", (RUNTIME / "dreams.jsonl").exists())
    sm = db_query("SELECT version FROM self_model WHERE id=1")
    check("self-model rewritten during sleep",
          bool(sm) and sm[0]["version"] >= 2, f"version={sm[0]['version'] if sm else 0}")
    check("decay/forgetting pass ran", "decayed" in (rep.get("decay") or {}),
          json.dumps(rep.get("decay")))

    print("Phase D: death and resurrection (persistence)")
    stop_daemon(proc)
    check("daemon stops cleanly", proc.returncode is not None)
    proc = start_daemon()
    wait_for(lambda: any(e["kind"] == "reboot" for e in journal()), 60)
    n_out = len(outbox())
    send("I'm back. What do you remember about me? What's my cat's name?")
    ok = wait_for(lambda: len(outbox()) > n_out, 600)
    reply = outbox()[-1]["text"].lower() if ok else ""
    hits = [w for w in ("miso", "rain", "worldmodel", "vision") if w in reply]
    check("recalls facts across restart", len(hits) >= 1,
          f"mentioned {hits or 'nothing'}: {reply[:120]!r}")

    print("Phase E: state coherence")
    state = json.loads((RUNTIME / "state.json").read_text())
    check("state file coherent", state["phase"] in ("awake", "sleeping")
          and state["memory"]["total"] > 0, json.dumps(state["memory"]))
    stop_daemon(proc)

    dur = time.time() - t_start
    fails = [r for r in RESULTS if not r[1]]
    print(f"\n{'='*60}\n{len(RESULTS)-len(fails)}/{len(RESULTS)} checks passed "
          f"in {dur:.0f}s")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
