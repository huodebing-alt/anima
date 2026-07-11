"""Anima CLI.

  python3 -m anima run [--max-ticks N] [--max-seconds S]   start the mind
  python3 -m anima talk [--thoughts]                        chat with it
  python3 -m anima say "text"                               one-shot message
  python3 -m anima control sleep|stop                       control command
  python3 -m anima status                                   state snapshot
  python3 -m anima dreams                                   dream journal
  python3 -m anima memories [--kind K] [--all]              inspect memory
  python3 -m anima self                                     show self-model
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid

from . import config as config_mod
from .store import MemoryStore


def _drop_inbox(cfg, text: str, kind: str = "message") -> None:
    cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.inbox_dir / f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json"
    p.write_text(json.dumps({"text": text, "kind": kind}))


def cmd_run(cfg, args) -> None:
    from .agent import Mind
    from .llm import Ollama
    llm = Ollama(cfg.ollama_url, cfg.model)
    if not llm.alive():
        print(f"error: no Ollama server at {cfg.ollama_url}", file=sys.stderr)
        sys.exit(1)
    mind = Mind(cfg)
    print(f"{cfg.agent_name} is awake (model={cfg.model}, "
          f"runtime={cfg.runtime_dir}). Ctrl-C to stop.")
    mind.run(max_ticks=args.max_ticks, max_seconds=args.max_seconds)
    print(f"{cfg.agent_name} stopped after {mind.tick_count} ticks.")


def cmd_ui(cfg, args) -> None:
    from .server import serve
    serve(args.port, getattr(args, "config", None))


def cmd_talk(cfg, args) -> None:
    print(f"Talking to {cfg.agent_name}. Type messages; /sleep forces sleep; "
          "/quit exits.")
    stop = threading.Event()

    def tail() -> None:
        # follow outbox (agent speech) and, with --thoughts, the journal
        files = [(cfg.outbox_path, "say")]
        if args.thoughts:
            files = [(cfg.journal_path, None)]
        handles = {}
        while not stop.is_set():
            for path, only in files:
                if path not in handles and path.exists():
                    h = open(path)
                    h.seek(0, 2)  # end: only new lines
                    handles[path] = (h, only)
            for h, only in list(handles.values()):
                line = h.readline()
                while line:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        e = None
                    if e:
                        if only == "say" or e.get("kind") == "say":
                            print(f"\n[{cfg.agent_name}] {e.get('text', '')}\n> ",
                                  end="", flush=True)
                        elif args.thoughts and e.get("kind") in (
                                "thought", "percept", "sleep.begin", "sleep.end",
                                "sleep.dream", "remember", "goal"):
                            print(f"\n  · {e.get('kind')}: "
                                  f"{str(e.get('text') or e.get('dream') or '')[:160]}"
                                  f"\n> ", end="", flush=True)
                    line = h.readline()
            time.sleep(0.3)

    t = threading.Thread(target=tail, daemon=True)
    t.start()
    try:
        while True:
            msg = input("> ").strip()
            if not msg:
                continue
            if msg in ("/quit", "/q"):
                break
            if msg == "/sleep":
                _drop_inbox(cfg, "sleep", kind="control")
                print("(sleep requested)")
                continue
            _drop_inbox(cfg, msg)
    except (EOFError, KeyboardInterrupt):
        pass
    stop.set()


def cmd_say(cfg, args) -> None:
    _drop_inbox(cfg, args.text)
    print("delivered.")


def cmd_control(cfg, args) -> None:
    _drop_inbox(cfg, args.command, kind="control")
    print(f"control '{args.command}' delivered.")


def cmd_status(cfg, args) -> None:
    if cfg.state_path.exists():
        state = json.loads(cfg.state_path.read_text())
        age = time.time() - state.get("ts", 0)
        state["state_age_s"] = round(age, 1)
        state["daemon_looks_alive"] = age < 600
        print(json.dumps(state, indent=2))
    else:
        print("no state file — the mind has never run here.")


def cmd_dreams(cfg, args) -> None:
    if not cfg.dream_journal_path.exists():
        print("no dreams yet.")
        return
    for line in cfg.dream_journal_path.read_text().splitlines():
        e = json.loads(line)
        print(f"--- {time.strftime('%Y-%m-%d %H:%M', time.localtime(e['ts']))}")
        print(e["dream"], "\n")


def _open_store(cfg) -> MemoryStore:
    return MemoryStore(cfg.db_path, embed_fn=lambda t: [])


def cmd_memories(cfg, args) -> None:
    store = _open_store(cfg)
    q = "SELECT * FROM memories"
    conds, params = [], []
    if not args.all:
        conds.append("archived_ts IS NULL")
    if args.kind:
        conds.append("kind=?")
        params.append(args.kind)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY created_ts DESC LIMIT ?"
    params.append(args.limit)
    for r in store.db.execute(q, params):
        flag = " [ARCHIVED]" if r["archived_ts"] else ""
        print(f"#{r['id']} {r['kind']:<10} imp={r['importance']:.2f} "
              f"str={r['strength']:.2f} acc={r['access_count']}{flag}\n"
              f"   {r['text'][:180]}")
    print("\nstats:", json.dumps(store.stats()))
    store.close()


def cmd_self(cfg, args) -> None:
    store = _open_store(cfg)
    text, version = store.get_self_model()
    print(f"self-model v{version}:\n\n{text or '(none)'}")
    store.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="anima")
    ap.add_argument("--config", default=None, help="path to config JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--max-ticks", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ui")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_ui)

    p = sub.add_parser("talk")
    p.add_argument("--thoughts", action="store_true",
                   help="also stream inner monologue")
    p.set_defaults(fn=cmd_talk)

    p = sub.add_parser("say")
    p.add_argument("text")
    p.set_defaults(fn=cmd_say)

    p = sub.add_parser("control")
    p.add_argument("command", choices=["sleep", "stop"])
    p.set_defaults(fn=cmd_control)

    for name, fn in (("status", cmd_status), ("dreams", cmd_dreams),
                     ("self", cmd_self)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)

    p = sub.add_parser("memories")
    p.add_argument("--kind", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(fn=cmd_memories)

    args = ap.parse_args()
    cfg = config_mod.load(args.config)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
