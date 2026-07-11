"""Anima web UI server — pure stdlib (http.server + SSE).

Serves the static UI from ui/ and a small JSON API in front of the mind's
file/SQLite state. The daemon is untouched: the server talks to it exactly
like any other client — inbox files in, journal/outbox/state out.

  GET  /                    the app
  GET  /api/state           daemon state snapshot
  GET  /api/settings        persona + granted powers
  POST /api/settings        update persona/powers (hot-reloaded by daemon)
  POST /api/message         {"text"} user chat -> inbox
  POST /api/percept         {"text","source","importance"} device percepts
  POST /api/control         {"command": "sleep"|"stop"}
  GET  /api/history         recent chat + mind events (for reload)
  GET  /api/stream          SSE: live journal entries + state heartbeats
  GET  /api/memories        long-term memory inspector
  GET  /api/dreams          dream journal
  GET  /api/self            self-model document
  POST /api/vision          {"image": base64-jpeg} -> description percept
                            (requires a vision-capable model in settings)

Run:  python3 -m anima ui  [--port 8765]
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):  # run directly: python3 anima/anima/server.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from anima import config as config_mod
else:
    from . import config as config_mod

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
CHAT_KINDS = ("say", "percept", "message", "thought", "sleep.begin",
              "sleep.end", "sleep.dream", "remember", "goal", "browse",
              "file", "grant", "awake", "reboot", "birth", "control")

DEFAULT_SETTINGS = {
    "agent_name": "Anima",
    "user_name": "you",
    "theme": "aurora",
    "mode": "dark",
    "avatar": "orb-aurora",
    "voice": "",
    "voice_rate": 1.0,
    "voice_pitch": 1.0,
    "auto_speak": True,
    "workspace_dir": "",
    "allow_browse": False,
    "allow_files": False,
    "vision_model": "",
}


class Api:
    """Shared state + helpers (the handler class is per-request)."""

    def __init__(self, cfg):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.settings_path = cfg.runtime / "settings.json"
        self.lock = threading.Lock()

    # -- settings ---------------------------------------------------------
    def settings(self) -> dict:
        s = dict(DEFAULT_SETTINGS)
        try:
            s.update(json.loads(self.settings_path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
        return s

    def save_settings(self, patch: dict) -> dict:
        with self.lock:
            s = self.settings()
            for k in DEFAULT_SETTINGS:
                if k in patch:
                    s[k] = patch[k]
            self.settings_path.write_text(json.dumps(s, indent=2))
        return s

    # -- daemon I/O ---------------------------------------------------------
    def drop_inbox(self, payload: dict) -> None:
        self.cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
        p = self.cfg.inbox_dir / f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False))

    def state(self) -> dict:
        try:
            st = json.loads(self.cfg.state_path.read_text())
            st["state_age_s"] = round(time.time() - st.get("ts", 0), 1)
            st["daemon_alive"] = st["state_age_s"] < max(
                600.0, self.cfg.tick_max_s * 2)
            return st
        except (OSError, json.JSONDecodeError):
            return {"phase": "not running", "daemon_alive": False}

    def journal_entries(self, limit: int = 80) -> list[dict]:
        try:
            lines = self.cfg.journal_path.read_text().splitlines()[-limit * 2:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") in CHAT_KINDS:
                out.append(e)
        return out[-limit:]

    # -- read-only db views ----------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(f"file:{self.cfg.db_path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def memories(self, kind: str = "", limit: int = 60) -> list[dict]:
        try:
            db = self._db()
        except sqlite3.Error:
            return []
        q = ("SELECT id, kind, text, importance, strength, access_count,"
             " created_ts, source FROM memories WHERE archived_ts IS NULL")
        args: list = []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY created_ts DESC LIMIT ?"
        args.append(limit)
        try:
            rows = [dict(r) for r in db.execute(q, args)]
        except sqlite3.Error:
            rows = []
        db.close()
        return rows

    def dreams(self, limit: int = 20) -> list[dict]:
        try:
            lines = self.cfg.dream_journal_path.read_text().splitlines()
            return [json.loads(l) for l in lines[-limit:]][::-1]
        except (OSError, json.JSONDecodeError):
            return []

    def self_model(self) -> dict:
        try:
            db = self._db()
            r = db.execute(
                "SELECT text, version, updated_ts FROM self_model WHERE id=1"
            ).fetchone()
            db.close()
            return dict(r) if r else {"text": "", "version": 0}
        except sqlite3.Error:
            return {"text": "", "version": 0}

    # -- vision --------------------------------------------------------------
    def describe_image(self, b64: str) -> str:
        model = self.settings().get("vision_model", "").strip()
        if not model:
            raise ValueError("no vision model configured")
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": "You are the eyes of an ambient assistant. In one"
                           " short sentence, describe what this camera frame"
                           " shows. Mention people, actions, and notable"
                           " changes only.",
                "images": [b64],
            }],
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 60, "temperature": 0.3},
        }
        req = urllib.request.Request(
            self.cfg.ollama_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode())
        desc = (out.get("message") or {}).get("content", "").strip()
        if not desc:
            raise ValueError(out.get("error") or "vision model gave no answer")
        return desc


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        # -- plumbing -----------------------------------------------------
        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n or n > 8_000_000:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        def _static(self, name: str) -> None:
            f = (UI_DIR / name).resolve()
            if UI_DIR.resolve() not in f.parents and f != UI_DIR.resolve():
                self.send_error(404)
                return
            if not f.is_file():
                self.send_error(404)
                return
            ctype = {"html": "text/html", "css": "text/css",
                     "js": "text/javascript", "svg": "image/svg+xml",
                     "png": "image/png"}.get(f.suffix.lstrip("."), "text/plain")
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # -- GET ------------------------------------------------------------
        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            qs = {}
            if "?" in self.path:
                for kv in self.path.split("?", 1)[1].split("&"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        qs[k] = urllib.parse.unquote(v)
            if path in ("/", "/index.html"):
                self._static("index.html")
            elif path in ("/app.js", "/style.css"):
                self._static(path.lstrip("/"))
            elif path == "/api/state":
                self._json(api.state())
            elif path == "/api/settings":
                self._json(api.settings())
            elif path == "/api/history":
                self._json({"entries": api.journal_entries(
                    int(qs.get("limit", "80")))})
            elif path == "/api/memories":
                self._json({"memories": api.memories(
                    qs.get("kind", ""), int(qs.get("limit", "60")))})
            elif path == "/api/dreams":
                self._json({"dreams": api.dreams()})
            elif path == "/api/self":
                self._json(api.self_model())
            elif path == "/api/stream":
                self._sse()
            else:
                self.send_error(404)

        # -- POST --------------------------------------------------------------
        def do_POST(self):  # noqa: N802
            body = self._read_body()
            if self.path == "/api/message":
                text = str(body.get("text", "")).strip()
                if not text:
                    self._json({"error": "empty"}, 400)
                    return
                api.drop_inbox({"text": text[:4000], "kind": "message",
                                "source": "user"})
                self._json({"ok": True})
            elif self.path == "/api/percept":
                text = str(body.get("text", "")).strip()
                source = str(body.get("source", "sensor"))[:24]
                if not text or source == "user":
                    self._json({"error": "bad percept"}, 400)
                    return
                api.drop_inbox({
                    "text": text[:2000], "kind": "observation",
                    "source": source,
                    "importance": float(body.get("importance", 0.4)),
                })
                self._json({"ok": True})
            elif self.path == "/api/control":
                cmd = str(body.get("command", ""))
                if cmd not in ("sleep", "stop"):
                    self._json({"error": "unknown command"}, 400)
                    return
                api.drop_inbox({"text": cmd, "kind": "control"})
                self._json({"ok": True})
            elif self.path == "/api/settings":
                self._json(api.save_settings(body))
            elif self.path == "/api/vision":
                b64 = str(body.get("image", ""))
                if "," in b64:  # tolerate data-URL prefix
                    b64 = b64.split(",", 1)[1]
                try:
                    desc = api.describe_image(b64)
                except Exception as e:  # noqa: BLE001
                    self._json({"error": str(e)}, 422)
                    return
                api.drop_inbox({"text": f"Through my camera I see: {desc}",
                                "kind": "observation", "source": "vision",
                                "importance": 0.5})
                self._json({"description": desc})
            else:
                self.send_error(404)

        # -- SSE -----------------------------------------------------------------
        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def emit(event: str, data) -> bool:
                try:
                    self.wfile.write(
                        f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                        .encode())
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return False

            if not emit("state", api.state()):
                return
            fh = None
            try:
                fh = open(api.cfg.journal_path)
                fh.seek(0, 2)
            except OSError:
                pass
            last_state = 0.0
            while True:
                sent_any = False
                if fh is None:
                    try:
                        fh = open(api.cfg.journal_path)
                    except OSError:
                        fh = None
                if fh is not None:
                    line = fh.readline()
                    while line:
                        try:
                            e = json.loads(line)
                            if e.get("kind") in CHAT_KINDS:
                                if not emit("entry", e):
                                    return
                                sent_any = True
                        except json.JSONDecodeError:
                            pass
                        line = fh.readline()
                if time.time() - last_state > 3.0:
                    last_state = time.time()
                    if not emit("state", api.state()):
                        return
                if not sent_any:
                    time.sleep(0.4)

    return Handler


def serve(port: int = 8765, config_path: str | None = None) -> None:
    cfg = config_mod.load(config_path)
    api = Api(cfg)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(api))
    httpd.daemon_threads = True
    print(f"Anima UI on http://localhost:{port}  (runtime: {cfg.runtime_dir})")
    if not api.state().get("daemon_alive"):
        print("note: the mind daemon is not running — start it with:"
              "  python3 -m anima run &")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    serve(args.port, args.config)
