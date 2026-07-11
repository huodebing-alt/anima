"""Perception: pluggable sensors that feed percepts into the mind.

A percept is a dict: {"source": str, "text": str, "importance": float,
"ts": float, "kind": "message"|"observation"|"control"}.

Shipped sensors:
  InboxSensor   — user messages dropped as JSON files into runtime/inbox/
  ClockSensor   — coarse time awareness (day-part transitions)
  WatchSensor   — notices files appearing/changing in a watched folder
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator


class InboxSensor:
    """Messages, control commands, and injected percepts — one JSON file each.

    Besides user chat ({"text": ...}) and control ({"kind": "control"}),
    external device layers (e.g. the web UI's camera/microphone) can inject
    arbitrary percepts by specifying source/kind/importance:
      {"text": "...", "source": "vision", "kind": "observation",
       "importance": 0.5}
    """

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir

    def poll(self) -> Iterator[dict]:
        for p in sorted(self.dir.glob("*.json")):
            born = 0.0
            try:  # filenames start with the drop timestamp
                born = float(p.name.split("-")[0])
            except ValueError:
                pass
            try:
                msg = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                msg = None
            try:
                p.unlink()
            except OSError:
                pass
            if not msg or not msg.get("text"):
                continue
            # a stale control command must not fire on a later boot: the
            # daemon it addressed is gone (e.g. "stop" outliving a restart)
            if msg.get("kind") == "control" and born and \
                    time.time() - born > 120:
                continue
            try:
                importance = max(0.0, min(1.0, float(msg.get("importance", 0.6))))
            except (TypeError, ValueError):
                importance = 0.6
            yield {
                "source": str(msg.get("source", "user"))[:24],
                "kind": msg.get("kind", "message"),
                "text": str(msg["text"])[:4000],
                "importance": importance,
                "ts": time.time(),
            }


class ClockSensor:
    """Emits a percept when the part of day changes (morning/afternoon/...)."""

    PARTS = ((5, "early morning"), (9, "morning"), (12, "midday"),
             (14, "afternoon"), (18, "evening"), (22, "night"), (24, "late night"))

    def __init__(self):
        self.last_part = self._part()

    def _part(self) -> str:
        h = time.localtime().tm_hour
        for edge, name in self.PARTS:
            if h < edge:
                return name
        return "late night"

    def poll(self) -> Iterator[dict]:
        part = self._part()
        if part != self.last_part:
            self.last_part = part
            yield {
                "source": "clock",
                "kind": "observation",
                "text": f"It is now {part} ({time.strftime('%H:%M, %A %B %d')}).",
                "importance": 0.1,
                "ts": time.time(),
            }


class WatchSensor:
    """Notices files created or modified in a watched directory."""

    MAX_PREVIEW = 800

    def __init__(self, watch_dir: Path):
        self.dir = watch_dir
        self.seen: dict[str, float] = {}
        if self.dir.exists():  # existing files are old news at boot
            for p in self.dir.iterdir():
                if p.is_file():
                    self.seen[p.name] = p.stat().st_mtime

    def poll(self) -> Iterator[dict]:
        if not self.dir.exists():
            return
        for p in sorted(self.dir.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            mtime = p.stat().st_mtime
            if self.seen.get(p.name) == mtime:
                continue
            new = p.name not in self.seen
            self.seen[p.name] = mtime
            preview = ""
            try:
                raw = p.read_bytes()[: self.MAX_PREVIEW * 4]
                preview = raw.decode("utf-8", errors="replace")[: self.MAX_PREVIEW]
            except OSError:
                pass
            verb = "appeared in" if new else "changed in"
            text = f"A file '{p.name}' {verb} my watch folder."
            if preview.strip():
                text += f" It begins: {preview.strip()!r}"
            yield {
                "source": "watch",
                "kind": "observation",
                "text": text,
                "importance": 0.5,
                "ts": time.time(),
            }
