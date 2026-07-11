"""Gated capabilities: web browsing and workspace file creation.

Both are OFF by default and enabled per-capability by the user (via the web
UI's Senses & Powers panel, which writes runtime/settings.json). Both are
bounded: browsing fetches one page at a time with size/time caps and returns
a plain-text extract; file writes are confined to the user-granted workspace
folder with sanitized names and capped sizes.
"""
from __future__ import annotations

import html
import html.parser
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

BROWSE_MAX_BYTES = 500_000
BROWSE_EXTRACT_CHARS = 1500
FILE_MAX_BYTES = 100_000
UA = "Mozilla/5.0 (compatible; Anima/0.1; +https://github.com/huodebing-alt/anima)"


class _TextExtractor(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def browse(url: str, timeout_s: float = 25.0) -> str:
    """Fetch one web page and return a compact plain-text digest."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not a web URL: {url[:80]!r}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(BROWSE_MAX_BYTES)
    text = raw.decode("utf-8", errors="replace")
    if "html" in ctype or text.lstrip()[:1] == "<":
        p = _TextExtractor()
        try:
            p.feed(text)
        except Exception:  # malformed HTML: fall back to tag stripping
            p.parts = [re.sub(r"<[^>]+>", " ", text)]
        body = html.unescape(" ".join(p.parts))
        title = html.unescape(p.title.strip())
    else:
        body, title = text, ""
    body = re.sub(r"\s+", " ", body).strip()[:BROWSE_EXTRACT_CHARS]
    head = f"[{title}] " if title else ""
    return f"{head}{body}" if body else "(the page had no readable text)"


def write_file(workspace: Path, name: str, content: str) -> Path:
    """Create/overwrite a file inside the granted workspace. Nowhere else."""
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise ValueError("no workspace folder granted")
    clean = os.path.basename(name.strip()).lstrip(".").replace("\x00", "")
    clean = re.sub(r"[^\w.\- ]", "_", clean)[:80] or "note.txt"
    if "." not in clean:
        clean += ".txt"
    path = (workspace / clean).resolve()
    if path.parent != workspace:  # paranoia: never escape the workspace
        raise ValueError("path escapes workspace")
    path.write_text(str(content)[:FILE_MAX_BYTES])
    return path
