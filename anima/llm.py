"""Minimal Ollama client. Pure stdlib (urllib) — no dependencies."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str, model: str, embed_model: str = "",
                 timeout_s: float = 120.0, keep_alive: str = "2h"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model or model
        self.timeout_s = timeout_s
        # an always-on mind must keep its substrate resident: without this,
        # Ollama unloads the model between quiet ticks and every wake-up pays
        # a ~40s reload on a memory-pressured machine
        self.keep_alive = keep_alive

    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise LLMError(f"ollama {path}: {e}") from e

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, model: str = "",
             json_mode: bool = False, temperature: float = 0.8,
             max_tokens: int = 512) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        out = self._post("/api/chat", payload)
        return (out.get("message") or {}).get("content", "")

    @staticmethod
    def _parse_json_lenient(text: str) -> dict | None:
        """Parse model JSON, tolerating chatter around it and truncation.

        Retrying a slow local generation is expensive; repairing a truncated
        tail is nearly free, so we try hard here before the caller retries.
        """
        candidates = [text]
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            candidates.append(text[s:e + 1])
        if s >= 0:  # truncated? try closing what's open
            body = text[s:]
            for suffix in ('"}', '"]}', "}", '"}}', "]}", "}}"):
                candidates.append(body + suffix)
        for c in candidates:
            try:
                obj = json.loads(c)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    def chat_json(self, system: str, user: str, *, model: str = "",
                  temperature: float = 0.6, max_tokens: int = 512,
                  retries: int = 1) -> dict:
        """Chat with JSON grammar enforcement; lenient parse, then retry."""
        last = ""
        for _ in range(retries + 1):
            last = self.chat(system, user, model=model, json_mode=True,
                             temperature=temperature, max_tokens=max_tokens)
            obj = self._parse_json_lenient(last)
            if obj is not None:
                return obj
        raise LLMError(f"unparseable JSON from model: {last[:200]!r}")

    def embed(self, text: str) -> list[float]:
        out = self._post("/api/embed", {"model": self.embed_model,
                                        "keep_alive": self.keep_alive,
                                        "input": text[:4000]})
        embs = out.get("embeddings") or []
        if not embs:
            raise LLMError(f"no embedding returned: {out.get('error')}")
        return embs[0]

    def alive(self) -> bool:
        try:
            req = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False
