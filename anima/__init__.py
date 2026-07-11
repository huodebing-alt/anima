"""Anima — an always-awake agent with persistent memory, sleep, and dreams.

Runs a continuous observe→think→act loop on a small local LLM (via Ollama),
stores everything it experiences in a long-term memory with human-like
retrieval, reinforcement, decay and forgetting, and periodically sleeps to
consolidate episodic memories into semantic knowledge, reflect, and dream.
"""

__version__ = "0.1.0"
