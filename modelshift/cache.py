"""On-disk response cache so repeated live calls cost nothing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .core import LLMRequest, LLMResponse
from .providers import Provider


class CachedProvider(Provider):
    """Wrap any Provider; identical (provider, model, request) returns the stored response.

    Put CostTracker *inside* this wrapper so cache hits are never charged:
    CachedProvider(CostTracker(real_provider, ...), path).
    """

    def __init__(self, inner: Provider, path: str | Path):
        self.inner = inner
        self.name = inner.name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self._db.commit()
        self._lock = threading.Lock()  # replay calls providers from many threads
        self.hits = self.misses = 0

    def key(self, req: LLMRequest, model: str) -> str:
        blob = json.dumps([self.inner.name, model, req.to_dict()], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, req: LLMRequest, model: str) -> LLMResponse:
        k = self.key(req, model)
        with self._lock:
            row = self._db.execute("SELECT response FROM cache WHERE key = ?", (k,)).fetchone()
        if row:
            self.hits += 1
            return LLMResponse.from_dict(json.loads(row[0]))
        # Errors propagate and are never cached.
        resp = self.inner.complete(req, model)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(resp.to_dict())))
            self._db.commit()
            self.misses += 1
        return resp

    _complete = complete

    def close(self) -> None:
        self._db.close()
