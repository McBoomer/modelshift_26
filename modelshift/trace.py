"""Trace storage: every captured production LLM call lands here (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field

from .core import LLMRequest, LLMResponse


@dataclass
class Trace:
    request: LLMRequest
    response: LLMResponse
    provider: str
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    tags: dict = field(default_factory=dict)


class TraceStore:
    def __init__(self, path: str = "modelshift_traces.db"):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                created_at REAL,
                provider TEXT,
                model TEXT,
                request TEXT,
                response TEXT,
                tags TEXT
            )"""
        )
        self._conn.commit()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.id,
                    trace.created_at,
                    trace.provider,
                    trace.model,
                    json.dumps(trace.request.to_dict(), default=str),
                    json.dumps(trace.response.to_dict(), default=str),
                    json.dumps(trace.tags),
                ),
            )
            self._conn.commit()

    def all(self, model: str | None = None, since: float | None = None) -> list[Trace]:
        sql = "SELECT id, created_at, provider, model, request, response, tags FROM traces WHERE 1=1"
        args: list = []
        if model:
            sql += " AND model = ?"
            args.append(model)
        if since:
            sql += " AND created_at >= ?"
            args.append(since)
        sql += " ORDER BY created_at"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            Trace(
                id=r[0],
                created_at=r[1],
                provider=r[2],
                model=r[3],
                request=LLMRequest.from_dict(json.loads(r[4])),
                response=LLMResponse.from_dict(json.loads(r[5])),
                tags=json.loads(r[6] or "{}"),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
