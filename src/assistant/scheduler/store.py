"""SQLite storage for conversation transcripts and assistant events."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from assistant.logging import get_logger

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'system',
                    payload TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT 'default',
                    source TEXT NOT NULL DEFAULT 'voice',
                    text TEXT NOT NULL,
                    response TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_transcripts_ts ON transcripts(ts DESC);
                """
            )
            self._conn.commit()

    def add_event(self, kind: str, payload: dict | None = None, source: str = "system") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (ts, kind, source, payload) VALUES (?, ?, ?, ?)",
                (_utcnow(), kind, source, json.dumps(payload or {}, ensure_ascii=False)),
            )
            self._conn.commit()

    def events(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, source, payload FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "ts": row["ts"],
                "kind": row["kind"],
                "source": row["source"],
                "payload": json.loads(row["payload"]),
            }
            for row in reversed(rows)
        ]

    def add_transcript(
        self,
        text: str,
        response: str = "",
        source: str = "voice",
        session_id: str = "default",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO transcripts (ts, session_id, source, text, response) "
                "VALUES (?, ?, ?, ?, ?)",
                (_utcnow(), session_id, source, text, response),
            )
            self._conn.commit()

    def update_response(self, text: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE transcripts SET response = ? "
                "WHERE id = (SELECT id FROM transcripts WHERE text = ? "
                "ORDER BY id DESC LIMIT 1)",
                (response, text),
            )
            self._conn.commit()

    def transcripts(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, session_id, source, text, response "
                "FROM transcripts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "ts": row["ts"],
                "session_id": row["session_id"],
                "source": row["source"],
                "text": row["text"],
                "response": row["response"],
            }
            for row in reversed(rows)
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
