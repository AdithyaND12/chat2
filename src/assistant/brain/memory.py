"""Conversation transcript persistence on top of the SQLite store."""

from __future__ import annotations

from assistant.scheduler.store import Database


class ConversationStore:
    def __init__(self, db: Database, session_id: str = "default") -> None:
        self.db = db
        self.session_id = session_id

    def log(self, text: str, response: str = "", source: str = "voice") -> None:
        self.db.add_transcript(
            text,
            response=response,
            source=source,
            session_id=self.session_id,
        )

    def history(self, limit: int = 100) -> list[dict]:
        return self.db.transcripts(limit=limit)
