"""
TalentScout — Session Storage
================================
Provides a simple, swappable persistence layer for candidate sessions.

默认 Implementation:
    ``SessionStorage`` writes each session as a JSON file in a local
    ``sessions/`` directory and keeps an in-memory registry for fast
    lookup.  No external database is required to run the application.

Extending:
    Subclass ``BaseSessionStorage`` and override ``save`` / ``load`` /
    ``list_sessions`` to plug in Redis, DynamoDB, SQLite, etc.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from models.candidate import CandidateSession

logger = logging.getLogger(__name__)

# Default storage directory, relative to project root
_DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class BaseSessionStorage(ABC):
    """Interface that all storage backends must implement."""

    @abstractmethod
    def save(self, session: CandidateSession) -> None:
        """Persist a ``CandidateSession``. Overwrites any existing entry with the same ID."""

    @abstractmethod
    def load(self, session_id: str) -> Optional[CandidateSession]:
        """
        Retrieve a session by its ID.

        Returns:
            A ``CandidateSession`` instance, or ``None`` if not found.
        """

    @abstractmethod
    def list_sessions(self) -> List[CandidateSession]:
        """Return all stored sessions, ordered by creation time (ascending)."""


# ---------------------------------------------------------------------------
# File-Backed + In-Memory Implementation
# ---------------------------------------------------------------------------

class SessionStorage(BaseSessionStorage):
    """
    Dual-layer storage: in-memory registry (fast) backed by JSON files (durable).

    Files are stored at ``{sessions_dir}/{session_id}.json``.

    Args:
        sessions_dir: Directory where session JSON files are written.
                      Defaults to ``<project_root>/sessions/``.
    """

    def __init__(self, sessions_dir: Optional[Path] = None) -> None:
        self._dir: Path = sessions_dir or _DEFAULT_SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CandidateSession] = {}

        # Hydrate cache from existing files on startup
        self._load_existing_sessions()
        logger.info(
            "SessionStorage ready — dir=%s, loaded=%d session(s)",
            self._dir,
            len(self._cache),
        )

    # ------------------------------------------------------------------
    # BaseSessionStorage interface
    # ------------------------------------------------------------------

    def save(self, session: CandidateSession) -> None:
        """
        Persist the session to memory and disk.

        Args:
            session: The ``CandidateSession`` to save.
        """
        self._cache[session.session_id] = session
        file_path = self._dir / f"{session.session_id}.json"
        try:
            file_path.write_text(
                json.dumps(session.model_dump_safe(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("Session written to %s", file_path)
        except OSError as exc:
            logger.error("Failed to write session file %s: %s", file_path, exc)
            raise

    def load(self, session_id: str) -> Optional[CandidateSession]:
        """
        Load a session from the in-memory cache or from disk as fallback.

        Args:
            session_id: UUID string of the target session.

        Returns:
            ``CandidateSession`` if found, else ``None``.
        """
        if session_id in self._cache:
            return self._cache[session_id]

        file_path = self._dir / f"{session_id}.json"
        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            session = CandidateSession.model_validate(data)
            self._cache[session_id] = session
            return session
        except Exception as exc:
            logger.error("Failed to load session %s: %s", session_id, exc)
            return None

    def list_sessions(self) -> List[CandidateSession]:
        """
        Return all stored sessions sorted by ``created_at`` ascending.
        Reads from disk to ensure completeness even after a restart.
        """
        self._load_existing_sessions()
        return sorted(self._cache.values(), key=lambda s: s.created_at)

    # ------------------------------------------------------------------
    # Additional Helpers
    # ------------------------------------------------------------------

    def delete(self, session_id: str) -> bool:
        """
        Remove a session from memory and disk.

        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        existed = session_id in self._cache
        self._cache.pop(session_id, None)

        file_path = self._dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            existed = True
        return existed

    def count(self) -> int:
        """Return the total number of stored sessions."""
        return len(list(self._dir.glob("*.json")))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_existing_sessions(self) -> None:
        """Scan the sessions directory and hydrate the in-memory cache."""
        for json_file in self._dir.glob("*.json"):
            session_id = json_file.stem
            if session_id not in self._cache:
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    self._cache[session_id] = CandidateSession.model_validate(data)
                except Exception as exc:
                    logger.warning("Skipping corrupt session file %s: %s", json_file, exc)
