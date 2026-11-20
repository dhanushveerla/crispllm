"""SQLite-backed tiered memory store with FTS5 search and zero external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

import platformdirs


@dataclass
class Fact:
    """Exact architectural constraint, constant, signature, or decision."""

    id: int
    session_id: str
    scope: str
    kind: str
    key: str
    value: str
    source_turn: int = 0
    pinned: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class Episode:
    """Turn summary or conversational episode."""

    id: int
    session_id: str
    role: str
    summary: str
    raw_tokens: int = 0
    created_at: float = 0.0


class MemoryStore:
    """Zero-external-dependency embedded SQLite memory store.

    Guarantees WAL mode, concurrent thread-safety via RLock, and synchronized FTS5 indexing.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            data_dir = platformdirs.user_data_dir("edgemem")
            self.db_path: Path | str = Path(data_dir) / "memory.db"
        elif isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        self._is_memory = str(self.db_path) == ":memory:"
        self._lock = threading.RLock()

        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_turn INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL,
                UNIQUE(session_id, kind, key)
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                summary TEXT NOT NULL,
                raw_tokens INTEGER DEFAULT 0,
                created_at REAL
            );
            """
        )
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                fact_id UNINDEXED,
                session_id UNINDEXED,
                content
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_session_pinned ON facts(session_id, pinned);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_session_kind ON facts(session_id, kind);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);"
        )
        self._conn.commit()

    @property
    def journal_mode(self) -> str:
        """Current SQLite journal mode."""
        with self._lock:
            cur = self._conn.execute("PRAGMA journal_mode;")
            row = cur.fetchone()
            return str(row[0]).lower() if row else ""

    def upsert_fact(
        self,
        session_id: str,
        key: str,
        value: str,
        kind: str = "note",
        scope: str = "session",
        source_turn: int = 0,
        pinned: bool = False,
    ) -> int:
        """Insert or update a fact and synchronize FTS5 within the same transaction."""
        now = time.time()
        pinned_int = 1 if pinned else 0

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO facts (session_id, scope, kind, key, value, source_turn, pinned, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, kind, key) DO UPDATE SET
                    scope=excluded.scope,
                    value=excluded.value,
                    source_turn=excluded.source_turn,
                    pinned=excluded.pinned,
                    updated_at=excluded.updated_at
                RETURNING id;
                """,
                (session_id, scope, kind, key, value, source_turn, pinned_int, now, now),
            )
            fact_id = int(cur.fetchone()[0])

            # Synchronize FTS5 virtual table explicitly inside the same transaction
            self._conn.execute("DELETE FROM memory_fts WHERE fact_id = ?;", (str(fact_id),))
            content = f"{key} {value}"
            self._conn.execute(
                "INSERT INTO memory_fts (fact_id, session_id, content) VALUES (?, ?, ?);",
                (str(fact_id), session_id, content),
            )
            self._conn.commit()
            return fact_id

    def get_pinned_facts(
        self,
        session_id: str,
        include_global: bool = True,
        limit: int = 20,
    ) -> list[Fact]:
        """Fetch pinned facts for a session ordered by recency."""
        with self._lock:
            if include_global:
                sql = """
                    SELECT * FROM facts
                    WHERE pinned = 1 AND (session_id = ? OR scope = 'global')
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?;
                """
                cur = self._conn.execute(sql, (session_id, limit))
            else:
                sql = """
                    SELECT * FROM facts
                    WHERE pinned = 1 AND session_id = ?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?;
                """
                cur = self._conn.execute(sql, (session_id, limit))
            return [self._row_to_fact(row) for row in cur.fetchall()]

    def search_facts(
        self,
        query: str,
        session_id: str | None = None,
        include_global: bool = True,
        limit: int = 10,
    ) -> list[Fact]:
        """Search facts using sanitized FTS5 BM25 ranking."""
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return []

        unique_tokens = list(dict.fromkeys(tokens))
        fts_match_query = " OR ".join(f'"{t}"' for t in unique_tokens)

        with self._lock:
            params: list[Any] = [fts_match_query]
            session_clauses = []
            if session_id is not None:
                if include_global:
                    session_clauses.append("(f.session_id = ? OR f.scope = 'global')")
                    params.append(session_id)
                else:
                    session_clauses.append("f.session_id = ?")
                    params.append(session_id)

            where_session = f"AND {' AND '.join(session_clauses)}" if session_clauses else ""
            params.append(limit)

            sql = f"""
                SELECT f.id, f.session_id, f.scope, f.kind, f.key, f.value,
                       f.source_turn, f.pinned, f.created_at, f.updated_at
                FROM memory_fts fts
                JOIN facts f ON f.id = CAST(fts.fact_id AS INTEGER)
                WHERE memory_fts MATCH ?
                {where_session}
                ORDER BY fts.rank ASC, f.updated_at DESC
                LIMIT ?;
            """
            cur = self._conn.execute(sql, params)
            return [self._row_to_fact(row) for row in cur.fetchall()]

    def list_facts(
        self,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """List stored facts with optional session and kind filters."""
        with self._lock:
            clauses = []
            params: list[Any] = []
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            if kind is not None:
                clauses.append("kind = ?")
                params.append(kind)

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"SELECT * FROM facts {where} ORDER BY updated_at DESC, id DESC LIMIT ?;"
            params.append(limit)

            cur = self._conn.execute(sql, params)
            return [self._row_to_fact(row) for row in cur.fetchall()]

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a fact and remove its FTS5 index entry."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?;", (fact_id,))
            deleted = cur.rowcount > 0
            if deleted:
                self._conn.execute("DELETE FROM memory_fts WHERE fact_id = ?;", (str(fact_id),))
                self._conn.commit()
            return deleted

    def add_episode(
        self,
        session_id: str,
        role: str,
        summary: str,
        raw_tokens: int = 0,
    ) -> int:
        """Insert a turn summary episode."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO episodes (session_id, role, summary, raw_tokens, created_at)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id;
                """,
                (session_id, role, summary, raw_tokens, now),
            )
            episode_id = int(cur.fetchone()[0])
            self._conn.commit()
            return episode_id

    def list_episodes(self, session_id: str | None = None, limit: int = 50) -> list[Episode]:
        """List stored episodes."""
        with self._lock:
            if session_id is not None:
                cur = self._conn.execute(
                    "SELECT * FROM episodes WHERE session_id = ? ORDER BY created_at ASC, id ASC LIMIT ?;",
                    (session_id, limit),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM episodes ORDER BY created_at ASC, id ASC LIMIT ?;",
                    (limit,),
                )
            return [
                Episode(
                    id=int(row["id"]),
                    session_id=str(row["session_id"]),
                    role=str(row["role"]),
                    summary=str(row["summary"]),
                    raw_tokens=int(row["raw_tokens"]),
                    created_at=float(row["created_at"]),
                )
                for row in cur.fetchall()
            ]

    def clear_session(self, session_id: str) -> int:
        """Delete all facts, FTS entries, and episodes for a session."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE session_id = ?;", (session_id,))
            deleted_count = cur.rowcount
            self._conn.execute("DELETE FROM memory_fts WHERE session_id = ?;", (session_id,))
            self._conn.execute("DELETE FROM episodes WHERE session_id = ?;", (session_id,))
            self._conn.commit()
            return deleted_count

    def clear_all(self) -> int:
        """Purge all facts, FTS entries, and episodes from the store."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts;")
            deleted_count = cur.rowcount
            self._conn.execute("DELETE FROM memory_fts;")
            self._conn.execute("DELETE FROM episodes;")
            self._conn.commit()
            return deleted_count

    def stats(self) -> dict[str, int]:
        """Return total counts of facts, episodes, and distinct sessions."""
        with self._lock:
            facts_count = self._conn.execute("SELECT COUNT(*) FROM facts;").fetchone()[0]
            episodes_count = self._conn.execute("SELECT COUNT(*) FROM episodes;").fetchone()[0]
            sessions_count = self._conn.execute(
                """
                SELECT COUNT(DISTINCT session_id) FROM (
                    SELECT session_id FROM facts
                    UNION
                    SELECT session_id FROM episodes
                );
                """
            ).fetchone()[0]
            return {
                "facts_count": int(facts_count),
                "episodes_count": int(episodes_count),
                "sessions_count": int(sessions_count),
            }

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if hasattr(self, "_conn") and self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            scope=str(row["scope"]),
            kind=str(row["kind"]),
            key=str(row["key"]),
            value=str(row["value"]),
            source_turn=int(row["source_turn"]),
            pinned=bool(row["pinned"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
