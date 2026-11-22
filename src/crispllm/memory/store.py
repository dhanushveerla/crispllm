"""SQLite-backed tiered memory store with FTS5 search and zero external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

import platformdirs

from crispllm.privacy.redactor import sanitize_fact, sanitize_for_memory


def compute_effective_salience(
    salience: float, age_days: float, access_count: int, half_life_days: float
) -> float:
    decay_rate = math.log(2.0) / max(half_life_days, 0.1)
    decay_factor = math.exp(-decay_rate * max(age_days, 0.0))
    reinforcement = 1.0 + 0.15 * math.log1p(max(access_count, 0))
    return salience * decay_factor * reinforcement


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
    tier: str = "episodic"
    status: str = "active"
    salience: float = 1.0
    access_count: int = 0
    last_accessed_at: float | None = None
    superseded_by: int | None = None
    valid_until: float | None = None
    source: str = "auto"


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
            data_dir = platformdirs.user_data_dir("crispllm")
            self.db_path: Path | str = Path(data_dir) / "memory.db"
        elif isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        self._is_memory = str(self.db_path) == ":memory:"
        self._lock = threading.RLock()

        if not self._is_memory:
            try:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.db_path = ":memory:"
                self._is_memory = True

        try:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        except (sqlite3.OperationalError, OSError):
            self.db_path = ":memory:"
            self._is_memory = True
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._conn.row_factory = sqlite3.Row

        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='facts';")
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                """
                CREATE TABLE facts (
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
                    tier TEXT NOT NULL DEFAULT 'episodic',
                    status TEXT NOT NULL DEFAULT 'active',
                    salience REAL NOT NULL DEFAULT 1.0,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at REAL,
                    superseded_by INTEGER,
                    valid_until REAL,
                    source TEXT NOT NULL DEFAULT 'auto'
                );
                """
            )
        else:
            self._migrate_schema(str(row[0]))

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
            """
            CREATE TABLE IF NOT EXISTS consolidation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ran_at REAL NOT NULL,
                mode TEXT NOT NULL,
                merged_count INTEGER NOT NULL DEFAULT 0,
                superseded_count INTEGER NOT NULL DEFAULT 0,
                promoted_count INTEGER NOT NULL DEFAULT 0,
                demoted_count INTEGER NOT NULL DEFAULT 0,
                created_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_session_status ON facts(session_id, status);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_key_status ON facts(key, status);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_tier_status ON facts(tier, status);"
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

    def _migrate_schema(self, table_sql: str) -> None:
        now = time.time()
        if "UNIQUE" in table_sql.upper():
            self._conn.execute("ALTER TABLE facts RENAME TO facts_legacy;")
            self._conn.execute(
                """
                CREATE TABLE facts (
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
                    tier TEXT NOT NULL DEFAULT 'episodic',
                    status TEXT NOT NULL DEFAULT 'active',
                    salience REAL NOT NULL DEFAULT 1.0,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at REAL,
                    superseded_by INTEGER,
                    valid_until REAL,
                    source TEXT NOT NULL DEFAULT 'auto'
                );
                """
            )
            self._conn.execute(
                """
                INSERT INTO facts (
                    id, session_id, scope, kind, key, value, source_turn, pinned,
                    created_at, updated_at, tier, status, salience, access_count,
                    last_accessed_at, superseded_by, valid_until, source
                )
                SELECT
                    id, session_id, scope, kind, key, value, source_turn, pinned,
                    created_at, updated_at,
                    CASE WHEN pinned = 1 THEN 'core' ELSE 'episodic' END,
                    'active',
                    1.0,
                    0,
                    COALESCE(updated_at, created_at, ?),
                    NULL,
                    NULL,
                    CASE WHEN pinned = 1 THEN 'user' ELSE 'auto' END
                FROM facts_legacy;
                """,
                (now,),
            )
            self._conn.execute("DROP TABLE facts_legacy;")
            return

        cur = self._conn.execute("PRAGMA table_info(facts);")
        existing_cols = {str(col["name"]).lower() for col in cur.fetchall()}

        if "tier" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN tier TEXT NOT NULL DEFAULT 'episodic';")
        if "status" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN status TEXT NOT NULL DEFAULT 'active';")
        if "salience" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN salience REAL NOT NULL DEFAULT 1.0;")
        if "access_count" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;")
        if "last_accessed_at" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN last_accessed_at REAL;")
        if "superseded_by" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN superseded_by INTEGER;")
        if "valid_until" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN valid_until REAL;")
        if "source" not in existing_cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN source TEXT NOT NULL DEFAULT 'auto';")

        self._conn.execute(
            """
            UPDATE facts
            SET tier = 'core', source = 'user'
            WHERE pinned = 1 AND (tier IS NULL OR tier = 'episodic');
            """
        )
        self._conn.execute(
            """
            UPDATE facts
            SET last_accessed_at = COALESCE(updated_at, created_at, ?)
            WHERE last_accessed_at IS NULL;
            """,
            (now,),
        )

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
        tier: str | None = None,
        source: str = "auto",
        salience: float = 1.0,
    ) -> int:
        now = time.time()
        key, value = sanitize_fact(key, value)
        if tier is None:
            tier = "core" if pinned else "episodic"
        if tier == "core":
            pinned = True
        pinned_int = 1 if pinned else 0
        if pinned and source == "auto":
            source = "user"

        norm_key = key.strip().lower()
        norm_val = " ".join(value.strip().split()).lower()

        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM facts
                WHERE LOWER(TRIM(key)) = ?
                  AND status = 'active'
                  AND (session_id = ? OR scope = 'global')
                ORDER BY (session_id = ?) DESC, updated_at DESC, id DESC;
                """,
                (norm_key, session_id, session_id),
            )
            existing_rows = cur.fetchall()

            if existing_rows:
                existing = existing_rows[0]
                existing_id = int(existing["id"])
                existing_norm_val = " ".join(str(existing["value"]).strip().split()).lower()

                if norm_val == existing_norm_val:
                    new_access = int(existing["access_count"]) + 1
                    new_salience = min(2.0, float(existing["salience"]) + 0.15)
                    target_tier = "core" if (tier == "core" or existing["tier"] == "core") else str(existing["tier"])
                    target_pinned = 1 if (pinned or bool(existing["pinned"]) or target_tier == "core") else 0
                    target_source = "user" if (source == "user" or existing["source"] == "user") else str(existing["source"])
                    self._conn.execute(
                        """
                        UPDATE facts SET
                            access_count = ?,
                            salience = ?,
                            last_accessed_at = ?,
                            updated_at = ?,
                            tier = ?,
                            pinned = ?,
                            source = ?,
                            source_turn = ?,
                            scope = ?
                        WHERE id = ?;
                        """,
                        (
                            new_access,
                            new_salience,
                            now,
                            now,
                            target_tier,
                            target_pinned,
                            target_source,
                            source_turn,
                            scope,
                            existing_id,
                        ),
                    )
                    self._conn.execute("DELETE FROM memory_fts WHERE fact_id = ?;", (str(existing_id),))
                    content = f"{key} {value}"
                    self._conn.execute(
                        "INSERT INTO memory_fts (fact_id, session_id, content) VALUES (?, ?, ?);",
                        (str(existing_id), session_id, content),
                    )
                    self._conn.commit()
                    return existing_id

                existing_high_trust = (
                    str(existing["tier"]) == "core"
                    or str(existing["source"]) == "user"
                    or bool(existing["pinned"])
                )
                incoming_high_trust = (tier == "core" or source == "user" or pinned)

                if existing_high_trust and not incoming_high_trust:
                    return existing_id

                cur_new = self._conn.execute(
                    """
                    INSERT INTO facts (
                        session_id, scope, kind, key, value, source_turn, pinned,
                        created_at, updated_at, tier, status, salience, access_count,
                        last_accessed_at, superseded_by, valid_until, source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, ?, NULL, NULL, ?)
                    RETURNING id;
                    """,
                    (
                        session_id,
                        scope,
                        kind,
                        key,
                        value,
                        source_turn,
                        pinned_int,
                        now,
                        now,
                        tier,
                        salience,
                        now,
                        source,
                    ),
                )
                new_id = int(cur_new.fetchone()[0])

                self._conn.execute(
                    """
                    UPDATE facts SET
                        status = 'superseded',
                        superseded_by = ?,
                        valid_until = ?
                    WHERE id = ?;
                    """,
                    (new_id, now, existing_id),
                )
                self._conn.execute("DELETE FROM memory_fts WHERE fact_id = ?;", (str(existing_id),))
                content = f"{key} {value}"
                self._conn.execute(
                    "INSERT INTO memory_fts (fact_id, session_id, content) VALUES (?, ?, ?);",
                    (str(new_id), session_id, content),
                )
                self._conn.commit()
                return new_id

            cur_ins = self._conn.execute(
                """
                INSERT INTO facts (
                    session_id, scope, kind, key, value, source_turn, pinned,
                    created_at, updated_at, tier, status, salience, access_count,
                    last_accessed_at, superseded_by, valid_until, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, ?, NULL, NULL, ?)
                RETURNING id;
                """,
                (
                    session_id,
                    scope,
                    kind,
                    key,
                    value,
                    source_turn,
                    pinned_int,
                    now,
                    now,
                    tier,
                    salience,
                    now,
                    source,
                ),
            )
            new_id = int(cur_ins.fetchone()[0])
            content = f"{key} {value}"
            self._conn.execute(
                "INSERT INTO memory_fts (fact_id, session_id, content) VALUES (?, ?, ?);",
                (str(new_id), session_id, content),
            )
            self._conn.commit()
            return new_id

    def record_access(self, fact_ids: list[int], now: float | None = None) -> None:
        """Increment access_count and set last_accessed_at for recalled/injected facts."""
        if not fact_ids:
            return
        ts = now if now is not None else time.time()
        with self._lock:
            placeholders = ",".join("?" for _ in fact_ids)
            self._conn.execute(
                f"""
                UPDATE facts
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE id IN ({placeholders});
                """,
                [ts, *fact_ids],
            )
            self._conn.commit()

    def get_fact_history(self, key: str, session_id: str | None = None) -> list[Fact]:
        """Return chronological lineage (active and superseded versions) for a key."""
        norm_key = key.strip().lower()
        with self._lock:
            params: list[Any] = [norm_key]
            session_clause = ""
            if session_id is not None:
                session_clause = "AND (session_id = ? OR scope = 'global')"
                params.append(session_id)
            sql = f"""
                SELECT * FROM facts
                WHERE LOWER(TRIM(key)) = ?
                {session_clause}
                ORDER BY created_at ASC, id ASC;
            """
            cur = self._conn.execute(sql, params)
            return [self._row_to_fact(row) for row in cur.fetchall()]

    def tier_counts(self) -> dict[str, int]:
        """Return counts of facts across tiers and statuses."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN status = 'active' AND tier = 'core' THEN 1 ELSE 0 END), 0) as core_cnt,
                    COALESCE(SUM(CASE WHEN status = 'active' AND tier = 'episodic' THEN 1 ELSE 0 END), 0) as episodic_cnt,
                    COALESCE(SUM(CASE WHEN tier = 'archival' OR status = 'archived' THEN 1 ELSE 0 END), 0) as archival_cnt,
                    COALESCE(SUM(CASE WHEN status = 'superseded' THEN 1 ELSE 0 END), 0) as superseded_cnt
                FROM facts;
                """
            )
            row = cur.fetchone()
            return {
                "core": int(row["core_cnt"]) if row else 0,
                "episodic": int(row["episodic_cnt"]) if row else 0,
                "archival": int(row["archival_cnt"]) if row else 0,
                "superseded": int(row["superseded_cnt"]) if row else 0,
            }

    def get_pinned_facts(
        self,
        session_id: str,
        include_global: bool = True,
        limit: int = 20,
    ) -> list[Fact]:
        """Fetch active pinned/core facts for a session ordered by recency."""
        with self._lock:
            if include_global:
                sql = """
                    SELECT * FROM facts
                    WHERE (pinned = 1 OR tier = 'core')
                      AND status = 'active'
                      AND (session_id = ? OR scope = 'global')
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?;
                """
                cur = self._conn.execute(sql, (session_id, limit))
            else:
                sql = """
                    SELECT * FROM facts
                    WHERE (pinned = 1 OR tier = 'core')
                      AND status = 'active'
                      AND session_id = ?
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
        tier: str | None = None,
        include_archival: bool = False,
        include_superseded: bool = False,
        limit: int = 10,
    ) -> list[Fact]:
        """Search facts using sanitized FTS5 BM25 ranking with tier and status filtering."""
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return []

        unique_tokens = list(dict.fromkeys(tokens))
        fts_match_query = " OR ".join(f'"{t}"' for t in unique_tokens)

        with self._lock:
            params: list[Any] = [fts_match_query]
            clauses = []
            if session_id is not None:
                if include_global:
                    clauses.append("(f.session_id = ? OR f.scope = 'global')")
                    params.append(session_id)
                else:
                    clauses.append("f.session_id = ?")
                    params.append(session_id)

            if tier is not None:
                clauses.append("f.tier = ?")
                params.append(tier)
            elif not include_archival:
                clauses.append("f.tier != 'archival' AND f.status != 'archived'")

            if not include_superseded:
                clauses.append("f.status = 'active'")

            where_extra = f"AND {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)

            sql = f"""
                SELECT f.*
                FROM memory_fts fts
                JOIN facts f ON f.id = CAST(fts.fact_id AS INTEGER)
                WHERE memory_fts MATCH ?
                {where_extra}
                ORDER BY fts.rank ASC, f.updated_at DESC
                LIMIT ?;
            """
            cur = self._conn.execute(sql, params)
            return [self._row_to_fact(row) for row in cur.fetchall()]

    def list_facts(
        self,
        session_id: str | None = None,
        kind: str | None = None,
        tier: str | None = None,
        include_archival: bool = False,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[Fact]:
        """List stored facts with tier, status, and kind filters."""
        with self._lock:
            clauses = []
            params: list[Any] = []
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            if kind is not None:
                clauses.append("kind = ?")
                params.append(kind)
            if tier is not None:
                clauses.append("tier = ?")
                params.append(tier)
            elif not include_archival:
                clauses.append("tier != 'archival' AND status != 'archived'")

            if not include_superseded:
                clauses.append("status != 'superseded'")

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
        summary = sanitize_for_memory(summary)
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
        """Return total counts of active facts, episodes, and distinct sessions."""
        with self._lock:
            facts_count = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE status = 'active' AND tier != 'archival';"
            ).fetchone()[0]
            episodes_count = self._conn.execute("SELECT COUNT(*) FROM episodes;").fetchone()[0]
            sessions_count = self._conn.execute(
                """
                SELECT COUNT(DISTINCT session_id) FROM (
                    SELECT session_id FROM facts WHERE status = 'active'
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
                self._conn = None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        keys = row.keys() if hasattr(row, "keys") else []
        return Fact(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            scope=str(row["scope"]),
            kind=str(row["kind"]),
            key=str(row["key"]),
            value=str(row["value"]),
            source_turn=int(row["source_turn"]),
            pinned=bool(row["pinned"]),
            created_at=float(row["created_at"]) if row["created_at"] is not None else 0.0,
            updated_at=float(row["updated_at"]) if row["updated_at"] is not None else 0.0,
            tier=str(row["tier"]) if "tier" in keys and row["tier"] is not None else ("core" if row["pinned"] else "episodic"),
            status=str(row["status"]) if "status" in keys and row["status"] is not None else "active",
            salience=float(row["salience"]) if "salience" in keys and row["salience"] is not None else 1.0,
            access_count=int(row["access_count"]) if "access_count" in keys and row["access_count"] is not None else 0,
            last_accessed_at=float(row["last_accessed_at"]) if "last_accessed_at" in keys and row["last_accessed_at"] is not None else None,
            superseded_by=int(row["superseded_by"]) if "superseded_by" in keys and row["superseded_by"] is not None else None,
            valid_until=float(row["valid_until"]) if "valid_until" in keys and row["valid_until"] is not None else None,
            source=str(row["source"]) if "source" in keys and row["source"] is not None else ("user" if row["pinned"] else "auto"),
        )
