"""edgemem.slicer.cache — OS-agnostic mtime-keyed SQLite tag cache.

Cache location: platformdirs.user_cache_dir("edgemem") / "tags_v1.db"

Schema:
  TABLE tags_cache
    abs_path TEXT PRIMARY KEY
    mtime    REAL      -- st_mtime of the file at cache time
    tags_json TEXT     -- JSON-serialised list of Tag dicts

Cache hit:  abs_path row exists AND stored mtime == current mtime
Cache miss: row absent, or mtime changed (file modified)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Sequence

from platformdirs import user_cache_dir

from edgemem.slicer.tags import Tag

logger = logging.getLogger("edgemem.slicer.cache")

_DB_VERSION = 1
_TABLE = "tags_cache"


def _db_path() -> Path:
    cache_dir = Path(user_cache_dir("edgemem"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"tags_v{_DB_VERSION}.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            abs_path  TEXT NOT NULL PRIMARY KEY,
            mtime     REAL NOT NULL,
            tags_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _tags_to_json(tags: list[Tag]) -> str:
    return json.dumps(
        [
            {
                "rel_path": t.rel_path,
                "abs_path": t.abs_path,
                "name":     t.name,
                "line":     t.line,
                "kind":     t.kind,
                "info":     t.info,
            }
            for t in tags
        ],
        separators=(",", ":"),
    )


def _json_to_tags(raw: str) -> list[Tag]:
    return [
        Tag(
            rel_path=d["rel_path"],
            abs_path=d["abs_path"],
            name=d["name"],
            line=d["line"],
            kind=d["kind"],
            info=d.get("info", ""),
        )
        for d in json.loads(raw)
    ]


class TagCache:
    """
    Persistent mtime-keyed SQLite cache for extracted tags.

    Thread-safe for single-process use (WAL mode + serialised writes).
    Not designed for concurrent multi-process write access.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _db_path()
        try:
            self._conn = _connect(self._db_path)
        except Exception as exc:
            logger.warning("TagCache: could not open SQLite at %s: %s", self._db_path, exc)
            self._conn = None  # degrade gracefully — always cache miss

    def get(self, abs_path: Path) -> list[Tag] | None:
        """Return cached tags if the file is unchanged, else None."""
        if self._conn is None:
            return None
        try:
            mtime = abs_path.stat().st_mtime
        except OSError:
            return None
        try:
            row = self._conn.execute(
                f"SELECT mtime, tags_json FROM {_TABLE} WHERE abs_path = ?",
                (str(abs_path),),
            ).fetchone()
        except Exception as exc:
            logger.debug("TagCache.get error: %s", exc)
            return None

        if row is None:
            return None
        cached_mtime, tags_json = row
        if abs(cached_mtime - mtime) > 1e-6:
            # File modified — cache miss
            return None
        try:
            return _json_to_tags(tags_json)
        except Exception as exc:
            logger.debug("TagCache.get deserialise error: %s", exc)
            return None

    def put(self, abs_path: Path, tags: list[Tag]) -> None:
        """Store tags for abs_path keyed by current mtime."""
        if self._conn is None:
            return
        try:
            mtime = abs_path.stat().st_mtime
            tags_json = _tags_to_json(tags)
            self._conn.execute(
                f"""
                INSERT INTO {_TABLE} (abs_path, mtime, tags_json)
                VALUES (?, ?, ?)
                ON CONFLICT(abs_path) DO UPDATE SET mtime=excluded.mtime, tags_json=excluded.tags_json
                """,
                (str(abs_path), mtime, tags_json),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("TagCache.put error: %s", exc)

    def invalidate(self, abs_path: Path) -> None:
        """Remove a specific entry from the cache."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                f"DELETE FROM {_TABLE} WHERE abs_path = ?",
                (str(abs_path),),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("TagCache.invalidate error: %s", exc)

    def clear(self) -> None:
        """Drop all cached entries."""
        if self._conn is None:
            return
        try:
            self._conn.execute(f"DELETE FROM {_TABLE}")
            self._conn.commit()
        except Exception as exc:
            logger.debug("TagCache.clear error: %s", exc)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "TagCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()
