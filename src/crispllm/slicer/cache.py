"""edgemem.slicer.cache — OS-agnostic mtime-keyed SQLite tag cache.

Cache location: platformdirs.user_cache_dir("crispllm") / "tags_v1.db"

Schema:
  TABLE tags_cache
    abs_path TEXT PRIMARY KEY
    mtime    REAL      -- st_mtime of the file at cache time
    tags_json TEXT     -- JSON-serialised list of Tag dicts

Cache hit:  abs_path row exists AND stored mtime == current mtime
Cache miss: row absent, or mtime changed (file modified)
"""
from __future__ import annotations

from collections import OrderedDict
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from platformdirs import user_cache_dir

from crispllm.slicer.tags import Tag

logger = logging.getLogger("crispllm.slicer.cache")

_DB_VERSION = 2
_TABLE = "tags_cache"


def _db_path() -> Path:
    cache_dir = Path(user_cache_dir("crispllm"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"tags_v{_DB_VERSION}.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            abs_path    TEXT NOT NULL PRIMARY KEY,
            mtime       REAL NOT NULL,
            tags_json   TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    try:
        conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outline_cache (
            rel_path    TEXT NOT NULL,
            mtime_ns    INTEGER NOT NULL,
            size        INTEGER NOT NULL,
            lines_key   TEXT NOT NULL,
            rendered    TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            PRIMARY KEY (rel_path, mtime_ns, size, lines_key)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outline_cache_rel_path ON outline_cache(rel_path)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS symbol_cache (
            rel_path     TEXT NOT NULL,
            mtime_ns     INTEGER NOT NULL,
            size         INTEGER NOT NULL,
            token_count  INTEGER NOT NULL,
            imports_json TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            PRIMARY KEY (rel_path, mtime_ns, size)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_cache_rel_path ON symbol_cache(rel_path)")
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


class OutlineCache:
    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        db_path: Path | None = None,
        max_entries: int = 256,
    ) -> None:
        self._max_entries = max_entries
        self._lru: OrderedDict[tuple[str, int, int, str], tuple[str, int]] = OrderedDict()
        self._db_path = db_path
        self._conn = conn
        if self._conn is None and self._db_path is not None:
            try:
                self._conn = _connect(self._db_path)
            except Exception as exc:
                logger.warning("OutlineCache: could not open SQLite at %s: %s", self._db_path, exc)
                self._conn = None

    def get(
        self,
        rel_path: str,
        mtime_ns: int,
        size: int,
        lines_key: str,
    ) -> tuple[str, int] | None:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        key = (rel_path, mtime_ns, size, lines_key)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        if self._conn is not None:
            try:
                row = self._conn.execute(
                    "SELECT rendered, token_count FROM outline_cache WHERE rel_path = ? AND mtime_ns = ? AND size = ? AND lines_key = ?",
                    (rel_path, mtime_ns, size, lines_key),
                ).fetchone()
                if row is not None:
                    rendered, token_count = row[0], int(row[1])
                    if len(self._lru) >= self._max_entries and key not in self._lru:
                        self._lru.popitem(last=False)
                    self._lru[key] = (rendered, token_count)
                    return rendered, token_count

                self._conn.execute(
                    "DELETE FROM outline_cache WHERE rel_path = ? AND (mtime_ns != ? OR size != ?)",
                    (rel_path, mtime_ns, size),
                )
                self._conn.commit()
                stale = [
                    k for k in self._lru
                    if k[0] == rel_path and (k[1] != mtime_ns or k[2] != size)
                ]
                for k in stale:
                    del self._lru[k]
            except Exception as exc:
                logger.debug("OutlineCache.get error: %s", exc)
        return None

    def put(
        self,
        rel_path: str,
        mtime_ns: int,
        size: int,
        lines_key: str,
        rendered: str,
        token_count: int,
    ) -> None:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        key = (rel_path, mtime_ns, size, lines_key)
        if self._conn is not None:
            try:
                self._conn.execute(
                    "DELETE FROM outline_cache WHERE rel_path = ? AND (mtime_ns != ? OR size != ?)",
                    (rel_path, mtime_ns, size),
                )
                self._conn.execute(
                    """
                    INSERT INTO outline_cache (rel_path, mtime_ns, size, lines_key, rendered, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(rel_path, mtime_ns, size, lines_key) DO UPDATE SET
                        rendered=excluded.rendered,
                        token_count=excluded.token_count
                    """,
                    (rel_path, mtime_ns, size, lines_key, rendered, int(token_count)),
                )
                self._conn.commit()
            except Exception as exc:
                logger.debug("OutlineCache.put error: %s", exc)

        stale = [
            k for k in self._lru
            if k[0] == rel_path and (k[1] != mtime_ns or k[2] != size)
        ]
        for k in stale:
            del self._lru[k]

        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            if len(self._lru) >= self._max_entries:
                self._lru.popitem(last=False)
            self._lru[key] = (rendered, int(token_count))

    def invalidate(self, rel_path: str) -> None:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        if self._conn is not None:
            try:
                self._conn.execute(
                    "DELETE FROM outline_cache WHERE rel_path = ? OR rel_path = ? OR rel_path LIKE ?",
                    (rel_path, Path(rel_path).name, f"%/{rel_path}"),
                )
                self._conn.commit()
            except Exception as exc:
                logger.debug("OutlineCache.invalidate error: %s", exc)
        keys_to_remove = [
            k for k in self._lru
            if k[0] == rel_path or k[0].endswith(rel_path) or rel_path.endswith(k[0])
        ]
        for k in keys_to_remove:
            del self._lru[k]

    def clear(self) -> None:
        if self._conn is not None:
            try:
                self._conn.execute("DELETE FROM outline_cache")
                self._conn.commit()
            except Exception as exc:
                logger.debug("OutlineCache.clear error: %s", exc)
        self._lru.clear()

    def __len__(self) -> int:
        return len(self._lru)

    def __bool__(self) -> bool:
        return True

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> OutlineCache:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class SymbolCache:
    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        db_path: Path | None = None,
        max_entries: int = 512,
    ) -> None:
        self._max_entries = max_entries
        self._lru: OrderedDict[tuple[str, int, int], tuple[int, list[str], list[dict[str, Any]]]] = OrderedDict()
        self._db_path = db_path
        self._conn = conn
        if self._conn is None and self._db_path is not None:
            try:
                self._conn = _connect(self._db_path)
            except Exception as exc:
                logger.warning("SymbolCache: could not open SQLite at %s: %s", self._db_path, exc)
                self._conn = None

    def get_all(
        self,
    ) -> dict[tuple[str, int, int], tuple[int, list[str], list[dict[str, Any]]]]:
        result: dict[tuple[str, int, int], tuple[int, list[str], list[dict[str, Any]]]] = {}
        if self._conn is not None:
            try:
                for row in self._conn.execute(
                    "SELECT rel_path, mtime_ns, size, token_count, imports_json, symbols_json FROM symbol_cache"
                ).fetchall():
                    rel, mt, sz, tok_cnt, imp_j, sym_j = row
                    item = (int(tok_cnt), json.loads(imp_j), json.loads(sym_j))
                    key = (rel, int(mt), int(sz))
                    result[key] = item
                    if key not in self._lru:
                        if len(self._lru) >= self._max_entries:
                            self._lru.popitem(last=False)
                        self._lru[key] = item
            except Exception as exc:
                logger.debug("SymbolCache.get_all error: %s", exc)
        return result

    def get(
        self,
        rel_path: str,
        mtime_ns: int,
        size: int,
    ) -> tuple[int, list[str], list[dict[str, Any]]] | None:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        key = (rel_path, mtime_ns, size)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        if self._conn is not None:
            try:
                row = self._conn.execute(
                    "SELECT token_count, imports_json, symbols_json FROM symbol_cache WHERE rel_path = ? AND mtime_ns = ? AND size = ?",
                    (rel_path, mtime_ns, size),
                ).fetchone()
                if row is not None:
                    tok_cnt = int(row[0])
                    imports = json.loads(row[1])
                    symbols = json.loads(row[2])
                    if len(self._lru) >= self._max_entries and key not in self._lru:
                        self._lru.popitem(last=False)
                    self._lru[key] = (tok_cnt, imports, symbols)
                    return tok_cnt, imports, symbols

                self._conn.execute(
                    "DELETE FROM symbol_cache WHERE rel_path = ? AND (mtime_ns != ? OR size != ?)",
                    (rel_path, mtime_ns, size),
                )
                self._conn.commit()
                stale = [
                    k for k in self._lru
                    if k[0] == rel_path and (k[1] != mtime_ns or k[2] != size)
                ]
                for k in stale:
                    del self._lru[k]
            except Exception as exc:
                logger.debug("SymbolCache.get error: %s", exc)
        return None

    def put(
        self,
        rel_path: str,
        mtime_ns: int,
        size: int,
        token_count: int,
        imports: list[str],
        symbols: list[dict[str, Any]],
    ) -> None:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        key = (rel_path, mtime_ns, size)
        imports_json = json.dumps(imports, separators=(",", ":"))
        symbols_json = json.dumps(symbols, separators=(",", ":"))
        if self._conn is not None:
            try:
                self._conn.execute(
                    "DELETE FROM symbol_cache WHERE rel_path = ? AND (mtime_ns != ? OR size != ?)",
                    (rel_path, mtime_ns, size),
                )
                self._conn.execute(
                    """
                    INSERT INTO symbol_cache (rel_path, mtime_ns, size, token_count, imports_json, symbols_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(rel_path, mtime_ns, size) DO UPDATE SET
                        token_count=excluded.token_count,
                        imports_json=excluded.imports_json,
                        symbols_json=excluded.symbols_json
                    """,
                    (rel_path, mtime_ns, size, int(token_count), imports_json, symbols_json),
                )
                self._conn.commit()
            except Exception as exc:
                logger.debug("SymbolCache.put error: %s", exc)

        stale = [
            k for k in self._lru
            if k[0] == rel_path and (k[1] != mtime_ns or k[2] != size)
        ]
        for k in stale:
            del self._lru[k]

        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            if len(self._lru) >= self._max_entries:
                self._lru.popitem(last=False)
            self._lru[key] = (int(token_count), imports, symbols)

    def invalidate(self, rel_path: str) -> None:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        if self._conn is not None:
            try:
                self._conn.execute(
                    "DELETE FROM symbol_cache WHERE rel_path = ? OR rel_path = ? OR rel_path LIKE ?",
                    (rel_path, Path(rel_path).name, f"%/{rel_path}"),
                )
                self._conn.commit()
            except Exception as exc:
                logger.debug("SymbolCache.invalidate error: %s", exc)
        keys_to_remove = [
            k for k in self._lru
            if k[0] == rel_path or k[0].endswith(rel_path) or rel_path.endswith(k[0])
        ]
        for k in keys_to_remove:
            del self._lru[k]

    def clear(self) -> None:
        if self._conn is not None:
            try:
                self._conn.execute("DELETE FROM symbol_cache")
                self._conn.commit()
            except Exception as exc:
                logger.debug("SymbolCache.clear error: %s", exc)
        self._lru.clear()

    def __len__(self) -> int:
        return len(self._lru)

    def __bool__(self) -> bool:
        return True

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> SymbolCache:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class TagCache:
    def __init__(
        self,
        db_path: Path | None = None,
        max_outline_entries: int = 256,
        max_symbol_entries: int = 512,
    ) -> None:
        self._db_path = db_path or _db_path()
        try:
            self._conn = _connect(self._db_path)
        except Exception as exc:
            logger.warning("TagCache: could not open SQLite at %s: %s", self._db_path, exc)
            self._conn = None
        self.outline_cache = OutlineCache(conn=self._conn, max_entries=max_outline_entries)
        self.symbol_cache = SymbolCache(conn=self._conn, max_entries=max_symbol_entries)

    def get_outline(
        self, rel_path: str, mtime_ns: int, size: int, lines_key: str
    ) -> tuple[str, int] | None:
        return self.outline_cache.get(rel_path, mtime_ns, size, lines_key)

    def put_outline(
        self,
        rel_path: str,
        mtime_ns: int,
        size: int,
        lines_key: str,
        rendered: str,
        token_count: int,
    ) -> None:
        self.outline_cache.put(
            rel_path, mtime_ns, size, lines_key, rendered, token_count
        )

    def get(self, abs_path: Path) -> tuple[list[Tag], int] | None:
        if self._conn is None:
            return None
        try:
            mtime = abs_path.stat().st_mtime
        except OSError:
            return None
        try:
            row = self._conn.execute(
                f"SELECT mtime, tags_json, token_count FROM {_TABLE} WHERE abs_path = ?",
                (str(abs_path),),
            ).fetchone()
        except Exception as exc:
            logger.debug("TagCache.get error: %s", exc)
            return None

        if row is None:
            return None
        cached_mtime, tags_json, token_count = row
        if abs(cached_mtime - mtime) > 1e-6:
            return None
        try:
            tags = _json_to_tags(tags_json)
            return tags, int(token_count or 0)
        except Exception as exc:
            logger.debug("TagCache.get deserialise error: %s", exc)
            return None

    def get_token_count(self, abs_path: Path) -> int | None:
        cached = self.get(abs_path)
        return cached[1] if cached is not None else None

    def put(self, abs_path: Path, tags: list[Tag], token_count: int = 0) -> None:
        if self._conn is None:
            return
        try:
            mtime = abs_path.stat().st_mtime
            tags_json = _tags_to_json(tags)
            self._conn.execute(
                f"""
                INSERT INTO {_TABLE} (abs_path, mtime, tags_json, token_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(abs_path) DO UPDATE SET
                    mtime=excluded.mtime,
                    tags_json=excluded.tags_json,
                    token_count=excluded.token_count
                """,
                (str(abs_path), mtime, tags_json, int(token_count)),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("TagCache.put error: %s", exc)

    def invalidate(self, abs_path: Path) -> None:
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
        self.outline_cache.invalidate(str(abs_path))
        self.outline_cache.invalidate(abs_path.name)
        self.symbol_cache.invalidate(str(abs_path))
        self.symbol_cache.invalidate(abs_path.name)

    def clear(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(f"DELETE FROM {_TABLE}")
            self._conn.commit()
        except Exception as exc:
            logger.debug("TagCache.clear error: %s", exc)
        self.outline_cache.clear()
        self.symbol_cache.clear()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if hasattr(self, "outline_cache") and self.outline_cache is not None:
            self.outline_cache._conn = None
        if hasattr(self, "symbol_cache") and self.symbol_cache is not None:
            self.symbol_cache._conn = None

    def __enter__(self) -> "TagCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()
