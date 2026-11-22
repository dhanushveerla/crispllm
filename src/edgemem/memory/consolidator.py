"""Async background consolidator for EdgeMem 3-tier synthetic memory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import math
import re
import time
from typing import Any

import httpx

from edgemem.config import EdgeMemConfig
from edgemem.memory.store import Fact, MemoryStore
from edgemem.privacy.redactor import sanitize_fact

logger = logging.getLogger("edgemem.memory.consolidator")


@dataclass
class ConsolidationReport:
    """Telemetry report of actions taken during a consolidation run."""

    merged: int = 0
    superseded: int = 0
    promoted: int = 0
    demoted: int = 0
    created: int = 0
    mode: str = "deterministic"
    dry_run: bool = False


def _tokenize_text(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _lexical_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize_text(a)
    tokens_b = _tokenize_text(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


class MemoryConsolidator:
    """Deterministic and LLM-assisted background memory consolidation."""

    def __init__(
        self,
        store: MemoryStore,
        config: EdgeMemConfig,
        llm_client: Any = None,
    ) -> None:
        self.store = store
        self.config = config
        self.llm_client = llm_client

    def consolidate_once(
        self, dry_run: bool = False, now: float | None = None
    ) -> ConsolidationReport:
        """Run a single consolidation sweep over the memory store."""
        ts = now if now is not None else time.time()
        report = ConsolidationReport(mode="deterministic", dry_run=dry_run)

        with self.store._lock:
            active_facts = self.store.list_facts(
                limit=1000, include_archival=True, include_superseded=False
            )

            episodic_active = [
                f for f in active_facts if f.status == "active" and f.tier == "episodic"
            ]

            groups: dict[tuple[str, str], list[Fact]] = {}
            for f in episodic_active:
                key_norm = f.key.strip().lower()
                groups.setdefault((f.session_id, key_norm), []).append(f)

            for (sess_id, key_norm), facts in groups.items():
                if len(facts) < 2:
                    continue

                clusters: list[list[Fact]] = []
                for f in facts:
                    placed = False
                    for cluster in clusters:
                        rep = cluster[0]
                        norm_f = " ".join(f.value.strip().split()).lower()
                        norm_rep = " ".join(rep.value.strip().split()).lower()
                        if norm_f == norm_rep or _lexical_similarity(f.value, rep.value) >= 0.85:
                            cluster.append(f)
                            placed = True
                            break
                    if not placed:
                        clusters.append([f])

                for cluster in clusters:
                    if len(cluster) < 2:
                        continue
                    cluster.sort(
                        key=lambda x: (x.salience, x.updated_at, x.id),
                        reverse=True,
                    )
                    survivor = cluster[0]
                    duplicates = cluster[1:]

                    additional_access = sum(d.access_count for d in duplicates)
                    boost = min(2.0, survivor.salience + 0.15 * len(duplicates))

                    if not dry_run:
                        self.store._conn.execute(
                            """
                            UPDATE facts
                            SET access_count = access_count + ?,
                                salience = ?,
                                updated_at = ?
                            WHERE id = ?;
                            """,
                            (additional_access, boost, ts, survivor.id),
                        )
                        for d in duplicates:
                            self.store._conn.execute(
                                """
                                UPDATE facts
                                SET status = 'superseded',
                                    superseded_by = ?,
                                    valid_until = ?
                                WHERE id = ?;
                                """,
                                (survivor.id, ts, d.id),
                            )
                            self.store._conn.execute(
                                "DELETE FROM memory_fts WHERE fact_id = ?;",
                                (str(d.id),),
                            )
                    report.merged += len(duplicates)

            remaining_active = self.store.list_facts(
                limit=1000, include_archival=True, include_superseded=False
            )
            key_groups: dict[tuple[str, str], list[Fact]] = {}
            for f in remaining_active:
                if f.status != "active":
                    continue
                k_norm = f.key.strip().lower()
                key_groups.setdefault((f.session_id, k_norm), []).append(f)

            for (sess_id, k_norm), facts in key_groups.items():
                if len(facts) < 2:
                    continue

                def trust_rank(item: Fact) -> int:
                    if item.tier == "core" or item.source == "user" or item.pinned:
                        return 3
                    if item.source == "consolidated":
                        return 2
                    return 1

                facts.sort(
                    key=lambda x: (trust_rank(x), x.updated_at, x.id),
                    reverse=True,
                )
                survivor = facts[0]
                conflicting = facts[1:]

                if not dry_run:
                    for cf in conflicting:
                        self.store._conn.execute(
                            """
                            UPDATE facts
                            SET status = 'superseded',
                                superseded_by = ?,
                                valid_until = ?
                            WHERE id = ?;
                            """,
                            (survivor.id, ts, cf.id),
                        )
                        self.store._conn.execute(
                            "DELETE FROM memory_fts WHERE fact_id = ?;",
                            (str(cf.id),),
                        )
                report.superseded += len(conflicting)

            current_episodic = [
                f
                for f in self.store.list_facts(
                    limit=1000, include_archival=True, include_superseded=False
                )
                if f.status == "active" and f.tier == "episodic"
            ]

            half_life = max(self.config.memory.decay_half_life_days, 0.1)
            decay_rate = math.log(2.0) / half_life

            for f in current_episodic:
                if (
                    f.access_count >= self.config.memory.promote_access_threshold
                    and f.salience >= 1.2
                ):
                    if not dry_run:
                        self.store._conn.execute(
                            """
                            UPDATE facts
                            SET tier = 'core',
                                pinned = 1,
                                updated_at = ?
                            WHERE id = ?;
                            """,
                            (ts, f.id),
                        )
                    report.promoted += 1
                    continue

                ref_time = (
                    f.last_accessed_at
                    if f.last_accessed_at is not None
                    else (f.updated_at or f.created_at or ts)
                )
                age_days = (ts - ref_time) / 86400.0

                if age_days >= 7.0:
                    decayed_salience = f.salience * math.exp(-decay_rate * age_days)
                    if decayed_salience < self.config.memory.archive_threshold:
                        if not dry_run:
                            self.store._conn.execute(
                                """
                                UPDATE facts
                                SET tier = 'archival',
                                    status = 'archived',
                                    updated_at = ?
                                WHERE id = ?;
                                """,
                                (ts, f.id),
                            )
                        report.demoted += 1

            if not dry_run:
                self.store._conn.execute(
                    """
                    INSERT INTO consolidation_runs (
                        ran_at, mode, merged_count, superseded_count,
                        promoted_count, demoted_count, created_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        ts,
                        report.mode,
                        report.merged,
                        report.superseded,
                        report.promoted,
                        report.demoted,
                        report.created,
                    ),
                )
                self.store._conn.commit()

        return report

    async def consolidate_with_llm_async(
        self, dry_run: bool = False, now: float | None = None
    ) -> ConsolidationReport:
        """Run consolidation with optional LLM synthesis; strictly fail-open."""
        if not self.config.memory.llm_synthesis:
            return await asyncio.to_thread(self.consolidate_once, dry_run=dry_run, now=now)

        ts = now if now is not None else time.time()
        episodes = self.store.list_episodes(limit=20)
        if not episodes:
            return await asyncio.to_thread(self.consolidate_once, dry_run=dry_run, now=now)

        try:
            prompt_content = "\n".join(
                f"[{ep.role}]: {ep.summary}" for ep in episodes
            )
            system_msg = (
                "You are an architectural memory consolidator. Analyze conversational episodes "
                "and return a JSON array of operations: "
                '[{"op": "CREATE"|"CORROBORATE"|"SUPERSEDE", "key": "...", "value": "...", "tier": "episodic"|"core"}]. '
                "Output JSON ONLY without commentary."
            )

            upstream_url = self.config.upstream_base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            api_key = self.config.upstream_api_key
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt_content},
                ],
                "temperature": 0.0,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(upstream_url, json=payload, headers=headers)
                if resp.status_code != 200:
                    raise ValueError(f"LLM synthesis HTTP error: {resp.status_code}")
                data = resp.json()

            choice_text = data["choices"][0]["message"]["content"]
            json_match = re.search(r"\[.*\]", choice_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in LLM response")

            ops = json.loads(json_match.group(0))
            if not isinstance(ops, list):
                raise ValueError("Parsed LLM output is not a list")

            created_count = 0
            merged_count = 0
            superseded_count = 0

            if not dry_run:
                for op in ops:
                    if not isinstance(op, dict):
                        continue
                    action = str(op.get("op", "")).upper()
                    k = str(op.get("key", "")).strip()
                    v = str(op.get("value", "")).strip()
                    t = str(op.get("tier", "episodic")).lower()
                    if not k or not v:
                        continue
                    if action == "CREATE":
                        clean_k, clean_v = sanitize_fact(k, v)
                        if clean_v != "[REDACTED_SECRET]":
                            self.store.upsert_fact(
                                session_id="global",
                                key=clean_k,
                                value=clean_v,
                                tier=t,
                                source="consolidated",
                            )
                            created_count += 1
                    elif action == "CORROBORATE":
                        clean_k, clean_v = sanitize_fact(k, v)
                        if clean_v != "[REDACTED_SECRET]":
                            self.store.upsert_fact(
                                session_id="global",
                                key=clean_k,
                                value=clean_v,
                                tier=t,
                                source="consolidated",
                            )
                            merged_count += 1
                    elif action == "SUPERSEDE":
                        clean_k, clean_v = sanitize_fact(k, v)
                        if clean_v != "[REDACTED_SECRET]":
                            self.store.upsert_fact(
                                session_id="global",
                                key=clean_k,
                                value=clean_v,
                                tier=t,
                                source="consolidated",
                            )
                            superseded_count += 1

            det_report = await asyncio.to_thread(
                self.consolidate_once, dry_run=dry_run, now=ts
            )
            det_report.mode = "llm_synthesis"
            det_report.created += created_count
            det_report.merged += merged_count
            det_report.superseded += superseded_count
            return det_report

        except Exception as exc:
            logger.warning("LLM synthesis failed, falling open to deterministic consolidation: %s", exc)
            return await asyncio.to_thread(self.consolidate_once, dry_run=dry_run, now=ts)


class ConsolidationWorker:
    """Async background worker managing debounce and periodic consolidation sweeps."""

    def __init__(
        self,
        consolidator: MemoryConsolidator,
        config: EdgeMemConfig,
    ) -> None:
        self.consolidator = consolidator
        self.config = config
        self.pending_turns: int = 0
        self._event = asyncio.Event()
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def start(self) -> None:
        """Start the background consolidation task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Cancel and await task shutdown before closing DB handles."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def notify_turn(self) -> None:
        """Record turn completion and trigger immediate sweep if turn threshold met."""
        self.pending_turns += 1
        if self.pending_turns >= self.config.memory.consolidation_every_n_turns:
            self._event.set()

    async def consolidate(self, dry_run: bool = False) -> ConsolidationReport:
        """Run consolidation immediately."""
        async with self._lock:
            if self.config.memory.llm_synthesis:
                return await self.consolidator.consolidate_with_llm_async(dry_run=dry_run)
            return await asyncio.to_thread(self.consolidator.consolidate_once, dry_run=dry_run)

    async def _run_loop(self) -> None:
        while self._running:
            try:
                timeout = max(self.config.memory.consolidation_idle_s, 0.5)
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass

                self._event.clear()
                if not self._running:
                    break

                if self.pending_turns > 0:
                    self.pending_turns = 0
                    await self.consolidate()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("ConsolidationWorker loop exception: %s", e)
                await asyncio.sleep(1.0)
