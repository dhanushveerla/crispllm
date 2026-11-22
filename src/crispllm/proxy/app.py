from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import threading
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from crispllm.config import CrispLLMConfig
from crispllm.memory.consolidator import ConsolidationWorker, MemoryConsolidator
from crispllm.memory.store import MemoryStore
from crispllm.privacy.redactor import (
    RedactionConfig,
    RedactionSession,
    SecretRedactor,
)
from crispllm.privacy.stream_restorer import StreamRestorer
from crispllm.proxy.auth import CrispLLMAuthMiddleware
from crispllm.proxy.budgeter import ContextBudgeter
from crispllm.proxy.mcp_server import create_mcp_server, create_mcp_starlette_app
from crispllm.proxy.middleware import MemoryMiddleware, _get_message_text
from crispllm.proxy.router import HybridRouter, RouteTarget
from crispllm.proxy.speculative import (
    PredictedOutputsEngine,
    PredictionDecision,
    SpeculativeStats,
)
from crispllm.proxy.streaming import (
    count_output_tokens_from_sse,
    estimate_tokens_from_messages,
    stream_sse_passthrough,
    stream_with_heartbeat,
)
from crispllm.proxy.telemetry import (
    RequestTelemetry,
    TelemetryLogger,
    TelemetryStore,
    Timer,
    get_rss_mb,
)
from crispllm.proxy.unloader import IdleUnloadWorker
from crispllm.proxy.watchdog import RamPressureWatchdog

logger = logging.getLogger("crispllm.proxy")


class ProxyCounters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.local_requests: int = 0
        self.cloud_requests: int = 0
        self.fallback_escalations: int = 0
        self.secrets_redacted_total: int = 0
        self.task_tier_fast: int = 0
        self.task_tier_balanced: int = 0
        self.task_tier_deep: int = 0
        self.prefix_cache_hits: int = 0
        self.prefix_cache_misses: int = 0

    def inc_local(self) -> None:
        with self._lock:
            self.local_requests += 1

    def inc_cloud(self) -> None:
        with self._lock:
            self.cloud_requests += 1

    def inc_fallback(self) -> None:
        with self._lock:
            self.fallback_escalations += 1

    def add_redacted(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self.secrets_redacted_total += count

    def inc_task_tier(self, tier: str) -> None:
        with self._lock:
            if tier == "fast":
                self.task_tier_fast += 1
            elif tier == "deep":
                self.task_tier_deep += 1
            else:
                self.task_tier_balanced += 1

    def inc_prefix_hit(self) -> None:
        with self._lock:
            self.prefix_cache_hits += 1

    def inc_prefix_miss(self) -> None:
        with self._lock:
            self.prefix_cache_misses += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "local_requests": self.local_requests,
                "cloud_requests": self.cloud_requests,
                "fallback_escalations": self.fallback_escalations,
                "secrets_redacted_total": self.secrets_redacted_total,
                "task_tiers": {
                    "fast": self.task_tier_fast,
                    "balanced": self.task_tier_balanced,
                    "deep": self.task_tier_deep,
                },
                "prefix_cache": {
                    "hits": self.prefix_cache_hits,
                    "misses": self.prefix_cache_misses,
                },
            }


_SESSION_PREFIX_CACHE: OrderedDict[str, str] = OrderedDict()
_MAX_SESSION_PREFIX_CACHE = 256
_SESSION_PREFIX_LOCK = threading.Lock()


def _format_server_timing(
    proxy_overhead_ms: float, existing_header: str | None = None
) -> str:
    metric = f"crispllm_proxy;dur={proxy_overhead_ms:.2f}, edgemem_proxy;dur={proxy_overhead_ms:.2f}"
    if existing_header:
        return f"{existing_header}, {metric}"
    return metric


def _prepare_payload_for_target(
    body: dict[str, Any],
    target: RouteTarget,
    redactor: SecretRedactor,
    router: HybridRouter,
) -> tuple[dict[str, Any], RedactionSession, dict[str, str]]:
    upstream = dict(body)
    upstream["model"] = target.model

    if target.extra_body:
        for k, v in target.extra_body.items():
            if k not in upstream:
                upstream[k] = v

    if target.tier == "local":
        if router.config.resources.llama_cache_prompt and router.budgeter.backend == "llama_server":
            upstream["cache_prompt"] = True
        else:
            upstream.pop("cache_prompt", None)
        if target.decision and target.decision.inject_ollama_num_ctx:
            raw_options = upstream.get("options")
            options = dict(raw_options) if isinstance(raw_options, dict) else {}
            if "num_ctx" in options and options["num_ctx"]:
                options["num_ctx"] = min(int(options["num_ctx"]), target.decision.pressure_ctx_cap)
            else:
                options["num_ctx"] = target.decision.selected_num_ctx
            upstream["options"] = options
    elif target.tier == "cloud":
        upstream.pop("cache_prompt", None)
        if "options" in upstream:
            if isinstance(upstream["options"], dict):
                options = dict(upstream["options"])
                options.pop("num_ctx", None)
                if options:
                    upstream["options"] = options
                else:
                    upstream.pop("options", None)
            else:
                upstream.pop("options", None)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    session = RedactionSession()
    if router.should_redact(target):
        if "messages" in upstream and isinstance(upstream["messages"], list):
            upstream["messages"] = redactor.redact_messages(
                upstream["messages"], session
            )

    return upstream, session, headers


def _is_empty_buffering_choice(choice: Any) -> bool:
    if not isinstance(choice, dict):
        return False
    if choice.get("finish_reason") is not None:
        return False
    if choice.get("role"):
        return False
    if choice.get("tool_calls"):
        return False
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False
    if delta.get("role"):
        return False
    if delta.get("tool_calls"):
        return False
    if delta.get("content") != "":
        return False
    for k, v in delta.items():
        if k != "content" and v:
            return False
    return True


async def _restoring_stream_wrapper(
    raw_stream: AsyncIterator[str],
    session: RedactionSession,
) -> AsyncIterator[str]:
    if not session.has_redactions:
        async for line in raw_stream:
            yield line
        return

    choice_restorers: dict[int, StreamRestorer] = {}
    tc_restorers: dict[tuple[int, int], StreamRestorer] = {}
    last_id = "chatcmpl-stream"
    last_model = "crispllm"

    async for sse_event in raw_stream:
        if not sse_event.startswith("data: "):
            yield sse_event
            continue

        raw_payload = sse_event[6:].strip()
        if raw_payload == "[DONE]":
            flushed_deltas: list[dict[str, Any]] = []
            for c_idx, restorer in choice_restorers.items():
                flushed_text = restorer.flush()
                if flushed_text:
                    flushed_deltas.append(
                        {
                            "index": c_idx,
                            "delta": {"content": flushed_text},
                            "finish_reason": None,
                        }
                    )
            for (c_idx, tc_idx), restorer in tc_restorers.items():
                flushed_text = restorer.flush()
                if flushed_text:
                    flushed_deltas.append(
                        {
                            "index": c_idx,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tc_idx,
                                        "function": {"arguments": flushed_text},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    )
            if flushed_deltas:
                synthetic_chunk = {
                    "id": last_id,
                    "object": "chat.completion.chunk",
                    "model": last_model,
                    "choices": flushed_deltas,
                }
                yield f"data: {json.dumps(synthetic_chunk)}\n\n"

            yield "data: [DONE]\n\n"
            continue

        try:
            chunk_data = json.loads(raw_payload)
        except Exception:
            yield sse_event
            continue

        if isinstance(chunk_data, dict):
            if "id" in chunk_data and chunk_data["id"]:
                last_id = chunk_data["id"]
            if "model" in chunk_data and chunk_data["model"]:
                last_model = chunk_data["model"]

            choices = chunk_data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    c_idx = choice.get("index", 0)
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if c_idx not in choice_restorers:
                                choice_restorers[c_idx] = StreamRestorer(session)
                            delta["content"] = choice_restorers[c_idx].feed(content)

                        tool_calls = delta.get("tool_calls")
                        if isinstance(tool_calls, list):
                            for tc in tool_calls:
                                if not isinstance(tc, dict):
                                    continue
                                tc_idx = tc.get("index", 0)
                                fn = tc.get("function")
                                if isinstance(fn, dict):
                                    args = fn.get("arguments")
                                    if isinstance(args, str) and args:
                                        key = (c_idx, tc_idx)
                                        if key not in tc_restorers:
                                            tc_restorers[key] = StreamRestorer(session)
                                        fn["arguments"] = tc_restorers[key].feed(args)

                        if choice.get("finish_reason") is not None:
                            if c_idx in choice_restorers:
                                flushed = choice_restorers[c_idx].flush()
                                if flushed:
                                    if "content" in delta and isinstance(
                                        delta["content"], str
                                    ):
                                        delta["content"] += flushed
                                    else:
                                        delta["content"] = flushed

            if (
                chunk_data.get("usage") is None
                and isinstance(choices, list)
                and len(choices) > 0
                and all(_is_empty_buffering_choice(c) for c in choices)
            ):
                continue

            yield f"data: {json.dumps(chunk_data)}\n\n"
        else:
            yield sse_event


def create_app(cfg: CrispLLMConfig | None = None) -> FastAPI:
    if cfg is None:
        from crispllm.config import load_config

        cfg = load_config()

    memory_store = MemoryStore(cfg.memory.db_path)
    telemetry_store = TelemetryStore(cfg.memory.db_path)
    memory_middleware = MemoryMiddleware(memory_store, cfg.memory)
    watchdog = RamPressureWatchdog(cfg.resources)
    budgeter = ContextBudgeter(cfg)
    router = HybridRouter(cfg, budgeter=budgeter, watchdog=watchdog)
    unloader = IdleUnloadWorker(cfg, http_client=None)
    pred_engine = PredictedOutputsEngine(cfg.speculative)
    spec_stats = SpeculativeStats()
    redaction_config = RedactionConfig(
        enabled=cfg.privacy.enabled,
        redact_secrets=cfg.privacy.redact_secrets,
        redact_emails=cfg.privacy.redact_emails,
        redact_ips=cfg.privacy.redact_ips,
        extra_patterns=tuple(cfg.privacy.extra_patterns),
    )
    redactor = SecretRedactor(redaction_config)
    counters = ProxyCounters()

    consolidator = MemoryConsolidator(store=memory_store, config=cfg)
    consolidation_worker = ConsolidationWorker(consolidator=consolidator, config=cfg)

    mcp_server = create_mcp_server(memory_store=memory_store, config=cfg)
    mcp_app = create_mcp_starlette_app(mcp_server)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        tel = TelemetryLogger(cfg.telemetry_path)
        client = httpx.AsyncClient(
            base_url=cfg.upstream_base_url,
            timeout=httpx.Timeout(
                connect=10.0, read=600.0, write=30.0, pool=10.0
            ),
        )

        unloader._http_client = client
        if hasattr(mcp_server, "_crispllm_set_telemetry"):
            mcp_server._crispllm_set_telemetry(tel, cfg)
        if hasattr(mcp_server, "_edgemem_set_telemetry"):
            mcp_server._edgemem_set_telemetry(tel, cfg)

        app.state.crispllm_config = app.state.edgemem_config = cfg
        app.state.crispllm_http_client = app.state.edgemem_http_client = client
        app.state.crispllm_telemetry = app.state.edgemem_telemetry = tel
        app.state.crispllm_telemetry_store = app.state.edgemem_telemetry_store = telemetry_store
        app.state.crispllm_memory_store = app.state.edgemem_memory_store = memory_store
        app.state.crispllm_memory_middleware = app.state.edgemem_memory_middleware = memory_middleware
        app.state.crispllm_consolidation_worker = app.state.edgemem_consolidation_worker = consolidation_worker
        app.state.crispllm_router = app.state.edgemem_router = router
        app.state.crispllm_redactor = app.state.edgemem_redactor = redactor
        app.state.crispllm_counters = app.state.edgemem_counters = counters
        app.state.crispllm_watchdog = app.state.edgemem_watchdog = watchdog
        app.state.crispllm_budgeter = app.state.edgemem_budgeter = budgeter
        app.state.crispllm_unloader = app.state.edgemem_unloader = unloader
        app.state.crispllm_pred_engine = app.state.edgemem_pred_engine = pred_engine
        app.state.crispllm_spec_stats = app.state.edgemem_spec_stats = spec_stats
        app.state.crispllm_last_num_ctx = app.state.edgemem_last_num_ctx = None

        if cfg.memory.enabled and cfg.memory.consolidation_enabled:
            consolidation_worker.start()

        if cfg.resources.enabled and cfg.resources.idle_unload_enabled:
            unloader.start()

        try:
            session_mgr = mcp_server.session_manager
            async with session_mgr.run():
                yield
        finally:
            await unloader.stop()
            await consolidation_worker.stop()
            await client.aclose()
            memory_store.close()
            telemetry_store.close()

    app = FastAPI(title="CrispLLM Proxy", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CrispLLMAuthMiddleware, config=cfg.security)
    app.state.crispllm_telemetry_store = app.state.edgemem_telemetry_store = telemetry_store
    app.state.crispllm_config = app.state.edgemem_config = cfg
    app.state.crispllm_memory_store = app.state.edgemem_memory_store = memory_store
    app.state.crispllm_memory_middleware = app.state.edgemem_memory_middleware = memory_middleware
    app.state.crispllm_consolidation_worker = app.state.edgemem_consolidation_worker = consolidation_worker
    app.state.crispllm_router = app.state.edgemem_router = router
    app.state.crispllm_redactor = app.state.edgemem_redactor = redactor
    app.state.crispllm_counters = app.state.edgemem_counters = counters
    app.state.crispllm_watchdog = app.state.edgemem_watchdog = watchdog
    app.state.crispllm_budgeter = app.state.edgemem_budgeter = budgeter
    app.state.crispllm_unloader = app.state.edgemem_unloader = unloader
    app.state.crispllm_pred_engine = app.state.edgemem_pred_engine = pred_engine
    app.state.crispllm_spec_stats = app.state.edgemem_spec_stats = spec_stats
    app.state.crispllm_last_num_ctx = app.state.edgemem_last_num_ctx = None

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        c: CrispLLMConfig = request.app.state.edgemem_config
        mem_store: MemoryStore = request.app.state.edgemem_memory_store
        cnt: ProxyCounters = request.app.state.edgemem_counters
        watchdog: RamPressureWatchdog | None = getattr(
            request.app.state, "edgemem_watchdog", None
        )
        budgeter: ContextBudgeter | None = getattr(
            request.app.state, "edgemem_budgeter", None
        )
        unloader: IdleUnloadWorker | None = getattr(
            request.app.state, "edgemem_unloader", None
        )
        spec_stats: SpeculativeStats | None = getattr(
            request.app.state, "edgemem_spec_stats", None
        )
        last_num_ctx = getattr(request.app.state, "edgemem_last_num_ctx", None)

        mem_stats = mem_store.stats()
        stats = cnt.snapshot()

        system_memory_data = None
        if watchdog is not None:
            snap = watchdog.snapshot()
            system_memory_data = {
                "total_mb": round(snap.memory.total_mb, 2) if snap.memory else None,
                "available_mb": round(snap.memory.available_mb, 2) if snap.memory else None,
                "used_percent": round(snap.memory.used_percent, 2) if snap.memory else None,
                "pressure": snap.pressure,
                "source": snap.memory.source if snap.memory else None,
            }

        resources_data = None
        if budgeter is not None:
            current_pressure = watchdog.snapshot().pressure if watchdog is not None else "normal"
            resources_data = {
                "local_backend": budgeter.backend,
                "effective_max_ctx": budgeter.pressure_cap(current_pressure),
                "effective_escalate_tokens": budgeter.escalate_threshold(current_pressure),
                "last_num_ctx": last_num_ctx,
                "idle_unload_count": unloader.unload_count if unloader is not None else 0,
            }

        payload: dict[str, Any] = {
            "status": "ok",
            "upstream_base_url": c.upstream_base_url,
            "model": c.model,
            "rss_mb": round(get_rss_mb(), 2),
            "memory_enabled": c.memory.enabled,
            "facts_count": mem_stats["facts_count"],
            "memory_tiers": mem_store.tier_counts(),
            "privacy_enabled": c.privacy.enabled,
            "cloud_enabled": c.cloud.enabled,
            "local_requests": stats["local_requests"],
            "cloud_requests": stats["cloud_requests"],
            "fallback_escalations": stats["fallback_escalations"],
            "secrets_redacted_total": stats["secrets_redacted_total"],
            "task_tiers": stats["task_tiers"],
            "prefix_cache": {
                "hits": stats["prefix_cache"]["hits"],
                "misses": stats["prefix_cache"]["misses"],
                "hit_rate_pct": round(
                    (stats["prefix_cache"]["hits"] / (stats["prefix_cache"]["hits"] + stats["prefix_cache"]["misses"]) * 100.0)
                    if (stats["prefix_cache"]["hits"] + stats["prefix_cache"]["misses"]) > 0
                    else 0.0,
                    1,
                ),
            },
            "auth_enabled": bool(c.security.resolve_auth_token()),
            "allow_remote_v1": c.security.allow_remote_v1,
        }
        if system_memory_data is not None:
            payload["system_memory"] = system_memory_data
        if resources_data is not None:
            payload["resources"] = resources_data
        if spec_stats is not None:
            payload["speculative"] = spec_stats.snapshot()

        return JSONResponse(payload)

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        c: CrispLLMConfig = request.app.state.edgemem_config
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": c.model,
                        "object": "model",
                        "owned_by": "crispllm",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
    ) -> StreamingResponse | JSONResponse | Response:
        t_ingress = time.perf_counter()
        c: CrispLLMConfig = request.app.state.edgemem_config
        client: httpx.AsyncClient = request.app.state.edgemem_http_client
        tel: TelemetryLogger = request.app.state.edgemem_telemetry
        tel_store: TelemetryStore = request.app.state.edgemem_telemetry_store
        middleware: MemoryMiddleware = request.app.state.edgemem_memory_middleware
        app_router: HybridRouter = request.app.state.edgemem_router
        app_redactor: SecretRedactor = request.app.state.edgemem_redactor
        cnt: ProxyCounters = request.app.state.edgemem_counters
        pred_engine: PredictedOutputsEngine = request.app.state.edgemem_pred_engine
        spec_stats: SpeculativeStats = request.app.state.edgemem_spec_stats

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Malformed JSON body", "type": "invalid_request_error"}},
                status_code=400,
            )

        if not isinstance(body, dict):
            return JSONResponse(
                {"error": {"message": "Malformed request body: expected JSON object", "type": "invalid_request_error"}},
                status_code=400,
            )
        is_stream = body.get("stream", False)

        est_tokens_raw = estimate_tokens_from_messages(body.get("messages", []))
        body, mem_telemetry = middleware.process_request(body, request.headers)
        est_tokens_sent = estimate_tokens_from_messages(body.get("messages", []))
        tokens_saved = max(0, est_tokens_raw - est_tokens_sent)
        compacted_turns = mem_telemetry.get("compacted_turns", 0)

        if (
            mem_telemetry.get("facts_extracted", 0) > 0
            or mem_telemetry.get("memory_injected_tokens", 0) > 0
            or mem_telemetry.get("compacted_turns", 0) > 0
        ):
            logger.info(
                "Memory middleware applied: extracted=%d facts, injected=%d tokens, compacted=%d turns",
                mem_telemetry["facts_extracted"],
                mem_telemetry["memory_injected_tokens"],
                mem_telemetry["compacted_turns"],
            )

        input_tokens = est_tokens_sent
        target = app_router.select_target(body, request.headers, input_tokens)
        if target.decision is not None:
            request.app.state.crispllm_last_num_ctx = app.state.edgemem_last_num_ctx = target.decision.selected_num_ctx
            if target.decision.trigger_local_unload:
                active_unloader: IdleUnloadWorker | None = getattr(
                    request.app.state, "edgemem_unloader", None
                )
                if active_unloader is not None:
                    asyncio.create_task(active_unloader.unload_now())

        upstream_body, session, headers = _prepare_payload_for_target(
            body, target, app_redactor, app_router
        )
        header_pred = request.headers.get("X-CrispLLM-Prediction") or request.headers.get("x-crispllm-prediction") or request.headers.get("X-EdgeMem-Prediction") or request.headers.get("x-edgemem-prediction")
        spec_decision = pred_engine.evaluate(
            body,
            target_is_cloud=target.is_cloud,
            base_url=target.base_url,
            model=target.model,
            header_mode=header_pred,
        )
        if spec_decision.inject:
            upstream_body["prediction"] = spec_decision.prediction_payload
            spec_header = "predicted_outputs"
            spec_stats.inc_injected()
        else:
            if not target.is_cloud and c.resources.local_backend == "llama_server":
                spec_header = "local_preset"
            else:
                spec_header = f"skipped:{spec_decision.reason}"
            spec_stats.inc_skipped()
        proxy_overhead_ms = (time.perf_counter() - t_ingress) * 1000.0

        session_id = middleware._derive_session_id(body.get("messages", []), request.headers)
        first_msg_text = ""
        sent_messages = body.get("messages", [])
        if sent_messages and isinstance(sent_messages, list):
            first_msg_text = _get_message_text(sent_messages[0])
        prefix_hash = hashlib.sha256(first_msg_text.encode("utf-8")).hexdigest()[:12]

        with _SESSION_PREFIX_LOCK:
            prev_hash = _SESSION_PREFIX_CACHE.get(session_id)
            if prev_hash is not None and prev_hash == prefix_hash:
                prefix_status = "hit"
                cnt.inc_prefix_hit()
            else:
                prefix_status = "miss"
                cnt.inc_prefix_miss()
            _SESSION_PREFIX_CACHE[session_id] = prefix_hash
            _SESSION_PREFIX_CACHE.move_to_end(session_id)
            if len(_SESSION_PREFIX_CACHE) > _MAX_SESSION_PREFIX_CACHE:
                _SESSION_PREFIX_CACHE.popitem(last=False)

        cnt.inc_task_tier(target.task_tier)

        logger.info(
            "Incoming request: stream=%s, est_prompt_tokens=%d, model=%s, route=%s (%s)",
            is_stream,
            input_tokens,
            upstream_body.get("model", "?"),
            target.tier,
            target.reason,
        )

        worker: ConsolidationWorker | None = getattr(
            request.app.state, "edgemem_consolidation_worker", None
        )
        active_unloader = getattr(
            request.app.state, "edgemem_unloader", None
        )
        if is_stream:
            return await _handle_streaming_request(
                client,
                body,
                target,
                upstream_body,
                headers,
                session,
                input_tokens,
                tel,
                app_router,
                app_redactor,
                cnt,
                tel_store,
                est_tokens_raw,
                est_tokens_sent,
                tokens_saved,
                compacted_turns,
                worker=worker,
                unloader=active_unloader,
                proxy_overhead_ms=proxy_overhead_ms,
                prefix_status=prefix_status,
                pred_engine=pred_engine,
                spec_stats=spec_stats,
                spec_decision=spec_decision,
                spec_header=spec_header,
            )
        else:
            resp = await _handle_non_streaming_request(
                client,
                body,
                target,
                upstream_body,
                headers,
                session,
                input_tokens,
                tel,
                app_router,
                app_redactor,
                cnt,
                tel_store,
                est_tokens_raw,
                est_tokens_sent,
                tokens_saved,
                compacted_turns,
                unloader=active_unloader,
                proxy_overhead_ms=proxy_overhead_ms,
                prefix_status=prefix_status,
                pred_engine=pred_engine,
                spec_stats=spec_stats,
                spec_decision=spec_decision,
                spec_header=spec_header,
            )
            if worker is not None:
                worker.notify_turn()
            return resp

    app.mount("", mcp_app)
    return app


def _is_local_fallback_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in (404, 429)


async def _handle_non_streaming_request(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    target: RouteTarget,
    upstream_body: dict[str, Any],
    headers: dict[str, str],
    session: RedactionSession,
    input_tokens: int,
    tel: TelemetryLogger,
    router: HybridRouter,
    redactor: SecretRedactor,
    counters: ProxyCounters,
    tel_store: TelemetryStore,
    est_tokens_raw: int,
    est_tokens_sent: int,
    tokens_saved: int,
    compacted_turns: int,
    unloader: IdleUnloadWorker | None = None,
    proxy_overhead_ms: float = 0.0,
    prefix_status: str = "miss",
    pred_engine: PredictedOutputsEngine | None = None,
    spec_stats: SpeculativeStats | None = None,
    spec_decision: PredictionDecision | None = None,
    spec_header: str = "",
) -> Response:
    timer = Timer()
    timer.__enter__()
    resp: httpx.Response | None = None
    fallback_target = (
        router.cloud_fallback_target(body, task_tier=target.task_tier) if target.tier == "local" else None
    )
    fallback_used = False

    target_url = f"{target.base_url.rstrip('/')}/chat/completions"

    if fallback_target is not None:
        try:
            resp = await client.post(
                target_url,
                json=upstream_body,
                headers=headers,
                timeout=target.timeout_s,
            )
            if _is_local_fallback_status(resp.status_code):
                resp = None
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
            resp = None

        if resp is None:
            counters.inc_fallback()
            fallback_used = True
            target = fallback_target
            upstream_body, session, headers = _prepare_payload_for_target(
                body, target, redactor, router
            )
            target_url = f"{target.base_url.rstrip('/')}/chat/completions"
            resp = await client.post(
                target_url,
                json=upstream_body,
                headers=headers,
                timeout=target.timeout_s,
            )
    else:
        resp = await client.post(
            target_url,
            json=upstream_body,
            headers=headers,
            timeout=target.timeout_s,
        )

    if (
        target.tier == "cloud"
        and spec_decision is not None
        and spec_decision.inject
        and resp is not None
        and resp.status_code in (400, 422)
    ):
        if pred_engine is not None:
            pred_engine.mark_unsupported(target.base_url, target.model)
        upstream_body.pop("prediction", None)
        if spec_stats is not None:
            spec_stats.inc_fallback_retry()
        spec_header = "fallback_retry"
        resp = await client.post(
            target_url,
            json=upstream_body,
            headers=headers,
            timeout=target.timeout_s,
        )

    timer.mark_first_token()
    timer.__exit__(None, None, None)

    if target.tier == "local":
        counters.inc_local()
        if unloader is not None:
            unloader.record_local_activity(target.model)
    else:
        counters.inc_cloud()

    redacted_count = session.stats().total_redacted
    if redacted_count > 0:
        counters.add_redacted(redacted_count)

    logger.info(
        "Route completed: tier=%s, reason=%s, redacted=%d",
        target.tier,
        target.reason,
        redacted_count,
    )

    existing_timing = resp.headers.get("server-timing") if resp is not None else None
    server_timing = _format_server_timing(proxy_overhead_ms, existing_timing)
    resp_headers = {
        "X-CrispLLM-Route": target.tier,
        "X-CrispLLM-Route-Reason": target.reason,
        "X-CrispLLM-Task-Tier": target.task_tier,
        "X-CrispLLM-Prefix-Stable": prefix_status,
        "X-CrispLLM-Redacted": str(redacted_count),
        "X-CrispLLM-Overhead-Ms": f"{proxy_overhead_ms:.2f}",
        "X-CrispLLM-Speculative": spec_header,
        "X-EdgeMem-Route": target.tier,
        "X-EdgeMem-Route-Reason": target.reason,
        "X-EdgeMem-Task-Tier": target.task_tier,
        "X-EdgeMem-Prefix-Stable": prefix_status,
        "X-EdgeMem-Redacted": str(redacted_count),
        "X-EdgeMem-Overhead-Ms": f"{proxy_overhead_ms:.2f}",
        "X-EdgeMem-Speculative": spec_header,
        "Server-Timing": server_timing,
    }

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        logger.warning(
            "Upstream error response (status=%d): %s",
            resp.status_code,
            snippet,
        )
        tel.log(
            RequestTelemetry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=target.model,
                input_tokens=input_tokens,
                output_tokens=0,
                ttft_ms=round(timer.ttft_ms, 2),
                total_latency_ms=round(timer.total_ms, 2),
                proxy_rss_mb=round(get_rss_mb(), 2),
                stream=False,
            )
        )
        tel_store.record(
            route_tier=target.tier,
            route_reason=target.reason,
            model=target.model,
            est_tokens_raw=est_tokens_raw,
            est_tokens_sent=est_tokens_sent,
            tokens_saved=tokens_saved,
            compacted_turns=compacted_turns,
            secrets_redacted=redacted_count,
            fallback_used=1 if fallback_used else 0,
            latency_ms=round(timer.total_ms, 2),
        )
        try:
            error_data = resp.json()
            return JSONResponse(
                content=error_data,
                status_code=resp.status_code,
                headers=resp_headers,
            )
        except Exception:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "text/plain"),
                headers=resp_headers,
            )

    try:
        data = resp.json()
    except Exception:
        data = None

    if isinstance(data, dict) and session.has_redactions:
        data = redactor.restore_response(data, session)

    output_tokens = 0
    if isinstance(data, dict):
        if spec_stats is not None:
            spec_stats.record_usage_payload(data, is_cloud=target.is_cloud)
        usage = data.get("usage")
        if isinstance(usage, dict):
            output_tokens = usage.get("completion_tokens", 0)

    tel.log(
        RequestTelemetry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=target.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=round(timer.ttft_ms, 2),
            total_latency_ms=round(timer.total_ms, 2),
            proxy_rss_mb=round(get_rss_mb(), 2),
            stream=False,
        )
    )
    tel_store.record(
        route_tier=target.tier,
        route_reason=target.reason,
        model=target.model,
        est_tokens_raw=est_tokens_raw,
        est_tokens_sent=est_tokens_sent,
        tokens_saved=tokens_saved,
        compacted_turns=compacted_turns,
        secrets_redacted=redacted_count,
        fallback_used=1 if fallback_used else 0,
        latency_ms=round(timer.total_ms, 2),
    )

    if isinstance(data, (dict, list)):
        return JSONResponse(
            content=data,
            status_code=resp.status_code,
            headers=resp_headers,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/plain"),
        headers=resp_headers,
    )


async def _handle_streaming_request(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    target: RouteTarget,
    upstream_body: dict[str, Any],
    headers: dict[str, str],
    session: RedactionSession,
    input_tokens: int,
    tel: TelemetryLogger,
    router: HybridRouter,
    redactor: SecretRedactor,
    counters: ProxyCounters,
    tel_store: TelemetryStore,
    est_tokens_raw: int,
    est_tokens_sent: int,
    tokens_saved: int,
    compacted_turns: int,
    worker: ConsolidationWorker | None = None,
    unloader: IdleUnloadWorker | None = None,
    proxy_overhead_ms: float = 0.0,
    prefix_status: str = "miss",
    pred_engine: PredictedOutputsEngine | None = None,
    spec_stats: SpeculativeStats | None = None,
    spec_decision: PredictionDecision | None = None,
    spec_header: str = "",
) -> StreamingResponse:
    fallback_target = (
        router.cloud_fallback_target(body, task_tier=target.task_tier) if target.tier == "local" else None
    )
    fallback_used = False
    opened_resp: httpx.Response | None = None
    target_url = f"{target.base_url.rstrip('/')}/chat/completions"

    if fallback_target is not None:
        try:
            req = client.build_request(
                "POST",
                target_url,
                json=upstream_body,
                headers=headers,
                timeout=target.timeout_s,
            )
            opened_resp = await client.send(req, stream=True)
            if _is_local_fallback_status(opened_resp.status_code):
                await opened_resp.aclose()
                opened_resp = None
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
            opened_resp = None

        if opened_resp is None:
            counters.inc_fallback()
            fallback_used = True
            target = fallback_target
            upstream_body, session, headers = _prepare_payload_for_target(
                body, target, redactor, router
            )
            target_url = f"{target.base_url.rstrip('/')}/chat/completions"

    if (
        target.tier == "cloud"
        and spec_decision is not None
        and spec_decision.inject
        and opened_resp is None
    ):
        try:
            req = client.build_request(
                "POST",
                target_url,
                json=upstream_body,
                headers=headers,
                timeout=target.timeout_s,
            )
            opened_resp = await client.send(req, stream=True)
            if opened_resp.status_code in (400, 422):
                await opened_resp.aclose()
                opened_resp = None
                if pred_engine is not None:
                    pred_engine.mark_unsupported(target.base_url, target.model)
                upstream_body.pop("prediction", None)
                if spec_stats is not None:
                    spec_stats.inc_fallback_retry()
                spec_header = "fallback_retry"
                req2 = client.build_request(
                    "POST",
                    target_url,
                    json=upstream_body,
                    headers=headers,
                    timeout=target.timeout_s,
                )
                opened_resp = await client.send(req2, stream=True)
        except Exception:
            opened_resp = None

    timer = Timer()
    timer.__enter__()

    existing_timing = (
        opened_resp.headers.get("server-timing")
        if opened_resp is not None
        else None
    )
    server_timing = _format_server_timing(proxy_overhead_ms, existing_timing)
    resp_headers = {
        "X-CrispLLM-Route": target.tier,
        "X-CrispLLM-Route-Reason": target.reason,
        "X-CrispLLM-Task-Tier": target.task_tier,
        "X-CrispLLM-Prefix-Stable": prefix_status,
        "X-CrispLLM-Redacted": str(session.stats().total_redacted),
        "X-CrispLLM-Overhead-Ms": f"{proxy_overhead_ms:.2f}",
        "X-CrispLLM-Speculative": spec_header,
        "X-EdgeMem-Route": target.tier,
        "X-EdgeMem-Route-Reason": target.reason,
        "X-EdgeMem-Task-Tier": target.task_tier,
        "X-EdgeMem-Prefix-Stable": prefix_status,
        "X-EdgeMem-Redacted": str(session.stats().total_redacted),
        "X-EdgeMem-Overhead-Ms": f"{proxy_overhead_ms:.2f}",
        "X-EdgeMem-Speculative": spec_header,
        "Server-Timing": server_timing,
    }

    async def raw_generator() -> AsyncIterator[str]:
        if opened_resp is not None:
            try:
                async for sse_line in stream_sse_passthrough(
                    opened_resp, on_first_data=timer.mark_first_token
                ):
                    yield sse_line
            finally:
                await opened_resp.aclose()
        else:
            async for sse_line in stream_with_heartbeat(
                client,
                upstream_body,
                headers,
                on_first_data=timer.mark_first_token,
                url=target_url,
                timeout=target.timeout_s,
            ):
                yield sse_line

    async def generate() -> AsyncIterator[str]:
        output_tokens = 0
        try:
            wrapped = _restoring_stream_wrapper(raw_generator(), session)
            async for sse_line in wrapped:
                output_tokens += count_output_tokens_from_sse(sse_line)
                if spec_stats is not None and sse_line.startswith("data: "):
                    payload_txt = sse_line[6:].strip()
                    if payload_txt and payload_txt != "[DONE]":
                        try:
                            parsed_chunk = json.loads(payload_txt)
                            if isinstance(parsed_chunk, dict) and ("usage" in parsed_chunk or "timings" in parsed_chunk):
                                spec_stats.record_usage_payload(parsed_chunk, is_cloud=target.is_cloud)
                        except Exception:
                            pass
                yield sse_line
        except Exception:
            logger.exception("Error during streaming response")
            raise
        finally:
            timer.__exit__(None, None, None)
            if target.tier == "local":
                counters.inc_local()
                if unloader is not None:
                    unloader.record_local_activity(target.model)
            else:
                counters.inc_cloud()

            redacted_count = session.stats().total_redacted
            if redacted_count > 0:
                counters.add_redacted(redacted_count)

            logger.info(
                "Route completed: tier=%s, reason=%s, redacted=%d",
                target.tier,
                target.reason,
                redacted_count,
            )

            tel.log(
                RequestTelemetry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=target.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=round(timer.ttft_ms, 2),
                    total_latency_ms=round(timer.total_ms, 2),
                    proxy_rss_mb=round(get_rss_mb(), 2),
                    stream=True,
                )
            )

            tel_store.record(
                route_tier=target.tier,
                route_reason=target.reason,
                model=target.model,
                est_tokens_raw=est_tokens_raw,
                est_tokens_sent=est_tokens_sent,
                tokens_saved=tokens_saved,
                compacted_turns=compacted_turns,
                secrets_redacted=redacted_count,
                fallback_used=1 if fallback_used else 0,
                latency_ms=round(timer.total_ms, 2),
            )

            if worker is not None:
                worker.notify_turn()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=resp_headers,
    )
