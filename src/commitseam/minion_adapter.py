from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Literal

from openminion import (
    APIRuntime,
    Agent,
    AgentOutputValidationError,
    Handoff,
    OpenMinionConfig,
    ProviderError,
    TurnTimeoutError,
    tool,
)
from openminion.base.config.core import ActionPolicyConfig, AgentProfileConfig, StorageConfig
from openminion.base.config.providers import (
    GroqProviderConfig,
    OpenAIProviderConfig,
    ProvidersConfig,
)

from commitseam.audit import export_memory_bundle, redact_secrets
from commitseam.junit_parser import FailureItem, parse_junit_failures
from commitseam.ledger import EnqueueResult, FaultMode, InjectedFaultError, QueueLedger

TOOL_ENQUEUE_RERUN = "enqueue_rerun"
TOOL_INSPECT_FAILURE = "inspect_failure"
HANDOFF_TRIAGE = "triage_failure"

_ITEM_ID_RE = re.compile(r"item_id=([A-Za-z0-9_.:-]+)")
_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


class MissingCredentialsError(RuntimeError):
    """Raised when live mode is requested without LLM provider credentials."""


@dataclass(frozen=True)
class ProviderSettings:
    provider: Literal["groq", "openai", "offline"]
    model: str
    api_key: str
    base_url: str | None = None
    timeout_seconds: int = 45


def default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "sample_junit.xml"


def resolve_provider_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: int = 45,
) -> ProviderSettings:
    selected = (provider or "").strip().lower()
    if selected == "offline":
        return ProviderSettings(
            provider="offline",
            model=model or "offline-deterministic-local",
            api_key="offline-local-key",
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not selected:
        if openai_key:
            selected = "openai"
        elif groq_key:
            selected = "groq"
        else:
            raise MissingCredentialsError(
                "No live LLM credentials found. Set GROQ_API_KEY or OPENAI_API_KEY "
                "(or pass --provider offline for deterministic local replay)."
            )

    if selected == "groq":
        if not groq_key:
            raise MissingCredentialsError(
                "GROQ_API_KEY is required when --provider groq is selected."
            )
        # openminion==0.1.24's GroqProviderConfig default (llama-3.3-70b-versatile)
        # returns HTTP 404 on Groq; default to an active tool-calling Groq model.
        return ProviderSettings(
            provider="groq",
            model=model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=groq_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    if selected == "openai":
        if not openai_key:
            raise MissingCredentialsError(
                "OPENAI_API_KEY is required when --provider openai is selected."
            )
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL", "").strip() or None
        return ProviderSettings(
            provider="openai",
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=openai_key,
            base_url=resolved_base,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"Unsupported provider: {provider!r} (expected groq, openai, or offline)")


class _OfflineLoopbackServer:
    """Deterministic local OpenAI-compatible HTTP server so offline tests run real OpenMinion 0.1.24."""

    def __init__(self) -> None:
        self._call_count = 0
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = body.get("messages", [])
                tools = body.get("tools") or []
                tool_names = [
                    t.get("function", {}).get("name", "")
                    for t in tools
                    if isinstance(t, dict)
                ]

                with outer._lock:
                    outer._call_count += 1
                    call_id = f"call_offline_{outer._call_count}"

                user_text = " ".join(
                    str(m.get("content", ""))
                    for m in messages
                    if isinstance(m, dict) and m.get("role") == "user"
                )
                tool_msgs = [
                    m for m in messages if isinstance(m, dict) and m.get("role") == "tool"
                ]
                item_match = _ITEM_ID_RE.search(user_text)
                item_id = item_match.group(1) if item_match else "unknown_item"

                if tool_names and not tool_msgs:
                    chosen = (
                        TOOL_ENQUEUE_RERUN
                        if TOOL_ENQUEUE_RERUN in tool_names
                        else (
                            TOOL_INSPECT_FAILURE
                            if TOOL_INSPECT_FAILURE in tool_names
                            else tool_names[0]
                        )
                    )
                    reply_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": chosen,
                                    "arguments": json.dumps({"item_id": item_id}),
                                },
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                else:
                    last_tool = str(tool_msgs[-1].get("content", "")) if tool_msgs else ""
                    reply_msg = {
                        "role": "assistant",
                        "content": last_tool or json.dumps({"item_id": item_id, "status": "ok"}),
                    }
                    finish_reason = "stop"

                payload = {
                    "id": f"chatcmpl-{call_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "offline-deterministic-local"),
                    "choices": [
                        {
                            "index": 0,
                            "message": reply_msg,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 24,
                        "total_tokens": 144,
                    },
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return _Handler


def _build_runtime_config(
    *,
    om_db_path: Path,
    settings: ProviderSettings,
    override_base_url: str | None = None,
) -> OpenMinionConfig:
    if settings.provider == "groq":
        provider_name = "groq"
        providers = ProvidersConfig(
            groq=GroqProviderConfig(
                api_key=settings.api_key,
                model=settings.model,
                timeout_seconds=settings.timeout_seconds,
                tool_call_strategy="native",
            )
        )
    else:
        provider_name = "openai"
        kwargs: dict[str, Any] = {
            "api_key": settings.api_key,
            "model": settings.model,
            "timeout_seconds": settings.timeout_seconds,
            "tool_call_strategy": "native",
        }
        effective_url = override_base_url or settings.base_url
        if effective_url:
            kwargs["base_url"] = effective_url
        providers = ProvidersConfig(openai=OpenAIProviderConfig(**kwargs))

    return OpenMinionConfig(
        storage=StorageConfig(path=str(om_db_path)),
        providers=providers,
        action_policy=ActionPolicyConfig(
            mode="confirm",
            allow_read_only_without_prompt=True,
        ),
        agents={
            "coord": AgentProfileConfig(
                name="coord",
                provider=provider_name,
                default_channel="console",
            ),
            "triage": AgentProfileConfig(
                name="triage",
                provider=provider_name,
                default_channel="console",
            ),
        },
        default_agent="coord",
    )


def _inspect_openminion_store(om_db_path: Path, session_ids: list[str]) -> dict[str, Any]:
    """Read-only inspection of OpenMinion's session SQLite store for context_traces availability."""
    if not om_db_path.exists():
        return {"available": False, "reason": "session_store_missing", "trace_count": 0, "sessions_found": []}

    conn = sqlite3.connect(f"file:{om_db_path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        sessions_found: list[str] = []
        if "sessions" in tables and session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT id FROM sessions WHERE id IN ({placeholders}) ORDER BY id ASC",
                session_ids,
            ).fetchall()
            sessions_found = [str(r[0]) for r in rows]

        trace_count = 0
        if "session_events" in tables and session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM session_events
                WHERE session_id IN ({placeholders}) AND event_type = 'context_manifest_created'
                """,
                session_ids,
            ).fetchone()
            trace_count = int(row[0]) if row else 0
        elif "events" in tables and session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(events)").fetchall()
            }
            if {"session_id", "event_type"}.issubset(cols):
                row = conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM events
                    WHERE session_id IN ({placeholders}) AND event_type = 'context_manifest_created'
                    """,
                    session_ids,
                ).fetchone()
                trace_count = int(row[0]) if row else 0

        return {
            "available": trace_count > 0,
            "reason": "ok" if trace_count > 0 else "no_context_manifest_created_events_recorded",
            "trace_count": trace_count,
            "sessions_found": sessions_found,
        }
    finally:
        conn.close()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _extract_retry_sleep(exc: Exception) -> float:
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return min(float(match.group(1)) + 1.5, 45.0)
    return 15.0


def run_demo(
    *,
    run_dir: Path,
    fixture_path: Path | None = None,
    request_id: str = "demo-run",
    provider_settings: ProviderSettings | None = None,
    approve_callback: Callable[[str, list[str], Path], bool] | None = None,
    max_attempts: int = 3,
    receipt_out: Path | None = None,
) -> dict[str, Any]:
    """Execute the 3 CommitSeam scenarios through real OpenMinion 0.1.24 + isolated SQLite queue."""
    logging.getLogger("openminion").setLevel(logging.WARNING)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = provider_settings or resolve_provider_settings()
    mode = "OFFLINE_REPLAY" if settings.provider == "offline" else "LIVE"
    resolved_fixture = Path(fixture_path) if fixture_path is not None else default_fixture_path()
    items = parse_junit_failures(resolved_fixture)
    if len(items) < 3:
        raise ValueError(f"Fixture {resolved_fixture} must contain at least 3 failing testcases")

    queue_db = run_dir / "queue.sqlite"
    om_db = run_dir / "openminion.sqlite"
    ledger = QueueLedger(queue_db)

    # Confirm operator approval up front if a custom callback was supplied.
    operator_approved = True
    if approve_callback is not None:
        operator_approved = bool(
            approve_callback(request_id, [it.item_id for it in items], queue_db)
        )

    loopback: _OfflineLoopbackServer | None = None
    if settings.provider == "offline":
        loopback = _OfflineLoopbackServer()
        loopback.start()

    runtime: APIRuntime | None = None
    runtime_closed = False
    t0 = time.monotonic()

    try:
        cfg = _build_runtime_config(
            om_db_path=om_db,
            settings=settings,
            override_base_url=loopback.base_url if loopback is not None else None,
        )
        runtime = APIRuntime.from_config(
            config=cfg,
            home_root=run_dir / "om_home",
            data_root=run_dir / "om_data",
        )

        items_by_id: dict[str, FailureItem] = {it.item_id: it for it in items}

        # Mutable per-attempt context consumed by the coordinator tool closure.
        ctx: dict[str, Any] = {
            "request_id": "",
            "attempt_id": "",
            "session_id": "",
            "fault_seam": FaultMode.NONE,
            "cancel_event": None,
            "last_result": None,
            "last_fault": None,
        }

        @tool(min_scope="READ_ONLY", idempotent=True)
        def inspect_failure(item_id: str) -> str:
            """Read-only lookup of a parsed JUnit failure item."""
            item = items_by_id.get(item_id)
            if item is None:
                return json.dumps({"item_id": item_id, "found": False})
            return json.dumps(
                {
                    "item_id": item.item_id,
                    "failure_kind": item.failure_kind,
                    "message": item.message,
                    "payload_digest": item.payload_digest,
                }
            )

        @tool(min_scope="WRITE_SAFE", dangerous=True, idempotent=False)
        def enqueue_rerun(item_id: str) -> str:
            """Enqueue a failed pytest item_id into CommitSeam's durable SQLite rerun queue."""
            item = items_by_id.get(item_id)
            if item is None:
                raise ValueError(f"Unknown failure item_id: {item_id}")
            try:
                res: EnqueueResult = ledger.enqueue(
                    request_id=ctx["request_id"],
                    item_id=item.item_id,
                    payload_digest=item.payload_digest,
                    attempt_id=ctx["attempt_id"],
                    session_id=ctx["session_id"],
                    fault_seam=ctx["fault_seam"],
                )
                ctx["last_result"] = res
                return json.dumps(
                    {
                        "action": res.action,
                        "request_id": res.request_id,
                        "item_id": res.item_id,
                        "job_id": res.job_id,
                        "recovered_job_id": res.recovered_job_id,
                        "created_by_attempt_id": res.created_by_attempt_id,
                    }
                )
            except InjectedFaultError as exc:
                ctx["last_fault"] = exc
                cancel_ev = ctx.get("cancel_event")
                if isinstance(cancel_ev, threading.Event):
                    cancel_ev.set()
                raise

        triage_agent = Agent(
            runtime=runtime,
            agent_id="triage",
            instructions="Inspect failed JUnit testcases read-only using inspect_failure.",
            tools=[inspect_failure],
        )
        triage_handoff = Handoff(
            triage_agent,
            name=HANDOFF_TRIAGE,
            description="Read-only JUnit failure classifier (no write tools).",
        )
        handoff_write_tools = [
            name for name in triage_agent.tools if name == TOOL_ENQUEUE_RERUN
        ]
        handoff_peer_read_only = (
            len(handoff_write_tools) == 0 and TOOL_INSPECT_FAILURE in triage_agent.tools
        )

        coord_agent = Agent(
            runtime=runtime,
            agent_id="coord",
            instructions=(
                f"Call {TOOL_ENQUEUE_RERUN} exactly once with the requested item_id, "
                "then briefly report the tool output."
            ),
            tools=[enqueue_rerun, inspect_failure],
            forced_tools=[TOOL_ENQUEUE_RERUN],
            handoffs=[triage_handoff],
        )

        def _invoke_attempt(
            *,
            req_id: str,
            item: FailureItem,
            attempt_id: str,
            session_id: str,
            fault_seam: FaultMode,
            allow_write: bool,
        ) -> dict[str, Any]:
            approval_calls: list[dict[str, Any]] = []
            cancel_ev = threading.Event()
            ctx.update(
                {
                    "request_id": req_id,
                    "attempt_id": attempt_id,
                    "session_id": session_id,
                    "fault_seam": fault_seam,
                    "cancel_event": cancel_ev,
                    "last_result": None,
                    "last_fault": None,
                }
            )

            def decide(tool_name: str, args: dict[str, Any], approval_id: str) -> bool:
                bound_item = str(args.get("item_id", ""))
                approved = bool(
                    allow_write
                    and operator_approved
                    and tool_name == TOOL_ENQUEUE_RERUN
                    and bound_item == item.item_id
                )
                approval_calls.append(
                    {
                        "tool_name": tool_name,
                        "request_id": req_id,
                        "item_id": bound_item,
                        "queue_path": queue_db.name,
                        "approval_id": approval_id,
                        "approved": approved,
                    }
                )
                if not approved:
                    ledger.record_denial(
                        request_id=req_id,
                        item_id=item.item_id,
                        attempt_id=attempt_id,
                        session_id=session_id,
                        detail=f"approval_callback_denied:{approval_id}",
                    )
                return approved

            prompt = (
                f"Enqueue failed test item_id={item.item_id} for request_id={req_id} "
                f"into the rerun queue using {TOOL_ENQUEUE_RERUN}."
            )
            stats: dict[str, Any] = {}
            sdk_error: str | None = None
            reply_text = ""

            for retry_idx in range(max_attempts):
                try:
                    res = coord_agent.run(
                        prompt,
                        session_id=session_id if retry_idx == 0 else f"{session_id}-rl{retry_idx}",
                        approval_callback=decide,
                        cancel_event=cancel_ev,
                        timeout_seconds=float(settings.timeout_seconds),
                    )
                    stats = dict(res.stats or {})
                    reply_text = res.text
                    break
                except (ProviderError, TurnTimeoutError) as exc:
                    sdk_error = f"{type(exc).__name__}: {exc}"
                    # If the tool already fired an injected fault OR already committed before
                    # a real provider error (e.g. Groq TPM post-tool rate limit), stop retrying
                    # this attempt and let outer SQLite reconciliation handle recovery!
                    if ctx["last_fault"] is not None or ctx["last_result"] is not None:
                        break
                    if "RATE_LIMITED" in str(exc) and retry_idx + 1 < max_attempts:
                        time.sleep(_extract_retry_sleep(exc))
                        continue
                    raise
                except AgentOutputValidationError as exc:
                    sdk_error = f"AgentOutputValidationError: {exc}"
                    break
                except Exception as exc:
                    if ctx["last_fault"] is not None:
                        sdk_error = f"{type(exc).__name__}: {exc}"
                        break
                    raise

            last_fault: InjectedFaultError | None = ctx["last_fault"]
            last_result: EnqueueResult | None = ctx["last_result"]
            return {
                "attempt_id": attempt_id,
                "session_id": session_id,
                "fault_seam": fault_seam.value,
                "approval_calls": approval_calls,
                "injected_fault": last_fault.seam.value if last_fault is not None else None,
                "injected_committed_job_id": (
                    last_fault.committed_job_id if last_fault is not None else None
                ),
                "tool_result": (
                    {
                        "action": last_result.action,
                        "job_id": last_result.job_id,
                        "recovered_job_id": last_result.recovered_job_id,
                        "created_by_attempt_id": last_result.created_by_attempt_id,
                        "error_code": last_result.error_code,
                    }
                    if last_result is not None
                    else None
                ),
                "sdk_error": sdk_error,
                "stats": stats,
                "reply_excerpt": reply_text[:160],
            }

        # ------------------------------------------------------------------
        # Scenario 1: Approval Denial (0 rows added to rerun_jobs)
        # ------------------------------------------------------------------
        deny_req = f"{request_id}-deny"
        deny_item = items[0]
        ledger.register_expected(deny_req, [deny_item])
        deny_att = _invoke_attempt(
            req_id=deny_req,
            item=deny_item,
            attempt_id="att-deny-1",
            session_id=f"{deny_req}-s1",
            fault_seam=FaultMode.NONE,
            allow_write=False,
        )
        deny_rec = ledger.reconcile(deny_req)
        _, deny_bundle_json = export_memory_bundle(
            ledger, request_id=deny_req, mode=mode, metadata={"scenario": "approval_denial"}
        )
        (run_dir / f"memory-bundle-{deny_req}.json").write_text(deny_bundle_json + "\n", encoding="utf-8")

        # ------------------------------------------------------------------
        # Scenario 2: Seam B — Pre-Commit Crash (0 -> 1 across fresh session_id)
        # ------------------------------------------------------------------
        pre_req = f"{request_id}-pre"
        pre_item = items[1]
        ledger.register_expected(pre_req, [pre_item])
        pre_att1 = _invoke_attempt(
            req_id=pre_req,
            item=pre_item,
            attempt_id="att-pre-1",
            session_id=f"{pre_req}-s1",
            fault_seam=FaultMode.PRE_COMMIT,
            allow_write=True,
        )
        pre_after_crash = len(ledger.committed_jobs(pre_req))
        pending_pre = ledger.pending_items(pre_req)
        pre_attempts = [pre_att1]
        if pending_pre and operator_approved:
            pre_att2 = _invoke_attempt(
                req_id=pre_req,
                item=items_by_id[pending_pre[0]["item_id"]],
                attempt_id="att-pre-2",
                session_id=f"{pre_req}-s2",
                fault_seam=FaultMode.NONE,
                allow_write=True,
            )
            pre_attempts.append(pre_att2)
        pre_rec = ledger.reconcile(pre_req)
        _, pre_bundle_json = export_memory_bundle(
            ledger, request_id=pre_req, mode=mode, metadata={"scenario": "pre_commit_crash"}
        )
        (run_dir / f"memory-bundle-{pre_req}.json").write_text(pre_bundle_json + "\n", encoding="utf-8")

        # ------------------------------------------------------------------
        # Scenario 3: Seam A — Post-Commit Crash (1 -> 1 identical job_id)
        # ------------------------------------------------------------------
        post_req = f"{request_id}-post"
        post_item = items[2]
        ledger.register_expected(post_req, [post_item])
        post_att1 = _invoke_attempt(
            req_id=post_req,
            item=post_item,
            attempt_id="att-post-1",
            session_id=f"{post_req}-s1",
            fault_seam=FaultMode.POST_COMMIT,
            allow_write=True,
        )
        post_after_crash_jobs = ledger.committed_jobs(post_req)
        post_after_crash = len(post_after_crash_jobs)
        initial_job_id = post_after_crash_jobs[0]["job_id"] if post_after_crash_jobs else None
        initial_creator = (
            post_after_crash_jobs[0]["created_by_attempt_id"] if post_after_crash_jobs else None
        )
        post_attempts = [post_att1]
        if operator_approved:
            # Retry on a fresh SDK session_id under the same business request_id;
            # ledger must return action="reused" and identical recovered_job_id.
            post_att2 = _invoke_attempt(
                req_id=post_req,
                item=post_item,
                attempt_id="att-post-2",
                session_id=f"{post_req}-s2",
                fault_seam=FaultMode.NONE,
                allow_write=True,
            )
            post_attempts.append(post_att2)
        post_rec = ledger.reconcile(post_req)
        _, post_bundle_json = export_memory_bundle(
            ledger, request_id=post_req, mode=mode, metadata={"scenario": "post_commit_crash"}
        )
        (run_dir / f"memory-bundle-{post_req}.json").write_text(post_bundle_json + "\n", encoding="utf-8")

        all_sessions = (
            [deny_att["session_id"]]
            + [a["session_id"] for a in pre_attempts]
            + [a["session_id"] for a in post_attempts]
        )
        context_traces_info = _inspect_openminion_store(om_db, all_sessions)
    finally:
        if runtime is not None:
            runtime.close()
            runtime_closed = True
        if loopback is not None:
            loopback.close()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    post_replay = post_attempts[-1]["tool_result"] if len(post_attempts) > 1 else None

    receipt: dict[str, Any] = redact_secrets(
        {
            "mode": mode,
            "openminion_version": "0.1.24",
            "provider": settings.provider,
            "model": settings.model,
            "base_request_id": request_id,
            "fixture_name": resolved_fixture.name,
            "elapsed_ms": elapsed_ms,
            "runtime_closed": runtime_closed,
            "handoff_peer_read_only": handoff_peer_read_only,
            "handoff_peer_write_tools": handoff_write_tools,
            "context_traces": context_traces_info,
            "scenarios": {
                "approval_denial": {
                    "request_id": deny_req,
                    "item_id": deny_item.item_id,
                    "session_ids": [deny_att["session_id"]],
                    "approval_callback_invoked": len(deny_att["approval_calls"]) > 0,
                    "committed_enqueue_count": deny_rec["committed_enqueue_count"],
                    "denied_events_count": sum(
                        1 for ev in deny_rec["events"] if ev["result"] == "denied"
                    ),
                    "pending_item_ids": deny_rec["pending_item_ids"],
                    "memory_bundle_file": f"memory-bundle-{deny_req}.json",
                    "attempts": [deny_att],
                },
                "pre_commit_crash": {
                    "request_id": pre_req,
                    "item_id": pre_item.item_id,
                    "session_ids": [a["session_id"] for a in pre_attempts],
                    "after_crash_committed_count": pre_after_crash,
                    "committed_enqueue_count": pre_rec["committed_enqueue_count"],
                    "exact_once_ok": pre_rec["exact_once_ok"],
                    "job_id": pre_rec["jobs"][0]["job_id"] if pre_rec["jobs"] else None,
                    "created_by_attempt_id": (
                        pre_rec["jobs"][0]["created_by_attempt_id"] if pre_rec["jobs"] else None
                    ),
                    "memory_bundle_file": f"memory-bundle-{pre_req}.json",
                    "attempts": pre_attempts,
                },
                "post_commit_crash": {
                    "request_id": post_req,
                    "item_id": post_item.item_id,
                    "session_ids": [a["session_id"] for a in post_attempts],
                    "after_crash_committed_count": post_after_crash,
                    "committed_enqueue_count": post_rec["committed_enqueue_count"],
                    "exact_once_ok": post_rec["exact_once_ok"],
                    "initial_committed_job_id": initial_job_id,
                    "initial_attempt_id": initial_creator,
                    "replay_action": post_replay["action"] if post_replay else None,
                    "recovered_job_id": post_replay["recovered_job_id"] if post_replay else None,
                    "created_by_attempt_id": (
                        post_rec["jobs"][0]["created_by_attempt_id"] if post_rec["jobs"] else None
                    ),
                    "memory_bundle_file": f"memory-bundle-{post_req}.json",
                    "attempts": post_attempts,
                },
            },
        }
    )

    _atomic_write_json(run_dir / "receipts.json", receipt)
    if receipt_out is not None:
        _atomic_write_json(Path(receipt_out), receipt)
    return receipt


def diagnose_run(
    run_dir: Path,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Offline read-only reconciliation of SQLite queue rows, MemoryBundle, and OpenMinion traces."""
    run_dir = Path(run_dir)
    queue_db = run_dir / "queue.sqlite"
    om_db = run_dir / "openminion.sqlite"
    receipt_path = run_dir / "receipts.json"

    if not queue_db.exists():
        raise FileNotFoundError(f"Queue database not found: {queue_db}")

    saved_receipt: dict[str, Any] = {}
    if receipt_path.exists():
        saved_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    mode = str(saved_receipt.get("mode", "OFFLINE_REPLAY"))
    ledger = QueueLedger(queue_db)

    def _build_single(req_id: str) -> dict[str, Any]:
        rec = ledger.reconcile(req_id)
        session_ids = sorted(
            {str(ev["session_id"]) for ev in rec["events"] if ev.get("session_id")}
        )
        reused_events = [ev for ev in rec["events"] if ev.get("result") == "reused"]
        recovered_job_id = reused_events[-1]["job_id"] if reused_events else None
        last_action = rec["events"][-1]["result"] if rec["events"] else None

        bundle_path = run_dir / f"memory-bundle-{req_id}.json"
        bundle_verified = False
        if bundle_path.exists():
            try:
                parsed_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                bundle_verified = (
                    isinstance(parsed_bundle, dict)
                    and parsed_bundle.get("metadata", {}).get("request_id") == req_id
                )
            except json.JSONDecodeError:
                bundle_verified = False

        traces = _inspect_openminion_store(om_db, session_ids)
        return {
            "mode": mode,
            "request_id": req_id,
            "expected_count": rec["expected_count"],
            "committed_enqueue_count": rec["committed_enqueue_count"],
            "pending_item_ids": rec["pending_item_ids"],
            "has_duplicates": rec["has_duplicates"],
            "exact_once_ok": rec["exact_once_ok"],
            "action": last_action,
            "recovered_job_id": recovered_job_id,
            "jobs": rec["jobs"],
            "events": rec["events"],
            "session_ids": session_ids,
            "context_traces": traces,
            "memory_bundle_verified": bundle_verified,
        }

    if request_id is not None:
        return redact_secrets(_build_single(request_id))

    conn = sqlite3.connect(f"file:{queue_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT request_id FROM expected_items ORDER BY request_id ASC"
        ).fetchall()
        req_ids = [str(r[0]) for r in rows]
    finally:
        conn.close()

    return redact_secrets(
        {
            "mode": mode,
            "model": saved_receipt.get("model"),
            "provider": saved_receipt.get("provider"),
            "requests": {rid: _build_single(rid) for rid in req_ids},
        }
    )
