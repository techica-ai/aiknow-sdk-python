"""
WebhookListener — SDK-side FastAPI server for Platform → App callbacks.

This is a lightweight async HTTP server (uvicorn + FastAPI) that:
    1. Listens on a configurable host:port.
    2. Receives POST /steps/{step_name} (or /skills/{step_name} for legacy compat)
       from the Platform's HttpRemoteStepInvoker.
    3. Verifies the HMAC-SHA256 signature (if hmac_secret is configured).
    4. Dispatches to the matching @step function via SkillDispatcher.
    5. Returns the StepResult as JSON.
    6. Receives POST /hooks/{event_type} for @hook event delivery.

Architecture:
    AiknowApp
        └── WebhookListener
                ├── SkillDispatcher (skill routing)
                │       └── LocalExtensionRegistry (skill lookup)
                └── HookDispatcher  (hook routing — inline in this module)

Lifecycle:
    Managed by AiknowApp.start() / AiknowApp.stop().
    start() creates the listener and launches uvicorn in a background asyncio task.
    stop() cancels the task and waits for clean shutdown.

Security:
    If hmac_secret is set, all incoming requests MUST carry a valid
    X-AIKnow-Signature header. Requests without a valid signature get 403.
    In development (hmac_secret=None), signature checking is skipped with
    a WARNING log so developers are notified.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from aiknow._hmac import SignatureVerificationError, verify_signature
from aiknow._skill_dispatcher import SkillDispatcher
from aiknow.extensions._local_registry import LocalExtensionRegistry
from aiknow.extensions.types import HookEvent

logger = logging.getLogger(__name__)

_SIGNATURE_HEADER = "X-AIKnow-Signature"


# ---------------------------------------------------------------------------
# Request schemas (wire format from Platform)
# ---------------------------------------------------------------------------


class StepInvocationPayload(BaseModel):
    """JSON body sent by the Platform's HttpRemoteStepInvoker.

    Accepts both ``step_name`` (canonical) and ``skill_name`` (legacy backward-compat).
    The ``model_validator`` runs at parse time so any missing-name error surfaces
    as HTTP 422 (Pydantic validation) rather than HTTP 500 (unhandled exception).

    After validation, ``step_name`` is always populated with the resolved name;
    callers should read ``payload.step_name`` directly.
    """

    execution_id: str
    # step_name is the canonical field sent by current Platform versions.
    # skill_name is accepted for backward compat with older Platform releases.
    step_name: str = ""
    skill_name: str = ""  # legacy; normalized to step_name by validator below
    tenant_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    locale: str = "vi"
    permissions: list[str] = Field(default_factory=list)
    user_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Seed data from start_execution.initial_inputs — forwarded verbatim to AppSkillDeps.user_context",
    )

    @model_validator(mode="after")
    def _resolve_step_name(self) -> "StepInvocationPayload":
        """Normalize: if step_name is empty, copy skill_name (legacy) into it.

        Raises ValidationError (→ HTTP 422) at parse time if both are absent.
        """
        if not self.step_name and not self.skill_name:
            raise ValueError("Either 'step_name' or 'skill_name' must be provided in the request body.")
        if not self.step_name:
            self.step_name = self.skill_name  # promote legacy field to canonical
        return self


# Keep the old name as an alias so existing code that imports SkillInvocationPayload
# continues to work without changes (remove in v3.0).
SkillInvocationPayload = StepInvocationPayload  # noqa: N816 — deprecated alias


class HookEventPayload(BaseModel):
    """JSON body sent by the Platform for hook event delivery."""

    event_type: str
    workflow_id: str | None = None
    execution_id: str | None = None
    tenant_id: str = ""
    state_id: str | None = None
    skill_name: str | None = None
    outcome: str | None = None
    event_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebhookListener
# ---------------------------------------------------------------------------


class WebhookListener:
    """FastAPI-based HTTP server receiving Platform → App callbacks.

    Args:
        registry:    The app's LocalExtensionRegistry (skills + hooks).
        host:        Bind host (default: '0.0.0.0').
        port:        Listen port (default: 9000).
        hmac_secret: Shared secret for HMAC-SHA256 signature verification.
                     If None, signature checking is skipped (development mode).
         path_prefix: URL prefix for all routes (default: '').
    """

    def __init__(
        self,
        registry: LocalExtensionRegistry,
        host: str = "0.0.0.0",
        port: int = 9000,
        hmac_secret: str | None = None,
        path_prefix: str = "",
        advertise_host: str | None = None,
    ) -> None:
        self._registry = registry
        self._host = host
        self._port = port
        self._hmac_secret = hmac_secret
        self._path_prefix = path_prefix.rstrip("/")
        self._advertise_host = advertise_host  # override for manifest URL
        self._dispatcher = SkillDispatcher(registry)
        self._server_task: asyncio.Task | None = None
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the uvicorn server in a background asyncio task.

        Returns immediately — the server runs in the background.
        """
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",  # Suppress uvicorn access logs (use our own)
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        logger.info(
            "WebhookListener started: http://%s:%d%s",
            self._host, self._port, self._path_prefix or "/",
        )

    async def stop(self) -> None:
        """Cancel the background server task and wait for shutdown."""
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            self._server_task = None
        logger.info("WebhookListener stopped.")

    @property
    def base_url(self) -> str:
        """Public base URL for this listener (used in AppManifest.webhook_url).

        If ``advertise_host`` was provided at construction, it is used directly.
        Otherwise, when the listener is bound to a wildcard address (0.0.0.0 or ::),
        we return 127.0.0.1 instead — the wildcard is a bind address, not a
        connectable target.
        """
        if self._advertise_host:
            host = self._advertise_host
        else:
            host = self._host
            if host in ("0.0.0.0", "::"):
                host = "127.0.0.1"
        return f"http://{host}:{self._port}{self._path_prefix}"

    @property
    def app(self) -> FastAPI:
        """Expose the FastAPI app for testing with httpx.ASGITransport."""
        return self._app

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        """Build the FastAPI application with all routes."""
        app = FastAPI(
            title="AIKnow WebhookListener",
            description="Internal: receives skill invocations and hook events from the Platform.",
            version="1.0.0",
            docs_url=None,   # No Swagger in production listeners
            redoc_url=None,
        )

        prefix = self._path_prefix

        async def _handle_step_invocation(
            path_name: str,
            request: Request,
        ) -> JSONResponse:
            """Shared handler for both /steps/{name} and /skills/{name} routes."""
            body = await request.body()
            await self._verify(body, request)

            try:
                payload = StepInvocationPayload.model_validate_json(body)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid request body: {exc}",
                ) from exc

            # After model_validator, payload.step_name is always populated.
            # Safety check: name in URL path must match name in body.
            if payload.step_name != path_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Path name '{path_name}' does not match "
                        f"body step_name '{payload.step_name}'."
                    ),
                )

            logger.info(
                "WebhookListener: invoke step='%s' execution_id='%s' tenant='%s'",
                payload.step_name, payload.execution_id, payload.tenant_id,
            )

            result = await self._dispatcher.dispatch(
                skill_name=payload.step_name,
                tenant_id=payload.tenant_id,
                execution_id=payload.execution_id,
                params=payload.params,
                session_id=payload.session_id,
                locale=payload.locale,
                permissions=payload.permissions,
                user_context=payload.user_context,
            )

            return JSONResponse(content=result.to_dict())

        # Primary route: /steps/{step_name} — current Platform naming
        @app.post(
            f"{prefix}/steps/{{step_name}}",
            summary="Invoke a step (called by Platform — current)",
        )
        async def invoke_step(
            step_name: str,
            request: Request,
        ) -> JSONResponse:
            return await _handle_step_invocation(step_name, request)

        # Legacy route: /skills/{skill_name} — backward compat
        @app.post(
            f"{prefix}/skills/{{skill_name}}",
            summary="Invoke a skill (legacy — backward compat)",
        )
        async def invoke_skill(
            skill_name: str,
            request: Request,
        ) -> JSONResponse:
            return await _handle_step_invocation(skill_name, request)

        @app.post(
            f"{prefix}/hooks/{{event_type:path}}",
            summary="Deliver a workflow hook event (called by Platform)",
            status_code=status.HTTP_202_ACCEPTED,
        )
        async def deliver_hook(
            event_type: str,
            request: Request,
        ) -> dict:
            body = await request.body()
            await self._verify(body, request)
            
            logger.info("WebhookListener: raw hook body: %s", body.decode("utf-8"))

            try:
                payload = HookEventPayload.model_validate_json(body)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid hook payload: {exc}",
                ) from exc

            # Merge flat fields into the payload dict to preserve backward compatibility
            # with app hooks that extract them from hook_event.payload
            extended_payload = {**payload.payload}
            if payload.state_id is not None:
                extended_payload["state_id"] = payload.state_id
            if payload.skill_name is not None:
                extended_payload["skill_name"] = payload.skill_name
            if payload.outcome is not None:
                extended_payload["outcome"] = payload.outcome
            if payload.event_id is not None:
                extended_payload["event_id"] = payload.event_id

            logger.info("WebhookListener: extended_payload parsed: %s", extended_payload)

            hook_event = HookEvent(
                event_type=payload.event_type,
                workflow_id=payload.workflow_id,
                execution_id=payload.execution_id,
                tenant_id=payload.tenant_id,
                payload=extended_payload,
            )

            # Fire all matching @hook handlers (best-effort, non-fatal)
            await self._dispatch_hooks(hook_event)

            return {"accepted": True, "event_type": event_type}

        @app.get(f"{prefix}/health", summary="Health check")
        async def health() -> dict:
            # list_by_type("step") includes both @step and @skill (legacy) entries
            steps = self._registry.list_by_type("step")
            return {
                "status": "ok",
                "steps": len(steps),
                "skills": len(steps),  # backward-compat alias — same value
                "hooks": len(self._registry.list_by_type("hook")),
            }

        return app

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _verify(self, body: bytes, request: Request) -> None:
        """Verify HMAC signature if secret is configured.

        Raises HTTPException(403) on invalid/missing signature.
        Logs a WARNING and skips if no secret is configured (dev mode).
        """
        if not self._hmac_secret:
            logger.warning(
                "WebhookListener: HMAC secret not configured — "
                "signature verification DISABLED (development mode only)."
            )
            return

        sig_header = request.headers.get(_SIGNATURE_HEADER)
        try:
            verify_signature(body, sig_header, self._hmac_secret)
        except SignatureVerificationError as exc:
            logger.warning(
                "WebhookListener: signature verification failed: %s", exc
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

    async def _dispatch_hooks(self, event: HookEvent) -> None:
        """Call all @hook handlers registered for event.event_type.

        Errors in individual hook handlers are caught and logged
        so one failing hook does not block delivery to others.
        """
        entries = [
            e for e in self._registry.list_by_type("hook")
            if e.name == event.event_type
        ]
        for entry in entries:
            try:
                await entry.fn(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "WebhookListener: hook handler '%s' raised: %s",
                    entry.name, exc,
                )
