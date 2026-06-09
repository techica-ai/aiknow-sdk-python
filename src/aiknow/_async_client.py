"""
AIKNOW SDK — asynchronous client.
"""
from __future__ import annotations

import os
from typing import Self

import httpx

from .resources.chat import AsyncChatResource
from .resources.conversation import AsyncConversationResource
from .resources.extensions import AsyncExtensionsResource
from .resources.ingestion import AsyncIngestionResource
from .resources.knowledge import AsyncKnowledgeResource, KnowledgeChunk as KnowledgeChunk  # re-export
from .resources.observe import AsyncObserveResource
from .resources.workflows import AsyncWorkflowsResource
from .resources.auth import AsyncAuthResource
from ._auth_flow import AIKnowAuth
from ._span_builder import SpanBuilder as SpanBuilder  # re-export

_DEFAULT_BASE_URL = "http://localhost:8000/api/v1"


class AsyncAIKnowClient:
    """Asynchronous AIKNOW Platform client.

    Usage::

        async with AsyncAIKnowClient(api_key="...") as client:
            response = await client.chat.ask("What is AIKNOW?", tenant_id="acme")

    Args:
        base_url:   Base URL of the AIKNOW API.
                    Defaults to ``AIKNOW_BASE_URL`` env var, then localhost.
        api_key:    Bearer token for end-user API endpoints.
                    Reads ``AIKNOW_API_KEY`` env var if not provided.
        admin_key:  Admin key for observability endpoints.
                    Reads ``AIKNOW_ADMIN_KEY`` env var if not provided.
                    When present, ``.observe`` resource is available.
        tenant_id:  Default tenant identifier sent as ``X-Tenant-Id`` header
                    on every request. Can be overridden per-call via
                    :meth:`set_tenant_id`.
        timeout:    Request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        admin_key: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        resolved_base = (
            base_url
            or os.environ.get("AIKNOW_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        resolved_api_key = api_key or os.environ.get("AIKNOW_API_KEY")
        resolved_admin_key = admin_key or os.environ.get("AIKNOW_ADMIN_KEY")

        self.auth = AsyncAuthResource(self)
        if resolved_api_key:
            self.auth.access_token = resolved_api_key

        self._auth_flow = AIKnowAuth(self.auth, auto_refresh=True)

        self._raw_client = httpx.AsyncClient(
            base_url=resolved_base,
            timeout=httpx.Timeout(timeout),
            transport=httpx.AsyncHTTPTransport(retries=3),
        )

        self._client = httpx.AsyncClient(
            base_url=resolved_base,
            auth=self._auth_flow,
            timeout=httpx.Timeout(timeout),
            transport=httpx.AsyncHTTPTransport(retries=3),
        )

        # Set default X-Tenant-Id header if provided
        if tenant_id:
            self._client.headers["X-Tenant-Id"] = tenant_id
            self._raw_client.headers["X-Tenant-Id"] = tenant_id

        self.chat = AsyncChatResource(self._client)
        self.conversation = AsyncConversationResource(self._client)
        self.ingestion = AsyncIngestionResource(self._client)
        self.extensions = AsyncExtensionsResource(self._client)
        self.workflows = AsyncWorkflowsResource(self._client)
        self.knowledge = AsyncKnowledgeResource(
            self._client,
            tenant_id=tenant_id,  # Propagate default tenant for KB searches
        )
        self.observe: AsyncObserveResource | None = (
            AsyncObserveResource(self._client, resolved_admin_key)
            if resolved_admin_key
            else None
        )

    def set_tenant_id(self, tenant_id: str) -> None:
        """Set or update the default ``X-Tenant-Id`` header.

        Args:
            tenant_id: Tenant identifier to send on every subsequent request.
        """
        self._client.headers["X-Tenant-Id"] = tenant_id
        self._raw_client.headers["X-Tenant-Id"] = tenant_id

    async def ping(self) -> bool:
        """Check connectivity to the AIKNOW API.

        Returns ``True`` if the server responds with HTTP 200, ``False``
        on any network or HTTP error.
        """
        try:
            res = await self._client.get("/health")
            return res.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def push_trace(
        self,
        trace: dict,
        spans: list[dict],
    ) -> dict:
        """Push a conversation trace and its spans to the Platform.

        Designed to be called after :class:`SpanBuilder.flush()`::

            builder = SpanBuilder(trace_type="conversation", session_id=thread_id)
            with builder.span("intent.detect") as s:
                result = await detect_intent(text)
                s.set("intent.workflow_id", result.workflow_id)

            trace_dict, spans_list = builder.flush()
            await client.push_trace(trace_dict, spans_list)

        Auth: Uses ``api_key`` (Bearer token) + ``X-Tenant-Id`` header.
        No admin key required — tenants record their own traces.

        Args:
            trace:  Trace metadata dict from :meth:`SpanBuilder.flush`.
            spans:  List of span dicts from :meth:`SpanBuilder.flush`.

        Returns:
            Dict with ``trace_id`` and ``spans_accepted``.
        """
        payload = {**trace, "spans": spans}
        try:
            res = await self._client.post("/observe/push", json=payload)
            res.raise_for_status()
            return res.json()
        except Exception as exc:
            # Fire-and-forget: log but never raise so callers aren't disrupted
            import logging
            logging.getLogger(__name__).warning(
                "push_trace failed (trace_id=%s): %s", trace.get("trace_id"), exc
            )
            return {"trace_id": trace.get("trace_id"), "spans_accepted": 0}

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()
        await self._raw_client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()
