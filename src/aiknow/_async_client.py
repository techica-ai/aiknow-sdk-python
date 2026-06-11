"""
AIKNOW SDK — asynchronous client.
"""
from __future__ import annotations

import os
import threading as _threading
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
from .resources.users import AsyncUsersResource
from ._auth_flow import AIKnowAuth
from ._span_builder import SpanBuilder as SpanBuilder  # re-export

# ---------------------------------------------------------------------------
# Shared HTTP transport singleton
# ---------------------------------------------------------------------------
# Một httpx.AsyncClient duy nhất cho tất cả for_request() instances.
# Tái dụng TCP connection pool → không establish kết nối mới mỗi request.
#
# threading.Lock (không asyncio.Lock) vì:
#   - Được tạo ở module level, trước khi event loop start
#   - Double-checked locking chỉ chạy trong _get_shared_http_client() (sync init)
#   - Lock giữ < 1ms (chỉ trong block tạo client), không ảnh hưởng event loop
_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None
_SHARED_HTTP_LOCK = _threading.Lock()


def _get_shared_http_client() -> httpx.AsyncClient:
    """
    Return (hoặc tạo) shared httpx.AsyncClient với connection pool.

    base_url đọc từ AIKNOW_BASE_URL env var — không nhận argument để tránh
    "bỏ qua base_url sau lần đầu" bug (singleton chỉ tạo một lần).

    BFF chỉ có một upstream URL → singleton pattern là safe và đúng.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        with _SHARED_HTTP_LOCK:  # Double-checked locking
            if _SHARED_HTTP_CLIENT is None:
                base = (
                    os.environ.get("AIKNOW_BASE_URL") or _DEFAULT_BASE_URL
                ).rstrip("/")
                _SHARED_HTTP_CLIENT = httpx.AsyncClient(
                    base_url=base,
                    timeout=httpx.Timeout(60.0),
                    transport=httpx.AsyncHTTPTransport(
                        retries=3,
                        limits=httpx.Limits(
                            max_connections=100,
                            max_keepalive_connections=20,
                            keepalive_expiry=30.0,
                        ),
                    ),
                )
    return _SHARED_HTTP_CLIENT


async def close_shared_http_client() -> None:
    """
    Close shared httpx transport. Gọi trong app lifespan shutdown.

    Sau khi gọi, for_request() sẽ tạo lại client mới khi được gọi tiếp.
    Thường chỉ gọi một lần khi app tắt.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is not None:
        await _SHARED_HTTP_CLIENT.aclose()
        _SHARED_HTTP_CLIENT = None

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
        self.users = AsyncUsersResource(self._client)
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
        """
        Close owned HTTP resources.

        for_request() instances KHÔNG own resources (dùng shared transport) →
        close() là no-op, detect bằng explicit `_is_per_request` flag.

        Shared transport được close bởi close_shared_http_client() trong app lifespan.
        """
        if getattr(self, "_is_per_request", False):
            return  # Shared resources — không owned, không close
        await self._client.aclose()
        await self._raw_client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    # ---------------------------------------------------------------------------
    # BFF / Proxy factory
    # ---------------------------------------------------------------------------

    @classmethod
    def for_request(
        cls,
        bearer_token: str,
        tenant_id: str,
    ) -> "AsyncAIKnowClient":
        """
        Tạo lightweight client với per-request auth headers.

        SYNC classmethod — KHÔNG dùng ``await``:

            # ✅ Đúng:
            client = AsyncAIKnowClient.for_request(token, tenant_id)

            # ❌ Sai — TypeError at runtime:
            client = await AsyncAIKnowClient.for_request(token, tenant_id)

        KHÔNG tạo httpx.AsyncClient mới — dùng shared singleton với
        ``_PerRequestClient`` wrapper (duck typing). Không cần close() hay
        context manager.

        base_url được đọc từ ``AIKNOW_BASE_URL`` env var (same as shared client).

        Args:
            bearer_token: JWT access token từ httpOnly cookie.
            tenant_id:    Tenant UUID từ token introspection.

        Returns:
            AsyncAIKnowClient instance với resources được bind vào per-request client.
        """
        from ._per_request_client import _PerRequestClient

        shared = _get_shared_http_client()
        per_req = _PerRequestClient(shared, bearer_token, tenant_id)

        instance = cls.__new__(cls)
        instance._is_per_request = True   # Explicit flag — không suy luận từ _auth_flow

        instance.chat = AsyncChatResource(per_req)              # type: ignore[arg-type]
        instance.ingestion = AsyncIngestionResource(per_req)    # type: ignore[arg-type]
        instance.conversation = AsyncConversationResource(per_req)  # type: ignore[arg-type]
        instance.users = AsyncUsersResource(per_req)            # type: ignore[arg-type]
        instance.extensions = AsyncExtensionsResource(per_req)  # type: ignore[arg-type]
        instance.workflows = AsyncWorkflowsResource(per_req)    # type: ignore[arg-type]
        instance.knowledge = AsyncKnowledgeResource(            # type: ignore[arg-type]
            per_req, tenant_id=tenant_id
        )
        instance.observe = None    # Admin key required — dùng get_admin_client() riêng
        instance.auth = None
        instance._auth_flow = None
        instance._client = per_req      # type: ignore[assignment]
        instance._raw_client = per_req  # type: ignore[assignment]

        return instance
