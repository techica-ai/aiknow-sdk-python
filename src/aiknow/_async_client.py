"""
AIKNOW SDK — asynchronous client.
"""
from __future__ import annotations

import os
import threading as _threading
from typing import Any, Self, cast

import httpx
from _per_request_client import _PerRequestClient
from aiknow_contracts.knowledge import Chunk, ParsedDocument

from ._auth_flow import AIKnowAuth
from ._http import _DEFAULT_BASE_URL
from ._span_builder import SpanBuilder as SpanBuilder  # re-export
from .resources.ai import AsyncAiResource
from .resources.auth import AsyncAuthResource
from .resources.chat import AsyncChatResource
from .resources.consumer_vector import AsyncConsumerVectorResource
from .resources.conversation import AsyncConversationResource
from .resources.extensions import AsyncExtensionsResource
from .resources.graphs import AsyncGraphsResource
from .resources.ingestion import AsyncIngestionResource
from .resources.knowledge import AsyncKnowledgeResource  # re-export
from .resources.knowledge import KnowledgeChunk as KnowledgeChunk
from .resources.kv_store import AsyncKVStoreResource
from .resources.observe import AsyncObserveResource
from .resources.prompt_configs import AsyncPromptConfigResource
from .resources.users import AsyncUsersResource
from .resources.workflows import AsyncWorkflowsResource

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
    Close shared httpx transport. Called during app lifespan shutdown.

    After calling, for_request() will recreate a new client when called again.
    Usually only called once when the app shuts down.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is not None:
        await _SHARED_HTTP_CLIENT.aclose()
        _SHARED_HTTP_CLIENT = None


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

        self._is_per_request = False
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
        self.graphs = AsyncGraphsResource(self._client)
        self.knowledge = AsyncKnowledgeResource(
            self._client,
            tenant_id=tenant_id,  # Propagate default tenant for KB searches
        )
        self.observe: AsyncObserveResource | None = (
            AsyncObserveResource(self._client, resolved_admin_key)
            if resolved_admin_key
            else None
        )
        # Phase 1 SPCC consumer resources
        self.ai = AsyncAiResource(self._client)
        self.kv = AsyncKVStoreResource(self._client)
        self.consumer_vector = AsyncConsumerVectorResource(self._client)
        self.prompt_configs = AsyncPromptConfigResource(self._client)

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
        trace: dict[str, Any],
        spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
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
            res = await self._client.post("/api/v1/observe/push", json=payload)
            res.raise_for_status()
            return cast(dict[str, Any], res.json())
        except Exception as exc:
            # Fire-and-forget: log but never raise so callers aren't disrupted
            import logging  # noqa: PLC0415
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

    async def parse(
        self,
        source_id: str,
        parser: str = "markitdown",
        config: dict[str, Any] | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Parse an existing registered source document (async)."""
        payload = {
            "source_id": source_id,
            "parser": parser,
            "config": config,
            "persist": persist,
        }
        res = await self._client.post("/pipeline/parse", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    async def chunk(
        self,
        state_token: str | None = None,
        document: dict[str, Any] | None = None,
        strategy: str = "recursive",
        chunk_size: int | None = None,
        overlap: int | None = None,
        dialog_config: dict[str, Any] | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Chunk a parsed document (async)."""
        payload = {
            "state_token": state_token,
            "document": document,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "dialog_config": dialog_config,
            "persist": persist,
        }
        res = await self._client.post("/pipeline/chunk", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    async def get_state(self, state_token: str) -> Any:
        """Fetch intermediate pipeline state from token and deserialize it (async)."""
        res = await self._client.get(f"/pipeline/state/{state_token}")
        res.raise_for_status()
        payload = res.json()
        data_type = payload["type"]
        data = payload["data"]

        # Deserialization logic
        if data_type == "parsed_document":
            return ParsedDocument.model_validate(data)
        elif data_type == "chunks":
            return [Chunk.model_validate(c) for c in data]
        return data

    async def list_states(self) -> list[dict[str, Any]]:
        """List all active and persistent states (async)."""
        res = await self._client.get("/pipeline/states")
        res.raise_for_status()
        return cast(list[dict[str, Any]], res.json())

    async def embed(self, texts: list[str], model: str = "default") -> list[list[float]]:
        """Embed a list of text strings (async)."""
        payload = {"texts": texts, "model": model}
        res = await self._client.post("/pipeline/embed", json=payload)
        res.raise_for_status()
        return cast(list[list[float]], res.json())

    async def embed_chunks(self, state_token: str, model: str = "default") -> dict[str, Any]:
        """Embed document chunks referenced by a state token (async)."""
        payload = {"state_token": state_token, "model": model}
        res = await self._client.post("/pipeline/embed-chunks", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    async def extract(
        self,
        text: str,
        extract_entities: bool = True,
        extract_relations: bool = True,
    ) -> dict[str, Any]:
        """Extract entities and relations from text (async)."""
        payload = {
            "text": text,
            "extract_entities": extract_entities,
            "extract_relations": extract_relations,
        }
        res = await self._client.post("/pipeline/extract", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    async def extract_chunks(
        self,
        state_token: str,
        extract_entities: bool = True,
        extract_relations: bool = True,
    ) -> dict[str, Any]:
        """Extract entities and relations from chunks referenced by state token (async)."""
        payload = {
            "state_token": state_token,
            "extract_entities": extract_entities,
            "extract_relations": extract_relations,
        }
        res = await self._client.post("/pipeline/extract-chunks", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    async def retrieve(
        self,
        query: str,
        strategy: str = "vector",
        top_k: int = 5,
        min_score: float | None = None,
        filters: dict[str, Any] | None = None,
        vector_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a search query (async)."""
        payload = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "min_score": min_score,
            "filters": filters,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
        }
        res = await self._client.post("/pipeline/retrieve", json=payload)
        res.raise_for_status()
        return cast(list[dict[str, Any]], res.json())

    # ---------------------------------------------------------------------------
    # BFF / Proxy factory
    # ---------------------------------------------------------------------------

    @classmethod
    def for_request(
        cls,
        bearer_token: str,
        tenant_id: str,
    ) -> AsyncAIKnowClient:
        """
        Create a lightweight client with per-request auth headers.

        SYNC classmethod — Do NOT use ``await``:

            # ✅ Correct:
            client = AsyncAIKnowClient.for_request(token, tenant_id)

            # ❌ Incorrect — TypeError at runtime:
            client = await AsyncAIKnowClient.for_request(token, tenant_id)

        Does NOT create a new httpx.AsyncClient — uses a shared singleton with
        ``_PerRequestClient`` wrapper (duck typing). No need to close() or use
        a context manager.

        base_url is read from the ``AIKNOW_BASE_URL`` env var (same as shared client).

        Args:
            bearer_token: JWT access token from httpOnly cookie.
            tenant_id:    Tenant UUID from token introspection.

        Returns:
            AsyncAIKnowClient instance with resources bound to the per-request client.
        """

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
        instance.graphs = AsyncGraphsResource(per_req)          # type: ignore[arg-type]
        instance.knowledge = AsyncKnowledgeResource(
            per_req, tenant_id=tenant_id  # type: ignore[arg-type]
        )
        instance.observe = None    # Admin key required — dùng get_admin_client() riêng
        instance.auth = None  # type: ignore[assignment]
        instance._auth_flow = None  # type: ignore[assignment]
        
        # Add new resources
        instance.ai = AsyncAiResource(per_req)                  # type: ignore[arg-type]
        instance.kv_store = AsyncKVStoreResource(per_req)       # type: ignore[arg-type]
        instance.consumer_vector = AsyncConsumerVectorResource(per_req) # type: ignore[arg-type]
        instance.prompt_configs = AsyncPromptConfigResource(per_req)    # type: ignore[arg-type]
        instance._client = per_req      # type: ignore[assignment]
        instance._raw_client = per_req  # type: ignore[assignment]

        return instance
