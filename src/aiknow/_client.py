"""
AIKNOW SDK — synchronous client.
"""
from __future__ import annotations

import os
from typing import Any, Self, cast

import httpx

from ._auth_flow import AIKnowAuth
from ._http import _DEFAULT_BASE_URL
from .resources.auth import AuthResource
from .resources.chat import ChatResource
from .resources.conversation import ConversationResource
from .resources.extensions import ExtensionsResource
from .resources.ingestion import IngestionResource
from .resources.knowledge import KnowledgeResource
from .resources.observe import ObserveResource
from .resources.users import UsersResource
from .resources.workflows import WorkflowsResource
from .resources.graphs import GraphsResource


class AIKnowClient:
    """Synchronous AIKNOW Platform client.

    Usage::

        with AIKnowClient(api_key="...") as client:
            response = client.chat.ask("What is AIKNOW?", tenant_id="acme")

    Args:
        base_url:   Base URL of the AIKNOW API.
                    Defaults to ``AIKNOW_BASE_URL`` env var, then localhost.
        api_key:    Bearer token for end-user API endpoints.
                    Reads ``AIKNOW_API_KEY`` env var if not provided.
        admin_key:  Admin key for observability endpoints.
                    Reads ``AIKNOW_ADMIN_KEY`` env var if not provided.
                    When present, ``.observe`` resource is available.
        tenant_id:  Default tenant identifier sent as ``X-Tenant-Id`` header
                    on every request. Can be overridden per-request via
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

        self.auth = AuthResource(self)
        if resolved_api_key:
            self.auth.access_token = resolved_api_key

        self._auth_flow = AIKnowAuth(self.auth, auto_refresh=True)

        self._raw_client = httpx.Client(
            base_url=resolved_base,
            timeout=httpx.Timeout(timeout),
            transport=httpx.HTTPTransport(retries=3),
        )

        self._client = httpx.Client(
            base_url=resolved_base,
            auth=self._auth_flow,
            timeout=httpx.Timeout(timeout),
            transport=httpx.HTTPTransport(retries=3),
        )

        # Set default X-Tenant-Id header if provided
        if tenant_id:
            self._client.headers["X-Tenant-Id"] = tenant_id
            self._raw_client.headers["X-Tenant-Id"] = tenant_id

        self.chat = ChatResource(self._client)
        self.conversation = ConversationResource(self._client)
        self.ingestion = IngestionResource(self._client)
        self.users = UsersResource(self._client)
        self.extensions = ExtensionsResource(self._client)
        self.workflows = WorkflowsResource(self._client)
        self.graphs = GraphsResource(self._client)
        self.knowledge = KnowledgeResource(self._client)
        self.observe: ObserveResource | None = (
            ObserveResource(self._client, resolved_admin_key)
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

    def ping(self) -> bool:
        """Check connectivity to the AIKNOW API.

        Returns ``True`` if the server responds with HTTP 200, ``False``
        on any network or HTTP error.
        """
        try:
            res = self._client.get("/health")
            return res.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()
        self._raw_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def parse(
        self,
        source_id: str,
        parser: str = "markitdown",
        config: dict[str, Any] | None = None,
        persist: bool = False,
     ) -> dict[str, Any]:
        """Parse an existing registered source document (sync)."""
        payload = {
            "source_id": source_id,
            "parser": parser,
            "config": config,
            "persist": persist,
        }
        res = self._client.post("/pipeline/parse", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    def chunk(
        self,
        state_token: str | None = None,
        document: dict[str, Any] | None = None,
        strategy: str = "recursive",
        chunk_size: int | None = None,
        overlap: int | None = None,
        dialog_config: dict[str, Any] | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Chunk a parsed document (sync)."""
        payload = {
            "state_token": state_token,
            "document": document,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "dialog_config": dialog_config,
            "persist": persist,
        }
        res = self._client.post("/pipeline/chunk", json=payload)
        res.raise_for_status()
        return cast(dict[str, Any], res.json())

    def get_state(self, state_token: str) -> Any:
        """Fetch intermediate pipeline state from token and deserialize it (sync)."""
        res = self._client.get(f"/pipeline/state/{state_token}")
        res.raise_for_status()
        payload = res.json()
        data_type = payload["type"]
        data = payload["data"]

        # Deserialization logic
        if data_type == "parsed_document":
            from aiknow_core.common.models.knowledge import ParsedDocument
            return ParsedDocument.model_validate(data)
        elif data_type == "chunks":
            from aiknow_core.common.models.knowledge import Chunk
            return [Chunk.model_validate(c) for c in data]
        return data

    def list_states(self) -> list[dict[str, Any]]:
        """List all active and persistent states (sync)."""
        res = self._client.get("/pipeline/states")
        res.raise_for_status()
        return cast(list[dict[str, Any]], res.json())
