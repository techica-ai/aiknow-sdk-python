"""
AIKNOW SDK — synchronous client.
"""
from __future__ import annotations

import os
from typing import Self

import httpx

from ._exceptions import AIKnowConnectionError, AIKnowTimeoutError
from .resources.chat import ChatResource
from .resources.ingestion import IngestionResource
from .resources.observe import ObserveResource

_DEFAULT_BASE_URL = "http://localhost:8000/api/v1"


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

        headers: dict[str, str] = {}
        if resolved_api_key:
            headers["Authorization"] = f"Bearer {resolved_api_key}"

        self._client = httpx.Client(
            base_url=resolved_base,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=httpx.HTTPTransport(retries=3),
        )

        # Set default X-Tenant-Id header if provided
        if tenant_id:
            self._client.headers["X-Tenant-Id"] = tenant_id

        self.chat = ChatResource(self._client)
        self.ingestion = IngestionResource(self._client)
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
