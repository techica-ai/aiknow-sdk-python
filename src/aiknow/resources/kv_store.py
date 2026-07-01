"""
AsyncKVStoreResource — SDK resource for consumer session KV store.

Maps to the /api/v1/consumer/sessions/* endpoints backed by Redis.

Methods:
    save(key, data, ttl_seconds)  → PUT  /api/v1/consumer/sessions/{key}
    load(key)                     → GET  /api/v1/consumer/sessions/{key}
    delete(key)                   → DELETE /api/v1/consumer/sessions/{key}
    exists(key)                   → HEAD /api/v1/consumer/sessions/{key}

Design:
    - load() returns dict | None (None when key not found — never raises 404).
      The API returns {data: null, exists: false} on miss; this is unwrapped here.
    - exists() uses HEAD for lightweight polling (no JSON body).
    - No aiknow_core or aiknow_adapters imports.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_PATH = "/api/v1/consumer/sessions"


class AsyncKVStoreResource:
    """SDK resource for consumer session state (Redis-backed KV store).

    Usage::

        async with AsyncAIKnowClient(...) as client:
            await client.kv.save("session-001", {"step": 3, "confirmed": True})
            data = await client.kv.load("session-001")  # {"step": 3, ...}
            deleted = await client.kv.delete("session-001")
    """

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def save(
        self,
        key: str,
        data: dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> None:
        """Save a data blob under ``key`` with TTL.

        Args:
            key: Storage key (namespaced by tenant on the server side).
            data: JSON-serializable dict to store.
            ttl_seconds: Time-to-live in seconds (default: 1 hour).

        Raises:
            httpx.HTTPStatusError: On HTTP error (4xx/5xx).
        """
        resp = await self._http.put(
            f"{_BASE_PATH}/{key}",
            json={"data": data, "ttl_seconds": ttl_seconds},
        )
        resp.raise_for_status()

    async def load(self, key: str) -> dict[str, Any] | None:
        """Load the data blob stored under ``key``.

        Args:
            key: Storage key.

        Returns:
            The stored dict, or ``None`` if the key does not exist or has expired.
            Never raises an HTTP 404 error.

        Raises:
            httpx.HTTPStatusError: On HTTP errors other than missing keys.
        """
        resp = await self._http.get(f"{_BASE_PATH}/{key}")
        resp.raise_for_status()
        payload = resp.json()
        # API returns {data: null, exists: false} on miss — unwrap here
        if not payload.get("exists"):
            return None
        return payload.get("data")

    async def delete(self, key: str) -> bool:
        """Delete the data blob stored under ``key``.

        Args:
            key: Storage key.

        Returns:
            ``True`` if the key existed and was deleted, ``False`` otherwise.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        resp = await self._http.delete(f"{_BASE_PATH}/{key}")
        resp.raise_for_status()
        return resp.json().get("deleted", False)

    async def exists(self, key: str) -> bool:
        """Check whether ``key`` exists without fetching the data.

        Uses HTTP HEAD for minimal overhead (no response body).

        Args:
            key: Storage key.

        Returns:
            ``True`` if the key exists, ``False`` otherwise.

        Raises:
            httpx.HTTPStatusError: On HTTP errors other than 404.
        """
        resp = await self._http.head(f"{_BASE_PATH}/{key}")
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return resp.status_code == 204
