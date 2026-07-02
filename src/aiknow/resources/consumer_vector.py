"""
AsyncConsumerVectorResource — SDK resource for consumer vector store operations.

Maps to the consumer/vector/* endpoints backed by Qdrant.

Methods:
    create_collection(name, vector_size, distance)  → POST   consumer/vector/collections
    delete_collection(name)                         → DELETE consumer/vector/collections/{name}
    upsert(collection, points)                      → POST   consumer/vector/upsert
    search(collection, query, top_k, threshold)     → POST   consumer/vector/search

Design:
    - Collections are auto-namespaced consumer_{tenant_id}_{name} on the server.
    - upsert points without vectors trigger auto-embedding on the server.
    - search query can be str (auto-embedded) or list[float] (used directly).
    - No aiknow_core or aiknow_adapters imports.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AsyncConsumerVectorResource:
    """SDK resource for consumer-facing Qdrant vector operations.

    Collections are namespaced per-tenant on the server side.
    Auto-embedding is handled by the platform if points have no pre-computed vectors.

    Usage::

        async with AsyncAIKnowClient(...) as client:
            await client.consumer_vector.create_collection("sop-docs", vector_size=1536)
            await client.consumer_vector.upsert(
                "sop-docs",
                [{"id": "1", "text": "How to onboard a new employee"}],
            )
            hits = await client.consumer_vector.search("sop-docs", "employee onboarding")
            for hit in hits:
                print(hit["score"], hit["payload"])
    """

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Create a consumer vector collection (idempotent).

        Args:
            name: Collection name. Auto-namespaced as ``consumer_{tenant}_{name}`` on the server.
            vector_size: Dimension of the embedding vectors.
            distance: Distance metric — ``"Cosine"`` (default), ``"Euclid"``, or ``"Dot"``.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        resp = await self._http.post(
            "consumer/vector/collections",
            json={"name": name, "vector_size": vector_size, "distance": distance.capitalize()},
        )
        resp.raise_for_status()

    async def delete_collection(self, name: str) -> None:
        """Delete a consumer vector collection.

        Args:
            name: Collection name (WITHOUT the ``consumer_{tenant}_`` prefix —
                  the server applies the namespace automatically).

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        resp = await self._http.delete(
            f"consumer/vector/collections/{name}",
        )
        resp.raise_for_status()

    async def upsert(
        self,
        collection: str,
        points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Upsert points into a consumer collection.

        Points without a ``vector`` field are auto-embedded by the server
        in a single batch call (not per-point).

        Args:
            collection: Target collection name (without namespace prefix).
            points: List of point dicts. Each point may include:
                    - ``id``: str | int (required)
                    - ``text``: str — used for auto-embedding if ``vector`` is absent
                    - ``vector``: list[float] — pre-computed vector (optional)
                    - ``payload``: dict — arbitrary metadata (optional)

        Returns:
            Dict with ``upserted`` count.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        resp = await self._http.post(
            "consumer/vector/upsert",
            json={"collection": collection, "points": points},
        )
        resp.raise_for_status()
        return resp.json()

    async def search(
        self,
        collection: str,
        query: str | list[float],
        *,
        top_k: int = 10,
        threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search a consumer collection for semantically similar content.

        Args:
            collection: Collection name (without namespace prefix).
            query: Search query — either a plain string (auto-embedded by server)
                   or a pre-computed float vector.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score filter (0.0–1.0). None = no filter.
            filters: Key-value exact-match filters applied to the Qdrant payload.

        Returns:
            List of hit dicts with ``id``, ``score``, and ``payload``.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        body: dict[str, Any] = {
            "collection": collection,
            "query": query,
            "top_k": top_k,
        }
        if threshold is not None:
            body["threshold"] = threshold
        if filters is not None:
            body["filters"] = filters

        resp = await self._http.post(
            "consumer/vector/search",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()
