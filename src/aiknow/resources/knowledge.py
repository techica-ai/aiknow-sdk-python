"""
AsyncKnowledgeResource — SDK resource for searching the tenant's Knowledge Base.

The AIKnow Platform stores documents in a vector store and provides a RAG
retrieval pipeline. This resource gives agents a simple ``search()`` interface
to retrieve relevant document chunks without going through the full chat
pipeline.

Typical use in AI Copilot:
    chunks = await sdk_client.knowledge.search(
        query=agent_question,
        top_k=3,
        tenant_id="demo-call-center",
    )
    context = "\\n\\n".join(f"[{c.source}]\\n{c.content}" for c in chunks)

Architecture
------------
Two endpoints are used depending on availability:

1. ``POST /retrieval/search`` (preferred) — dedicated chunk-search endpoint
   that returns raw document chunks with scores. Available when the Platform
   retrieval router is implemented.

2. ``POST /chat/ask`` (fallback) — the full RAG chat endpoint. We send the
   query and parse citations from the ``DoneEvent``. Lower quality for pure
   retrieval (LLM overhead) but always available.

This resource transparently falls back to strategy 2 when strategy 1 returns
HTTP 501 (Not Implemented).

Thread Safety
-------------
``AsyncKnowledgeResource`` is safe to share across asyncio tasks — it holds
no mutable state, only a reference to the shared ``httpx.AsyncClient``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A single retrieved document chunk from the Knowledge Base.

    Attributes:
        content:  The relevant text extracted from the document.
        source:   Human-readable source identifier (document name, URL, etc.).
        score:    Similarity score from the vector store (0.0–1.0).
                  Higher is more relevant. May be 0.0 when using fallback mode.
        metadata: Additional metadata from the source document
                  (e.g. page number, section, tags).
    """

    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if len(self.content) > 120:
            return f"[{self.source}] {self.content[:120]}…"
        return f"[{self.source}] {self.content}"


class AsyncKnowledgeResource:
    """Async resource for searching the tenant's Knowledge Base.

    Access via ``client.knowledge.search()``.

    Args:
        client:    Shared ``httpx.AsyncClient`` (lifecycle owned by AsyncAIKnowClient).
        tenant_id: Default tenant. Can be overridden per-call.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        tenant_id: str | None = None,
    ) -> None:
        import warnings
        warnings.warn(
            "client.knowledge is deprecated and will be removed in v5.0.0. "
            "Use Pipeline API (POST /pipelines) with retrieval stage instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._client = client
        self._tenant_id = tenant_id
        # Track whether the dedicated /retrieval/search endpoint is available
        self._has_retrieval_endpoint: bool | None = None  # None = not yet probed

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tenant_id: str | None = None,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[KnowledgeChunk]:
        """Search the Knowledge Base for documents relevant to *query*.

        Tries the dedicated retrieval endpoint first; falls back to the RAG
        chat pipeline on HTTP 501 / 404.

        Args:
            query:     Natural language query to search for.
            top_k:     Maximum number of chunks to return (default: 5).
            tenant_id: Tenant to search. Defaults to the client-level tenant.
            filters:   Optional metadata filters (implementation-dependent).
                       Example: ``{"document_type": "policy"}``.
            min_score: Minimum similarity score threshold (0.0–1.0).
                       Chunks below this score are excluded.

        Returns:
            List of :class:`KnowledgeChunk`, ordered by relevance (best first).
            Returns an empty list if no results or on any retrieval error.
        """
        effective_tenant = tenant_id or self._tenant_id
        if not query.strip():
            return []

        try:
            # Strategy 1: dedicated retrieval endpoint
            if self._has_retrieval_endpoint is not False:
                chunks = await self._search_via_retrieval(
                    query=query,
                    top_k=top_k,
                    tenant_id=effective_tenant,
                    filters=filters,
                    min_score=min_score,
                )
                if chunks is not None:
                    return chunks

            # Strategy 2: fallback via RAG chat endpoint
            return await self._search_via_chat(
                query=query,
                top_k=top_k,
                tenant_id=effective_tenant,
                min_score=min_score,
            )
        except Exception as exc:
            # KB search is non-critical — always return empty on any error
            logger.warning("Knowledge.search: unexpected error (query=%r): %s", query[:50], exc)
            return []

    # ------------------------------------------------------------------
    # Strategy 1 — dedicated retrieval endpoint
    # ------------------------------------------------------------------

    async def _search_via_retrieval(
        self,
        query: str,
        top_k: int,
        tenant_id: str | None,
        filters: dict[str, Any] | None,
        min_score: float,
    ) -> list[KnowledgeChunk] | None:
        """Call POST /retrieval/search. Returns None if endpoint unavailable."""
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if filters:
            payload["filters"] = filters

        try:
            response = await self._client.post("/retrieval/search", json=payload, timeout=5.0)
        except Exception as exc:
            logger.debug("Knowledge[retrieval]: request failed (%s) — switching to fallback", exc)
            self._has_retrieval_endpoint = False
            return None

        if response.status_code in (404, 501):
            # Endpoint not implemented — mark and skip in future calls
            if self._has_retrieval_endpoint is None:
                logger.debug(
                    "Knowledge: /retrieval/search not available (HTTP %d) — "
                    "switching to chat fallback permanently.",
                    response.status_code,
                )
                self._has_retrieval_endpoint = False
            return None

        if not response.is_success:
            logger.warning(
                "Knowledge._search_via_retrieval: HTTP %d — %s",
                response.status_code, response.text[:200],
            )
            return None

        self._has_retrieval_endpoint = True
        try:
            data = response.json()
            chunks = data.get("chunks", data) if isinstance(data, dict) else data
            result = []
            for item in chunks:
                score = float(item.get("score", 0.0))
                if score < min_score:
                    continue
                result.append(KnowledgeChunk(
                    content=item.get("content", item.get("text", "")),
                    source=item.get("source", item.get("document_name", "Unknown")),
                    score=score,
                    metadata=item.get("metadata", {}),
                ))
            logger.debug(
                "Knowledge[retrieval]: query=%r top_k=%d → %d chunks",
                query[:50], top_k, len(result),
            )
            return result
        except Exception as parse_exc:
            logger.warning("Knowledge: failed to parse retrieval response: %s", parse_exc)
            return None

    # ------------------------------------------------------------------
    # Strategy 2 — fallback via /chat/ask
    # ------------------------------------------------------------------

    async def _search_via_chat(
        self,
        query: str,
        top_k: int,
        tenant_id: str | None,
        min_score: float,
    ) -> list[KnowledgeChunk]:
        """Fallback: POST /chat/ask and extract citations from DoneEvent."""
        payload: dict[str, Any] = {"query": query}
        if tenant_id:
            payload["tenant_id"] = tenant_id

        try:
            response = await self._client.post("/chat/ask", json=payload, timeout=8.0)
        except Exception as exc:
            logger.warning("Knowledge[chat-fallback]: request failed: %s", exc)
            return []

        if not response.is_success:
            logger.warning(
                "Knowledge._search_via_chat: HTTP %d — %s",
                response.status_code, response.text[:200],
            )
            return []

        try:
            data = response.json()
            # Platform /chat/ask returns {answer, citations: [{source, content, score}]}
            citations: list[dict[str, Any]] = data.get("citations", [])
            result = []
            for cit in citations[:top_k]:
                score = float(cit.get("score", 0.0))
                if score < min_score:
                    continue
                result.append(KnowledgeChunk(
                    content=cit.get("content", cit.get("text", data.get("answer", ""))),
                    source=cit.get("source", cit.get("document_name", "KB")),
                    score=score,
                    metadata=cit.get("metadata", {}),
                ))
            # If no citations, use the full answer as a single chunk
            if not result and data.get("answer"):
                result.append(KnowledgeChunk(
                    content=data["answer"],
                    source="AIKnow KB",
                    score=0.5,
                ))
            logger.debug(
                "Knowledge[chat-fallback]: query=%r top_k=%d → %d chunks",
                query[:50], top_k, len(result),
            )
            return result
        except Exception as parse_exc:
            logger.warning("Knowledge: failed to parse chat response: %s", parse_exc)
            return []
