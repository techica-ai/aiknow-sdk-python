"""
Sync and Async Observability resources.

Provides access to traces, spans, LLM calls, analytics, jobs, sources,
vector DB (Qdrant) and graph DB (Memgraph) endpoints.
All requests require ``admin_key`` (X-Admin-Key header).
"""
from __future__ import annotations

from typing import Any, cast

import httpx
from aiknow_contracts.graph import (
    GraphNeighborsResponse,
    GraphNodeListResponse,
    GraphQueryResponse,
    GraphStatsResponse,
)
from aiknow_contracts.jobs import JobListResponse, JobStatusResponse
from aiknow_contracts.observe import (
    CleanupResponse,
    LatencyStatsResponse,
    LlmCallItem,
    LlmCallListResponse,
    PolicyStatsResponse,
    SlowSpansResponse,
    ToolUsageResponse,
    TraceDetailResponse,
    TraceListResponse,
)
from aiknow_contracts.sources import SourceDetailResponse, SourceListResponse
from aiknow_contracts.vector import (
    VectorChunk,
    VectorCollectionDetail,
    VectorCollectionListResponse,
    VectorSearchResponse,
)

from .._http import raise_for_status, wrap_httpx_errors

_BASE = "/observe"


def _build_params(base: dict[str, Any], optional: list[tuple[str, Any]]) -> dict[str, Any]:
    """Merge required params with non-None optional params."""
    result = dict(base)
    for key, value in optional:
        if value is not None:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Synchronous resource
# ---------------------------------------------------------------------------

class ObserveResource:
    """Synchronous observability resource (requires admin_key)."""

    def __init__(self, client: httpx.Client, admin_key: str) -> None:
        self._client = client
        self._headers = {"X-Admin-Key": admin_key}

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.get(path, params=params, headers=self._headers)
        except Exception as exc:
            wrap_httpx_errors(f"GET {path}", exc)

    def _post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(path, json=json, headers=self._headers)
        except Exception as exc:
            wrap_httpx_errors(f"POST {path}", exc)

    def _delete(self, path: str, params: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.delete(path, params=params, headers=self._headers)
        except Exception as exc:
            wrap_httpx_errors(f"DELETE {path}", exc)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def list_traces(
        self,
        tenant_id: str,
        trace_type: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TraceListResponse:
        """List traces with optional filters."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            [("trace_type", trace_type), ("status", status),
             ("session_id", session_id), ("from_ts", from_ts), ("to_ts", to_ts)],
        )
        res = self._get(f"{_BASE}/traces", params)
        raise_for_status("Observe.list_traces", res)
        return TraceListResponse.model_validate(res.json())

    def get_trace(self, trace_id: str, tenant_id: str) -> TraceDetailResponse:
        """Get full trace detail including spans, LLM calls and waterfall."""
        res = self._get(f"{_BASE}/traces/{trace_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_trace", res)
        return TraceDetailResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------

    def get_slow_spans(
        self,
        tenant_id: str,
        name: str | None = None,
        min_duration_ms: float = 3000,
        limit: int = 50,
    ) -> SlowSpansResponse:
        """List spans exceeding the given duration threshold."""
        params = _build_params(
            {"tenant_id": tenant_id, "min_duration_ms": min_duration_ms, "limit": limit},
            [("name", name)],
        )
        res = self._get(f"{_BASE}/spans/slow", params)
        raise_for_status("Observe.get_slow_spans", res)
        return SlowSpansResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # LLM Calls
    # ------------------------------------------------------------------

    def list_llm_calls(
        self,
        tenant_id: str,
        trace_id: str | None = None,
        model: str | None = None,
        call_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> LlmCallListResponse:
        """List LLM calls with optional filters."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            [("trace_id", trace_id), ("model", model), ("call_type", call_type)],
        )
        res = self._get(f"{_BASE}/llm-calls", params)
        raise_for_status("Observe.list_llm_calls", res)
        return LlmCallListResponse.model_validate(res.json())

    def get_llm_call(self, call_id: str, tenant_id: str) -> LlmCallItem:
        """Get full detail for a single LLM call."""
        res = self._get(f"{_BASE}/llm-calls/{call_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_llm_call", res)
        return LlmCallItem.model_validate(res.json())

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_latency_stats(
        self,
        tenant_id: str,
        trace_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> LatencyStatsResponse:
        """Retrieve trace latency percentiles (p50/p90/p95/p99)."""
        params = _build_params(
            {"tenant_id": tenant_id},
            [("trace_type", trace_type), ("from_ts", from_ts), ("to_ts", to_ts)],
        )
        res = self._get(f"{_BASE}/analytics/latency", params)
        raise_for_status("Observe.get_latency_stats", res)
        return LatencyStatsResponse.model_validate(res.json())

    def get_tool_usage(
        self,
        tenant_id: str,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> ToolUsageResponse:
        """Retrieve tool invocation stats aggregated by tool name."""
        params = _build_params({"tenant_id": tenant_id}, [("from_ts", from_ts), ("to_ts", to_ts)])
        res = self._get(f"{_BASE}/analytics/tools", params)
        raise_for_status("Observe.get_tool_usage", res)
        return ToolUsageResponse.model_validate(res.json())

    def get_policy_stats(
        self,
        tenant_id: str,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> PolicyStatsResponse:
        """Retrieve policy allow/deny statistics."""
        params = _build_params({"tenant_id": tenant_id}, [("from_ts", from_ts), ("to_ts", to_ts)])
        res = self._get(f"{_BASE}/analytics/policy", params)
        raise_for_status("Observe.get_policy_stats", res)
        return PolicyStatsResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, retention_days: int | None = None) -> CleanupResponse:
        """Delete observability data older than *retention_days*."""
        params: dict[str, Any] = {}
        if retention_days is not None:
            params["retention_days"] = retention_days
        res = self._delete(f"{_BASE}/traces/cleanup", params)
        raise_for_status("Observe.cleanup", res)
        return CleanupResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def list_jobs(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> JobListResponse:
        """List ingestion jobs visible to admins, optionally filtered by status."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            [("status", status)],
        )
        res = self._get(f"{_BASE}/jobs", params)
        raise_for_status("Observe.list_jobs", res)
        return JobListResponse.model_validate(res.json())

    def get_job(self, job_id: str, tenant_id: str) -> JobStatusResponse:
        """Get full detail for a single job by ID."""
        res = self._get(f"{_BASE}/jobs/{job_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_job", res)
        return JobStatusResponse.model_validate(res.json())


    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def list_sources(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> SourceListResponse:
        """List uploaded source documents for a tenant."""
        res = self._get(
            f"{_BASE}/sources",
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
        )
        raise_for_status("Observe.list_sources", res)
        return SourceListResponse.model_validate(res.json())

    def get_source(self, source_id: str, tenant_id: str) -> SourceDetailResponse:
        """Get full detail for a single source document including linked jobs."""
        res = self._get(f"{_BASE}/sources/{source_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_source", res)
        return SourceDetailResponse.model_validate(res.json())

    def delete_source(self, source_id: str, tenant_id: str) -> dict[str, Any]:
        """Delete a single source document, associated vectors and files."""
        res = self._delete(f"{_BASE}/sources/{source_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.delete_source", res)
        return cast(dict[str, Any], res.json())


    def reset_tenant_data(self, tenant_id: str) -> dict[str, Any]:
        """Reset all demo data for a tenant (sources, vectors, files, graph nodes, sessions)."""
        try:
            res = self._client.post(
                f"{_BASE}/reset",
                params={"tenant_id": tenant_id},
                headers=self._headers,
            )
        except Exception as exc:
            wrap_httpx_errors(f"POST {_BASE}/reset", exc)
        raise_for_status("Observe.reset_tenant_data", res)
        return cast(dict[str, Any], res.json())

    # ------------------------------------------------------------------
    # Vector DB (Qdrant)
    # ------------------------------------------------------------------

    def list_vector_collections(
        self,
        tenant_id: str | None = None,
    ) -> VectorCollectionListResponse:
        """List Qdrant collections, optionally filtered by tenant prefix."""
        params: dict[str, Any] = {}
        if tenant_id is not None:
            params["tenant_id"] = tenant_id
        res = self._get(f"{_BASE}/vector/collections", params)
        raise_for_status("Observe.list_vector_collections", res)
        return VectorCollectionListResponse.model_validate(res.json())

    def get_vector_collection(self, collection_name: str) -> VectorCollectionDetail:
        """Get Qdrant collection stats and configuration."""
        res = self._get(f"{_BASE}/vector/collections/{collection_name}", {})
        raise_for_status("Observe.get_vector_collection", res)
        return VectorCollectionDetail.model_validate(res.json())

    def search_vector(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
    ) -> VectorSearchResponse:
        """Execute a semantic vector search against a tenant's Qdrant collection."""
        res = self._post(
            f"{_BASE}/vector/search",
            {"query": query, "tenant_id": tenant_id, "top_k": top_k},
        )
        raise_for_status("Observe.search_vector", res)
        return VectorSearchResponse.model_validate(res.json())

    def get_vector_chunk(self, chunk_id: str, collection_name: str) -> VectorChunk:
        """Retrieve the full payload for a single Qdrant point (chunk)."""
        res = self._get(
            f"{_BASE}/vector/chunks/{chunk_id}",
            {"collection_name": collection_name},
        )
        raise_for_status("Observe.get_vector_chunk", res)
        return VectorChunk.model_validate(res.json())

    # ------------------------------------------------------------------
    # Graph DB (Memgraph)
    # ------------------------------------------------------------------

    def get_graph_stats(self, tenant_id: str) -> GraphStatsResponse:
        """Get node and edge counts for a tenant's graph partition."""
        res = self._get(f"{_BASE}/graph/stats", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_graph_stats", res)
        return GraphStatsResponse.model_validate(res.json())

    def list_graph_nodes(
        self,
        tenant_id: str,
        label: str | None = None,
        search: str | None = None,
        limit: int = 50,
    ) -> GraphNodeListResponse:
        """List graph nodes for a tenant, optionally filtered by Cypher label or search text."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit},
            [("label", label), ("search", search)],
        )
        res = self._get(f"{_BASE}/graph/nodes", params)
        raise_for_status("Observe.list_graph_nodes", res)
        return GraphNodeListResponse.model_validate(res.json())

    def get_graph_neighbors(
        self,
        node_id: str,
        tenant_id: str,
        depth: int = 1,
    ) -> GraphNeighborsResponse:
        """Get neighbors of a graph node up to *depth* hops away."""
        res = self._get(
            f"{_BASE}/graph/neighbors/{node_id}",
            {"tenant_id": tenant_id, "depth": depth},
        )
        raise_for_status("Observe.get_graph_neighbors", res)
        return GraphNeighborsResponse.model_validate(res.json())

    def execute_graph_query(
        self,
        query: str,
        tenant_id: str,
        limit: int = 100,
    ) -> GraphQueryResponse:
        """Execute a read-only Cypher query against Memgraph."""
        res = self._post(
            f"{_BASE}/graph/query",
            {"query": query, "tenant_id": tenant_id, "limit": limit},
        )
        raise_for_status("Observe.execute_graph_query", res)
        return GraphQueryResponse.model_validate(res.json())


# ---------------------------------------------------------------------------
# Asynchronous resource
# ---------------------------------------------------------------------------

class AsyncObserveResource:
    """Asynchronous observability resource (requires admin_key)."""

    def __init__(self, client: httpx.AsyncClient, admin_key: str) -> None:
        self._client = client
        self._headers = {"X-Admin-Key": admin_key}

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.get(path, params=params, headers=self._headers)
        except Exception as exc:
            wrap_httpx_errors(f"GET {path}", exc)

    async def _post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post(path, json=json, headers=self._headers)
        except Exception as exc:
            wrap_httpx_errors(f"POST {path}", exc)

    async def _delete(self, path: str, params: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.delete(path, params=params, headers=self._headers)
        except Exception as exc:
            wrap_httpx_errors(f"DELETE {path}", exc)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    async def list_traces(
        self,
        tenant_id: str,
        trace_type: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TraceListResponse:
        """List traces with optional filters."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            [("trace_type", trace_type), ("status", status),
             ("session_id", session_id), ("from_ts", from_ts), ("to_ts", to_ts)],
        )
        res = await self._get(f"{_BASE}/traces", params)
        raise_for_status("Observe.list_traces", res)
        return TraceListResponse.model_validate(res.json())

    async def get_trace(self, trace_id: str, tenant_id: str) -> TraceDetailResponse:
        """Get full trace detail including spans, LLM calls and waterfall."""
        res = await self._get(f"{_BASE}/traces/{trace_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_trace", res)
        return TraceDetailResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------

    async def get_slow_spans(
        self,
        tenant_id: str,
        name: str | None = None,
        min_duration_ms: float = 3000,
        limit: int = 50,
    ) -> SlowSpansResponse:
        """List spans exceeding the given duration threshold."""
        params = _build_params(
            {"tenant_id": tenant_id, "min_duration_ms": min_duration_ms, "limit": limit},
            [("name", name)],
        )
        res = await self._get(f"{_BASE}/spans/slow", params)
        raise_for_status("Observe.get_slow_spans", res)
        return SlowSpansResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # LLM Calls
    # ------------------------------------------------------------------

    async def list_llm_calls(
        self,
        tenant_id: str,
        trace_id: str | None = None,
        model: str | None = None,
        call_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> LlmCallListResponse:
        """List LLM calls with optional filters."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            [("trace_id", trace_id), ("model", model), ("call_type", call_type)],
        )
        res = await self._get(f"{_BASE}/llm-calls", params)
        raise_for_status("Observe.list_llm_calls", res)
        return LlmCallListResponse.model_validate(res.json())

    async def get_llm_call(self, call_id: str, tenant_id: str) -> LlmCallItem:
        """Get full detail for a single LLM call."""
        res = await self._get(f"{_BASE}/llm-calls/{call_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_llm_call", res)
        return LlmCallItem.model_validate(res.json())

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    async def get_latency_stats(
        self,
        tenant_id: str,
        trace_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> LatencyStatsResponse:
        """Retrieve trace latency percentiles (p50/p90/p95/p99)."""
        params = _build_params(
            {"tenant_id": tenant_id},
            [("trace_type", trace_type), ("from_ts", from_ts), ("to_ts", to_ts)],
        )
        res = await self._get(f"{_BASE}/analytics/latency", params)
        raise_for_status("Observe.get_latency_stats", res)
        return LatencyStatsResponse.model_validate(res.json())

    async def get_tool_usage(
        self,
        tenant_id: str,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> ToolUsageResponse:
        """Retrieve tool invocation stats aggregated by tool name."""
        params = _build_params({"tenant_id": tenant_id}, [("from_ts", from_ts), ("to_ts", to_ts)])
        res = await self._get(f"{_BASE}/analytics/tools", params)
        raise_for_status("Observe.get_tool_usage", res)
        return ToolUsageResponse.model_validate(res.json())

    async def get_policy_stats(
        self,
        tenant_id: str,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> PolicyStatsResponse:
        """Retrieve policy allow/deny statistics."""
        params = _build_params({"tenant_id": tenant_id}, [("from_ts", from_ts), ("to_ts", to_ts)])
        res = await self._get(f"{_BASE}/analytics/policy", params)
        raise_for_status("Observe.get_policy_stats", res)
        return PolicyStatsResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, retention_days: int | None = None) -> CleanupResponse:
        """Delete observability data older than *retention_days*."""
        params: dict[str, Any] = {}
        if retention_days is not None:
            params["retention_days"] = retention_days
        res = await self._delete(f"{_BASE}/traces/cleanup", params)
        raise_for_status("Observe.cleanup", res)
        return CleanupResponse.model_validate(res.json())

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def list_jobs(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> JobListResponse:
        """List ingestion jobs visible to admins, optionally filtered by status."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            [("status", status)],
        )
        res = await self._get(f"{_BASE}/jobs", params)
        raise_for_status("Observe.list_jobs", res)
        return JobListResponse.model_validate(res.json())

    async def get_job(self, job_id: str, tenant_id: str) -> JobStatusResponse:
        """Get full detail for a single job by ID."""
        res = await self._get(f"{_BASE}/jobs/{job_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_job", res)
        return JobStatusResponse.model_validate(res.json())


    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    async def list_sources(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> SourceListResponse:
        """List uploaded source documents for a tenant."""
        res = await self._get(
            f"{_BASE}/sources",
            {"tenant_id": tenant_id, "limit": limit, "offset": offset},
        )
        raise_for_status("Observe.list_sources", res)
        return SourceListResponse.model_validate(res.json())

    async def get_source(self, source_id: str, tenant_id: str) -> SourceDetailResponse:
        """Get full detail for a single source document including linked jobs."""
        res = await self._get(f"{_BASE}/sources/{source_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_source", res)
        return SourceDetailResponse.model_validate(res.json())

    async def delete_source(self, source_id: str, tenant_id: str) -> dict[str, Any]:
        """Delete a single source document, associated vectors and files."""
        res = await self._delete(f"{_BASE}/sources/{source_id}", {"tenant_id": tenant_id})
        raise_for_status("Observe.delete_source", res)
        return cast(dict[str, Any], res.json())


    async def reset_tenant_data(self, tenant_id: str) -> dict[str, Any]:
        """Reset all demo data for a tenant (sources, vectors, files, graph nodes, sessions)."""
        try:
            res = await self._client.post(
                f"{_BASE}/reset",
                params={"tenant_id": tenant_id},
                headers=self._headers,
            )
        except Exception as exc:
            wrap_httpx_errors(f"POST {_BASE}/reset", exc)
        raise_for_status("Observe.reset_tenant_data", res)
        return cast(dict[str, Any], res.json())

    # ------------------------------------------------------------------
    # Vector DB (Qdrant)
    # ------------------------------------------------------------------

    async def list_vector_collections(
        self,
        tenant_id: str | None = None,
    ) -> VectorCollectionListResponse:
        """List Qdrant collections, optionally filtered by tenant prefix."""
        params: dict[str, Any] = {}
        if tenant_id is not None:
            params["tenant_id"] = tenant_id
        res = await self._get(f"{_BASE}/vector/collections", params)
        raise_for_status("Observe.list_vector_collections", res)
        return VectorCollectionListResponse.model_validate(res.json())

    async def get_vector_collection(self, collection_name: str) -> VectorCollectionDetail:
        """Get Qdrant collection stats and configuration."""
        res = await self._get(f"{_BASE}/vector/collections/{collection_name}", {})
        raise_for_status("Observe.get_vector_collection", res)
        return VectorCollectionDetail.model_validate(res.json())

    async def search_vector(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
    ) -> VectorSearchResponse:
        """Execute a semantic vector search against a tenant's Qdrant collection."""
        res = await self._post(
            f"{_BASE}/vector/search",
            {"query": query, "tenant_id": tenant_id, "top_k": top_k},
        )
        raise_for_status("Observe.search_vector", res)
        return VectorSearchResponse.model_validate(res.json())

    async def get_vector_chunk(self, chunk_id: str, collection_name: str) -> VectorChunk:
        """Retrieve the full payload for a single Qdrant point (chunk)."""
        res = await self._get(
            f"{_BASE}/vector/chunks/{chunk_id}",
            {"collection_name": collection_name},
        )
        raise_for_status("Observe.get_vector_chunk", res)
        return VectorChunk.model_validate(res.json())

    # ------------------------------------------------------------------
    # Graph DB (Memgraph)
    # ------------------------------------------------------------------

    async def get_graph_stats(self, tenant_id: str) -> GraphStatsResponse:
        """Get node and edge counts for a tenant's graph partition."""
        res = await self._get(f"{_BASE}/graph/stats", {"tenant_id": tenant_id})
        raise_for_status("Observe.get_graph_stats", res)
        return GraphStatsResponse.model_validate(res.json())

    async def list_graph_nodes(
        self,
        tenant_id: str,
        label: str | None = None,
        search: str | None = None,
        limit: int = 50,
    ) -> GraphNodeListResponse:
        """List graph nodes for a tenant, optionally filtered by Cypher label or search text."""
        params = _build_params(
            {"tenant_id": tenant_id, "limit": limit},
            [("label", label), ("search", search)],
        )
        res = await self._get(f"{_BASE}/graph/nodes", params)
        raise_for_status("Observe.list_graph_nodes", res)
        return GraphNodeListResponse.model_validate(res.json())

    async def get_graph_neighbors(
        self,
        node_id: str,
        tenant_id: str,
        depth: int = 1,
    ) -> GraphNeighborsResponse:
        """Get neighbors of a graph node up to *depth* hops away."""
        res = await self._get(
            f"{_BASE}/graph/neighbors/{node_id}",
            {"tenant_id": tenant_id, "depth": depth},
        )
        raise_for_status("Observe.get_graph_neighbors", res)
        return GraphNeighborsResponse.model_validate(res.json())

    async def execute_graph_query(
        self,
        query: str,
        tenant_id: str,
        limit: int = 100,
    ) -> GraphQueryResponse:
        """Execute a read-only Cypher query against Memgraph."""
        res = await self._post(
            f"{_BASE}/graph/query",
            {"query": query, "tenant_id": tenant_id, "limit": limit},
        )
        raise_for_status("Observe.execute_graph_query", res)
        return GraphQueryResponse.model_validate(res.json())
