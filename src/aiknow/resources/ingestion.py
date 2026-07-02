"""
Sync and Async Ingestion resources.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import uuid
import warnings
from typing import Any

import httpx
from aiknow_contracts.documents import DocumentResponse
from aiknow_contracts.jobs import JobStatusResponse

from .._http import raise_for_status, wrap_httpx_errors


def _filename(file_path: str) -> str:
    """Extract the OS-agnostic basename from a file path."""
    return os.path.basename(file_path)


def _read_file_bytes(file_path: str) -> bytes:
    """Read file bytes synchronously — run via asyncio.to_thread."""
    with open(file_path, "rb") as f:
        return f.read()


class _IngestionResourceBase:
    """Base class with shared ingestion logic."""

    def _prepare_upload(
        self,
        file_path: str,
        tenant_id: str,
        source_id: str | None = None,
        pipeline_config: dict | None = None,
    ) -> tuple[str, str, dict[str, str]]:
        """Prepare the upload payload and metadata."""
        sid = source_id or str(uuid.uuid4())
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        data = {"tenant_id": tenant_id, "source_id": sid}
        if pipeline_config is not None:
            data["pipeline_config_json"] = json.dumps(pipeline_config)
        return sid, mime_type, data

    def _parse_upload_response(self, response_json: Any) -> DocumentResponse:
        """Parse the upload response."""
        return DocumentResponse.model_validate(response_json)


class IngestionResource(_IngestionResourceBase):
    """Synchronous ingestion resource."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    @staticmethod
    def _warn_deprecated() -> None:
        """Emit deprecation warning on actual method use, not construction (SDK-8)."""
        warnings.warn(
            "client.ingestion is deprecated and will be removed in v5.0.0. "
            "Use Pipeline API (POST /pipelines) with IngestPipelineConfig instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def upload(
        self,
        file_path: str,
        tenant_id: str,
        source_id: str | None = None,
        pipeline_config: dict | None = None,
    ) -> DocumentResponse:
        """Upload a document for ingestion processing (sync).

        Args:
            file_path: Absolute or relative path to the file on disk.
            tenant_id: Tenant identifier for data isolation.
            source_id: Optional stable ID for the source document.
                       A UUID is generated automatically if not provided.

        Returns:
            DocumentResponse with `source_id`, `status`, and `metadata`.

        Raises:
            AuthenticationError: on 401/403.
            AIKnowAPIError:      on other HTTP errors.
            AIKnowConnectionError: if the server is unreachable.
            AIKnowTimeoutError:  if the request times out.
            FileNotFoundError:   if *file_path* does not exist.
        """
        self._warn_deprecated()
        sid, mime_type, data = self._prepare_upload(
            file_path, tenant_id, source_id, pipeline_config
        )
        with open(file_path, "rb") as f:
            files = {"file": (_filename(file_path), f, mime_type)}
            try:
                res = self._client.post("/documents", data=data, files=files)
            except Exception as exc:
                wrap_httpx_errors("Ingestion.upload", exc)
        raise_for_status("Ingestion.upload", res)
        return self._parse_upload_response(res.json())

    def get_job(
        self, job_id: str, tenant_id: str,
    ) -> JobStatusResponse:
        """Get ingestion job status by ID (sync).

        Args:
            job_id: Job identifier.
            tenant_id: Tenant identifier for data isolation.

        Returns:
            JobStatusResponse with status, progress, and metadata.
        """
        params = {"tenant_id": tenant_id}
        try:
            res = self._client.get(f"/jobs/{job_id}", params=params)
        except Exception as exc:
            wrap_httpx_errors("Ingestion.get_job", exc)
        raise_for_status("Ingestion.get_job", res)
        return JobStatusResponse.model_validate(res.json())

    def list_jobs(
        self,
        tenant_id: str,
        status_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobStatusResponse]:
        """List ingestion jobs for a tenant (sync).

        Args:
            tenant_id: Tenant identifier.
            status_filter: Optional status filter (pending, processing,
                           completed, failed).
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of JobStatusResponse items.
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id, "limit": limit, "offset": offset,
        }
        if status_filter:
            params["status"] = status_filter
        try:
            res = self._client.get("/jobs", params=params)
        except Exception as exc:
            wrap_httpx_errors("Ingestion.list_jobs", exc)
        raise_for_status("Ingestion.list_jobs", res)
        return [JobStatusResponse.model_validate(j) for j in res.json()]


class AsyncIngestionResource(_IngestionResourceBase):
    """Asynchronous ingestion resource."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @staticmethod
    def _warn_deprecated() -> None:
        """Emit deprecation warning on actual method use, not construction (SDK-8)."""
        warnings.warn(
            "client.ingestion is deprecated and will be removed in v5.0.0. "
            "Use Pipeline API (POST /pipelines) with IngestPipelineConfig instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    async def upload(
        self,
        file_path: str,
        tenant_id: str,
        source_id: str | None = None,
        pipeline_config: dict | None = None,
    ) -> DocumentResponse:
        """Upload a document for ingestion processing (async).

        Uses `asyncio.to_thread` to read the file without blocking the
        event loop — critical for high-concurrency async servers.

        Args:
            file_path: Absolute or relative path to the file on disk.
            tenant_id: Tenant identifier for data isolation.
            source_id: Optional stable ID for the source document.
                       A UUID is generated automatically if not provided.

        Returns:
            DocumentResponse with `source_id`, `status`, and `metadata`.

        Raises:
            AuthenticationError: on 401/403.
            AIKnowAPIError:      on other HTTP errors.
            AIKnowConnectionError: if the server is unreachable.
            AIKnowTimeoutError:  if the request times out.
            FileNotFoundError:   if *file_path* does not exist.
        """
        self._warn_deprecated()
        sid, mime_type, data = self._prepare_upload(
            file_path, tenant_id, source_id, pipeline_config
        )
        file_bytes = await asyncio.to_thread(_read_file_bytes, file_path)
        files = {"file": (_filename(file_path), file_bytes, mime_type)}
        try:
            res = await self._client.post("/documents", data=data, files=files)
        except Exception as exc:
            wrap_httpx_errors("Ingestion.upload", exc)
        raise_for_status("Ingestion.upload", res)
        return self._parse_upload_response(res.json())

    async def get_job(
        self, job_id: str, tenant_id: str,
    ) -> JobStatusResponse:
        """Get ingestion job status by ID (async).

        Args:
            job_id: Job identifier.
            tenant_id: Tenant identifier for data isolation.

        Returns:
            JobStatusResponse with status, progress, and metadata.
        """
        params = {"tenant_id": tenant_id}
        try:
            res = await self._client.get(f"/jobs/{job_id}", params=params)
        except Exception as exc:
            wrap_httpx_errors("Ingestion.get_job", exc)
        raise_for_status("Ingestion.get_job", res)
        return JobStatusResponse.model_validate(res.json())

    async def list_jobs(
        self,
        tenant_id: str,
        status_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobStatusResponse]:
        """List ingestion jobs for a tenant (async).

        Args:
            tenant_id: Tenant identifier.
            status_filter: Optional status filter (pending, processing,
                           completed, failed).
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of JobStatusResponse items.
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id, "limit": limit, "offset": offset,
        }
        if status_filter:
            params["status"] = status_filter
        try:
            res = await self._client.get("/jobs", params=params)
        except Exception as exc:
            wrap_httpx_errors("Ingestion.list_jobs", exc)
        raise_for_status("Ingestion.list_jobs", res)
        return [JobStatusResponse.model_validate(j) for j in res.json()]
