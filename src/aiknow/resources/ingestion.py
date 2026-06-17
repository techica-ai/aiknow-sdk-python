"""
Sync and Async Ingestion resources.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import httpx
from aiknow_contracts.documents import DocumentResponse

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
    ) -> tuple[str, str, dict[str, str]]:
        """Prepare the upload payload and metadata."""
        import mimetypes
        sid = source_id or str(uuid.uuid4())
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        data = {"tenant_id": tenant_id, "source_id": sid}
        return sid, mime_type, data

    def _parse_upload_response(self, response_json: Any) -> DocumentResponse:
        """Parse the upload response."""
        return DocumentResponse.model_validate(response_json)


class IngestionResource(_IngestionResourceBase):
    """Synchronous ingestion resource."""

    def __init__(self, client: httpx.Client) -> None:
        import warnings
        warnings.warn(
            "client.ingestion is deprecated and will be removed in v5.0.0. "
            "Use Pipeline API (POST /pipelines) with IngestPipelineConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._client = client

    def upload(
        self,
        file_path: str,
        tenant_id: str,
        source_id: str | None = None,
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
        sid, mime_type, data = self._prepare_upload(file_path, tenant_id, source_id)
        with open(file_path, "rb") as f:
            files = {"file": (_filename(file_path), f, mime_type)}
            try:
                res = self._client.post("/documents", data=data, files=files)
            except Exception as exc:
                wrap_httpx_errors("Ingestion.upload", exc)
        raise_for_status("Ingestion.upload", res)
        return self._parse_upload_response(res.json())


class AsyncIngestionResource(_IngestionResourceBase):
    """Asynchronous ingestion resource."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        import warnings
        warnings.warn(
            "client.ingestion is deprecated and will be removed in v5.0.0. "
            "Use Pipeline API (POST /pipelines) with IngestPipelineConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._client = client

    async def upload(
        self,
        file_path: str,
        tenant_id: str,
        source_id: str | None = None,
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
        sid, mime_type, data = self._prepare_upload(file_path, tenant_id, source_id)
        file_bytes = await asyncio.to_thread(_read_file_bytes, file_path)
        files = {"file": (_filename(file_path), file_bytes, mime_type)}
        try:
            res = await self._client.post("/documents", data=data, files=files)
        except Exception as exc:
            wrap_httpx_errors("Ingestion.upload", exc)
        raise_for_status("Ingestion.upload", res)
        return self._parse_upload_response(res.json())
