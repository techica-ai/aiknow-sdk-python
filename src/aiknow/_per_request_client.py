"""
_PerRequestClient — per-request header injection via duck typing.

Vấn đề giải quyết:
  BFF cần forward user's JWT token + tenant_id vào mỗi request lên aiknow-api,
  nhưng mỗi incoming request có token khác nhau. Nếu tạo AsyncAIKnowClient mới
  mỗi request → tạo httpx.AsyncClient mới → không tái dụng TCP connection pool.

Giải pháp:
  _PerRequestClient wrap một shared httpx.AsyncClient và inject auth headers
  per-call. SDK Resources (AsyncChatResource, AsyncIngestionResource, ...) nhận
  vào httpx.AsyncClient-like object — _PerRequestClient satisfy interface đó via
  duck typing (không cần kế thừa).

Ownership:
  _PerRequestClient KHÔNG own shared_client → không có close()/aclose().
  Shared client được quản lý bởi AsyncAIKnowClient._SHARED_HTTP_CLIENT
  và được close bởi close_shared_http_client() trong app lifespan.
"""
from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx


class _PerRequestClient:
    """
    Duck-typed wrapper inject per-request auth headers vào shared httpx.AsyncClient.

    Implement subset of httpx.AsyncClient interface mà SDK Resources sử dụng:
      - get(), post(), delete(): async HTTP methods
      - stream(): async context manager cho SSE streaming

    Không implement close()/aclose() vì không owned resources.
    Lifetime: tạo per-request, garbage collected sau khi request xử lý xong.

    Args:
        shared_client: Shared httpx.AsyncClient với connection pool. Không owned.
        bearer_token:  JWT access token để inject vào Authorization header.
        tenant_id:     Tenant identifier để inject vào X-Tenant-Id header.
    """

    __slots__ = ("_client", "_inject")

    def __init__(
        self,
        shared_client: httpx.AsyncClient,
        bearer_token: str,
        tenant_id: str,
    ) -> None:
        self._client = shared_client  # Shared — not owned, not closed
        self._inject: dict[str, str] = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tenant-Id": tenant_id,
        }

    def _merge(self, extra: dict[str, str] | None) -> dict[str, str]:
        """
        Merge injected auth headers với per-call headers.
        Per-call headers có priority (override injected nếu trùng key).
        """
        if extra:
            return {**self._inject, **extra}
        return dict(self._inject)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return await self._client.get(url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return await self._client.post(url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -\u003e httpx.Response:
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return await self._client.delete(url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -\u003e httpx.Response:
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return await self._client.put(url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -\u003e httpx.Response:
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return await self._client.patch(url, **kwargs)

    async def head(self, url: str, **kwargs: Any) -\u003e httpx.Response:
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return await self._client.head(url, **kwargs)

    def stream(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[httpx.Response]:
        """
        Delegate SSE streaming tới shared client.

        Return type rõ ràng: AbstractAsyncContextManager[httpx.Response]
        để caller (SDK resources) biết đây là async context manager,
        không phải coroutine hay regular function.

        Usage:
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    yield line
        """
        kwargs["headers"] = self._merge(kwargs.pop("headers", None))
        return self._client.stream(method, url, **kwargs)
