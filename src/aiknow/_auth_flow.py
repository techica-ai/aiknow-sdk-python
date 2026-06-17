"""
Custom httpx.Auth implementation for AIKnow multi-tenant JWT token management.
Supports both synchronous and asynchronous request pipelines.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx


class AIKnowAuth(httpx.Auth):
    """Custom httpx authentication class managing bearer token and automatic refresh."""

    def __init__(self, auth_resource: Any, auto_refresh: bool = True) -> None:
        self.auth_resource = auth_resource
        self.auto_refresh = auto_refresh

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        # Inject access token if present
        if self.auth_resource.access_token:
            request.headers["Authorization"] = f"Bearer {self.auth_resource.access_token}"

        response = yield request

        # If unauthorized, try refreshing the token
        if response.status_code == 401 and self.auto_refresh:
            refreshed = self.auth_resource.refresh()
            if refreshed and self.auth_resource.access_token:
                # Update authorization header with new token and retry request
                request.headers["Authorization"] = f"Bearer {self.auth_resource.access_token}"
                yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        # Inject access token if present
        if self.auth_resource.access_token:
            request.headers["Authorization"] = f"Bearer {self.auth_resource.access_token}"

        response = yield request

        # If unauthorized, try refreshing the token asynchronously
        if response.status_code == 401 and self.auto_refresh:
            refreshed = await self.auth_resource.refresh()
            if refreshed and self.auth_resource.access_token:
                request.headers["Authorization"] = f"Bearer {self.auth_resource.access_token}"
                yield request
