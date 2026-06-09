"""
Sync and Async Authentication resources.
"""
from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from .._http import raise_for_status, wrap_httpx_errors

if TYPE_CHECKING:
    from .._client import AIKnowClient
    from .._async_client import AsyncAIKnowClient


class AuthResource:
    """Synchronous authentication resource."""

    def __init__(self, client: AIKnowClient) -> None:
        self._sdk_client = client
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    def login(self, tenant_slug: str, email: str, password: str) -> dict:
        """Authenticate user and store access/refresh tokens in-memory (sync)."""
        payload = {
            "tenant_slug": tenant_slug,
            "email": email,
            "password": password,
        }
        try:
            # Use _raw_client to avoid infinite redirection in custom auth flow
            res = self._sdk_client._raw_client.post("/auth/login", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.login", exc)
        raise_for_status("Auth.login", res)
        
        data = res.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return data

    def refresh(self) -> bool:
        """Refresh the current access token using the refresh token (sync)."""
        if not self.refresh_token:
            return False
        payload = {"refresh_token": self.refresh_token}
        try:
            res = self._sdk_client._raw_client.post("/auth/refresh", json=payload)
        except Exception:
            return False
        
        if res.status_code != 200:
            return False
            
        data = res.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return True

    def logout(self) -> dict:
        """Revoke current refresh token and clear tokens (sync)."""
        if not self.refresh_token:
            self.access_token = None
            return {"message": "Already logged out."}
        
        payload = {"refresh_token": self.refresh_token}
        try:
            # Use _client to perform authenticated logout
            res = self._sdk_client._client.post("/auth/logout", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.logout", exc)
        
        self.access_token = None
        self.refresh_token = None
        
        raise_for_status("Auth.logout", res)
        return res.json()

    def logout_all(self) -> dict:
        """Revoke all sessions/refresh tokens for the authenticated user (sync)."""
        try:
            res = self._sdk_client._client.post("/auth/logout/all")
        except Exception as exc:
            wrap_httpx_errors("Auth.logout_all", exc)
        
        self.access_token = None
        self.refresh_token = None
        
        raise_for_status("Auth.logout_all", res)
        return res.json()


class AsyncAuthResource:
    """Asynchronous authentication resource."""

    def __init__(self, client: AsyncAIKnowClient) -> None:
        self._sdk_client = client
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    async def login(self, tenant_slug: str, email: str, password: str) -> dict:
        """Authenticate user and store access/refresh tokens in-memory (async)."""
        payload = {
            "tenant_slug": tenant_slug,
            "email": email,
            "password": password,
        }
        try:
            res = await self._sdk_client._raw_client.post("/auth/login", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.login", exc)
        raise_for_status("Auth.login", res)
        
        data = res.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return data

    async def refresh(self) -> bool:
        """Refresh the current access token using the refresh token (async)."""
        if not self.refresh_token:
            return False
        payload = {"refresh_token": self.refresh_token}
        try:
            res = await self._sdk_client._raw_client.post("/auth/refresh", json=payload)
        except Exception:
            return False
        
        if res.status_code != 200:
            return False
            
        data = res.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return True

    async def logout(self) -> dict:
        """Revoke current refresh token and clear tokens (async)."""
        if not self.refresh_token:
            self.access_token = None
            return {"message": "Already logged out."}
        
        payload = {"refresh_token": self.refresh_token}
        try:
            res = await self._sdk_client._client.post("/auth/logout", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.logout", exc)
        
        self.access_token = None
        self.refresh_token = None
        
        raise_for_status("Auth.logout", res)
        return res.json()

    async def logout_all(self) -> dict:
        """Revoke all sessions/refresh tokens for the authenticated user (async)."""
        try:
            res = await self._sdk_client._client.post("/auth/logout/all")
        except Exception as exc:
            wrap_httpx_errors("Auth.logout_all", exc)
        
        self.access_token = None
        self.refresh_token = None
        
        raise_for_status("Auth.logout_all", res)
        return res.json()
