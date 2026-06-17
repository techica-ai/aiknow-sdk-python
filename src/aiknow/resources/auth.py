"""
Sync and Async Authentication resources.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .._http import raise_for_status, wrap_httpx_errors

if TYPE_CHECKING:
    from .._async_client import AsyncAIKnowClient
    from .._client import AIKnowClient


class _AuthResourceBase:
    """Base class with shared auth logic."""

    def __init__(self) -> None:
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    def _build_login_payload(self, tenant_slug: str, email: str, password: str) -> dict[str, str]:
        """Build request payload for login."""
        return {
            "tenant_slug": tenant_slug,
            "email": email,
            "password": password,
        }

    def _update_tokens(self, data: dict[str, Any]) -> None:
        """Update access and refresh tokens."""
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")

    def _clear_tokens(self) -> None:
        """Clear access and refresh tokens."""
        self.access_token = None
        self.refresh_token = None


class AuthResource(_AuthResourceBase):
    """Synchronous authentication resource."""

    def __init__(self, client: AIKnowClient) -> None:
        super().__init__()
        self._sdk_client = client

    def login(self, tenant_slug: str, email: str, password: str) -> dict[str, Any]:
        """Authenticate user and store access/refresh tokens in-memory (sync).

        Args:
            tenant_slug: Tenant slug string.
            email: User email address.
            password: User password.

        Returns:
            A dictionary containing authentication tokens.

        Raises:
            Exception: If the login HTTP request fails.
        """
        payload = self._build_login_payload(tenant_slug, email, password)
        try:
            # Use _raw_client to avoid infinite redirection in custom auth flow
            res = self._sdk_client._raw_client.post("/auth/login", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.login", exc)
        raise_for_status("Auth.login", res)
        
        data = res.json()
        self._update_tokens(data)
        return cast(dict[str, Any], data)

    def refresh(self) -> bool:
        """Refresh the current access token using the refresh token (sync).

        Returns:
            True if the token was successfully refreshed, False otherwise.

        Raises:
            None.
        """
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
        self._update_tokens(data)
        return True

    def logout(self) -> dict[str, Any]:
        """Revoke current refresh token and clear tokens (sync).

        Returns:
            A dictionary containing the response message.

        Raises:
            Exception: If the logout HTTP request fails.
        """
        if not self.refresh_token:
            self._clear_tokens()
            return {"message": "Already logged out."}
        
        payload = {"refresh_token": self.refresh_token}
        try:
            # Use _client to perform authenticated logout
            res = self._sdk_client._client.post("/auth/logout", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.logout", exc)
        
        self._clear_tokens()
        raise_for_status("Auth.logout", res)
        return cast(dict[str, Any], res.json())

    def logout_all(self) -> dict[str, Any]:
        """Revoke all sessions/refresh tokens for the authenticated user (sync).

        Returns:
            A dictionary containing the response message.

        Raises:
            Exception: If the logout HTTP request fails.
        """
        try:
            res = self._sdk_client._client.post("/auth/logout/all")
        except Exception as exc:
            wrap_httpx_errors("Auth.logout_all", exc)
        
        self._clear_tokens()
        raise_for_status("Auth.logout_all", res)
        return cast(dict[str, Any], res.json())


class AsyncAuthResource(_AuthResourceBase):
    """Asynchronous authentication resource."""

    def __init__(self, client: AsyncAIKnowClient) -> None:
        super().__init__()
        self._sdk_client = client

    async def login(self, tenant_slug: str, email: str, password: str) -> dict[str, Any]:
        """Authenticate user and store access/refresh tokens in-memory (async).

        Args:
            tenant_slug: Tenant slug string.
            email: User email address.
            password: User password.

        Returns:
            A dictionary containing authentication tokens.

        Raises:
            Exception: If the login HTTP request fails.
        """
        payload = self._build_login_payload(tenant_slug, email, password)
        try:
            res = await self._sdk_client._raw_client.post("/auth/login", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.login", exc)
        raise_for_status("Auth.login", res)
        
        data = res.json()
        self._update_tokens(data)
        return cast(dict[str, Any], data)

    async def refresh(self) -> bool:
        """Refresh the current access token using the refresh token (async).

        Returns:
            True if the token was successfully refreshed, False otherwise.

        Raises:
            None.
        """
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
        self._update_tokens(data)
        return True

    async def logout(self) -> dict[str, Any]:
        """Revoke current refresh token and clear tokens (async).

        Returns:
            A dictionary containing the response message.

        Raises:
            Exception: If the logout HTTP request fails.
        """
        if not self.refresh_token:
            self._clear_tokens()
            return {"message": "Already logged out."}
        
        payload = {"refresh_token": self.refresh_token}
        try:
            res = await self._sdk_client._client.post("/auth/logout", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Auth.logout", exc)
        
        self._clear_tokens()
        raise_for_status("Auth.logout", res)
        return cast(dict[str, Any], res.json())

    async def logout_all(self) -> dict[str, Any]:
        """Revoke all sessions/refresh tokens for the authenticated user (async).

        Returns:
            A dictionary containing the response message.

        Raises:
            Exception: If the logout HTTP request fails.
        """
        try:
            res = await self._sdk_client._client.post("/auth/logout/all")
        except Exception as exc:
            wrap_httpx_errors("Auth.logout_all", exc)
        
        self._clear_tokens()
        raise_for_status("Auth.logout_all", res)
        return cast(dict[str, Any], res.json())
