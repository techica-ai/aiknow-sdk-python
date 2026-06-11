"""
Sync and Async Users resources.
"""
from __future__ import annotations

import httpx

from .._http import raise_for_status, wrap_httpx_errors


class UsersResource:
    """Synchronous users management resource."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def list(self, limit: int = 20, offset: int = 0) -> dict:
        """List users in the tenant (sync)."""
        params = {"limit": limit, "offset": offset}
        try:
            res = self._client.get("/users", params=params)
        except Exception as exc:
            wrap_httpx_errors("Users.list", exc)
        raise_for_status("Users.list", res)
        return res.json()

    def get(self, user_id: str) -> dict:
        """Get a single user by ID (sync)."""
        try:
            res = self._client.get(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.get", exc)
        raise_for_status("Users.get", res)
        return res.json()

    def invite(
        self,
        email: str,
        password: str,
        role: str = "member",
        full_name: str | None = None,
    ) -> dict:
        """Invite a user to the tenant (sync)."""
        payload = {
            "email": email,
            "password": password,
            "role": role,
            "full_name": full_name,
        }
        try:
            res = self._client.post("/users/invite", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.invite", exc)
        raise_for_status("Users.invite", res)
        return res.json()

    def update(
        self,
        user_id: str,
        role: str | None = None,
        full_name: str | None = None,
    ) -> dict:
        """Update a user's details or role (sync)."""
        payload = {}
        if role is not None:
            payload["role"] = role
        if full_name is not None:
            payload["full_name"] = full_name
        try:
            res = self._client.put(f"/users/{user_id}", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.update", exc)
        raise_for_status("Users.update", res)
        return res.json()

    def deactivate(self, user_id: str) -> dict:
        """Deactivate a user from the tenant (sync)."""
        try:
            res = self._client.delete(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.deactivate", exc)
        raise_for_status("Users.deactivate", res)
        return res.json()


class AsyncUsersResource:
    """Asynchronous users management resource."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def list(self, limit: int = 20, offset: int = 0) -> dict:
        """List users in the tenant (async)."""
        params = {"limit": limit, "offset": offset}
        try:
            res = await self._client.get("/users", params=params)
        except Exception as exc:
            wrap_httpx_errors("Users.list", exc)
        raise_for_status("Users.list", res)
        return res.json()

    async def get(self, user_id: str) -> dict:
        """Get a single user by ID (async)."""
        try:
            res = await self._client.get(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.get", exc)
        raise_for_status("Users.get", res)
        return res.json()

    async def invite(
        self,
        email: str,
        password: str,
        role: str = "member",
        full_name: str | None = None,
    ) -> dict:
        """Invite a user to the tenant (async)."""
        payload = {
            "email": email,
            "password": password,
            "role": role,
            "full_name": full_name,
        }
        try:
            res = await self._client.post("/users/invite", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.invite", exc)
        raise_for_status("Users.invite", res)
        return res.json()

    async def update(
        self,
        user_id: str,
        role: str | None = None,
        full_name: str | None = None,
    ) -> dict:
        """Update a user's details or role (async)."""
        payload = {}
        if role is not None:
            payload["role"] = role
        if full_name is not None:
            payload["full_name"] = full_name
        try:
            res = await self._client.put(f"/users/{user_id}", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.update", exc)
        raise_for_status("Users.update", res)
        return res.json()

    async def deactivate(self, user_id: str) -> dict:
        """Deactivate a user from the tenant (async)."""
        try:
            res = await self._client.delete(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.deactivate", exc)
        raise_for_status("Users.deactivate", res)
        return res.json()
