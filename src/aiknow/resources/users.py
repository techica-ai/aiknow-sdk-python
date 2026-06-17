"""
Sync and Async Users resources.
"""
from __future__ import annotations

from typing import Any, cast

import httpx

from .._http import raise_for_status, wrap_httpx_errors


class _UsersResourceBase:
    """Base class with shared users logic."""

    def _build_list_params(self, limit: int = 20, offset: int = 0) -> dict[str, int]:
        """Build parameters for listing users."""
        return {"limit": limit, "offset": offset}

    def _build_invite_payload(
        self,
        email: str,
        password: str,
        role: str = "member",
        full_name: str | None = None,
    ) -> dict[str, Any]:
        """Build payload for inviting a user."""
        return {
            "email": email,
            "password": password,
            "role": role,
            "full_name": full_name,
        }

    def _build_update_payload(
        self,
        role: str | None = None,
        full_name: str | None = None,
    ) -> dict[str, Any]:
        """Build payload for updating user details."""
        payload = {}
        if role is not None:
            payload["role"] = role
        if full_name is not None:
            payload["full_name"] = full_name
        return payload


class UsersResource(_UsersResourceBase):
    """Synchronous users management resource."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def list(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List users in the tenant (sync).

        Args:
            limit: Maximum number of users to retrieve (default 20).
            offset: Offset for pagination (default 0).

        Returns:
            A dictionary containing the list of users.

        Raises:
            Exception: If the HTTP request fails.
        """
        params = self._build_list_params(limit, offset)
        try:
            res = self._client.get("/users", params=params)
        except Exception as exc:
            wrap_httpx_errors("Users.list", exc)
        raise_for_status("Users.list", res)
        return cast(dict[str, Any], res.json())

    def get(self, user_id: str) -> dict[str, Any]:
        """Get a single user by ID (sync).

        Args:
            user_id: The ID of the user to retrieve.

        Returns:
            A dictionary containing the user's details.

        Raises:
            Exception: If the HTTP request fails.
        """
        try:
            res = self._client.get(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.get", exc)
        raise_for_status("Users.get", res)
        return cast(dict[str, Any], res.json())

    def invite(
        self,
        email: str,
        password: str,
        role: str = "member",
        full_name: str | None = None,
    ) -> dict[str, Any]:
        """Invite a user to the tenant (sync).

        Args:
            email: User's email address.
            password: User's temporary password.
            role: User's role (default "member").
            full_name: User's full name.

        Returns:
            A dictionary containing the invited user's information.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_invite_payload(email, password, role, full_name)
        try:
            res = self._client.post("/users/invite", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.invite", exc)
        raise_for_status("Users.invite", res)
        return cast(dict[str, Any], res.json())

    def update(
        self,
        user_id: str,
        role: str | None = None,
        full_name: str | None = None,
    ) -> dict[str, Any]:
        """Update a user's details or role (sync).

        Args:
            user_id: The ID of the user to update.
            role: Optional new role.
            full_name: Optional new full name.

        Returns:
            A dictionary containing the updated user details.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_update_payload(role, full_name)
        try:
            res = self._client.put(f"/users/{user_id}", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.update", exc)
        raise_for_status("Users.update", res)
        return cast(dict[str, Any], res.json())

    def deactivate(self, user_id: str) -> dict[str, Any]:
        """Deactivate a user from the tenant (sync).

        Args:
            user_id: The ID of the user to deactivate.

        Returns:
            A dictionary containing the deactivation result.

        Raises:
            Exception: If the HTTP request fails.
        """
        try:
            res = self._client.delete(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.deactivate", exc)
        raise_for_status("Users.deactivate", res)
        return cast(dict[str, Any], res.json())


class AsyncUsersResource(_UsersResourceBase):
    """Asynchronous users management resource."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def list(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List users in the tenant (async).

        Args:
            limit: Maximum number of users to retrieve (default 20).
            offset: Offset for pagination (default 0).

        Returns:
            A dictionary containing the list of users.

        Raises:
            Exception: If the HTTP request fails.
        """
        params = self._build_list_params(limit, offset)
        try:
            res = await self._client.get("/users", params=params)
        except Exception as exc:
            wrap_httpx_errors("Users.list", exc)
        raise_for_status("Users.list", res)
        return cast(dict[str, Any], res.json())

    async def get(self, user_id: str) -> dict[str, Any]:
        """Get a single user by ID (async).

        Args:
            user_id: The ID of the user to retrieve.

        Returns:
            A dictionary containing the user's details.

        Raises:
            Exception: If the HTTP request fails.
        """
        try:
            res = await self._client.get(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.get", exc)
        raise_for_status("Users.get", res)
        return cast(dict[str, Any], res.json())

    async def invite(
        self,
        email: str,
        password: str,
        role: str = "member",
        full_name: str | None = None,
    ) -> dict[str, Any]:
        """Invite a user to the tenant (async).

        Args:
            email: User's email address.
            password: User's temporary password.
            role: User's role (default "member").
            full_name: User's full name.

        Returns:
            A dictionary containing the invited user's information.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_invite_payload(email, password, role, full_name)
        try:
            res = await self._client.post("/users/invite", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.invite", exc)
        raise_for_status("Users.invite", res)
        return cast(dict[str, Any], res.json())

    async def update(
        self,
        user_id: str,
        role: str | None = None,
        full_name: str | None = None,
    ) -> dict[str, Any]:
        """Update a user's details or role (async).

        Args:
            user_id: The ID of the user to update.
            role: Optional new role.
            full_name: Optional new full name.

        Returns:
            A dictionary containing the updated user details.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_update_payload(role, full_name)
        try:
            res = await self._client.put(f"/users/{user_id}", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Users.update", exc)
        raise_for_status("Users.update", res)
        return cast(dict[str, Any], res.json())

    async def deactivate(self, user_id: str) -> dict[str, Any]:
        """Deactivate a user from the tenant (async).

        Args:
            user_id: The ID of the user to deactivate.

        Returns:
            A dictionary containing the deactivation result.

        Raises:
            Exception: If the HTTP request fails.
        """
        try:
            res = await self._client.delete(f"/users/{user_id}")
        except Exception as exc:
            wrap_httpx_errors("Users.deactivate", exc)
        raise_for_status("Users.deactivate", res)
        return cast(dict[str, Any], res.json())
