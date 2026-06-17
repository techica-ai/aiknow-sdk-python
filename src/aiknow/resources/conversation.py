"""
Sync and Async Conversation resources.
"""
from __future__ import annotations

from typing import Any, cast

import httpx

from .._http import raise_for_status, wrap_httpx_errors


class _ConversationResourceBase:
    """Base class with shared conversation logic."""

    def _build_converse_payload(
        self,
        session_id: str,
        message: str,
        channel: str = "web",
        caller_id: str | None = None,
        locale: str = "vi",
        workflow_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build request payload for converse."""
        return {
            "session_id": session_id,
            "message": message,
            "channel": channel,
            "caller_id": caller_id,
            "locale": locale,
            "workflow_types": workflow_types or [],
        }


class ConversationResource(_ConversationResourceBase):
    """Synchronous conversation resource."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def converse(
        self,
        session_id: str,
        message: str,
        channel: str = "web",
        caller_id: str | None = None,
        locale: str = "vi",
        workflow_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a message in a conversation session and get the updated state (sync).

        Args:
            session_id: Conversation session identifier.
            message: The user message string.
            channel: The channel from which the message was sent (default "web").
            caller_id: Optional caller identifier.
            locale: Optional language locale code (default "vi").
            workflow_types: Optional list of workflow types to restrict matching.

        Returns:
            A dictionary containing the updated conversation state and output.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_converse_payload(
            session_id=session_id,
            message=message,
            channel=channel,
            caller_id=caller_id,
            locale=locale,
            workflow_types=workflow_types,
        )
        try:
            res = self._client.post("/conversation", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.converse", exc)
        raise_for_status("Conversation.converse", res)
        return cast(dict[str, Any], res.json())


class AsyncConversationResource(_ConversationResourceBase):
    """Asynchronous conversation resource."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def converse(
        self,
        session_id: str,
        message: str,
        channel: str = "web",
        caller_id: str | None = None,
        locale: str = "vi",
        workflow_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a message in a conversation session and get the updated state (async).

        Args:
            session_id: Conversation session identifier.
            message: The user message string.
            channel: The channel from which the message was sent (default "web").
            caller_id: Optional caller identifier.
            locale: Optional language locale code (default "vi").
            workflow_types: Optional list of workflow types to restrict matching.

        Returns:
            A dictionary containing the updated conversation state and output.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_converse_payload(
            session_id=session_id,
            message=message,
            channel=channel,
            caller_id=caller_id,
            locale=locale,
            workflow_types=workflow_types,
        )
        try:
            res = await self._client.post("/conversation", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.converse", exc)
        raise_for_status("Conversation.converse", res)
        return cast(dict[str, Any], res.json())
