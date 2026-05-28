"""
Sync and Async Conversation resources.
"""
from __future__ import annotations

import httpx

from .._http import raise_for_status, wrap_httpx_errors


class ConversationResource:
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
    ) -> dict:
        """Send a message in a conversation session and get the updated state (sync)."""
        payload = {
            "session_id": session_id,
            "message": message,
            "channel": channel,
            "caller_id": caller_id,
            "locale": locale,
            "workflow_types": workflow_types or [],
        }
        try:
            res = self._client.post("/conversation", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.converse", exc)
        raise_for_status("Conversation.converse", res)
        return res.json()


class AsyncConversationResource:
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
    ) -> dict:
        """Send a message in a conversation session and get the updated state (async)."""
        payload = {
            "session_id": session_id,
            "message": message,
            "channel": channel,
            "caller_id": caller_id,
            "locale": locale,
            "workflow_types": workflow_types or [],
        }
        try:
            res = await self._client.post("/conversation", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.converse", exc)
        raise_for_status("Conversation.converse", res)
        return res.json()
