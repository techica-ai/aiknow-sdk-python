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

    @staticmethod
    def _build_graph_turn_payload(
        session_id: str,
        graph_id: str,
        message: str = "",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Build request payload for P3 graph turn."""
        return {
            "session_id": session_id,
            "graph_id": graph_id,
            "message": message,
            "tenant_id": tenant_id,
        }

    @staticmethod
    def _build_approve_payload(
        session_id: str,
        decision: str,
        approver_id: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Build request payload for P3 checkpoint approval."""
        return {
            "session_id": session_id,
            "decision": decision,
            "approver_id": approver_id,
            "tenant_id": tenant_id,
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

    def graph_turn(
        self,
        session_id: str,
        graph_id: str,
        message: str = "",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Process one P3 graph engine turn (sync).

        Starts a new session (message="") or advances an existing one.

        Args:
            session_id: Conversation session identifier.
            graph_id: P3 conversation graph ID.
            message: User's text input. Empty string starts a new session.
            tenant_id: Tenant identifier.

        Returns:
            Dictionary with outcome, stop_reason, is_frozen, current_node, etc.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_graph_turn_payload(
            session_id=session_id,
            graph_id=graph_id,
            message=message,
            tenant_id=tenant_id,
        )
        try:
            res = self._client.post("/conversation/graph/turn", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.graph_turn", exc)
        raise_for_status("Conversation.graph_turn", res)
        return cast(dict[str, Any], res.json())

    def approve_checkpoint(
        self,
        session_id: str,
        decision: str,
        approver_id: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Approve or reject a P3 graph checkpoint (sync).

        Args:
            session_id: Conversation session identifier (must be frozen).
            decision: Either "approve" or "reject".
            approver_id: Identifier of the human approver.
            tenant_id: Tenant identifier.

        Returns:
            Dictionary with the updated turn result after approval.

        Raises:
            Exception: If the HTTP request fails or session is not frozen.
        """
        payload = self._build_approve_payload(
            session_id=session_id,
            decision=decision,
            approver_id=approver_id,
            tenant_id=tenant_id,
        )
        try:
            res = self._client.post("/conversation/graph/approve", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.approve_checkpoint", exc)
        raise_for_status("Conversation.approve_checkpoint", res)
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

    async def graph_turn(
        self,
        session_id: str,
        graph_id: str,
        message: str = "",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Process one P3 graph engine turn (async).

        Starts a new session (message="") or advances an existing one.

        Args:
            session_id: Conversation session identifier.
            graph_id: P3 conversation graph ID.
            message: User's text input. Empty string starts a new session.
            tenant_id: Tenant identifier.

        Returns:
            Dictionary with outcome, stop_reason, is_frozen, current_node, etc.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_graph_turn_payload(
            session_id=session_id,
            graph_id=graph_id,
            message=message,
            tenant_id=tenant_id,
        )
        try:
            res = await self._client.post("/conversation/graph/turn", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.graph_turn", exc)
        raise_for_status("Conversation.graph_turn", res)
        return cast(dict[str, Any], res.json())

    async def approve_checkpoint(
        self,
        session_id: str,
        decision: str,
        approver_id: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Approve or reject a P3 graph checkpoint (async).

        Args:
            session_id: Conversation session identifier (must be frozen).
            decision: Either "approve" or "reject".
            approver_id: Identifier of the human approver.
            tenant_id: Tenant identifier.

        Returns:
            Dictionary with the updated turn result after approval.

        Raises:
            Exception: If the HTTP request fails or session is not frozen.
        """
        payload = self._build_approve_payload(
            session_id=session_id,
            decision=decision,
            approver_id=approver_id,
            tenant_id=tenant_id,
        )
        try:
            res = await self._client.post("/conversation/graph/approve", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Conversation.approve_checkpoint", exc)
        raise_for_status("Conversation.approve_checkpoint", res)
        return cast(dict[str, Any], res.json())
