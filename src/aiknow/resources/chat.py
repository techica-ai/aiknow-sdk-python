"""
Sync and Async Chat resources.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import httpx
from aiknow_contracts.chat import (
    ChatRequest,
    ChatResponse,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamTokenEvent,
)

from .._http import raise_for_status, wrap_httpx_errors

StreamEvent = StreamTokenEvent | StreamDoneEvent | StreamErrorEvent

_DONE_SENTINEL = object()


class _ChatResourceBase:
    """Base class with shared chat logic (no I/O, just business logic)."""

    def _build_ask_payload(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build request payload for ask and ask_stream."""
        payload = ChatRequest(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context or {},
        )
        return payload.model_dump(exclude_none=True)

    def _parse_ask_response(self, response_json: Any) -> ChatResponse:
        """Parse chat response."""
        return ChatResponse.model_validate(response_json)

    def _parse_stream_line(self, line: bytes | str) -> StreamEvent | None | object:
        """Parse a single SSE stream line."""
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            return None
        data = line_str[6:]
        if data == "[DONE]":
            return _DONE_SENTINEL
        event = json.loads(data)
        return _parse_stream_event(event)

    def _parse_history_response(self, response_json: Any) -> list[dict[str, Any]]:
        """Parse chat history response."""
        return cast(list[dict[str, Any]], response_json.get("messages", []))


class ChatResource(_ChatResourceBase):
    """Synchronous chat resource."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def ask(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a question to the AIKNOW Agent (sync).

        Args:
            query: The question string to ask the agent.
            tenant_id: Tenant identifier for data isolation.
            session_id: Optional session ID to track conversation history.
            source_ids: Optional list of specific source document IDs to query.
            user_context: Optional dictionary containing user context or metadata.

        Returns:
            The parsed ChatResponse from the AIKNOW Agent.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_ask_payload(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context,
        )
        try:
            res = self._client.post("/chat/ask", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Chat.ask", exc)
        raise_for_status("Chat.ask", res)
        return self._parse_ask_response(res.json())

    def ask_stream(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Send a question to the AIKNOW Agent with SSE streaming (sync).

        Args:
            query: The question string to ask the agent.
            tenant_id: Tenant identifier for data isolation.
            session_id: Optional session ID to track conversation history.
            source_ids: Optional list of specific source document IDs to query.
            user_context: Optional dictionary containing user context or metadata.

        Returns:
            An iterator yielding StreamEvents representing the streaming response.

        Raises:
            Exception: If the stream fails.
        """
        payload = self._build_ask_payload(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context,
        )
        with self._client.stream(
            "POST",
            "/chat/ask/stream",
            json=payload,
            timeout=httpx.Timeout(120.0),
        ) as response:
            raise_for_status("Chat.ask_stream", response)
            for line in response.iter_lines():
                parsed = self._parse_stream_line(line)
                if parsed is None:
                    continue
                if parsed is _DONE_SENTINEL:
                    break
                yield parsed  # type: ignore

    def get_history(
        self,
        session_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Get UI message history to display on frontend (sync).

        Args:
            session_id: Session identifier to retrieve history for.
            tenant_id: Tenant identifier for data isolation.

        Returns:
            A list of dictionary objects representing message history.

        Raises:
            Exception: If the request fails.
        """
        try:
            res = self._client.get(
                "/chat/history",
                params={"session_id": session_id, "tenant_id": tenant_id},
                timeout=httpx.Timeout(10.0),  # Short timeout — only Redis read
            )
        except Exception as exc:
            wrap_httpx_errors("Chat.get_history", exc)
        raise_for_status("Chat.get_history", res)
        return self._parse_history_response(res.json())

    def delete_history(
        self,
        session_id: str,
        tenant_id: str,
    ) -> None:
        """Delete chat history of a session in Redis (sync).

        Args:
            session_id: Session identifier to delete history for.
            tenant_id: Tenant identifier for data isolation.

        Returns:
            None.

        Raises:
            Exception: If the request fails.
        """
        try:
            res = self._client.delete(
                "/chat/history",
                params={"session_id": session_id, "tenant_id": tenant_id},
                timeout=httpx.Timeout(10.0),
            )
        except Exception as exc:
            wrap_httpx_errors("Chat.delete_history", exc)
        raise_for_status("Chat.delete_history", res)


class AsyncChatResource(_ChatResourceBase):
    """Asynchronous chat resource."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def ask(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a question to the AIKNOW Agent (async).

        Args:
            query: The question string to ask the agent.
            tenant_id: Tenant identifier for data isolation.
            session_id: Optional session ID to track conversation history.
            source_ids: Optional list of specific source document IDs to query.
            user_context: Optional dictionary containing user context or metadata.

        Returns:
            The parsed ChatResponse from the AIKNOW Agent.

        Raises:
            Exception: If the HTTP request fails.
        """
        payload = self._build_ask_payload(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context,
        )
        try:
            res = await self._client.post("/chat/ask", json=payload)
        except Exception as exc:
            wrap_httpx_errors("Chat.ask", exc)
        raise_for_status("Chat.ask", res)
        return self._parse_ask_response(res.json())

    async def ask_stream(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a question to the AIKNOW Agent with SSE streaming (async).

        Args:
            query: The question string to ask the agent.
            tenant_id: Tenant identifier for data isolation.
            session_id: Optional session ID to track conversation history.
            source_ids: Optional list of specific source document IDs to query.
            user_context: Optional dictionary containing user context or metadata.

        Returns:
            An async iterator yielding StreamEvents representing the streaming response.

        Raises:
            Exception: If the stream fails.
        """
        payload = self._build_ask_payload(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context,
        )
        async with self._client.stream(
            "POST",
            "/chat/ask/stream",
            json=payload,
            timeout=httpx.Timeout(120.0),
        ) as response:
            raise_for_status("Chat.ask_stream", response)
            async for line in response.aiter_lines():
                parsed = self._parse_stream_line(line)
                if parsed is None:
                    continue
                if parsed is _DONE_SENTINEL:
                    break
                yield parsed  # type: ignore

    async def get_history(
        self,
        session_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Get UI message history to display on frontend (async).

        Args:
            session_id: Session identifier to retrieve history for.
            tenant_id: Tenant identifier for data isolation.

        Returns:
            A list of dictionary objects representing message history.

        Raises:
            Exception: If the request fails.
        """
        try:
            res = await self._client.get(
                "/chat/history",
                params={"session_id": session_id, "tenant_id": tenant_id},
                timeout=httpx.Timeout(10.0),  # Short timeout — only Redis read
            )
        except Exception as exc:
            wrap_httpx_errors("Chat.get_history", exc)
        raise_for_status("Chat.get_history", res)
        return self._parse_history_response(res.json())

    async def delete_history(
        self,
        session_id: str,
        tenant_id: str,
    ) -> None:
        """Delete chat history of a session in Redis (async).

        Args:
            session_id: Session identifier to delete history for.
            tenant_id: Tenant identifier for data isolation.

        Returns:
            None.

        Raises:
            Exception: If the request fails.
        """
        try:
            res = await self._client.delete(
                "/chat/history",
                params={"session_id": session_id, "tenant_id": tenant_id},
                timeout=httpx.Timeout(10.0),
            )
        except Exception as exc:
            wrap_httpx_errors("Chat.delete_history", exc)
        raise_for_status("Chat.delete_history", res)


def _parse_stream_event(event: dict[str, Any]) -> StreamEvent:
    """Parse a SSE data payload into a typed event model.

    Args:
        event: Dict object representing the SSE payload.

    Returns:
        The corresponding parsed StreamEvent subclass.
    """
    match event.get("type"):
        case "token":
            return StreamTokenEvent(content=event["content"])
        case "done":
            return StreamDoneEvent(
                answer=event.get("answer", ""),
                citations=event.get("citations", []),
                trace_id=event.get("trace_id"),
                graph=event.get("graph"),
            )
        case "error":
            return StreamErrorEvent(message=event.get("message", "Unknown error"))
        case _:
            return StreamErrorEvent(message=f"Unknown event type: {event.get('type')}")
