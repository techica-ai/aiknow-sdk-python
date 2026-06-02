"""
Sync and Async Chat resources.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

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


class ChatResource:
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
        """Send a question to the AIKNOW Agent (sync)."""
        payload = ChatRequest(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context or {},
        )
        try:
            res = self._client.post("/chat/ask", json=payload.model_dump(exclude_none=True))
        except Exception as exc:
            wrap_httpx_errors("Chat.ask", exc)
        raise_for_status("Chat.ask", res)
        return ChatResponse.model_validate(res.json())

    def ask_stream(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Send a question to the AIKNOW Agent with SSE streaming (sync)."""
        payload = ChatRequest(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context or {},
        )
        with self._client.stream(
            "POST",
            "/chat/ask/stream",
            json=payload.model_dump(exclude_none=True),
            timeout=httpx.Timeout(120.0),
        ) as response:
            raise_for_status("Chat.ask_stream", response)
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                yield _parse_stream_event(event)

    def get_history(
        self,
        session_id: str,
        tenant_id: str,
    ) -> list[dict]:
        """Lấy UI message history để hiển thị trên frontend (sync)."""
        try:
            res = self._client.get(
                "/chat/history",
                params={"session_id": session_id, "tenant_id": tenant_id},
                timeout=httpx.Timeout(10.0),  # Short timeout — chỉ là Redis read
            )
        except Exception as exc:
            wrap_httpx_errors("Chat.get_history", exc)
        raise_for_status("Chat.get_history", res)
        return res.json().get("messages", [])


class AsyncChatResource:
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
        """Send a question to the AIKNOW Agent (async)."""
        payload = ChatRequest(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context or {},
        )
        try:
            res = await self._client.post("/chat/ask", json=payload.model_dump(exclude_none=True))
        except Exception as exc:
            wrap_httpx_errors("Chat.ask", exc)
        raise_for_status("Chat.ask", res)
        return ChatResponse.model_validate(res.json())

    async def ask_stream(
        self,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        source_ids: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a question to the AIKNOW Agent with SSE streaming (async)."""
        payload = ChatRequest(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
            source_ids=source_ids,
            user_context=user_context or {},
        )
        async with self._client.stream(
            "POST",
            "/chat/ask/stream",
            json=payload.model_dump(exclude_none=True),
            timeout=httpx.Timeout(120.0),
        ) as response:
            raise_for_status("Chat.ask_stream", response)
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                yield _parse_stream_event(event)

    async def get_history(
        self,
        session_id: str,
        tenant_id: str,
    ) -> list[dict]:
        """Lấy UI message history để hiển thị trên frontend (async)."""
        try:
            res = await self._client.get(
                "/chat/history",
                params={"session_id": session_id, "tenant_id": tenant_id},
                timeout=httpx.Timeout(10.0),  # Short timeout — chỉ là Redis read
            )
        except Exception as exc:
            wrap_httpx_errors("Chat.get_history", exc)
        raise_for_status("Chat.get_history", res)
        return res.json().get("messages", [])


def _parse_stream_event(event: dict) -> StreamEvent:
    """Parse a SSE data payload into a typed event model."""
    match event.get("type"):
        case "token":
            return StreamTokenEvent(content=event["content"])
        case "done":
            return StreamDoneEvent(
                answer=event.get("answer", ""),
                citations=event.get("citations", []),
                trace_id=event.get("trace_id"),
            )
        case "error":
            return StreamErrorEvent(message=event.get("message", "Unknown error"))
        case _:
            return StreamErrorEvent(message=f"Unknown event type: {event.get('type')}")
