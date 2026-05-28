"""
AIKnow Conversation Agent — Phase 5 AG-UI Bridge.

This module provides ``AIKnowConversationAgent``, a reusable SDK agent that
bridges CopilotKit's AG-UI protocol to the Platform's ``POST /conversation`` API.

Architecture:

    Frontend (CopilotKit)
        │  AG-UI SSE events
        ▼
    AIKnowConversationAgent  ← SDK provides this base class
        │  POST /conversation
        ▼
    Platform ConversationAPI
        │  WorkflowContext + UIHints
        ▼
    ConversationTurn
        │  StateSnapshot + TextMessage events
        ▼
    Frontend (React useCoAgent() state)

Design:
  - One-to-one mapping: AG-UI ``threadId`` → Platform ``session_id``.
  - ``StateSnapshot`` carries WorkflowContext + UIHints for real-time slot UI.
  - Full SSE event sequence: RunStarted → TextMessageStarted → TextMessageContent
    → TextMessageEnd → StateSnapshot → RunFinished.
  - No local LLM call — all intelligence is in the Platform.
  - Subclasses can override ``_extra_state()`` to merge App-specific state.
  - Registration: Use ``@copilot_agent("aiknow_agent")`` OR instantiate directly.

Usage (App backend)::

    # Option A: decorator (auto-registers with SDK)
    @copilot_agent("aiknow_agent")
    class AIKnowAgent(AIKnowConversationAgent):
        platform_url = "http://platform:8000"

    # Option B: standalone (no decorator needed in simple apps)
    agent = AIKnowConversationAgent(
        name="aiknow_agent",
        platform_url=os.environ["PLATFORM_URL"],
        tenant_id=os.environ["TENANT_ID"],
    )
    app.add_route("/agents/aiknow_agent", agent.handle_agui_request)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class AIKnowConversationAgent:
    """AG-UI bridge agent that proxies chat to the Platform ConversationAPI.

    Emits the full AG-UI event sequence on each turn:
        RunStarted
        TextMessageStarted
        TextMessageContent  (full text, not streamed — Platform is synchronous)
        TextMessageEnd
        StateSnapshot       (workflowContext + uiHints + sessionState)
        RunFinished

    Args:
        name:            Agent name registered in CopilotKit (default "aiknow_agent").
        platform_url:    Base URL of the Platform API (e.g. "http://platform:8000").
                         Reads from AIKNOW_PLATFORM_URL env var if not set.
        tenant_id:       Platform tenant identifier.
                         Reads from AIKNOW_TENANT_ID env var if not set.
        timeout:         HTTP timeout for Platform requests (default 30s).
        channel:         Default channel name (default "web").
    """

    name: str = "aiknow_agent"
    platform_url: str = ""
    tenant_id: str = ""
    channel: str = "web"
    timeout: float = 30.0

    def __init__(
        self,
        name: str | None = None,
        platform_url: str | None = None,
        tenant_id: str | None = None,
        timeout: float | None = None,
        channel: str | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if platform_url is not None:
            self.platform_url = platform_url
        if tenant_id is not None:
            self.tenant_id = tenant_id
        if timeout is not None:
            self.timeout = timeout
        if channel is not None:
            self.channel = channel

        # Resolve from env vars if still unset
        if not self.platform_url:
            self.platform_url = os.environ.get("AIKNOW_PLATFORM_URL", "http://localhost:8000")
        if not self.tenant_id:
            self.tenant_id = os.environ.get("AIKNOW_TENANT_ID", "default")

    # ------------------------------------------------------------------
    # Public interface — FastAPI / Starlette handler
    # ------------------------------------------------------------------

    async def handle_agui_request(self, body: dict[str, Any]) -> Any:
        """Handle an AG-UI request and return a StreamingResponse.

        Args:
            body: Parsed AG-UI request JSON (messages, state, threadId, runId, etc.).

        Returns:
            starlette.responses.StreamingResponse with text/event-stream content type.
        """
        from starlette.responses import StreamingResponse

        return StreamingResponse(
            self._stream_agui(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Core SSE generator
    # ------------------------------------------------------------------

    async def _stream_agui(self, body: dict[str, Any]) -> AsyncIterator[str]:
        """Generate AG-UI SSE events for one conversation turn.

        Maps AG-UI request → Platform /conversation → AG-UI events.
        """
        run_id: str = body.get("runId", str(uuid.uuid4()))
        thread_id: str = body.get("threadId", run_id)
        messages: list[dict] = body.get("messages", [])
        incoming_state: dict = body.get("state", {})

        # 1. Extract last user message
        user_message = self._extract_last_user_message(messages)
        if not user_message:
            # No user message → emit empty turn
            yield self._sse("RunStarted", {"runId": run_id, "threadId": thread_id})
            yield self._sse("RunFinished", {"runId": run_id, "threadId": thread_id})
            return

        # 2. Determine session_id (stable per thread)
        session_id = incoming_state.get("session_id") or thread_id

        # 3. Extract extra metadata from incoming state
        caller_id: str | None = incoming_state.get("caller_id")
        locale: str = incoming_state.get("locale", "vi")

        # 4. RunStarted
        yield self._sse("RunStarted", {"runId": run_id, "threadId": thread_id})

        # 5. Call Platform /conversation
        try:
            turn = await self._call_platform(
                session_id=session_id,
                message=user_message,
                locale=locale,
                caller_id=caller_id,
            )
        except Exception as exc:
            logger.error(
                "AIKnowConversationAgent: Platform call failed [session=%s]: %s",
                session_id, exc,
            )
            turn = {
                "session_id": session_id,
                "turn_id": str(uuid.uuid4()),
                "message": f"Xin lỗi, không thể kết nối với hệ thống. ({exc})",
                "session_state": "idle",
                "workflow_context": None,
                "ui_hints": None,
            }

        # 6. Emit text message
        msg_id = str(uuid.uuid4())
        reply_text: str = turn.get("message", "")
        yield self._sse("TextMessageStarted", {"messageId": msg_id, "role": "assistant"})
        if reply_text:
            yield self._sse("TextMessageContent", {"messageId": msg_id, "delta": reply_text})
        yield self._sse("TextMessageEnd", {"messageId": msg_id})

        # 7. StateSnapshot — carry full workflow state for frontend
        agent_state = self._build_agent_state(turn, session_id, incoming_state)
        agent_state.update(self._extra_state(turn, incoming_state))
        yield self._sse("StateSnapshot", {"state": agent_state})

        # 8. RunFinished
        yield self._sse("RunFinished", {"runId": run_id, "threadId": thread_id})

    # ------------------------------------------------------------------
    # Platform API call
    # ------------------------------------------------------------------

    async def _call_platform(
        self,
        session_id: str,
        message: str,
        locale: str = "vi",
        caller_id: str | None = None,
    ) -> dict:
        """POST to Platform /api/v1/conversation.

        Args:
            session_id: Stable session identifier (= CopilotKit threadId).
            message:    User's message text.
            locale:     BCP-47 locale.
            caller_id:  Optional caller identification.

        Returns:
            Parsed ConversationTurn dict from Platform.

        Raises:
            httpx.HTTPStatusError: If Platform returns 4xx/5xx.
            httpx.ConnectError:    If Platform is unreachable.
        """
        url = f"{self.platform_url.rstrip('/')}/api/v1/conversation"
        payload = {
            "session_id": session_id,
            "message": message,
            "channel": self.channel,
            "locale": locale,
        }
        if caller_id:
            payload["caller_id"] = caller_id

        headers = {
            "Content-Type": "application/json",
            "X-Tenant-Id": self.tenant_id,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # State building
    # ------------------------------------------------------------------

    def _build_agent_state(
        self,
        turn: dict,
        session_id: str,
        incoming_state: dict,
    ) -> dict:
        """Build the AG-UI StateSnapshot state dict from a ConversationTurn.

        The state is what ``useCoAgent<AIKnowAgentState>()`` sees in React.

        Schema::

            {
              session_id: str,
              session_state: "idle" | "in_workflow" | "in_conversation",
              workflow_context: WorkflowContext | null,
              ui_hints: UIHints | null,
            }

        Args:
            turn:           Parsed ConversationTurn from Platform.
            session_id:     Current session identifier.
            incoming_state: Previous state from incoming AG-UI request.

        Returns:
            State dict suitable for StateSnapshot event.
        """
        return {
            "session_id": session_id,
            "session_state": turn.get("session_state", "idle"),
            "workflow_context": turn.get("workflow_context"),
            "ui_hints": turn.get("ui_hints"),
        }

    def _extra_state(
        self,
        turn: dict,
        incoming_state: dict,
    ) -> dict:
        """Hook for subclasses to merge App-specific state into StateSnapshot.

        Override this to add custom fields to the agent state.

        Example::

            class MyAgent(AIKnowConversationAgent):
                def _extra_state(self, turn, incoming_state):
                    return {"branch_id": incoming_state.get("branch_id", "HN-001")}

        Args:
            turn:           ConversationTurn from Platform.
            incoming_state: Previous state from incoming AG-UI body.

        Returns:
            Dict merged into StateSnapshot state (shallow merge).
        """
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_user_message(messages: list[dict]) -> str:
        """Return the content of the last user role message in the thread.

        Args:
            messages: AG-UI messages list (role + content).

        Returns:
            Content string of the last user message, or "" if none found.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # AG-UI multi-part content
                    parts = [
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    return " ".join(parts).strip()
                return str(content).strip()
        return ""

    @staticmethod
    def _sse(event_type: str, data: dict[str, Any]) -> str:
        """Format a single AG-UI SSE event.

        Args:
            event_type: AG-UI event type string.
            data:       Event payload dict.

        Returns:
            SSE-formatted string ending with double newline.
        """
        payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
        return f"data: {payload}\n\n"
