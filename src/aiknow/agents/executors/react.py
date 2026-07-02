"""
agents/executors/react.py — ReActAgentExecutor.

General-purpose executor for rag, autonomous, and conversation agents.
Implements ReAct (Reason + Act) loop via synchronous LLM calls (non-streaming REST).

Use this executor when:
    - You want a REST API (not SSE streaming)
    - Agent type is rag, autonomous, or conversation
    - You need a simpler request/response pattern

For streaming AG-UI protocol use CopilotAgentExecutor instead.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, cast

from aiknow.agents.executors.base import BaseAgentExecutor

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class ReActAgentExecutor(BaseAgentExecutor):
    """Executor for ``rag``, ``autonomous``, and ``conversation`` agent types.

    Provides:
    - ``handle_agui_request()`` — AG-UI SSE endpoint (delegates to CopilotAgentExecutor)
    - ``chat()`` — Plain REST endpoint, returns full JSON response
    - ``chat_stream()`` — REST endpoint with streaming chunks

    Note:
        For simplicity, ``handle_agui_request()`` delegates to the CopilotAgentExecutor
        so all agent types can serve the AG-UI protocol. This avoids code duplication.
    """

    async def handle_agui_request(self, body: dict[str, Any]) -> Any:
        """Handle AG-UI protocol request (same as CopilotAgentExecutor).

        Delegates to CopilotAgentExecutor for SSE streaming.
        """
        from aiknow.agents.executors.copilot import CopilotAgentExecutor  # noqa: PLC0415

        copilot_executor = CopilotAgentExecutor(
            cls=self._cls,
            meta=self.meta,
            tool_registry=self._tool_registry,
            llm_gateway_url=self._llm_gateway_url,
            runtime=self._runtime,
        )
        return await copilot_executor.handle_agui_request(body)

    async def chat(
        self,
        message: str,
        *,
        ctx: Any = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Single-turn chat — returns full text response.

        Useful for programmatic agent calls without streaming.

        Args:
            message: User message.
            ctx:     Optional AgentContext. Creates empty one if not provided.
            history: Prior conversation history in OpenAI format.

        Returns:
            Full assistant text response.
        """
        from aiknow.agents.base import AgentContext  # noqa: PLC0415

        if ctx is None:
            ctx = AgentContext(tenant_id="")

        system_prompt = await self._runtime.build_system_prompt(self.meta, ctx)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in (history or []):
            messages.append(m)
        messages.append({"role": "user", "content": message})

        tool_schemas = self._resolve_tools_schemas()

        # ReAct loop
        for _ in range(self.meta.max_steps):
            response_data = await self._llm_call(messages, tool_schemas)
            choice = (response_data.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                return content or ""

            messages.append(msg)
            for tc in tool_calls:
                tc_name = tc.get("function", {}).get("name", "")
                tc_args_str = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", str(uuid.uuid4()))
                try:
                    tc_args = json.loads(tc_args_str)
                except json.JSONDecodeError:
                    tc_args = {}
                result = await self._call_tool(ctx, tc_name, tc_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })

        return "[Max steps reached]"

    async def _llm_call(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Non-streaming LLM call via LiteLLM Gateway.

        Args:
            messages:    Full message history.
            tool_schemas: Tool schemas for function calling.

        Returns:
            Raw OpenAI-format response dict.
        """
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("httpx is required for ReActAgentExecutor") from exc

        payload: dict[str, Any] = {
            "model": self.meta.model,
            "messages": messages,
            "stream": False,
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._llm_gateway_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
