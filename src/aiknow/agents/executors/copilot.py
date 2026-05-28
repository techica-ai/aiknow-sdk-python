"""
agents/executors/copilot.py — CopilotAgentExecutor.

Handles AG-UI protocol requests for @copilot_agent decorated classes.

AG-UI Protocol:
    - Request:  POST /agents/{name}  with JSON body {messages, state, ...}
    - Response: SSE stream with AG-UI events

AG-UI Event Sequence:
    RunStarted
    TextMessageStarted
    TextMessageContent  (streamed chunks)
    TextMessageEnd
    [ToolCallStarted → ToolCallArgs → ToolCallEnd → ToolCallResult]  (0..N)
    StateSnapshot  (updated agent state)
    RunFinished

LLM calls route through LiteLLM Gateway (NOT directly to provider).
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from aiknow.agents.executors.base import BaseAgentExecutor

logger = logging.getLogger(__name__)


class CopilotAgentExecutor(BaseAgentExecutor):
    """Executor for ``@copilot_agent`` decorated classes.

    Implements ReAct loop with tool calling over LiteLLM SSE streaming.
    Emits AG-UI protocol events.
    """

    async def handle_agui_request(self, body: dict[str, Any]) -> Any:
        """Handle an AG-UI request and return a StreamingResponse.

        Args:
            body: Parsed AG-UI request JSON (messages, state, runId, etc.).

        Returns:
            StreamingResponse with SSE content-type.
        """
        from starlette.responses import StreamingResponse

        ctx = self._extract_context(body)
        messages: list[dict] = body.get("messages", [])
        run_id: str = body.get("runId", str(uuid.uuid4()))

        return StreamingResponse(
            self._stream_agui(run_id, messages, ctx, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_agui(
        self,
        run_id: str,
        messages: list[dict],
        ctx: Any,
        body: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Core AG-UI SSE event generator.

        Performs:
        1. Build system prompt (static + context providers)
        2. Emit RunStarted
        3. ReAct loop:
           a. Call LLM (streaming)
           b. Emit TextMessageContent chunks
           c. If tool_calls: execute tools, emit ToolCall events, continue
           d. If no tool_calls: done
        4. Emit StateSnapshot with updated state
        5. Emit RunFinished
        """
        msg_id = str(uuid.uuid4())
        thread_id = body.get("threadId", run_id)

        # ── 1. Build system prompt ──────────────────────────────────────
        system_prompt = await self._runtime.build_system_prompt(self.meta, ctx)

        # ── 2. RunStarted ───────────────────────────────────────────────
        yield self._sse_event("RunStarted", {"runId": run_id, "threadId": thread_id})

        # ── 3. Build LLM messages ───────────────────────────────────────
        llm_messages: list[dict] = []
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})

        # Convert AG-UI messages to OpenAI format
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                llm_messages.append({"role": role, "content": content})

        tool_schemas = self._resolve_tools_schemas()

        # ── 4. ReAct loop ───────────────────────────────────────────────
        full_text = ""
        max_steps = self.meta.max_steps

        for step in range(max_steps):
            text_chunks: list[str] = []
            tool_calls_raw: list[dict] = []
            is_first_chunk = True

            try:
                async for chunk, tool_calls in self._llm_stream(
                    llm_messages, tool_schemas
                ):
                    if chunk:
                        if is_first_chunk:
                            yield self._sse_event("TextMessageStarted", {
                                "messageId": msg_id, "role": "assistant"
                            })
                            is_first_chunk = False
                        yield self._sse_event("TextMessageContent", {
                            "messageId": msg_id, "delta": chunk
                        })
                        text_chunks.append(chunk)
                    if tool_calls:
                        tool_calls_raw.extend(tool_calls)

            except Exception as exc:  # noqa: BLE001
                logger.error("CopilotAgentExecutor: LLM call failed: %s", exc)
                if is_first_chunk:
                    yield self._sse_event("TextMessageStarted", {
                        "messageId": msg_id, "role": "assistant"
                    })
                err_text = f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. ({exc})"
                yield self._sse_event("TextMessageContent", {
                    "messageId": msg_id, "delta": err_text
                })
                text_chunks.append(err_text)
                yield self._sse_event("TextMessageEnd", {"messageId": msg_id})
                break

            # Close text message if we emitted any
            if text_chunks:
                full_text = "".join(text_chunks)
                yield self._sse_event("TextMessageEnd", {"messageId": msg_id})
                llm_messages.append({"role": "assistant", "content": full_text})

            # ── Tool calls ──────────────────────────────────────────────
            if not tool_calls_raw:
                break  # No tools → done

            tool_results: list[dict] = []
            for tc in tool_calls_raw:
                tc_id = tc.get("id", str(uuid.uuid4()))
                tc_name = tc.get("function", {}).get("name", "")
                tc_args_str = tc.get("function", {}).get("arguments", "{}")

                yield self._sse_event("ToolCallStarted", {
                    "toolCallId": tc_id, "toolCallName": tc_name
                })
                yield self._sse_event("ToolCallArgs", {
                    "toolCallId": tc_id, "delta": tc_args_str
                })

                try:
                    tc_args = json.loads(tc_args_str) if tc_args_str else {}
                except json.JSONDecodeError:
                    tc_args = {}

                result_str = await self._call_tool(ctx, tc_name, tc_args)

                yield self._sse_event("ToolCallEnd", {"toolCallId": tc_id})
                yield self._sse_event("ToolCallResult", {
                    "toolCallId": tc_id,
                    "toolCallName": tc_name,
                    "result": result_str,
                })

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

            # Add assistant message with tool_calls and tool results
            llm_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_raw,
            })
            llm_messages.extend(tool_results)

            # Reset msg_id for next assistant message
            msg_id = str(uuid.uuid4())

        # ── 5. StateSnapshot ────────────────────────────────────────────
        yield self._sse_event("StateSnapshot", {
            "state": {
                "tenant_id": ctx.tenant_id,
                "session_id": ctx.session_id,
                "execution_id": ctx.execution_id,
                "workflow_id": ctx.workflow_id,
                "current_state": ctx.current_state,
                "execution_status": ctx.execution_status,
            }
        })

        # ── 6. RunFinished ──────────────────────────────────────────────
        yield self._sse_event("RunFinished", {"runId": run_id, "threadId": thread_id})

    async def _llm_stream(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
    ) -> AsyncIterator[tuple[str, list[dict]]]:
        """Stream from LiteLLM Gateway.

        Yields:
            (text_chunk, tool_calls) tuples.
            text_chunk is empty when tool_calls is non-empty and vice versa.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for CopilotAgentExecutor") from exc

        model = self.meta.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"

        collected_tool_calls: dict[int, dict] = {}
        finish_reason: str | None = None

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self._llm_gateway_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Text content
                    content = delta.get("content")
                    if content:
                        yield (content, [])

                    # Tool calls (streaming — accumulate)
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        existing = collected_tool_calls[idx]
                        fn_chunk = tc.get("function", {})
                        if fn_chunk.get("name"):
                            existing["function"]["name"] += fn_chunk["name"]
                        if fn_chunk.get("arguments"):
                            existing["function"]["arguments"] += fn_chunk["arguments"]
                        if tc.get("id"):
                            existing["id"] = tc["id"]

        if collected_tool_calls:
            yield ("", list(collected_tool_calls.values()))

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sse_event(event_type: str, data: dict[str, Any]) -> str:
        """Format a single AG-UI SSE event.

        Args:
            event_type: AG-UI event type string.
            data:       Event payload dict.

        Returns:
            SSE-formatted string (ending with double newline).
        """
        payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
        return f"data: {payload}\n\n"
