"""
Tests for SDK-3: Copilot + ReAct agent executors.

Coverage:
    - ReActAgentExecutor.chat(): text-only, single tool, multi-step, max_steps
    - ReActAgentExecutor._llm_call: error handling
    - CopilotAgentExecutor._sse_event: event formatting
    - CopilotAgentExecutor._stream_agui: basic SSE flow (text-only + error)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiknow.agents.base import AgentContext, AgentMeta
from aiknow.agents.executors.copilot import CopilotAgentExecutor
from aiknow.agents.executors.react import ReActAgentExecutor

# ---------------------------------------------------------------------------
# Fixtures — minimal mock objects to construct executors
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_meta() -> AgentMeta:
    return AgentMeta(
        name="test-agent",
        agent_type="rag",
        model="test-model",
        system_prompt="You are a helpful assistant.",
        tools=[],
        builtin_tools=[],
        max_steps=5,
    )


@pytest.fixture
def mock_tool_registry() -> MagicMock:
    reg = MagicMock()
    reg.get = MagicMock(return_value=None)
    reg.get_schema = MagicMock(return_value=None)
    return reg


@pytest.fixture
def mock_runtime() -> MagicMock:
    rt = MagicMock()
    rt.build_system_prompt = AsyncMock(return_value="System prompt")
    return rt


@pytest.fixture
def react_executor(
    mock_meta: AgentMeta,
    mock_tool_registry: MagicMock,
    mock_runtime: MagicMock,
) -> ReActAgentExecutor:
    return ReActAgentExecutor(
        cls=type("FakeAgent", (), {}),
        meta=mock_meta,
        tool_registry=mock_tool_registry,
        llm_gateway_url="http://mock-litellm:4000",
        runtime=mock_runtime,
    )


@pytest.fixture
def copilot_executor(
    mock_meta: AgentMeta,
    mock_tool_registry: MagicMock,
    mock_runtime: MagicMock,
) -> CopilotAgentExecutor:
    return CopilotAgentExecutor(
        cls=type("FakeAgent", (), {}),
        meta=mock_meta,
        tool_registry=mock_tool_registry,
        llm_gateway_url="http://mock-litellm:4000",
        runtime=mock_runtime,
    )


def _mock_llm_response(
    content: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    """Build a mock LiteLLM /v1/chat/completions response."""
    message: dict = {"role": "assistant"}
    if content:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {"choices": [{"message": message}]}


# ---------------------------------------------------------------------------
# ReActAgentExecutor.chat() — text-only response
# ---------------------------------------------------------------------------

class TestReActTextOnly:
    """ReAct loop with no tool calls — returns text immediately."""

    @pytest.mark.asyncio
    async def test_text_only_response(self, react_executor):
        """LLM returns text only → no tool calls → return text."""
        with patch.object(
            react_executor, "_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = _mock_llm_response("Hello! How can I help?")
            result = await react_executor.chat("hi")
            assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(
        self, react_executor
    ):
        """LLM returns empty content, no tools → empty string."""
        with patch.object(
            react_executor, "_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = _mock_llm_response("")
            result = await react_executor.chat("hi")
            assert result == ""


# ---------------------------------------------------------------------------
# ReActAgentExecutor.chat() — tool calling
# ---------------------------------------------------------------------------

class TestReActToolCalling:
    """ReAct loop with tool calls."""

    @pytest.mark.asyncio
    async def test_single_tool_then_text(self, react_executor):
        """LLM → tool_call → tool_result → LLM → final text."""
        tool_call = {
            "id": "tc-1",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": "{}",
            },
        }
        with patch.object(
            react_executor, "_llm_call", new_callable=AsyncMock
        ) as mock_llm, patch.object(
            react_executor, "_call_tool", new_callable=AsyncMock
        ) as mock_tool:
            # First call: tool call, second call: final text
            mock_llm.side_effect = [
                _mock_llm_response(tool_calls=[tool_call]),
                _mock_llm_response("It is 3 PM."),
            ]
            mock_tool.return_value = "15:00"

            result = await react_executor.chat("What time is it?")

            assert result == "It is 3 PM."
            assert mock_llm.await_count == 2
            mock_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_step_tool_loop(self, react_executor):
        """Two sequential tool calls before final answer."""
        tc1 = {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        tc2 = {"id": "t2", "type": "function", "function": {"name": "format", "arguments": "{}"}}

        with patch.object(
            react_executor, "_llm_call", new_callable=AsyncMock
        ) as mock_llm, patch.object(
            react_executor, "_call_tool", new_callable=AsyncMock
        ) as mock_tool:
            mock_llm.side_effect = [
                _mock_llm_response(tool_calls=[tc1]),
                _mock_llm_response(tool_calls=[tc2]),
                _mock_llm_response("Final answer."),
            ]
            mock_tool.return_value = "data"

            result = await react_executor.chat("complex query")
            assert result == "Final answer."
            assert mock_llm.await_count == 3

    @pytest.mark.asyncio
    async def test_max_steps_exhaustion(self, react_executor):
        """Exceeding max_steps → returns '[Max steps reached]'."""
        # Force tool call on every step → never resolves
        tc = {"id": "t", "type": "function", "function": {"name": "loop", "arguments": "{}"}}

        with patch.object(
            react_executor, "_llm_call", new_callable=AsyncMock
        ) as mock_llm, patch.object(
            react_executor, "_call_tool", new_callable=AsyncMock
        ) as mock_tool:
            mock_llm.return_value = _mock_llm_response(tool_calls=[tc])
            mock_tool.return_value = "looping"

            result = await react_executor.chat("loop forever")
            assert "Max steps" in result


# ---------------------------------------------------------------------------
# ReActAgentExecutor._llm_call — error handling
# ---------------------------------------------------------------------------

class TestReActLLMCall:
    """_llm_call HTTP wrapper."""

    @pytest.mark.asyncio
    async def test_llm_call_returns_dict(self, react_executor):
        """Successful LLM call → returns response dict."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await react_executor._llm_call([], [])
            assert isinstance(result, dict)
            assert "choices" in result

    @pytest.mark.asyncio
    async def test_llm_call_raises_on_http_error(self, react_executor):
        """HTTP error from LiteLLM → raise_for_status propagates."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await react_executor._llm_call([], [])


# ---------------------------------------------------------------------------
# CopilotAgentExecutor — SSE event helpers
# ---------------------------------------------------------------------------

class TestCopilotSSEEvent:
    """CopilotAgentExecutor._sse_event static method."""

    def test_sse_event_format(self):
        """SSE event is properly formatted as data: JSON\\n\\n."""
        result = CopilotAgentExecutor._sse_event("RunStarted", {"runId": "r1"})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        payload = json.loads(result[6:].strip())
        assert payload["type"] == "RunStarted"
        assert payload["runId"] == "r1"

    def test_sse_event_with_special_chars(self):
        """Unicode content in SSE event."""
        result = CopilotAgentExecutor._sse_event(
            "TextMessageContent",
            {"delta": "Xin chào! 🎉"},
        )
        payload = json.loads(result[6:].strip())
        assert "Xin chào" in payload["delta"]


# ---------------------------------------------------------------------------
# CopilotAgentExecutor — streaming flow (text-only)
# ---------------------------------------------------------------------------

class TestCopilotStreaming:
    """CopilotAgentExecutor._stream_agui basic flow."""

    @pytest.mark.asyncio
    async def test_text_only_stream(self, copilot_executor):
        """LLM returns text → RunStarted, TextMessage*, RunFinished events."""
        # Mock _llm_stream to yield a single text chunk
        async def mock_stream(msgs, schemas):
            yield ("Hello!", [])

        with patch.object(
            copilot_executor, "_llm_stream", side_effect=mock_stream
        ):
            events = []
            async for event in copilot_executor._stream_agui(
                "run-1",
                [{"role": "user", "content": "hi"}],
                AgentContext(tenant_id="t1"),
                {},
            ):
                events.append(event)

        event_types = [json.loads(e[6:].strip())["type"] for e in events]
        assert "RunStarted" in event_types
        assert "TextMessageStarted" in event_types
        assert "TextMessageContent" in event_types
        assert "TextMessageEnd" in event_types
        assert "RunFinished" in event_types

    @pytest.mark.asyncio
    async def test_llm_error_emits_error_text(self, copilot_executor):
        """LLM raises mid-stream → error text in TextMessageContent."""
        async def mock_stream(msgs, schemas):
            raise RuntimeError("LLM is down")
            yield  # type: ignore[unreachable]

        with patch.object(
            copilot_executor, "_llm_stream", side_effect=mock_stream
        ):
            events = []
            async for event in copilot_executor._stream_agui(
                "run-err",
                [{"role": "user", "content": "hi"}],
                AgentContext(tenant_id="t1"),
                {},
            ):
                events.append(event)

        # Should still get RunFinished
        event_types = [json.loads(e[6:].strip())["type"] for e in events]
        assert "RunFinished" in event_types
        # Error message should be in one of the TextMessageContent events
        content_events = [
            e for e in events if "TextMessageContent" in e
        ]
        if content_events:
            payload = json.loads(content_events[0][6:].strip())
            assert "lỗi" in payload["delta"].lower() or "error" in payload["delta"].lower()
