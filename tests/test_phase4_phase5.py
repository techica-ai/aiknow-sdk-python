"""Unit tests for Phase 4 (@sop triggers) and Phase 5 (AIKnowConversationAgent).

Tests:
    Phase 4:
        - @sop with triggers=dict → WorkflowTriggers in sop_meta
        - @sop with triggers=WorkflowTriggers instance → passed through
        - @sop(triggers=None) → no triggers (existing behaviour unchanged)
        - @sop triggers propagated to cls.triggers attribute
    Phase 5:
        - AIKnowConversationAgent._extract_last_user_message
        - AIKnowConversationAgent._build_agent_state
        - AIKnowConversationAgent._sse formatting
        - AIKnowConversationAgent._extra_state hook
        - handle_agui_request: returns StreamingResponse
        - Full _stream_agui: happy-path with mocked platform
"""
from __future__ import annotations

import json
import pytest

from aiknow_contracts.workflow import WorkflowTriggers


# ---------------------------------------------------------------------------
# Phase 4: @sop triggers
# ---------------------------------------------------------------------------

class TestSopTriggers:
    def setup_method(self):
        """Use a fresh local registry per test to avoid cross-test pollution."""
        from aiknow.extensions._local_registry import LocalExtensionRegistry
        self._registry = LocalExtensionRegistry()

    def _make_sop(self, triggers=None, description=None):
        """Create a @sop-decorated class using the decorator directly."""
        from aiknow.extensions import _decorators as d
        orig_registry = d._default_registry
        d._default_registry = self._registry
        try:
            @d.sop(
                "refund_flow",
                initial_state="verify_customer",
                triggers=triggers,
                description=description,
            )
            class RefundFlow:
                transitions = {
                    "verify_customer": {"ok": "process_refund", "fail": "__end__"},
                    "process_refund": {"done": "__end__"},
                }
        finally:
            d._default_registry = orig_registry
        return RefundFlow

    def test_sop_no_triggers(self):
        """@sop without triggers → sop_meta.triggers is None."""
        cls = self._make_sop(triggers=None)
        assert cls._sop_meta["triggers"] is None
        assert not hasattr(cls, "triggers")  # not propagated

    def test_sop_triggers_dict_stored_in_meta(self):
        """@sop with triggers=dict → stored as dict in sop_meta."""
        triggers_dict = {
            "keywords": {"vi": ["đổi trả", "hoàn tiền"]},
            "description": {"vi": "Xử lý hoàn tiền"},
            "priority": 5,
        }
        cls = self._make_sop(triggers=triggers_dict)
        assert cls._sop_meta["triggers"] == triggers_dict

    def test_sop_triggers_propagated_to_class(self):
        """@sop with triggers → cls.triggers set."""
        triggers_dict = {"keywords": {"vi": ["hoàn tiền"]}}
        cls = self._make_sop(triggers=triggers_dict)
        assert hasattr(cls, "triggers")
        assert cls.triggers == triggers_dict

    def test_sop_triggers_workflowtriggers_passthrough(self):
        """@sop with triggers=WorkflowTriggers instance → stored as-is."""
        triggers_obj = WorkflowTriggers(
            keywords={"vi": ["khiếu nại"]},
            priority=10,
        )
        cls = self._make_sop(triggers=triggers_obj)
        assert cls._sop_meta["triggers"] is triggers_obj

    def test_sop_description_stored(self):
        """@sop description param stored in sop_meta."""
        cls = self._make_sop(description={"vi": "Hoàn tiền"})
        assert cls._sop_meta["description"] == {"vi": "Hoàn tiền"}

    def test_sop_without_triggers_backward_compat(self):
        """@sop without triggers/description works exactly as before."""
        from aiknow.extensions import _decorators as d
        orig_registry = d._default_registry
        d._default_registry = self._registry
        try:
            @d.sop("legacy_flow", initial_state="start")
            class LegacyFlow:
                transitions = {"start": {"done": "__end__"}}
        finally:
            d._default_registry = orig_registry

        assert LegacyFlow._sop_meta["id"] == "legacy_flow"
        assert LegacyFlow._sop_meta.get("triggers") is None


# ---------------------------------------------------------------------------
# Phase 5: AIKnowConversationAgent
# ---------------------------------------------------------------------------

class TestAIKnowConversationAgentHelpers:
    def _agent(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        return AIKnowConversationAgent(
            name="test_agent",
            platform_url="http://platform:8000",
            tenant_id="acme",
        )

    def test_extract_last_user_message_simple(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        messages = [
            {"role": "user", "content": "xin chào"},
            {"role": "assistant", "content": "Chào bạn!"},
            {"role": "user", "content": "đặt lịch khám"},
        ]
        result = AIKnowConversationAgent._extract_last_user_message(messages)
        assert result == "đặt lịch khám"

    def test_extract_last_user_message_multipart(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "cho tôi"},
                    {"type": "text", "text": "hẹn khám"},
                ],
            }
        ]
        result = AIKnowConversationAgent._extract_last_user_message(messages)
        assert result == "cho tôi hẹn khám"

    def test_extract_last_user_message_empty(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        result = AIKnowConversationAgent._extract_last_user_message([])
        assert result == ""

    def test_extract_no_user_message(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        messages = [{"role": "assistant", "content": "Chào!"}]
        result = AIKnowConversationAgent._extract_last_user_message(messages)
        assert result == ""

    def test_sse_format(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        sse = AIKnowConversationAgent._sse("RunStarted", {"runId": "r1", "threadId": "t1"})
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[6:])
        assert payload["type"] == "RunStarted"
        assert payload["runId"] == "r1"

    def test_build_agent_state(self):
        agent = self._agent()
        turn = {
            "session_state": "in_workflow",
            "workflow_context": {"workflow_id": "book_appointment", "status": "waiting_human"},
            "ui_hints": {"input_type": "text"},
        }
        state = agent._build_agent_state(turn, "sess-1", {})
        assert state["session_state"] == "in_workflow"
        assert state["workflow_context"]["workflow_id"] == "book_appointment"
        assert state["ui_hints"]["input_type"] == "text"
        assert state["session_id"] == "sess-1"

    def test_extra_state_hook_empty(self):
        agent = self._agent()
        extra = agent._extra_state({}, {})
        assert extra == {}

    def test_extra_state_subclass_hook(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent

        class MyAgent(AIKnowConversationAgent):
            def _extra_state(self, turn, incoming_state):
                return {"branch_id": "HN-001"}

        agent = MyAgent(platform_url="http://p:8000", tenant_id="t")
        extra = agent._extra_state({}, {})
        assert extra == {"branch_id": "HN-001"}

    def test_env_var_platform_url(self, monkeypatch):
        monkeypatch.setenv("AIKNOW_PLATFORM_URL", "http://env-platform:9000")
        monkeypatch.setenv("AIKNOW_TENANT_ID", "env-tenant")
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        agent = AIKnowConversationAgent()
        assert agent.platform_url == "http://env-platform:9000"
        assert agent.tenant_id == "env-tenant"


class TestAIKnowConversationAgentStreaming:
    """Integration-level tests for _stream_agui with mocked Platform."""

    @pytest.mark.asyncio
    async def test_stream_empty_messages_emits_minimal_events(self):
        from aiknow.agents.conversation_agent import AIKnowConversationAgent
        agent = AIKnowConversationAgent(platform_url="http://p:8000", tenant_id="t")

        body = {"runId": "r1", "threadId": "t1", "messages": [], "state": {}}
        events = []
        async for chunk in agent._stream_agui(body):
            events.append(json.loads(chunk[6:]))

        types = [e["type"] for e in events]
        assert types == ["RunStarted", "RunFinished"]

    @pytest.mark.asyncio
    async def test_stream_happy_path_mocked_platform(self, monkeypatch):
        """Full happy-path: user message → Platform mock → SSE events."""
        import httpx
        from aiknow.agents.conversation_agent import AIKnowConversationAgent

        platform_response = {
            "session_id": "t1",
            "turn_id": "turn-1",
            "message": "Vui lòng cho biết tên của bạn?",
            "session_state": "in_workflow",
            "workflow_context": {
                "workflow_id": "book_appointment",
                "workflow_label": "book_appointment",
                "execution_id": "exec-1",
                "status": "waiting_human",
                "current_slot": "customer_name",
                "slots": {},
            },
            "ui_hints": {"input_type": "text", "options": [], "input_disabled": False},
        }

        # Mock httpx.AsyncClient.post
        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return platform_response

        class MockClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def post(self, *args, **kwargs): return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: MockClient())

        agent = AIKnowConversationAgent(platform_url="http://p:8000", tenant_id="t")
        body = {
            "runId": "r1",
            "threadId": "t1",
            "messages": [{"role": "user", "content": "đặt lịch khám"}],
            "state": {},
        }

        events = []
        async for chunk in agent._stream_agui(body):
            events.append(json.loads(chunk[6:]))

        types = [e["type"] for e in events]
        assert types == [
            "RunStarted",
            "TextMessageStarted",
            "TextMessageContent",
            "TextMessageEnd",
            "StateSnapshot",
            "RunFinished",
        ]

        # TextMessageContent carries the reply
        content_event = next(e for e in events if e["type"] == "TextMessageContent")
        assert "tên" in content_event["delta"]

        # StateSnapshot carries workflowContext
        snapshot = next(e for e in events if e["type"] == "StateSnapshot")
        assert snapshot["state"]["session_state"] == "in_workflow"
        assert snapshot["state"]["workflow_context"]["workflow_id"] == "book_appointment"
        assert snapshot["state"]["ui_hints"]["input_type"] == "text"

    @pytest.mark.asyncio
    async def test_stream_platform_error_graceful(self, monkeypatch):
        """Platform call failure → error text emitted, no crash."""
        import httpx
        from aiknow.agents.conversation_agent import AIKnowConversationAgent

        class MockClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def post(self, *args, **kwargs):
                raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: MockClient())

        agent = AIKnowConversationAgent(platform_url="http://p:8000", tenant_id="t")
        body = {
            "runId": "r1",
            "threadId": "t1",
            "messages": [{"role": "user", "content": "xin chào"}],
            "state": {},
        }

        events = []
        async for chunk in agent._stream_agui(body):
            events.append(json.loads(chunk[6:]))

        types = [e["type"] for e in events]
        # Must still emit all required events
        assert "RunStarted" in types
        assert "TextMessageContent" in types
        assert "StateSnapshot" in types
        assert "RunFinished" in types

        # Error message should contain helpful text
        content = next(e for e in events if e["type"] == "TextMessageContent")
        assert "kết nối" in content["delta"] or "lỗi" in content["delta"]

    @pytest.mark.asyncio
    async def test_session_id_from_state_preferred(self, monkeypatch):
        """If incoming state has session_id, use it (not threadId)."""
        import httpx
        from aiknow.agents.conversation_agent import AIKnowConversationAgent

        captured = {}

        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return {
                "session_id": "stable-sess",
                "turn_id": "t1",
                "message": "ok",
                "session_state": "idle",
            }

        class MockClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def post(self, url, json=None, **kwargs):
                captured["payload"] = json
                return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: MockClient())

        agent = AIKnowConversationAgent(platform_url="http://p:8000", tenant_id="t")
        body = {
            "runId": "r1",
            "threadId": "thread-xyz",
            "messages": [{"role": "user", "content": "hi"}],
            "state": {"session_id": "stable-sess"},
        }

        async for _ in agent._stream_agui(body):
            pass

        assert captured["payload"]["session_id"] == "stable-sess"
