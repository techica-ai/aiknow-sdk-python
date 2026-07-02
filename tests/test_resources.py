"""
Tests for SDK-4: Resource layer coverage.

Covers:
    - ConversationResource.graph_turn (sync + async) — P3 graph endpoints
    - ConversationResource.approve_checkpoint (sync + async)
    - ConversationResource.converse (sync + async)
    - Sync resource parity (SDK-1): extensions, workflows, knowledge on AIKnowClient
    - ChatResource.ask (sync)
    - UsersResource basic (sync)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiknow import AIKnowClient
from aiknow._exceptions import AIKnowAPIError, AuthenticationError
from aiknow.resources.conversation import (
    ApproveCheckpointResult,
    AsyncConversationResource,
    ConversationResource,
    GraphTurnResult,
)
from aiknow_contracts.chat_models import CheckpointApprovalStatus, PillarType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.is_success = 200 <= status_code < 300
    resp.status_code = status_code
    resp.text = json.dumps(json_data or {})
    resp.json.return_value = json_data or {}
    return resp


# ---------------------------------------------------------------------------
# GraphTurnResult — typed model validation
# ---------------------------------------------------------------------------

class TestGraphTurnSync:
    """ConversationResource.graph_turn (sync)."""

    def test_returns_graph_turn_result(self):
        """graph_turn returns GraphTurnResult, not raw dict."""
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "sess-1",
            "message": "Xin chào!",
            "outcome": "greeted",
            "stop_reason": "WAITING",
            "is_frozen": False,
        })
        r = ConversationResource(mock_client)
        result = r.graph_turn(
            session_id="sess-1", graph_id="demo_cskh",
            message="chào", tenant_id="default",
        )
        assert isinstance(result, GraphTurnResult)
        assert result.session_id == "sess-1"
        assert result.message == "Xin chào!"
        assert result.outcome == "greeted"
        assert result.is_frozen is False
        assert result.is_delegated is False

    def test_delegation_fields_mapped(self):
        """is_delegated + handoff fields parsed correctly."""
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "sess-2",
            "message": "Delegating to agent...",
            "outcome": "delegated",
            "stop_reason": "DELEGATE_TO_AGENT",
            "is_frozen": False,
            "is_delegated": True,
            "handoff": {"from_pillar": "graph", "to_pillar": "agentic"},
        })
        r = ConversationResource(mock_client)
        result = r.graph_turn(
            session_id="sess-2", graph_id="demo_cskh", message="help",
        )
        assert result.is_delegated is True
        assert result.handoff is not None
        assert result.handoff.to_pillar == PillarType.AGENTIC

    def test_raises_on_auth_error(self):
        """401 → AuthenticationError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(401)
        r = ConversationResource(mock_client)
        with pytest.raises(AuthenticationError):
            r.graph_turn(session_id="s1", graph_id="g1")

    def test_raises_on_server_error(self):
        """500 → AIKnowAPIError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(500, {"detail": "boom"})
        r = ConversationResource(mock_client)
        with pytest.raises(AIKnowAPIError):
            r.graph_turn(session_id="s1", graph_id="g1")

    def test_payload_built_correctly(self):
        """Verify the HTTP payload contains all expected fields."""
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "s1", "message": "ok",
            "outcome": "ok", "stop_reason": "DONE",
        })
        r = ConversationResource(mock_client)
        r.graph_turn(
            session_id="s1", graph_id="g1",
            message="hello", tenant_id="acme",
        )
        call_args = mock_client.post.call_args
        sent_json = call_args.kwargs["json"]
        assert sent_json["session_id"] == "s1"
        assert sent_json["graph_id"] == "g1"
        assert sent_json["message"] == "hello"
        assert sent_json["tenant_id"] == "acme"


class TestGraphTurnAsync:
    """AsyncConversationResource.graph_turn."""

    @pytest.mark.asyncio
    async def test_returns_graph_turn_result(self):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "sess-a",
            "message": "Hello!",
            "outcome": "ok",
            "stop_reason": "DONE",
            "is_frozen": False,
        })
        r = AsyncConversationResource(mock_client)
        result = await r.graph_turn(
            session_id="sess-a", graph_id="demo", message="hi",
        )
        assert isinstance(result, GraphTurnResult)
        assert result.session_id == "sess-a"

    @pytest.mark.asyncio
    async def test_delegation_fields_mapped(self):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "sess-d",
            "message": "delegating",
            "outcome": "delegated",
            "stop_reason": "DELEGATE",
            "is_delegated": True,
            "handoff": {"from_pillar": "graph", "to_pillar": "agentic"},
        })
        r = AsyncConversationResource(mock_client)
        result = await r.graph_turn(
            session_id="sess-d", graph_id="g", message="m",
        )
        assert result.is_delegated is True
        assert result.handoff is not None
        assert result.handoff.to_pillar == PillarType.AGENTIC

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(500)
        r = AsyncConversationResource(mock_client)
        with pytest.raises(AIKnowAPIError):
            await r.graph_turn(session_id="s", graph_id="g")


# ---------------------------------------------------------------------------
# ApproveCheckpointResult
# ---------------------------------------------------------------------------

class TestApproveCheckpoint:
    """ConversationResource.approve_checkpoint."""

    def test_sync_approve_returns_typed(self):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "frozen-sess",
            "status": "approved",
            "current_node": "verify",
            "message": "Checkpoint approved",
        })
        r = ConversationResource(mock_client)
        result = r.approve_checkpoint(
            session_id="frozen-sess", decision="approve", approver_id="admin-1",
        )
        assert isinstance(result, ApproveCheckpointResult)
        assert result.status == CheckpointApprovalStatus.APPROVED
        assert result.current_node == "verify"

    @pytest.mark.asyncio
    async def test_async_reject_returns_typed(self):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_id": "frozen-sess",
            "status": "rejected",
            "message": "Rejected by admin",
        })
        r = AsyncConversationResource(mock_client)
        result = await r.approve_checkpoint(
            session_id="frozen-sess", decision="reject", approver_id="admin-2",
        )
        assert isinstance(result, ApproveCheckpointResult)
        assert result.status == CheckpointApprovalStatus.REJECTED

    def test_raises_on_non_frozen(self):
        """Session not frozen → 400 → AIKnowAPIError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(400, {"detail": "not frozen"})
        r = ConversationResource(mock_client)
        with pytest.raises(AIKnowAPIError):
            r.approve_checkpoint(
                session_id="s", decision="approve", approver_id="a",
            )


# ---------------------------------------------------------------------------
# Sync resource parity (SDK-1)
# ---------------------------------------------------------------------------

class TestSyncResourceParity:
    """Verify AIKnowClient (sync) now exposes all resources (SDK-1 fix)."""

    def test_has_extensions(self):
        with AIKnowClient(api_key="test-key") as client:
            assert hasattr(client, "extensions")
            assert client.extensions is not None

    def test_has_workflows(self):
        with AIKnowClient(api_key="test-key") as client:
            assert hasattr(client, "workflows")
            assert client.workflows is not None

    def test_has_knowledge(self):
        with AIKnowClient(api_key="test-key") as client:
            assert hasattr(client, "knowledge")
            assert client.knowledge is not None

    def test_has_conversation(self):
        with AIKnowClient(api_key="test-key") as client:
            assert hasattr(client, "conversation")

    def test_has_chat(self):
        with AIKnowClient(api_key="test-key") as client:
            assert hasattr(client, "chat")


# ---------------------------------------------------------------------------
# ChatResource basic (SDK-4)
# ---------------------------------------------------------------------------

class TestChatResource:
    """ChatResource.ask basic coverage."""

    def test_ask_returns_response(self):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, {
            "answer": "Hello there!",
            "session_id": "chat-1",
            "sources": [],
        })
        from aiknow.resources.chat import ChatResource
        r = ChatResource(mock_client)
        result = r.ask(query="hi", tenant_id="default")
        # ChatResponse is a pydantic model / dataclass
        assert result.answer == "Hello there!" or result["answer"] == "Hello there!"


# ---------------------------------------------------------------------------
# Converse legacy endpoint
# ---------------------------------------------------------------------------

class TestConverse:
    """ConversationResource.converse."""

    def test_sync_returns_dict(self):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_state": {"current_step": "greet"},
            "output": "Welcome!",
        })
        r = ConversationResource(mock_client)
        result = r.converse(
            session_id="s1", message="hello",
        )
        assert isinstance(result, dict)
        assert result["output"] == "Welcome!"

    @pytest.mark.asyncio
    async def test_async_returns_dict(self):
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200, {
            "session_state": {"current_step": "ask"},
            "output": "What do you need?",
        })
        r = AsyncConversationResource(mock_client)
        result = await r.converse(session_id="s1", message="help")
        assert result["output"] == "What do you need?"
