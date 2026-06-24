"""
Tests for SDK-2: RouterAgentExecutor — 2-layer intent routing.

Coverage:
    - Keyword scan (Layer 1): match, no-match, priority ordering
    - LLM classification (Layer 2): success, timeout, parse error, low confidence
    - Slot extraction from keyword match
    - RouteDecision structure
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiknow.agents.executors.router import RouterAgentExecutor
from aiknow.agents.routing import IntentSpec, RouteDecision

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_specs() -> list[IntentSpec]:
    return [
        IntentSpec(
            route_id="refund",
            keywords=[r"refund", r"hoàn\s+tiền"],
            llm_description="Customer wants a refund",
            slot_patterns={"order_id": r"ORD-\d+"},
            priority=10,
        ),
        IntentSpec(
            route_id="support",
            keywords=[r"help", r"hỗ\s+trợ"],
            llm_description="Customer needs general support",
            priority=5,
        ),
    ]


@pytest.fixture
def executor(basic_specs: list[IntentSpec]) -> RouterAgentExecutor:
    return RouterAgentExecutor(
        specs=basic_specs,
        model="test-model",
        llm_gateway_url="http://mock-litellm:4000",
        confidence_threshold=0.6,
        llm_timeout=5.0,
    )


def _make_llm_response(intent: str, confidence: float) -> MagicMock:
    """Create a mock that looks like an httpx response from LiteLLM."""
    content = json.dumps({"intent": intent, "confidence": confidence})
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# Layer 1 — Keyword scan
# ---------------------------------------------------------------------------

class TestKeywordScan:
    """Layer 1: deterministic keyword regex matching."""

    @pytest.mark.asyncio
    async def test_keyword_match_returns_route(self, executor):
        decision = await executor.route("I want a refund for my order")
        assert decision.route_id == "refund"
        assert decision.confidence == 1.0
        assert decision.method == "keyword"

    @pytest.mark.asyncio
    async def test_keyword_match_case_insensitive(self, executor):
        decision = await executor.route("REFUND please")
        assert decision.route_id == "refund"

    @pytest.mark.asyncio
    async def test_keyword_match_vietnamese(self, executor):
        decision = await executor.route("Tôi muốn hoàn tiền")
        assert decision.route_id == "refund"

    @pytest.mark.asyncio
    async def test_keyword_no_match_tries_llm(self, executor):
        """When no keyword matches, LLM fallback is attempted."""
        with patch.object(
            executor, "_llm_classify", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = RouteDecision(
                route_id="support", confidence=0.8, method="llm"
            )
            decision = await executor.route("something random")
            mock_llm.assert_awaited_once()
            assert decision.route_id == "support"
            assert decision.method == "llm"

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Higher priority spec is checked first."""
        specs = [
            IntentSpec(route_id="low", keywords=[r"test"], llm_description="low", priority=1),
            IntentSpec(route_id="high", keywords=[r"test"], llm_description="high", priority=99),
        ]
        exec_ = RouterAgentExecutor(specs=specs, model="m", llm_gateway_url="http://x")
        decision = await exec_.route("test input")
        assert decision.route_id == "high"


# ---------------------------------------------------------------------------
# Layer 2 — LLM classification (via _parse_llm_response unit tests)
# ---------------------------------------------------------------------------

class TestLLMResponseParsing:
    """Unit tests for _parse_llm_response — the core parsing logic."""

    def test_valid_response_high_confidence(self, executor):
        content_str = json.dumps({"intent": "refund", "confidence": 0.95})
        data = {"choices": [{"message": {"content": content_str}}]}
        decision = executor._parse_llm_response(data)
        assert decision.route_id == "refund"
        assert decision.confidence == 0.95
        assert decision.method == "llm"

    def test_low_confidence_rejected(self, executor):
        """Confidence below threshold → route_id=None."""
        content_str = json.dumps({"intent": "refund", "confidence": 0.3})
        data = {"choices": [{"message": {"content": content_str}}]}
        decision = executor._parse_llm_response(data)
        assert decision.route_id is None
        assert decision.confidence == 0.3

    def test_unknown_intent(self, executor):
        """Intent not in specs → route_id=None."""
        content_str = json.dumps({"intent": "unknown", "confidence": 0.95})
        data = {"choices": [{"message": {"content": content_str}}]}
        decision = executor._parse_llm_response(data)
        assert decision.route_id is None

    def test_none_intent(self, executor):
        """LLM says 'none' → route_id=None."""
        content_str = json.dumps({"intent": "none", "confidence": 0.5})
        data = {"choices": [{"message": {"content": content_str}}]}
        decision = executor._parse_llm_response(data)
        assert decision.route_id is None

    def test_malformed_json_returns_none(self, executor):
        """Malformed JSON content → graceful 'none'."""
        data = {"choices": [{"message": {"content": "not json at all"}}]}
        decision = executor._parse_llm_response(data)
        assert decision.route_id is None
        assert decision.method == "none"

    def test_markdown_fenced_json(self, executor):
        """LLM wraps JSON in ```json ... ``` fences — still parses."""
        content = '```json\n{"intent": "refund", "confidence": 0.9}\n```'
        data = {"choices": [{"message": {"content": content}}]}
        decision = executor._parse_llm_response(data)
        assert decision.route_id == "refund"
        assert decision.confidence == 0.9

    def test_missing_choices_key(self, executor):
        """Response without 'choices' key → graceful 'none'."""
        data = {"error": "something"}
        decision = executor._parse_llm_response(data)
        assert decision.route_id is None


class TestLLMTimeoutAndError:
    """Integration: LLM timeout/connection error → graceful fallback."""

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_none(self, executor):
        """LLM timeout → graceful fallback to 'none'."""
        async def slow_classify(text: str) -> RouteDecision:
            await asyncio.sleep(100)
            return RouteDecision("x", 1.0, "llm")

        with patch.object(executor, "_llm_classify", side_effect=slow_classify):
            decision = await executor.route(
                "unknown text that won't match keywords"
            )
            assert decision.route_id is None
            assert decision.method == "none"

    @pytest.mark.asyncio
    async def test_llm_returns_none_on_connection_error(self, executor):
        """_llm_classify catches connection errors → returns 'none'."""
        async def failing_classify(text: str) -> RouteDecision:
            raise ConnectionError("refused")

        # The executor's route() wraps _llm_classify in wait_for, but _llm_classify
        # itself catches exceptions internally. Let's test via patching httpx
        # to raise and verify _llm_classify doesn't propagate.
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = ConnectionError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            decision = await executor.route("something completely unrecognizable")
            assert decision.route_id is None
            assert decision.method == "none"


# ---------------------------------------------------------------------------
# Slot extraction
# ---------------------------------------------------------------------------

class TestSlotExtraction:
    """Slot patterns extract named values from text."""

    @pytest.mark.asyncio
    async def test_slot_extracted_on_keyword_match(self, executor):
        decision = await executor.route("refund for order ORD-12345")
        assert decision.route_id == "refund"
        assert "order_id" in decision.slots
        assert "ORD-12345" in str(decision.slots["order_id"])

    @pytest.mark.asyncio
    async def test_no_slot_when_pattern_missing(self, executor):
        decision = await executor.route("I want a refund")
        assert decision.route_id == "refund"
        assert "order_id" not in decision.slots

    @pytest.mark.asyncio
    async def test_no_slots_for_none_route(self, executor):
        with patch.object(
            executor, "_llm_classify", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = RouteDecision(
                route_id=None, confidence=0.0, method="none"
            )
            decision = await executor.route("unrecognizable text")
            assert decision.slots == {}

    def test_extract_slots_directly(self, executor):
        """Test _extract_slots with a matching route_id."""
        slots = executor._extract_slots("refund ORD-999", "refund")
        assert "order_id" in slots

    def test_extract_slots_unknown_route(self, executor):
        """Unknown route_id → empty slots."""
        slots = executor._extract_slots("text", "nonexistent")
        assert slots == {}

    def test_extract_slots_none_route(self, executor):
        """None route_id → empty slots."""
        slots = executor._extract_slots("text", None)
        assert slots == {}


# ---------------------------------------------------------------------------
# RouteDecision structure
# ---------------------------------------------------------------------------

class TestRouteDecisionDefaults:
    """RouteDecision default values."""

    def test_default_slots_empty(self):
        rd = RouteDecision(route_id="x", confidence=1.0, method="keyword")
        assert rd.slots == {}

    def test_none_route_has_empty_slots(self):
        rd = RouteDecision(route_id=None, confidence=0.0, method="none")
        assert rd.slots == {}

    def test_slots_with_values(self):
        rd = RouteDecision(
            route_id="r", confidence=0.9, method="llm",
            slots={"key": "value"},
        )
        assert rd.slots == {"key": "value"}
