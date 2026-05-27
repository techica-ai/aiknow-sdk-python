"""
Contract compatibility tests — verify the public API surface of the SDK.

These tests ensure that:
- All expected symbols are importable from the top-level `aiknow` package.
- Public Pydantic models have the expected fields.
"""
from __future__ import annotations

import aiknow


class TestPublicApiSurface:
    """Ensure all expected symbols are importable from `aiknow`."""

    def test_clients_exported(self):
        assert hasattr(aiknow, "AIKnowClient")
        assert hasattr(aiknow, "AsyncAIKnowClient")

    def test_exceptions_exported(self):
        assert hasattr(aiknow, "AIKnowAPIError")
        assert hasattr(aiknow, "AIKnowConnectionError")
        assert hasattr(aiknow, "AIKnowTimeoutError")
        assert hasattr(aiknow, "AuthenticationError")
        assert hasattr(aiknow, "TimeoutError")  # backward-compat alias

    def test_chat_types_exported(self):
        assert hasattr(aiknow, "ChatRequest")
        assert hasattr(aiknow, "ChatResponse")

    def test_ingestion_types_exported(self):
        assert hasattr(aiknow, "UploadResponse")

    def test_observe_types_exported(self):
        for name in [
            "TraceListResponse", "TraceDetailResponse",
            "SlowSpansResponse", "LlmCallItem", "LlmCallListResponse",
            "LatencyStatsResponse", "ToolUsageResponse",
            "PolicyStatsResponse", "CleanupResponse",
        ]:
            assert hasattr(aiknow, name), f"Missing export: {name}"


class TestModelFields:
    """Ensure Pydantic models have the expected fields."""

    def test_chat_response_has_citations_as_list_of_dict(self):
        from aiknow import ChatResponse
        import typing
        hints = typing.get_type_hints(ChatResponse)
        # citations should be list[dict[str, Any]], not list[str]
        assert hints["citations"] is not str

    def test_chat_response_has_trace_id(self):
        from aiknow import ChatResponse
        resp = ChatResponse(answer="hi", citations=[], trace_id="abc123")
        assert resp.trace_id == "abc123"

    def test_upload_response_fields(self):
        from aiknow import UploadResponse
        resp = UploadResponse(message="ok", job_id="j1", source_id="s1")
        assert resp.job_id == "j1"
        assert resp.source_id == "s1"
