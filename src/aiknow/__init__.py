"""
AIKNOW Python SDK.

Public API::

    from aiknow import AIKnowClient, AsyncAIKnowClient
    from aiknow import AiknowApp
    from aiknow.extensions import step, sop, hook, parser, tool
    from aiknow.extensions import StepDeps, StepResult, HookEvent
    from aiknow.extensions import copilot_agent, rag_agent, autonomous_agent
    from aiknow import AIKnowAPIError, AIKnowTimeoutError, AIKnowConnectionError
"""
from __future__ import annotations

from aiknow.app import AiknowApp
from aiknow.extensions import (
    AppSkillDeps,  # deprecated alias
    HookEvent,
    SkillResult,   # deprecated alias
    autonomous_agent,
    conversation_agent,
    copilot_agent,
    hook,
    parser,
    rag_agent,
    router_agent,
    skill,         # deprecated
    sop,
    step,
    supervisor_agent,
    tool,
    StepDeps,
    StepResult,
)

from aiknow_contracts.chat import ChatRequest, ChatResponse
from aiknow_contracts.graph import (
    GraphNeighbor,
    GraphNeighborsResponse,
    GraphNode,
    GraphNodeListResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphStatsResponse,
    VectorSearchRequest,
)
from aiknow_contracts.ingestion import UploadResponse
from aiknow_contracts.jobs import JobListResponse, JobStatusResponse
from aiknow_contracts.observe import (
    CleanupResponse,
    LatencyStatsResponse,
    LlmCallItem,
    LlmCallListResponse,
    PolicyStatsResponse,
    SlowSpansResponse,
    ToolUsageResponse,
    TraceDetailResponse,
    TraceListResponse,
)
from aiknow_contracts.sources import SourceDetailResponse, SourceItem, SourceListResponse
from aiknow_contracts.vector import (
    VectorChunk,
    VectorCollection,
    VectorCollectionDetail,
    VectorCollectionListResponse,
    VectorHit,
    VectorSearchResponse,
)

from ._async_client import AsyncAIKnowClient, close_shared_http_client
from ._client import AIKnowClient
from ._span_builder import SpanBuilder
from ._exceptions import (
    AIKnowAPIError,
    AIKnowConnectionError,
    AIKnowTimeoutError,
    AuthenticationError,
    TimeoutError,  # backward-compat alias
)
from .resources.workflows import ExecutionState
from .resources.knowledge import KnowledgeChunk

__all__ = [
    # App builder
    "AiknowApp",
    # Span builder for tracing
    "SpanBuilder",
    # Extension decorators — SOP
    "step",
    "sop",
    "hook",
    "parser",
    # Extension decorators — Agent Tools
    "tool",
    # Extension decorators — Agent Patterns
    "copilot_agent",
    "rag_agent",
    "autonomous_agent",
    "router_agent",
    "supervisor_agent",
    "conversation_agent",
    # Extension types
    "StepDeps",
    "StepResult",
    "HookEvent",
    # Deprecated aliases
    "skill",
    "AppSkillDeps",
    "SkillResult",
    # Clients
    "AIKnowClient",
    "AsyncAIKnowClient",
    "close_shared_http_client",
    # Exceptions
    "AIKnowAPIError",
    "AIKnowConnectionError",
    "AIKnowTimeoutError",
    "AuthenticationError",
    "TimeoutError",
    # Workflow execution
    "ExecutionState",
    # Knowledge Base
    "KnowledgeChunk",
    # Chat
    "ChatRequest",
    "ChatResponse",
    # Ingestion
    "UploadResponse",
    # Jobs
    "JobStatusResponse",
    "JobListResponse",
    # Sources
    "SourceItem",
    "SourceListResponse",
    "SourceDetailResponse",
    # Vector
    "VectorCollection",
    "VectorCollectionListResponse",
    "VectorCollectionDetail",
    "VectorHit",
    "VectorSearchResponse",
    "VectorChunk",
    # Graph
    "GraphQueryRequest",
    "VectorSearchRequest",
    "GraphStatsResponse",
    "GraphNode",
    "GraphNodeListResponse",
    "GraphNeighbor",
    "GraphNeighborsResponse",
    "GraphQueryResponse",
    # Observe
    "CleanupResponse",
    "LatencyStatsResponse",
    "LlmCallItem",
    "LlmCallListResponse",
    "PolicyStatsResponse",
    "SlowSpansResponse",
    "ToolUsageResponse",
    "TraceDetailResponse",
    "TraceListResponse",
]
