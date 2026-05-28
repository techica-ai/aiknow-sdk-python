"""
aiknow.agents — Agent layer for building AI agents on the AIKnow Platform.

Public API:
    AgentRuntime       — Runtime orchestrator (used internally by AiknowApp)
    AgentContext       — Runtime context injected into agents
    AgentMeta          — Metadata for @agent-decorated classes
    ToolMeta           — Metadata for @tool-decorated functions
    AppToolRegistry    — Registry for @tool functions
    ContextProvider    — Abstract base for context providers
    ExecutionStateProvider  — Built-in: inject SOP execution state
    SessionInfoProvider     — Built-in: inject session metadata

    AIKnowConversationAgent — Phase 5: AG-UI bridge to Platform ConversationAPI.
                              Subclass or instantiate directly for CopilotKit integration.
"""
from aiknow.agents.base import AgentContext, AgentMeta, ToolMeta
from aiknow.agents.context_providers import (
    ContextProvider,
    ExecutionStateProvider,
    SessionInfoProvider,
    BUILTIN_PROVIDERS,
)
from aiknow.agents.tool_registry import AppToolRegistry
from aiknow.agents.runtime import AgentRuntime
from aiknow.agents.conversation_agent import AIKnowConversationAgent

__all__ = [
    # Core types
    "AgentContext",
    "AgentMeta",
    "ToolMeta",
    # Runtime
    "AgentRuntime",
    # Tool registry
    "AppToolRegistry",
    # Context providers
    "ContextProvider",
    "ExecutionStateProvider",
    "SessionInfoProvider",
    "BUILTIN_PROVIDERS",
    # Phase 5: CopilotKit AG-UI bridge
    "AIKnowConversationAgent",
]
