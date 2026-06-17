"""
agents/base.py — Core data types for the AgentRuntime layer.

These data types are used in both the Agent Executor and the Context Provider
to pass context from platform → agent → LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """Runtime context injected into every agent invocation.

    Attributes:
        tenant_id:        Owning tenant.
        session_id:       Current WebSocket / HTTP session ID.
        execution_id:     Active workflow execution ID (None if no active execution).
        workflow_id:      Active workflow definition ID (None if no active execution).
        current_state:    Current FSM state label (None if no active execution).
        execution_status: ``running`` | ``waiting_human`` | ``completed`` | ``failed``.
        skill_outputs:    Accumulated step outputs keyed by state_id.
        user_context:     Merged dict of initial_inputs + all skill_outputs.
        metadata:         Arbitrary extra data (locale, caller_id, etc.).
        locale:           ISO 639-1 locale code (default ``"vi"``).
    """

    tenant_id: str
    session_id: str = ""
    execution_id: str | None = None
    workflow_id: str | None = None
    current_state: str | None = None
    execution_status: str = "idle"
    skill_outputs: dict[str, Any] = field(default_factory=dict)
    user_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    locale: str = "vi"


@dataclass
class AgentMeta:
    """Metadata attached to an @agent-decorated class.

    Set by agent decorator (e.g. ``@copilot_agent``) and read by AgentRuntime.

    Attributes:
        name:               Unique agent name used in routing.
        agent_type:         Agent pattern: ``copilot`` | ``rag`` | ``autonomous``
                            | ``router`` | ``supervisor`` | ``conversation``.
        system_prompt:      Optional static system prompt (overridden by
                            ContextProvider output at runtime).
        builtin_tools:      List of BuiltinTool names to load automatically.
        tools:              List of ``@tool``-decorated function names to include.
        context_providers:  List of ContextProvider names to run before each turn.
        engine:             Execution engine: ``litellm`` | ``pydantic_ai`` | ``langgraph``.
        model:              LLM model identifier (resolved through LiteLLM config).
        max_steps:          Max ReAct loop iterations.
        extra:              Reserved for future / engine-specific config.
    """

    name: str
    agent_type: str
    system_prompt: str = ""
    builtin_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    knowledge_bases: list[str] = field(default_factory=list)
    context_providers: list[str] = field(default_factory=list)
    engine: str = "litellm"
    model: str = ""
    max_steps: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMeta:
    """Metadata for a ``@tool``-decorated function registered on an App.

    Attributes:
        name:        Unique tool name (must match LLM function call name).
        description: i18n description dict ``{locale: text}``.
        tags:        Arbitrary string tags for filtering/grouping.
    """

    name: str
    description: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
