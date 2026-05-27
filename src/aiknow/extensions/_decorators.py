"""
Extension Decorators — public API for declaring steps, SOPs, hooks, parsers, tools, and agents.

Usage::

    from aiknow.extensions import step, sop, hook, parser, tool
    from aiknow.extensions.types import StepDeps, StepResult

    @step("verify_customer", description={"vi": "Xác thực KH"}, required_permissions=["crm:read"])
    async def verify_customer(deps: StepDeps, phone: str) -> StepResult:
        ...

    @sop("refund_flow", initial_state="verify_customer")
    class RefundFlow:
        transitions = {
            "verify_customer": {"verified": "fetch_order", "failed": "reject"},
            ...
        }

    @hook("escalation.triggered")
    async def on_escalation(event: HookEvent) -> None:
        ...

    @parser("audio/wav")
    async def audio_parser(data: bytes, metadata: dict) -> str:
        ...

    @tool("get_session_context", description={"vi": "Lấy thông tin session"})
    async def get_session_context(session_id: str) -> str:
        ...

    @copilot_agent("my_copilot", system_prompt="...", builtin_tools=["kb_search"])
    class MyCopilot:
        ...

Design:
- Decorators attach metadata to the callable via private attributes (_step_meta, _tool_meta, _agent_meta, etc.)
- Decorators also register into the module-level _default_registry
- AiknowApp reads from its own LocalExtensionRegistry instance
- No import of aiknow-core — SDK stays self-contained

Deprecated:
- @skill → use @step instead (kept for backward compatibility, removed in v3.0)
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from aiknow.extensions._local_registry import ExtensionEntry, _default_registry


# ---------------------------------------------------------------------------
# @step  (replaces deprecated @skill in SOP context)
# ---------------------------------------------------------------------------


def step(
    name: str,
    description: dict[str, str] | None = None,
    required_permissions: list[str] | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Declare a function as an AIKnow SOP Workflow Step.

    A workflow step is called by the AIKnow Platform when the SOP FSM
    reaches the corresponding state node. The function must return a
    StepResult whose .status drives the FSM transition.

    Args:
        name:                 Unique step identifier. Must match the state name
                              in the SOP transitions dict.
        description:          Localised description dict, e.g. {"vi": "...", "en": "..."}.
        required_permissions: RBAC permissions the caller must hold (e.g. ["crm:read"]).
        tags:                 Freeform tags for grouping/filtering.

    The decorated function must be an async function with signature::

        async def fn(deps: StepDeps, **params) -> StepResult

    Example::

        @step("verify_customer", required_permissions=["crm:read"])
        async def verify_customer(deps: StepDeps, phone: str) -> StepResult:
            customer = await crm.find(phone)
            if customer:
                return StepResult(status="verified", data={"name": customer.name})
            return StepResult(status="failed", message="Customer not found")

    Note:
        The Platform emits ``step.completed`` / ``step.failed`` lifecycle events
        (previously ``skill.completed`` / ``skill.failed``).
    """
    meta: dict[str, Any] = {
        "name": name,
        "description": description or {},
        "required_permissions": required_permissions or [],
        "tags": tags or [],
    }

    def decorator(fn: Callable) -> Callable:
        fn._step_meta = meta  # type: ignore[attr-defined]
        _default_registry.register(
            ExtensionEntry(name=name, ext_type="step", fn=fn, metadata=meta)
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# @skill  (DEPRECATED — alias for @step)
# ---------------------------------------------------------------------------


def skill(
    name: str,
    description: dict[str, str] | None = None,
    required_permissions: list[str] | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Declare a function as an AIKnow SOP Workflow Step.

    .. deprecated:: 2.0
        Use :func:`step` instead. ``@skill`` will be removed in v3.0.
        Reason: "skill" is ambiguous — it collides with "agent tool" semantics.

    This is a backward-compatibility alias for :func:`step`.
    """
    warnings.warn(
        f"@skill('{name}') is deprecated. Use @step('{name}') instead. "
        "The @skill decorator will be removed in v3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return step(name, description=description, required_permissions=required_permissions, tags=tags)


# ---------------------------------------------------------------------------
# @tool  (Agent Tool — callable by LLM agents)
# ---------------------------------------------------------------------------


def tool(
    name: str,
    description: dict[str, str] | None = None,
    required_permissions: list[str] | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Declare an async function as an Agent Tool.

    An agent tool is a capability that LLM agents can invoke autonomously
    during their reasoning (ReAct) loop. Unlike workflow steps (which are
    called by the Platform FSM), tools are invoked by the agent's own
    decision-making based on the user's request.

    Args:
        name:                 Unique tool identifier used in @*_agent builtin_tools list.
        description:          Localised description exposed to the LLM.
        required_permissions: RBAC permissions required to call this tool.
        tags:                 Freeform tags for grouping/filtering.

    The decorated function must be an async function. Return type should be
    a str (description of the result for the LLM to reason about)::

        async def fn(**params) -> str

    Example::

        @tool("get_session_context", description={"vi": "Lấy thông tin session hiện tại"})
        async def get_session_context(session_id: str) -> str:
            session = await mgr.get_session(session_id)
            return f"Caller: {session.caller_id}, Status: {session.status}"
    """
    meta: dict[str, Any] = {
        "name": name,
        "description": description or {},
        "required_permissions": required_permissions or [],
        "tags": tags or [],
    }

    def decorator(fn: Callable) -> Callable:
        fn._tool_meta = meta  # type: ignore[attr-defined]
        _default_registry.register(
            ExtensionEntry(name=name, ext_type="tool", fn=fn, metadata=meta)
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# @sop
# ---------------------------------------------------------------------------


def sop(
    sop_id: str,
    initial_state: str | None = None,
    human_wait_states: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[type], type]:
    """Declare a class as an AIKnow SOP (Standard Operating Procedure).

    The class must define a ``transitions`` dict:
        transitions: dict[str, dict[str, str]]
            # state_id → {outcome → target_state}

    Optionally define ``initial_state`` as a class attribute (can also be
    passed to the decorator). If neither is provided, the first key in
    ``transitions`` is used.

    Args:
        sop_id:             Unique SOP identifier (e.g. 'refund_flow').
        initial_state:      First state to enter. Overrides class attribute.
        human_wait_states:  States where the workflow suspends for human input.
        metadata:           Extra metadata (description, sla_seconds, tags...).

    Example::

        @sop("refund_flow", initial_state="verify_customer")
        class RefundFlow:
            transitions = {
                "verify_customer": {
                    "verified": "fetch_order",
                    "failed":   "__end__",
                },
                "fetch_order": {
                    "found":   "process_refund",
                    "missing": "escalate_human",
                },
            }
            human_wait_states = ["wait_manager_approval"]
    """
    sop_meta: dict[str, Any] = {
        "id": sop_id,
        "initial_state": initial_state,
        "human_wait_states": human_wait_states or [],
        **(metadata or {}),
    }

    def decorator(cls: type) -> type:
        cls._sop_meta = sop_meta  # type: ignore[attr-defined]
        # Propagate initial_state to class if not already set
        if initial_state and not hasattr(cls, "initial_state"):
            cls.initial_state = initial_state
        if human_wait_states and not hasattr(cls, "human_wait_states"):
            cls.human_wait_states = human_wait_states
        _default_registry.register(
            ExtensionEntry(name=sop_id, ext_type="sop", fn=cls, metadata=sop_meta)
        )
        return cls

    return decorator


# ---------------------------------------------------------------------------
# @hook
# ---------------------------------------------------------------------------


def hook(event_type: str) -> Callable[[Callable], Callable]:
    """Declare a function as a lifecycle event handler.

    Args:
        event_type: The event to subscribe to, e.g. 'step.completed',
                    'workflow.completed', 'state.entered'.

    The decorated function must be an async function::

        async def fn(event: HookEvent) -> None

    Example::

        @hook("workflow.completed")
        async def on_workflow_done(event: HookEvent) -> None:
            await save_call_record(event.execution_id)

    Standard event_type values:
        workflow.started, state.entered, step.completed, step.failed,
        waiting.human, human.signal, workflow.completed, workflow.failed,
        workflow.escalated
    """
    meta: dict[str, Any] = {"event_type": event_type}

    def decorator(fn: Callable) -> Callable:
        fn._hook_meta = meta  # type: ignore[attr-defined]
        _default_registry.register(
            ExtensionEntry(name=event_type, ext_type="hook", fn=fn, metadata=meta)
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# @parser
# ---------------------------------------------------------------------------


def parser(content_type: str) -> Callable[[Callable], Callable]:
    """Declare a function as a custom document parser for a MIME type.

    Args:
        content_type: MIME type this parser handles, e.g. 'audio/wav',
                      'application/x-crm-export'.

    The decorated function signature::

        async def fn(data: bytes, metadata: dict) -> str
        (returns the extracted plain text)

    Example::

        @parser("audio/wav")
        async def audio_parser(data: bytes, metadata: dict) -> str:
            return await whisper.transcribe(data)
    """
    meta: dict[str, Any] = {"content_type": content_type}

    def decorator(fn: Callable) -> Callable:
        fn._parser_meta = meta  # type: ignore[attr-defined]
        _default_registry.register(
            ExtensionEntry(name=content_type, ext_type="parser", fn=fn, metadata=meta)
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Agent decorators — @copilot_agent, @rag_agent, @autonomous_agent, etc.
# These are defined here as thin wrappers that attach _agent_meta.
# Full runtime logic lives in aiknow.agents.
# ---------------------------------------------------------------------------

def _make_agent_decorator(agent_type: str) -> Callable:
    """Factory for agent decorators that differ only in default agent_type."""

    def decorator_factory(
        name: str,
        system_prompt: str = "",
        builtin_tools: list[str] | None = None,
        tools: list[str] | None = None,
        knowledge_bases: list[str] | None = None,
        context_providers: list[str] | None = None,
        model: str | None = None,
        max_steps: int = 5,
        **extra: Any,
    ) -> Callable[[type], type]:
        meta: dict[str, Any] = {
            "name": name,
            "agent_type": agent_type,   # canonical field used by AgentRuntime
            "engine": "litellm",        # default LLM engine
            "system_prompt": system_prompt,
            "builtin_tools": builtin_tools or [],
            "tools": tools or [],
            "knowledge_bases": knowledge_bases or [],
            "context_providers": context_providers or [],
            "model": model or "",
            "max_steps": max_steps,
            **extra,
        }

        def decorator(cls: type) -> type:
            cls._agent_meta = meta  # type: ignore[attr-defined]
            _default_registry.register(
                ExtensionEntry(name=name, ext_type="agent", fn=cls, metadata=meta)
            )
            return cls

        return decorator

    decorator_factory.__name__ = f"{agent_type}_agent"
    return decorator_factory


def copilot_agent(
    name: str,
    system_prompt: str = "",
    builtin_tools: list[str] | None = None,
    tools: list[str] | None = None,
    knowledge_bases: list[str] | None = None,
    context_providers: list[str] | None = None,
    model: str | None = None,
    max_steps: int = 5,
    **extra: Any,
) -> Callable[[type], type]:
    """Declare a class as a Copilot Agent (AG-UI / CopilotKit compatible).

    A copilot agent streams responses via Server-Sent Events using the AG-UI
    protocol. It is designed to assist human agents (not end customers) and
    runs its own ReAct tool loop on the backend.

    Args:
        name:              Unique agent identifier (e.g. 'call_center_copilot').
        system_prompt:     The agent's persona and behavioral instructions.
        builtin_tools:     List of SDK built-in tool names to enable.
        tools:             List of custom @tool function names from this app.
        knowledge_bases:   KB IDs to scope kb_search results.
        context_providers: ContextProvider names to inject dynamic context.
        model:             LiteLLM model alias (e.g. 'glm-5').
        max_steps:         Maximum ReAct loop iterations per turn.

    Example::

        @copilot_agent(
            name="call_center_copilot",
            system_prompt="Bạn là AI Copilot hỗ trợ nội bộ...",
            builtin_tools=["kb_search", "get_execution_status"],
            model="glm-5",
        )
        class CallCenterCopilot:
            async def enrich_context(self, ctx: AgentContext) -> AgentContext:
                ...
    """
    return _make_agent_decorator("copilot")(
        name, system_prompt=system_prompt,
        builtin_tools=builtin_tools, tools=tools,
        knowledge_bases=knowledge_bases, context_providers=context_providers,
        model=model, max_steps=max_steps, **extra,
    )


def rag_agent(
    name: str,
    system_prompt: str = "",
    knowledge_bases: list[str] | None = None,
    retrieval_strategy: str = "hybrid",
    rerank: bool = True,
    max_context_chunks: int = 5,
    model: str | None = None,
    **extra: Any,
) -> Callable[[type], type]:
    """Declare a class as a RAG Agent (Knowledge Base Q&A specialist).

    RAG agents automatically have ``kb_search`` as a built-in tool.
    """
    return _make_agent_decorator("rag")(
        name, system_prompt=system_prompt,
        knowledge_bases=knowledge_bases or [],
        builtin_tools=["kb_search"],
        retrieval_strategy=retrieval_strategy,
        rerank=rerank,
        max_context_chunks=max_context_chunks,
        model=model, **extra,
    )


def autonomous_agent(
    name: str,
    system_prompt: str = "",
    builtin_tools: list[str] | None = None,
    tools: list[str] | None = None,
    max_steps: int = 10,
    model: str | None = None,
    **extra: Any,
) -> Callable[[type], type]:
    """Declare a class as an Autonomous Agent (self-running, no human loop).

    Autonomous agents run a ReAct loop and complete tasks without waiting
    for human input. They expose a REST chat/stream interface (not AG-UI).
    """
    return _make_agent_decorator("autonomous")(
        name, system_prompt=system_prompt,
        builtin_tools=builtin_tools, tools=tools,
        model=model, max_steps=max_steps, **extra,
    )


def router_agent(
    name: str,
    intents: "list[Any]",
    model: str | None = None,
    confidence_threshold: float = 0.6,
    llm_timeout: float = 10.0,
    **extra: Any,
) -> Callable[[type], type]:
    """Declare a class as a Router Agent (intent classification + SOP routing).

    The decorated class is registered with AiknowApp and used to build a
    ``RouterAgentExecutor`` in-process. It does NOT expose an HTTP endpoint
    (unlike ``@copilot_agent``).

    Args:
        name:                 Unique agent name (e.g. ``"sop_router"``).
        intents:              List of :class:`~aiknow.agents.routing.IntentSpec`
                              objects declaring each routable intent. Written
                              by the App; the SDK treats ``route_id``,
                              ``keywords``, etc. as opaque data.
        model:                Optional LLM model alias for Layer-2 classification.
                              If ``None``, ``AgentRuntime`` uses the value of
                              the ``AIKNOW_DEFAULT_MODEL`` environment variable
                              (or its own default).
        confidence_threshold: Minimum LLM confidence to accept a result (0.0–1.0).
                              Results below this threshold are treated as ``"none"``.
        llm_timeout:          Seconds to wait for the LLM response before
                              falling back to ``method='none'``.

    Note:
        ``llm_gateway_url`` is **not** a parameter here — it is an SDK-internal
        infrastructure concern managed by ``AgentRuntime`` via the
        ``LITELLM_BASE_URL`` environment variable.

    Example::

        from aiknow.agents.routing import IntentSpec
        from aiknow.extensions import router_agent

        INTENT_SPECS = [
            IntentSpec(
                route_id="refund_flow",
                keywords=[r"refund", r"hoàn tiền"],
                llm_description="Customer wants a refund or return",
                slot_patterns={"order_id": r"ORD-\\d+"},
            ),
        ]

        @router_agent(
            name="sop_router",
            intents=INTENT_SPECS,
            confidence_threshold=0.7,
            # model not set → SDK uses AIKNOW_DEFAULT_MODEL
        )
        class SOPRouter:
            pass
    """
    return _make_agent_decorator("router")(
        name,
        intents=intents,
        model=model,
        confidence_threshold=confidence_threshold,
        llm_timeout=llm_timeout,
        **extra,
    )


def supervisor_agent(
    name: str,
    sub_agents: list[str],
    system_prompt: str = "",
    model: str | None = None,
    **extra: Any,
) -> Callable[[type], type]:
    """Declare a class as a Supervisor Agent (multi-agent orchestrator).

    Args:
        sub_agents: List of agent names this supervisor can delegate to.
    """
    return _make_agent_decorator("supervisor")(
        name, system_prompt=system_prompt,
        sub_agents=sub_agents, model=model, **extra,
    )


def conversation_agent(
    name: str,
    system_prompt: str = "",
    builtin_tools: list[str] | None = None,
    tools: list[str] | None = None,
    memory_strategy: str = "sliding_window",
    max_history_turns: int = 20,
    model: str | None = None,
    **extra: Any,
) -> Callable[[type], type]:
    """Declare a class as a Conversation Agent (stateful multi-turn).

    Args:
        memory_strategy:   "sliding_window" | "summary" | "full" | "none"
        max_history_turns: Max conversation turns to keep in context.
    """
    return _make_agent_decorator("conversation")(
        name, system_prompt=system_prompt,
        builtin_tools=builtin_tools, tools=tools,
        memory_strategy=memory_strategy,
        max_history_turns=max_history_turns,
        model=model, **extra,
    )
