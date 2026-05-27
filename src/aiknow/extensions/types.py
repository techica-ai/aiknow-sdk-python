"""
SDK Extension Types — public types for app developer-written steps, tools, and hooks.

These are intentionally SIMPLER than the internal aiknow-core types:
- StepDeps:  no infrastructure ports (retriever, graph_store)
  App code should not directly access Qdrant/Memgraph.
- StepResult: rich return type that drives FSM transitions.
- HookEvent:  parameter type for @hook handlers.

Zero dependency on aiknow-core or aiknow-adapters.

Terminology:
- StepDeps / StepResult — context and return type for @step (SOP workflow steps)
- AppSkillDeps / SkillResult — deprecated aliases, will be removed in v3.0
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Step dependencies (injected by Platform at invocation time)
# ---------------------------------------------------------------------------


@dataclass
class StepDeps:
    """Runtime context injected into every app-defined @step function.

    The Platform creates this from its internal context before calling
    a developer-written step. Unlike the internal ``AgentToolDeps``, this
    does NOT expose infrastructure ports — app steps are pure functions
    that receive structured inputs and return a ``StepResult``.

    If a step needs to search the knowledge base, it should do so via
    the Platform API (e.g., ``client.knowledge.search()``) rather than
    accessing Qdrant directly.

    Attributes:
        tenant_id:    Tenant this call belongs to.
        session_id:   Conversation/call session ID (if any).
        locale:       BCP-47 language tag (e.g., 'vi', 'en').
        permissions:  Flat list of RBAC permissions granted to the caller.
        user_context: Arbitrary metadata from the upstream request
                      (e.g., channel, customer tier, agent ID).
    """

    tenant_id: str
    session_id: str | None = None
    locale: str = "vi"
    permissions: list[str] = field(default_factory=list)
    user_context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step return type — drives FSM transitions
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """Return type for app-defined @step functions.

    The ``status`` field drives FSM state transitions::

        @step("verify_customer")
        async def verify_customer(deps: StepDeps, phone: str) -> StepResult:
            if crm.find(phone):
                return StepResult(status="verified", message="OK", data={...})
            return StepResult(status="failed", message="Not found")

    The SOP then looks up ``transitions["verify_customer"]["verified"]``
    to determine the next state.

    Attributes:
        status:     The outcome label. Must match a key in the SOP transitions dict.
        message:    Human-readable description (logged, shown in agent desktop).
        data:       Structured data carried forward in workflow state.
        next_state: Optional hard override — skip the transition table and jump
                    directly to this state. Use sparingly (emergency escalations only).
    """

    status: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    next_state: str | None = None

    # Convenience constructors
    @classmethod
    def ok(cls, message: str = "OK", data: dict | None = None) -> "StepResult":
        return cls(status="ok", message=message, data=data or {})

    @classmethod
    def error(cls, message: str, data: dict | None = None) -> "StepResult":
        return cls(status="error", message=message, data=data or {})

    @classmethod
    def escalate(cls, message: str = "Escalating to human agent") -> "StepResult":
        return cls(status="escalate", message=message, next_state="escalate_human")


# ---------------------------------------------------------------------------
# Deprecated aliases — AppSkillDeps, SkillResult
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy deprecated alias access."""
    if name == "AppSkillDeps":
        warnings.warn(
            "AppSkillDeps is deprecated and will be removed in v3.0. "
            "Use StepDeps instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return StepDeps
    if name == "SkillResult":
        warnings.warn(
            "SkillResult is deprecated and will be removed in v3.0. "
            "Use StepResult instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return StepResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Hook event — parameter type for @hook handlers
# ---------------------------------------------------------------------------


class HookEvent(BaseModel):
    """Event delivered to @hook handlers at workflow lifecycle points.

    Standard event_type values (matches WorkflowEvent in aiknow-contracts):
        workflow.started, state.entered, step.completed, step.failed,
        waiting.human, human.signal, workflow.completed, workflow.failed,
        workflow.escalated

    Attributes:
        event_type:    What happened.
        workflow_id:   SOP definition ID (e.g. 'refund_flow').
        execution_id:  Specific execution instance ID.
        tenant_id:     Which tenant.
        payload:       Event-specific data (state names, outcomes, etc).
        timestamp:     When the event was emitted (UTC).
    """

    event_type: str
    workflow_id: str | None = None
    execution_id: str | None = None
    tenant_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)  # noqa: UP017
    )
