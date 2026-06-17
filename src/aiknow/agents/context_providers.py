"""
agents/context_providers.py — ContextProvider abstraction.

ContextProviders run before each agent turn and inject structured
information into the system prompt / agent state.

Built-in providers:
    - ExecutionStateProvider  — inject active SOP execution state
    - SessionInfoProvider     — inject session metadata (caller_id, locale, etc.)

App developers can implement custom ContextProvider subclasses and register
them via ``app.register_context_provider(provider)``.

Usage::

    from aiknow.agents.context_providers import ContextProvider, ExecutionStateProvider

    class MyProvider(ContextProvider):
        name = "my_custom_context"

        async def provide(self, ctx: AgentContext) -> str:
            return f"Customer tier: {ctx.metadata.get('tier', 'standard')}"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiknow.agents.base import AgentContext


class ContextProvider(ABC):
    """Abstract base for all context providers.

    Subclass this and implement :meth:`provide` to inject text into
    the agent's system prompt at runtime.

    Attributes:
        name: Unique identifier used when registering the provider.
              Must be unique within an App.
    """

    name: str = ""

    @abstractmethod
    async def provide(self, ctx: AgentContext) -> str:
        """Return a text block to inject into the system prompt.

        Args:
            ctx: Full AgentContext for the current turn.

        Returns:
            Non-empty string to append to system prompt, or empty string
            to skip injection for this turn.
        """
        ...


class ExecutionStateProvider(ContextProvider):
    """Injects active workflow execution state into the agent's system prompt.

    Generates a structured Markdown block with:
    - Active workflow name and current state
    - Execution status (running / waiting_human / completed)
    - Accumulated step outputs from previous states
    - Available transitions from current state

    This provider is auto-included by ``@copilot_agent`` unless
    ``include_execution_state=False`` is set.
    """

    name = "execution_state"

    async def provide(self, ctx: AgentContext) -> str:
        if not ctx.execution_id:
            return ""

        lines = [
            "## Active Workflow Execution",
            f"- **Workflow**: `{ctx.workflow_id or 'unknown'}`",
            f"- **Execution ID**: `{ctx.execution_id}`",
            f"- **Current State**: `{ctx.current_state or 'unknown'}`",
            f"- **Status**: `{ctx.execution_status}`",
        ]

        if ctx.skill_outputs:
            lines.append("\n### Step Outputs")
            for state_id, data in ctx.skill_outputs.items():
                if data:
                    lines.append(f"- **{state_id}**: {data}")

        if ctx.user_context:
            relevant = {
                k: v for k, v in ctx.user_context.items()
                if not k.endswith("_data") and v
            }
            if relevant:
                lines.append("\n### User Context")
                for k, v in relevant.items():
                    lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)


class SessionInfoProvider(ContextProvider):
    """Injects session-level metadata into the agent's system prompt.

    Injects:
    - Session ID and locale
    - Custom metadata fields (e.g. caller_id, channel, tier)

    This provider is auto-included by all agent types that have
    ``include_session_info=True`` (default for copilot and conversation agents).
    """

    name = "session_info"

    async def provide(self, ctx: AgentContext) -> str:
        lines = [
            "## Session Info",
            f"- **Tenant**: `{ctx.tenant_id}`",
            f"- **Locale**: `{ctx.locale}`",
        ]

        if ctx.session_id:
            lines.append(f"- **Session**: `{ctx.session_id[:16]}…`")

        for key in ("caller_id", "channel", "agent_id", "tier"):
            val = ctx.metadata.get(key)
            if val:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {val}")

        return "\n".join(lines)


# Registry of all built-in providers keyed by name
BUILTIN_PROVIDERS: dict[str, type[ContextProvider]] = {
    ExecutionStateProvider.name: ExecutionStateProvider,
    SessionInfoProvider.name: SessionInfoProvider,
}
