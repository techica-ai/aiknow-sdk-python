"""
agents/executors/base.py — BaseAgentExecutor abstract class.

All agent executors (CopilotAgentExecutor, ReActAgentExecutor, etc.)
extend this base class.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from aiknow.agents.base import AgentContext, AgentMeta
    from aiknow.agents.runtime import AgentRuntime
    from aiknow.agents.tool_registry import AppToolRegistry


class BaseAgentExecutor(ABC):
    """Abstract base for all agent executors.

    Args:
        cls:              The @agent-decorated class.
        meta:             Resolved AgentMeta.
        tool_registry:    AppToolRegistry with @tool functions.
        toolbox:          BuiltinToolBox with SDK built-in tools.
        llm_gateway_url:  LiteLLM proxy base URL.
        runtime:          Reference to parent AgentRuntime.
    """

    def __init__(
        self,
        cls: type,
        meta: AgentMeta,
        tool_registry: AppToolRegistry,
        llm_gateway_url: str,
        runtime: AgentRuntime,
        toolbox: Any = None,
    ) -> None:
        self._cls = cls
        self.meta = meta
        self._tool_registry = tool_registry
        self._toolbox = toolbox  # BuiltinToolBox | None
        self._llm_gateway_url = llm_gateway_url
        self._runtime = runtime

    @abstractmethod
    async def handle_agui_request(self, body: dict[str, Any]) -> Any:
        """Handle an incoming AG-UI protocol request.

        Args:
            body: Parsed JSON body from the HTTP request.

        Returns:
            A StreamingResponse (SSE) or plain Response.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers shared by all executors
    # ------------------------------------------------------------------

    def _extract_context(self, body: dict[str, Any]) -> AgentContext:
        """Extract AgentContext from AG-UI request body.

        The AG-UI protocol passes agent state in ``body["state"]``.
        We map known fields to AgentContext.

        Args:
            body: Parsed request body.

        Returns:
            AgentContext populated from state fields.
        """
        from aiknow.agents.base import AgentContext  # noqa: PLC0415

        state: dict[str, Any] = body.get("state") or {}

        # Try to determine locale from messages or state
        locale = state.get("locale", "vi")

        ctx = AgentContext(
            tenant_id=state.get("tenant_id", ""),
            session_id=state.get("session_id", ""),
            execution_id=state.get("execution_id") or state.get("activeExecutionId"),
            workflow_id=state.get("workflow_id"),
            current_state=state.get("current_state") or state.get("currentSopState"),
            execution_status=state.get("execution_status") or state.get("executionStatus", "idle"),
            skill_outputs=state.get("skill_outputs", {}),
            user_context=state.get("user_context", {}),
            metadata=state.get("metadata", {}),
            locale=locale,
        )
        return ctx

    def _resolve_tools_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-format schemas for all tools available to this agent.

        Combines:
        1. Built-in tools from BuiltinToolBox (filtered by agent's builtin_tools list)
        2. App-level @tool functions filtered by agent's tools list

        Returns:
            List of OpenAI function-calling schema dicts.
        """
        schemas: list[dict[str, Any]] = []

        # Built-in tools
        if self._toolbox and self.meta.builtin_tools:
            schemas.extend(self._toolbox.get_schemas(self.meta.builtin_tools))

        # App-level custom tools
        for tool_name in self.meta.tools:
            schema = self._tool_registry.get_schema(tool_name)
            if schema:
                schemas.append(schema)

        return schemas

    async def _call_tool(
        self,
        ctx: AgentContext,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Invoke a registered tool by name (built-in or app-level).

        Checks built-in toolbox first, then app-level registry.
        """

        # Try built-in tool first
        if self._toolbox and self._toolbox.get(tool_name):
            return cast(str, await self._toolbox.execute(ctx, tool_name, tool_args))

        # Fall back to app-level @tool
        fn = self._tool_registry.get(tool_name)
        if fn is None:
            return f"[Tool '{tool_name}' not found]"

        try:
            from aiknow.extensions.types import StepDeps  # noqa: PLC0415
            deps = StepDeps(
                tenant_id=ctx.tenant_id,
                locale=ctx.locale,
                user_context=ctx.user_context,
                execution_id=ctx.execution_id,
            )
            result = await fn(deps, **tool_args)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"[Tool '{tool_name}' error: {exc}]"
