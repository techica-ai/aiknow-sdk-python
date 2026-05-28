"""
agents/builtin_tools/_toolbox.py — BuiltinToolBox and BuiltinToolContext.

BuiltinToolBox is the central registry of all SDK-provided tools.
It is instantiated by AgentRuntime and injected into executors.

BuiltinToolContext is the infra-access object passed to each tool's execute().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from aiknow.agents.builtin_tools._base import BuiltinTool

if TYPE_CHECKING:
    from aiknow.agents.base import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class BuiltinToolContext:
    """Infra access object injected into every BuiltinTool.execute() call.

    Attributes:
        agent_ctx:       Full AgentContext for the current turn.
        http_client:     Optional async httpx.AsyncClient (shared, do NOT close).
        platform_url:    AIKnow Platform API base URL.
        litellm_url:     LiteLLM Gateway base URL.
        api_key:         Platform API key for authenticated requests.
        tenant_id:       Owning tenant identifier.
        extra:           Arbitrary extra context for custom tools.
    """

    agent_ctx: "AgentContext"
    http_client: Any = None          # httpx.AsyncClient — optional
    platform_url: str = "http://localhost:8000"
    litellm_url: str = "http://localhost:4000"
    api_key: str = ""
    tenant_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class BuiltinToolBox:
    """Registry and executor for all SDK built-in tools.

    Instantiate once per AgentRuntime. Tools are loaded lazily on first access.

    Usage::

        toolbox = BuiltinToolBox(platform_url="http://platform:8000")
        toolbox.load_all()  # or load specific: toolbox.load(["kb_search"])

        # Get OpenAI schemas for selected tools:
        schemas = toolbox.get_schemas(["kb_search", "get_execution_status"])

        # Execute a tool:
        result = await toolbox.execute(ctx, "kb_search", query="hoàn tiền")
    """

    def __init__(
        self,
        platform_url: str = "http://localhost:8000",
        litellm_url: str = "http://localhost:4000",
        api_key: str = "",
    ) -> None:
        self._platform_url = platform_url
        self._litellm_url = litellm_url
        self._api_key = api_key
        self._tools: dict[str, BuiltinTool] = {}
        self._loaded = False

    def load_all(self) -> None:
        """Load all built-in tools into the registry."""
        self._load_tools([
            # KB tools
            "kb_search",
            "kb_list_sources",
            "kb_get_document",
            # Platform tools
            "get_execution_status",
            "list_executions",
            "advance_execution",
            "escalate_to_supervisor",
            "get_workflow_definition",
            # Memory tools
            "get_session_info",
            "get_conversation_history",
            "save_agent_note",
            # Utility tools
            "format_currency",
            "current_datetime",
        ])
        self._loaded = True

    def load(self, tool_names: list[str]) -> None:
        """Load only specific built-in tools.

        Args:
            tool_names: List of tool names to load.
        """
        self._load_tools(tool_names)

    def get(self, name: str) -> BuiltinTool | None:
        """Get a BuiltinTool by name."""
        return self._tools.get(name)

    def get_schemas(self, names: list[str]) -> list[dict[str, Any]]:
        """Get OpenAI function schemas for the specified tool names.

        Args:
            names: Tool names to include.

        Returns:
            List of OpenAI function-calling schema dicts.
        """
        schemas = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                schemas.append(tool.to_openai_schema())
            else:
                logger.warning("BuiltinToolBox: tool '%s' not loaded.", name)
        return schemas

    def list_loaded(self) -> list[str]:
        """Return names of all loaded tools."""
        return list(self._tools.keys())

    async def execute(
        self,
        agent_ctx: "AgentContext",
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        http_client: Any = None,
    ) -> str:
        """Execute a built-in tool.

        Args:
            agent_ctx:   Current AgentContext.
            tool_name:   Name of the tool to call.
            tool_args:   Keyword arguments from LLM function call.
            http_client: Optional shared httpx.AsyncClient.

        Returns:
            Tool result as string.
        """
        import json

        tool = self._tools.get(tool_name)
        if tool is None:
            return f"[BuiltinTool '{tool_name}' not found]"

        ctx = BuiltinToolContext(
            agent_ctx=agent_ctx,
            http_client=http_client,
            platform_url=self._platform_url,
            litellm_url=self._litellm_url,
            api_key=self._api_key,
            tenant_id=agent_ctx.tenant_id,
        )

        try:
            result = await tool.execute(ctx, **tool_args)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.error("BuiltinTool '%s' error: %s", tool_name, exc)
            return f"[Tool error: {exc}]"

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load_tools(self, names: list[str]) -> None:
        """Lazy-import and instantiate tools by name."""
        _TOOL_MODULES: dict[str, str] = {
            # KB
            "kb_search":              "aiknow.agents.builtin_tools.kb.kb_search",
            "kb_list_sources":        "aiknow.agents.builtin_tools.kb.kb_list_sources",
            "kb_get_document":        "aiknow.agents.builtin_tools.kb.kb_get_document",
            # Platform
            "get_execution_status":   "aiknow.agents.builtin_tools.platform.get_execution_status",
            "list_executions":        "aiknow.agents.builtin_tools.platform.list_executions",
            "advance_execution":      "aiknow.agents.builtin_tools.platform.advance_execution",
            "escalate_to_supervisor": "aiknow.agents.builtin_tools.platform.escalate_to_supervisor",
            "get_workflow_definition": "aiknow.agents.builtin_tools.platform.get_execution_status",
            # Memory
            "get_session_info":       "aiknow.agents.builtin_tools.memory.get_session_info",
            "get_conversation_history": "aiknow.agents.builtin_tools.memory.get_conversation_history",
            "save_agent_note":        "aiknow.agents.builtin_tools.memory.save_agent_note",
            # Utility
            "format_currency":        "aiknow.agents.builtin_tools.utility.format_currency",
            "current_datetime":       "aiknow.agents.builtin_tools.utility.current_datetime",
        }

        for name in names:
            if name in self._tools:
                continue  # already loaded
            module_path = _TOOL_MODULES.get(name)
            if module_path is None:
                logger.warning("BuiltinToolBox: unknown tool '%s'.", name)
                continue
            try:
                import importlib
                mod = importlib.import_module(module_path)
                # Convention: module exports a class named after CamelCase of tool name
                class_name = "".join(w.capitalize() for w in name.split("_")) + "Tool"
                tool_cls = getattr(mod, class_name)
                self._tools[name] = tool_cls()
                logger.debug("BuiltinToolBox: loaded '%s'.", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("BuiltinToolBox: failed to load '%s': %s", name, exc)
