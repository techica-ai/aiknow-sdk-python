"""
agents/tool_registry.py — AppToolRegistry for managing @tool-decorated functions.

AppToolRegistry is different from aiknow_core's AgentToolRegistry:
    - AgentToolRegistry  (core)  — manages built-in Platform tools (search, graph, etc.)
    - AppToolRegistry    (sdk)   — manages @tool-decorated functions registered on an App

App-level tools are custom functions written by App developers, registered via
``app.register_tool(fn)`` and exposed to agents via the agent executor's tool
calling mechanism.

Usage::

    from aiknow.extensions import tool
    from aiknow.extensions.types import StepDeps

    @tool("get_session_context", description={"vi": "Lấy context phiên làm việc"})
    async def get_session_context(deps: StepDeps, session_id: str = "") -> dict:
        return {"session_id": deps.user_context.get("session_id", session_id)}

    app.register_tool(get_session_context)
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from aiknow.agents.base import ToolMeta

logger = logging.getLogger(__name__)


class AppToolRegistry:
    """Registry for App-level ``@tool``-decorated functions.

    Tools registered here are available to agent executors for function calling.
    Each entry stores the callable, its metadata, and its JSON schema for LLM calls.

    Usage::

        registry = AppToolRegistry()
        registry.register(my_tool_fn)
        tool = registry.get("my_tool")
        all_tools = registry.list_all()
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._meta: dict[str, ToolMeta] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(self, fn: Callable) -> None:
        """Register a ``@tool``-decorated function.

        Args:
            fn: Function decorated with ``@tool(...)``.

        Raises:
            TypeError: If ``fn`` is not decorated with ``@tool``.
        """
        meta_dict = getattr(fn, "_tool_meta", None)
        if meta_dict is None:
            raise TypeError(
                f"'{fn.__name__}' is not decorated with @tool. "
                "Use @tool('name') before registering."
            )
        meta = ToolMeta(
            name=meta_dict["name"],
            description=meta_dict.get("description", {}),
            tags=meta_dict.get("tags", []),
        )
        if meta.name in self._tools:
            logger.warning("AppToolRegistry: overwriting tool '%s'.", meta.name)

        self._tools[meta.name] = fn
        self._meta[meta.name] = meta
        self._schemas[meta.name] = self._build_schema(fn, meta)
        logger.debug("AppToolRegistry: registered tool '%s'.", meta.name)

    def get(self, name: str) -> Callable | None:
        """Get a registered tool function by name."""
        return self._tools.get(name)

    def get_meta(self, name: str) -> ToolMeta | None:
        """Get ToolMeta for a registered tool."""
        return self._meta.get(name)

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """Get JSON schema dict for a registered tool (for LLM function calling)."""
        return self._schemas.get(name)

    def list_all(self) -> list[ToolMeta]:
        """Return all registered tool metadata."""
        return list(self._meta.values())

    def list_schemas(self) -> list[dict[str, Any]]:
        """Return all tool JSON schemas (for LLM function calling config)."""
        return list(self._schemas.values())

    def size(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schema(fn: Callable, meta: ToolMeta) -> dict[str, Any]:
        """Generate OpenAI-compatible function schema from function signature.

        Args:
            fn:   The tool function.
            meta: ToolMeta with name and description.

        Returns:
            Dict in OpenAI function-calling schema format.
        """
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        _TYPE_MAP: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            # Skip deps (first param) and *args/**kwargs
            if param_name in ("deps", "self"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            ann = param.annotation
            json_type = "string"  # default
            if ann is not inspect.Parameter.empty:
                origin = getattr(ann, "__origin__", None)
                if origin is not None:
                    # handle Optional[X], list[X], etc. — use string fallback
                    json_type = "string"
                else:
                    json_type = _TYPE_MAP.get(ann, "string")

            prop: dict[str, Any] = {"type": json_type}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop["default"] = param.default

            properties[param_name] = prop

        # i18n description: prefer "en" or first available
        desc_dict = meta.description
        description = desc_dict.get("en") or next(iter(desc_dict.values()), meta.name)

        return {
            "type": "function",
            "function": {
                "name": meta.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
