"""
agents/builtin_tools/_base.py — BuiltinTool base class and decorator.

BuiltinTools are SDK-provided capabilities automatically available to agents.
They differ from @tool (app-level) in that:
    - Built-in tools are bundled with the SDK, not written by app developers.
    - They can access SDK infra (HTTP client, LiteLLM gateway URL, etc.) via
      an injected ``BuiltinToolContext``.
    - They are referenced by name in agent decorator args (builtin_tools=[...]).

Implementing a new BuiltinTool:
    1. Subclass ``BuiltinTool``.
    2. Set ``name``, ``description``, ``parameters``.
    3. Implement ``async execute(ctx, **kwargs) -> str``.
    4. Register in ``BuiltinToolBox.__init__``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from aiknow.agents.builtin_tools._toolbox import BuiltinToolContext


@dataclass
class ToolParam:
    """Schema descriptor for a single BuiltinTool parameter.

    Attributes:
        name:        Parameter name (matches ``**kwargs`` key).
        type:        JSON Schema type string (``"string"``, ``"integer"``, etc.).
        description: i18n description dict.
        required:    Whether the parameter is required.
        default:     Default value if not required.
        enum:        If set, the parameter must be one of these values.
    """

    name: str
    type: str = "string"
    description: dict[str, str] = field(default_factory=dict)
    required: bool = False
    default: Any = None
    enum: list[str] | None = None


class BuiltinTool(ABC):
    """Abstract base for all SDK built-in tools.

    Attributes:
        name:        Unique tool identifier (referenced in builtin_tools=[...]).
        description: i18n description dict ``{locale: text}``.
        parameters:  List of ToolParam describing accepted kwargs.
    """

    name: str = ""
    # SDK-9 fix: use ClassVar to signal these are meant to be overridden
    # per-subclass, not mutated at instance level. Mutable shared defaults
    # ({}, []) are footguns if a subclass ever does self.parameters.append().
    description: ClassVar[dict[str, str]] = {}
    parameters: ClassVar[list[ToolParam]] = []

    @abstractmethod
    async def execute(
        self, ctx: BuiltinToolContext, **kwargs: Any
    ) -> str:
        """Execute the tool and return a string result.

        Args:
            ctx:    BuiltinToolContext with SDK infra (client, LiteLLM URL, etc.).
            kwargs: Arguments from LLM function call (matching ``parameters``).

        Returns:
            Tool result as a plain string or JSON string.
        """
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function calling schema.

        Returns:
            Dict in OpenAI function-calling schema format.
        """
        desc = self.description.get("en") or next(iter(self.description.values()), self.name)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type}
            param_desc = param.description.get("en") or next(
                iter(param.description.values()), ""
            )
            if param_desc:
                prop["description"] = param_desc
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
