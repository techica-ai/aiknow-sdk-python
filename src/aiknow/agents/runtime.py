"""
agents/runtime.py — AgentRuntime: load, route, and execute agents.

AgentRuntime is the central orchestrator of the agent layer in aiknow-sdk.
It:
    1. Loads all registered @agent-decorated classes and resolves their executors.
    2. Builds FastAPI/Starlette route handlers for each agent endpoint.
    3. Dispatches incoming requests to the correct executor based on agent name.
    4. Manages ContextProvider chain execution before each agent turn.

Architecture::

    AiknowApp
        └── AgentRuntime
                ├── agents: dict[name → AgentEntry]
                │       ├── cls: the @agent-decorated class
                │       ├── meta: AgentMeta
                │       └── executor: BaseAgentExecutor
                ├── tool_registry: AppToolRegistry
                ├── context_providers: list[ContextProvider]
                └── _llm_gateway_url: str (LiteLLM base URL)

Usage (internal — called by AiknowApp)::

    runtime = AgentRuntime(
        llm_gateway_url="http://localhost:4000",
        tool_registry=app._tool_registry,
    )
    runtime.register_agent(MyAgentClass)
    runtime.register_context_provider(ExecutionStateProvider())
    router = runtime.build_router()
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiknow.agents.base import AgentMeta
from aiknow.agents.builtin_tools._toolbox import BuiltinToolBox
from aiknow.agents.context_providers import (
    ContextProvider,
    ExecutionStateProvider,
    SessionInfoProvider,
)
from aiknow.agents.executors.react import ReActAgentExecutor
from aiknow.agents.tool_registry import AppToolRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AgentEntry:
    """Internal registry entry for a loaded agent."""

    cls: type
    meta: AgentMeta
    executor: Any  # BaseAgentExecutor — avoid circular import


class AgentRuntime:
    """Loads, routes, and executes agents registered on an AiknowApp.

    Args:
        llm_gateway_url: Base URL of the LiteLLM proxy gateway.
                         Falls back to ``LITELLM_BASE_URL`` env var,
                         then ``http://localhost:4000``.
        tool_registry:   AppToolRegistry shared with AiknowApp.
        default_model:   Default LLM model for agents that don't specify one.
                         Falls back to ``AIKNOW_DEFAULT_MODEL`` env var,
                         then ``"gpt-4o-mini"``.
    """

    def __init__(
        self,
        llm_gateway_url: str | None = None,
        tool_registry: AppToolRegistry | None = None,
        default_model: str | None = None,
    ) -> None:
        self._llm_gateway_url = (
            llm_gateway_url
            or os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
        )
        self._default_model = (
            default_model
            or os.environ.get("AIKNOW_DEFAULT_MODEL", "gpt-4o-mini")
        )
        self._tool_registry = tool_registry or AppToolRegistry()
        self._agents: dict[str, AgentEntry] = {}
        self._context_providers: list[ContextProvider] = []

        # Built-in toolbox — lazy-loads tools on first agent registration
        self._toolbox = BuiltinToolBox(
            platform_url=os.environ.get("AIKNOW_BASE_URL", "http://localhost:8000"),
            litellm_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"),
            api_key=os.environ.get("AIKNOW_API_KEY", ""),
        )

        # Auto-register built-in providers (always active)
        self._builtin_providers: list[ContextProvider] = [
            ExecutionStateProvider(),
            SessionInfoProvider(),
        ]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, cls: type) -> None:
        """Load and register an @agent-decorated class.

        Args:
            cls: A class decorated with one of the agent decorators
                 (``@copilot_agent``, ``@rag_agent``, etc.).

        Raises:
            TypeError: If ``cls`` does not have ``_agent_meta`` set.
        """
        meta_dict = getattr(cls, "_agent_meta", None)
        if meta_dict is None:
            raise TypeError(
                f"'{cls.__name__}' has no _agent_meta attribute. "
                "It is not decorated with an agent decorator "
                "(e.g. @copilot_agent, @rag_agent). "
                "Use one of the agent decorators from aiknow.extensions."
            )

        # Extract only known AgentMeta fields — put the rest into extra
        known_fields = {f.name for f in dataclasses.fields(AgentMeta)}
        known_kwargs = {k: v for k, v in meta_dict.items() if k in known_fields}
        extra_kwargs = {k: v for k, v in meta_dict.items() if k not in known_fields}
        if extra_kwargs:
            # Merge extra keys into the extra dict
            existing_extra = known_kwargs.get("extra", {})
            known_kwargs["extra"] = {**extra_kwargs, **existing_extra}

        meta = AgentMeta(**known_kwargs)
        if not meta.model:
            meta.model = self._default_model

        # Load required builtin tools for this agent
        if meta.builtin_tools:
            self._toolbox.load(meta.builtin_tools)

        executor = self._build_executor(cls, meta)
        entry = AgentEntry(cls=cls, meta=meta, executor=executor)
        self._agents[meta.name] = entry
        logger.info("AgentRuntime: registered agent '%s' (type=%s).", meta.name, meta.agent_type)

    def register_context_provider(self, provider: ContextProvider) -> None:
        """Register a custom ContextProvider.

        Custom providers run AFTER the built-in providers (ExecutionState,
        SessionInfo) and their output is appended to the system prompt.

        Args:
            provider: A ContextProvider instance (must have a unique ``name``).
        """
        existing = [p.name for p in self._context_providers]
        if provider.name in existing:
            logger.warning(
                "AgentRuntime: overwriting context provider '%s'.", provider.name
            )
            self._context_providers = [
                p for p in self._context_providers if p.name != provider.name
            ]
        self._context_providers.append(provider)
        logger.debug("AgentRuntime: registered context provider '%s'.", provider.name)

    # ------------------------------------------------------------------
    # Router / HTTP
    # ------------------------------------------------------------------

    def build_router(self) -> Any:
        """Build a FastAPI APIRouter with one POST endpoint per agent.

        Endpoint pattern: ``POST /agents/{agent_name}``

        Each endpoint accepts AG-UI protocol requests and streams
        SSE responses back.

        Returns:
            ``fastapi.APIRouter`` instance (only imported if fastapi is installed).

        Raises:
            ImportError: If ``fastapi`` is not installed in the environment.
        """
        try:
            from fastapi import APIRouter  # noqa: PLC0415
            from fastapi.responses import StreamingResponse  # noqa: PLC0415

            from aiknow.agents._router_factory import make_agent_handler  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "fastapi is required to use AgentRuntime.build_router(). "
                "Install it with: pip install fastapi"
            ) from exc

        router = APIRouter(prefix="/agents", tags=["agents"])

        for name, entry in self._agents.items():
            _executor = entry.executor
            _name = name

            # make_agent_handler() is defined in _router_factory.py which has
            # NO `from __future__ import annotations`. This means its type hints
            # (Request, StreamingResponse) are concrete class objects at definition
            # time — FastAPI can resolve them without ForwardRef errors.
            _handle = make_agent_handler(_executor)

            router.add_api_route(
                f"/{_name}",
                _handle,
                methods=["POST"],
                name=f"agent_{_name}",
                summary=f"AG-UI SSE endpoint for agent '{_name}'",
                response_class=StreamingResponse,
                response_model=None,
                include_in_schema=True,
            )
            logger.debug("AgentRuntime: mounted endpoint POST /agents/%s", _name)

        return router

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    async def build_system_prompt(
        self, agent_meta: AgentMeta, ctx: AgentContext  # noqa: F821
    ) -> str:
        """Assemble full system prompt = static prompt + provider injections.

        Runs built-in providers first, then custom providers in registration order.
        Each provider output is appended as a Markdown section.

        Args:
            agent_meta: AgentMeta with static ``system_prompt``.
            ctx:        AgentContext for the current turn.

        Returns:
            Full system prompt string.
        """

        parts: list[str] = []
        if agent_meta.system_prompt:
            parts.append(agent_meta.system_prompt.strip())

        all_providers = self._builtin_providers + self._context_providers
        for provider in all_providers:
            # Only run providers requested by this agent (or run all if not specified)
            if (
                agent_meta.context_providers
                and provider.name not in agent_meta.context_providers
            ):
                continue
            try:
                block = await provider.provide(ctx)
                if block:
                    parts.append(block.strip())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AgentRuntime: context provider '%s' raised %s — skipping.",
                    provider.name, exc,
                )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_agents(self) -> list[AgentMeta]:
        """Return metadata for all registered agents."""
        return [entry.meta for entry in self._agents.values()]

    def get_agent(self, name: str) -> AgentEntry | None:
        """Get an AgentEntry by name."""
        return self._agents.get(name)

    # ------------------------------------------------------------------
    # Internal: executor factory
    # ------------------------------------------------------------------

    def _build_executor(self, cls: type, meta: AgentMeta) -> Any:
        """Instantiate the correct executor based on agent_type."""
        agent_type = meta.agent_type
        if agent_type == "copilot":
            from aiknow.agents.executors.copilot import CopilotAgentExecutor  # noqa: PLC0415
            return CopilotAgentExecutor(
                cls=cls,
                meta=meta,
                tool_registry=self._tool_registry,
                toolbox=self._toolbox,
                llm_gateway_url=self._llm_gateway_url,
                runtime=self,
            )

        if agent_type in ("rag", "autonomous", "conversation"):
            return ReActAgentExecutor(
                cls=cls,
                meta=meta,
                tool_registry=self._tool_registry,
                toolbox=self._toolbox,
                llm_gateway_url=self._llm_gateway_url,
                runtime=self,
            )

        if agent_type == "router":
            from aiknow.agents.executors.router import RouterAgentExecutor  # noqa: PLC0415
            intents = meta.extra.get("intents", [])
            threshold = float(meta.extra.get("confidence_threshold", 0.6))
            timeout = float(meta.extra.get("llm_timeout", 10.0))
            return RouterAgentExecutor(
                specs=intents,
                model=meta.model or self._default_model,
                llm_gateway_url=self._llm_gateway_url,  # SDK-internal, not from App
                confidence_threshold=threshold,
                llm_timeout=timeout,
            )

        logger.warning(
            "AgentRuntime: agent_type '%s' executor not implemented yet — "
            "using ReActAgentExecutor as fallback.",
            agent_type,
        )
        return ReActAgentExecutor(
            cls=cls,
            meta=meta,
            tool_registry=self._tool_registry,
            toolbox=self._toolbox,
            llm_gateway_url=self._llm_gateway_url,
            runtime=self,
        )


# Re-export AgentContext here for convenience
from aiknow.agents.base import AgentContext as AgentContext  # noqa: E402, F401
