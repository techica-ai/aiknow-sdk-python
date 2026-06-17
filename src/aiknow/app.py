"""
AiknowApp — entry point for building applications on the AIKnow Platform.

Usage::

    from aiknow import AiknowApp
    from aiknow.extensions import skill, sop, hook
    from aiknow.extensions.types import AppSkillDeps, SkillResult

    @skill("fetch_customer")
    async def fetch_customer(deps: AppSkillDeps, phone: str) -> SkillResult:
        ...

    @sop("refund_flow", initial_state="verify_customer")
    class RefundFlow:
        transitions = {...}

    @hook("workflow.completed")
    async def on_done(event):
        ...

    async def main():
        app = AiknowApp(tenant_id="acme-corp", api_key="ak_...")
        app.register_skill(fetch_customer)
        app.register_sop(RefundFlow)
        app.register_hook("workflow.completed", on_done)
        await app.start()
        print(app.describe())

Phase 1: start() only pings the Platform and logs registered extensions.
Phase 3: start() uploads manifest + starts webhook listener.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Literal

from aiknow._async_client import AsyncAIKnowClient
from aiknow.extensions._local_registry import ExtensionEntry, LocalExtensionRegistry

logger = logging.getLogger(__name__)


class AiknowApp:
    """Builder and entry point for an AIKnow application.

    Args:
        tenant_id: Tenant identifier for this application.
        api_key:   API key for authenticating with the Platform.
                   Falls back to ``AIKNOW_API_KEY`` env var.
        base_url:  Platform API base URL.
                   Falls back to ``AIKNOW_BASE_URL`` env var, then localhost.
    """

    def __init__(
        self,
        tenant_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        app_name: str = "aiknow-app",
        app_version: str = "1.0",
        webhook_host: str = "0.0.0.0",
        webhook_port: int = 9000,
        webhook_secret: str | None = None,
        webhook_advertise_host: str | None = None,
    ) -> None:
        """Initialise the AIKnow application.

        Args:
            tenant_id:               Owning tenant identifier.
            api_key:                 Platform API key. Reads ``AIKNOW_API_KEY`` env var.
            base_url:                Platform base URL. Reads ``AIKNOW_BASE_URL`` env var.
            app_name:                Human-readable name sent in AppManifest.
            app_version:             Version string sent in AppManifest.
            webhook_host:            Host for the WebhookListener to BIND to (e.g. '0.0.0.0').
            webhook_port:            Port for the WebhookListener (default 9000).
            webhook_secret:          Shared HMAC secret for request signing.
                                     Reads ``AIKNOW_WEBHOOK_SECRET`` env var if not set.
                                     None = dev mode (signature checking disabled).
            webhook_advertise_host:  Hostname to publish in the AppManifest webhook_url.
                                     Use when the bind host (0.0.0.0) differs from the
                                     externally-reachable hostname (e.g. Docker service name).
                                     Reads ``AIKNOW_WEBHOOK_ADVERTISE_HOST`` env var if not set.
        """
        self.tenant_id = tenant_id
        self._api_key = api_key or os.environ.get("AIKNOW_API_KEY")
        self._base_url = base_url or os.environ.get("AIKNOW_BASE_URL")
        self._app_name = app_name
        self._app_version = app_version
        self._webhook_host = webhook_host
        self._webhook_port = webhook_port
        self._webhook_secret = (
            webhook_secret or os.environ.get("AIKNOW_WEBHOOK_SECRET")
        )
        self._webhook_advertise_host = (
            webhook_advertise_host or os.environ.get("AIKNOW_WEBHOOK_ADVERTISE_HOST")
        )
        self._registry = LocalExtensionRegistry()
        self._client: AsyncAIKnowClient | None = None
        self._listener: Any | None = None  # WebhookListener — lazy import
        self._started = False
        self._platform_reachable = False

        # Agent layer — lazy init on first register_agent() call
        from aiknow.agents.tool_registry import AppToolRegistry
        self._tool_registry: AppToolRegistry = AppToolRegistry()
        self._agent_runtime: Any | None = None  # AgentRuntime — lazy init

    @classmethod
    def from_env(cls) -> AiknowApp:
        """Create an AiknowApp from environment variables.

        Required env vars:
            AIKNOW_TENANT_ID — tenant identifier
        Optional:
            AIKNOW_API_KEY   — API key (can also be set later)
            AIKNOW_BASE_URL  — Platform URL (defaults to localhost:8000)
        """
        tenant_id = os.environ.get("AIKNOW_TENANT_ID")
        if not tenant_id:
            raise OSError(
                "AIKNOW_TENANT_ID environment variable is not set. "
                "Set it before calling AiknowApp.from_env()."
            )
        return cls(
            tenant_id=tenant_id,
            api_key=os.environ.get("AIKNOW_API_KEY"),
            base_url=os.environ.get("AIKNOW_BASE_URL"),
        )

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register_step(self, fn: Callable[..., Any]) -> None:
        """Register a @step-decorated function (SOP workflow step).

        Args:
            fn: A callable decorated with ``@step(...)``.

        Raises:
            TypeError: If ``fn`` is not decorated with ``@step`` or ``@skill``.
        """
        # Accept both _step_meta (new) and _skill_meta (deprecated @skill)
        if hasattr(fn, "_step_meta"):
            meta: dict[str, Any] = fn._step_meta
            ext_type: Literal["step"] = "step"
        elif hasattr(fn, "_skill_meta"):
            import warnings
            warnings.warn(
                f"'{fn.__name__}' uses deprecated @skill. Use @step instead.",
                DeprecationWarning, stacklevel=2,
            )
            meta = fn._skill_meta
            ext_type = "step"  # normalise to "step" on register
        else:
            raise TypeError(
                f"'{fn.__name__}' is not decorated with @step. "
                "Apply @step(...) before calling register_step()."
            )
        entry = ExtensionEntry(
            name=meta["name"], ext_type=ext_type, fn=fn, metadata=meta
        )
        self._registry.register(entry)
        logger.debug("Registered step: %s", meta["name"])

    def register_skill(self, fn: Callable[..., Any]) -> None:
        """Register a @skill-decorated function.

        .. deprecated:: 2.0
            Use :meth:`register_step` instead. ``register_skill()`` will be
            removed in v3.0.
        """
        import warnings
        warnings.warn(
            "register_skill() is deprecated. Use register_step() instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.register_step(fn)

    def register_tool(self, fn: Callable[..., Any]) -> None:
        """Register a @tool-decorated function as an agent tool.

        Agent tools are callable by LLM agents during their reasoning loop.
        Unlike workflow steps, tools are invoked by agent decision-making
        rather than by the Platform FSM.

        Args:
            fn: A callable decorated with ``@tool(...)``.

        Raises:
            TypeError: If ``fn`` is not decorated with ``@tool``.
        """
        if not hasattr(fn, "_tool_meta"):
            raise TypeError(
                f"'{fn.__name__}' is not decorated with @tool. "
                "Apply @tool(...) before calling register_tool()."
            )
        meta: dict[str, Any] = fn._tool_meta
        entry = ExtensionEntry(
            name=meta["name"], ext_type="tool", fn=fn, metadata=meta
        )
        self._registry.register(entry)
        # Also register into AppToolRegistry for agent executors
        self._tool_registry.register(fn)
        logger.debug("Registered tool: %s", meta["name"])

    def register_agent(self, cls: type) -> None:
        """Register a @*_agent decorated class into the AgentRuntime.

        Supported decorators: @copilot_agent, @rag_agent, @autonomous_agent,
        @router_agent, @supervisor_agent, @conversation_agent.

        Args:
            cls: A class decorated with one of the @*_agent decorators.

        Raises:
            TypeError: If ``cls`` is not decorated with a recognised @*_agent.
        """
        if not hasattr(cls, "_agent_meta"):
            raise TypeError(
                f"'{cls.__name__}' is not decorated with an @*_agent decorator. "
                "Apply @copilot_agent / @rag_agent / etc. before calling register_agent()."
            )
        meta: dict[str, Any] = cls._agent_meta
        entry = ExtensionEntry(
            name=meta["name"], ext_type="agent", fn=cls, metadata=meta
        )
        self._registry.register(entry)
        # Also register into AgentRuntime
        self._ensure_agent_runtime().register_agent(cls)
        logger.debug("Registered agent: %s (type=%s)", meta["name"], meta.get("agent_type"))

    def register_sop(self, cls: type) -> None:
        """Register a @sop-decorated class.

        Args:
            cls: A class decorated with ``@sop(...)``.

        Raises:
            TypeError: If ``cls`` is not decorated with ``@sop``.
        """
        if not hasattr(cls, "_sop_meta"):
            raise TypeError(
                f"'{cls.__name__}' is not decorated with @sop. "
                "Apply @sop(...) before calling register_sop()."
            )
        meta: dict[str, Any] = cls._sop_meta
        entry = ExtensionEntry(
            name=meta["id"], ext_type="sop", fn=cls, metadata=meta
        )
        self._registry.register(entry)
        logger.debug("Registered SOP: %s", meta["id"])

    def register_dialog(self, cls: type) -> None:
        """Register a @dialog-decorated class.

        Args:
            cls: A class decorated with ``@dialog(...)``.

        Raises:
            TypeError: If ``cls`` is not decorated with ``@dialog``.
        """
        if not hasattr(cls, "_dialog_meta"):
            raise TypeError(
                f"'{cls.__name__}' is not decorated with @dialog. "
                "Apply @dialog(...) before calling register_dialog()."
            )
        meta: dict[str, Any] = cls._dialog_meta
        entry = ExtensionEntry(
            name=meta["id"], ext_type="dialog", fn=cls, metadata=meta
        )
        self._registry.register(entry)
        logger.debug("Registered Dialog: %s", meta["id"])

    def register_hook(self, event_type: str, fn: Callable[..., Any]) -> None:
        """Register a hook handler for a specific event type.

        Unlike ``@hook`` decorator (which auto-registers on import), this
        method gives explicit control over which event_type the handler fires for.

        Args:
            event_type: The event to subscribe to.
            fn:         An async callable ``(event: HookEvent) -> None``.
        """
        entry = ExtensionEntry(
            name=event_type,
            ext_type="hook",
            fn=fn,
            metadata={"event_type": event_type},
        )
        self._registry.register(entry)
        logger.debug("Registered hook: %s → %s", event_type, fn.__name__)

    def register_parser(self, fn: Callable[..., Any]) -> None:
        """Register a @parser-decorated function.

        Args:
            fn: A callable decorated with ``@parser(...)``.
        """
        if not hasattr(fn, "_parser_meta"):
            raise TypeError(
                f"'{fn.__name__}' is not decorated with @parser. "
                "Apply @parser(...) before calling register_parser()."
            )
        meta: dict[str, Any] = fn._parser_meta
        entry = ExtensionEntry(
            name=meta["content_type"], ext_type="parser", fn=fn, metadata=meta
        )
        self._registry.register(entry)
        logger.debug("Registered parser: %s", meta["content_type"])

    def load_sops(self, sops_dir: str) -> None:
        """Load SOP definitions from a directory of YAML files.

        Phase 2 feature — not yet implemented.
        Parses each .yaml file in ``sops_dir`` and registers the resulting
        SOP via ``register_sop()``.
        """
        # Phase 2: implement YAML SOP loading
        logger.info("load_sops('%s') — Phase 2 feature, not yet implemented.", sops_dir)

    def register_context_provider(self, provider: Any) -> None:
        """Register a custom ContextProvider for the AgentRuntime.

        Context providers inject dynamic text into the agent system prompt
        before each LLM turn.  Built-in providers (ExecutionStateProvider,
        SessionInfoProvider) are always active; use this to add custom ones.

        Args:
            provider: A :class:`~aiknow.agents.ContextProvider` instance.
        """
        self._ensure_agent_runtime().register_context_provider(provider)
        logger.debug("Registered context provider: %s", getattr(provider, "name", repr(provider)))

    def get_agent_router(self) -> Any:
        """Build and return a FastAPI APIRouter with all agent endpoints.

        Mount this on your FastAPI app to expose agent endpoints::

            from fastapi import FastAPI
            fastapi_app = FastAPI()
            fastapi_app.include_router(aiknow_app.get_agent_router())

        All registered agents are available at ``POST /agents/{agent_name}``.

        Returns:
            ``fastapi.APIRouter`` with agent endpoints mounted.

        Raises:
            RuntimeError: If no agents have been registered yet.
        """
        if self._agent_runtime is None:
            logger.warning(
                "get_agent_router() called but no agents registered. "
                "Call register_agent() first."
            )
        return self._ensure_agent_runtime().build_router()

    def build_router_executor(self, agent_name: str) -> Any:
        """Return the RouterAgentExecutor for a registered ``@router_agent`` class.

        Call this **after** ``app.start()`` — the executor is built during
        ``AgentRuntime.register_agent()`` which happens before ``start()``.

        The returned executor is an in-process Python object. Inject it into
        ``ConversationManager`` (or any other component that needs intent
        routing). It does **not** expose an HTTP endpoint.

        Args:
            agent_name: The ``name`` passed to ``@router_agent(name=...)``.

        Returns:
            :class:`~aiknow.agents.executors.router.RouterAgentExecutor` instance.

        Raises:
            KeyError:  If ``agent_name`` is not registered or is not a router agent.
            TypeError: If the registered agent is not a RouterAgentExecutor.

        Example::

            app.register_agent(SOPRouter)   # @router_agent class
            await app.start()
            executor = app.build_router_executor("sop_router")
            mgr = ConversationManager(client=sdk_client, router=executor)
        """
        from aiknow.agents.executors.router import RouterAgentExecutor

        runtime = self._ensure_agent_runtime()
        entry = runtime.get_agent(agent_name)
        if entry is None:
            raise KeyError(
                f"No agent named '{agent_name}' is registered. "
                "Call app.register_agent() with a @router_agent-decorated class first."
            )
        if entry.meta.agent_type != "router":
            raise KeyError(
                f"Agent '{agent_name}' has type '{entry.meta.agent_type}', not 'router'. "
                "Use @router_agent decorator to declare a router agent."
            )
        if not isinstance(entry.executor, RouterAgentExecutor):
            raise TypeError(
                f"Agent '{agent_name}' executor is {type(entry.executor).__name__}, "
                f"expected RouterAgentExecutor."
            )
        return entry.executor

    def _ensure_agent_runtime(self) -> Any:
        """Lazily initialise and return the AgentRuntime.

        Returns:
            The :class:`~aiknow.agents.AgentRuntime` instance.
        """
        if self._agent_runtime is None:
            from aiknow.agents.runtime import AgentRuntime
            self._agent_runtime = AgentRuntime(
                tool_registry=self._tool_registry,
            )
        return self._agent_runtime

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a structured description of all registered extensions.

        Useful for debugging and for generating the app manifest that will
        be uploaded to the Platform in Phase 3.

        Returns::

            {
                "tenant_id": "acme-corp",
                "skills":  [{"name": "fetch_customer", "tags": [...], ...}],
                "sops":    [{"id": "refund_flow", "initial_state": "...", ...}],
                "hooks":   [{"event_type": "workflow.completed"}],
                "parsers": [{"content_type": "audio/wav"}],
            }
        """
        desc: dict[str, Any] = dict(self._registry.describe())
        desc["tenant_id"] = self.tenant_id
        return desc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the application.

        Phase 3 behaviour (current):
        - Build the HTTP client.
        - Ping the Platform (warn if unreachable — offline mode).
        - Start WebhookListener for skill invocation + hook delivery.
        - Build AppManifest (with webhook_url) and upload to Platform.
        - Log all registered extensions.
        """
        self._client = AsyncAIKnowClient(
            base_url=self._base_url,
            api_key=self._api_key,
            tenant_id=self.tenant_id,
        )

        # --- Ping Platform ---
        try:
            self._platform_reachable = await self._client.ping()
            if self._platform_reachable:
                logger.info(
                    "AIKnow Platform reachable at %s",
                    self._base_url or "default",
                )
            else:
                logger.warning(
                    "AIKnow Platform not reachable at %s — running in offline mode.",
                    self._base_url or "default",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Platform ping failed: %s — offline mode.", exc)
            self._platform_reachable = False

        # --- Start WebhookListener ---
        await self._start_listener()

        # --- Upload manifest (non-fatal) ---
        # Guard: only attempt if platform is reachable AND api_key is set.
        if self._platform_reachable and self._api_key:
            try:
                await self._upload_manifest()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Manifest upload raised unexpectedly (offline mode): %s", exc
                )
        elif self._platform_reachable and not self._api_key:
            logger.info("No api_key provided — skipping manifest upload (read-only mode).")

        # --- Log summary ---
        desc = self.describe()
        steps = desc.get("steps", [])
        sops = desc.get("sops", [])
        dialogs = desc.get("dialogs", [])
        hooks = desc.get("hooks", [])
        parsers = desc.get("parsers", [])
        logger.info(
            "AiknowApp started [tenant=%s]: %d step(s), %d SOP(s), %d dialog(s), "
            "%d hook(s), %d parser(s)",
            self.tenant_id, len(steps), len(sops), len(dialogs), len(hooks), len(parsers),
        )
        if steps:
            logger.info("Steps:   %s", [s["name"] for s in steps])
        if sops:
            logger.info("SOPs:    %s", [s.get("id") for s in sops])
        if dialogs:
            logger.info("Dialogs: %s", [d.get("id") for d in dialogs])
        if hooks:
            logger.info("Hooks:   %s", [h.get("event_type") for h in hooks])

        self._started = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_manifest(self) -> Any:
        """Construct an AppManifest from the current local registry.

        SDK does NOT compile SOP/Dialog classes into WorkflowDefinition here.
        Instead, it extracts class attributes into SOPDeclaration / DialogDeclaration
        (plain Pydantic models from aiknow_contracts) and wraps them in
        WorkflowManifestContract. The Platform's compilers do the actual graph
        compilation server-side.

        This design keeps SDK dependency-free from aiknow-core, enabling
        future SDK implementations in JS, Go, etc.

        Returns:
            AppManifest Pydantic model ready for serialisation.
        """
        from aiknow_contracts.extensions import (
            AppManifest,
            HookManifestContract,
            SkillManifestContract,
            WorkflowManifestContract,
        )
        from aiknow_contracts.workflow import (
            DialogDeclaration,
            SlotDefinition,
            SOPDeclaration,
        )

        # Build skill manifests
        skill_manifests = [
            SkillManifestContract(
                name=entry.metadata.get("name", entry.name),
                version=entry.metadata.get("version", "1.0"),
                description=entry.metadata.get("description", {}),
                required_permissions=entry.metadata.get("required_permissions", []),
                tags=entry.metadata.get("tags", []),
            )
            for entry in self._registry.list_by_type("step")
        ]

        workflow_manifests: list[WorkflowManifestContract] = []

        # @sop → SOPDeclaration (extract class attributes, NO graph compilation)
        for entry in self._registry.list_by_type("sop"):
            cls = entry.fn
            try:
                # Resolve initial_state: decorator wins, then class attribute
                initial_state = (
                    entry.metadata.get("initial_state")
                    or getattr(cls, "initial_state", None)
                )
                transitions = getattr(cls, "transitions", None)
                if not transitions:
                    raise ValueError(
                        f"SOP '{entry.name}' has no transitions defined."
                    )
                # Resolve triggers from decorator or class attribute
                raw_triggers = (
                    entry.metadata.get("triggers")
                    or getattr(cls, "triggers", None)
                )
                triggers_obj = None
                if raw_triggers is not None:
                    from aiknow_contracts.workflow import WorkflowTriggers
                    if isinstance(raw_triggers, dict):
                        triggers_obj = WorkflowTriggers(**raw_triggers)
                    else:
                        triggers_obj = raw_triggers   # already WorkflowTriggers

                # Extract metadata (exclude reserved keys)
                extra_meta = {
                    k: v for k, v in entry.metadata.items()
                    if k not in ("id", "initial_state", "human_wait_states",
                                 "triggers", "description")
                }
                declaration = SOPDeclaration(
                    type="sop",
                    initial_state=initial_state or None,
                    transitions=transitions,
                    human_wait_states=list(getattr(cls, "human_wait_states", [])),
                    triggers=triggers_obj,
                    metadata=extra_meta,
                )
                workflow_manifests.append(
                    WorkflowManifestContract(
                        workflow_id=entry.name,
                        version=entry.metadata.get("version", "1.0"),
                        declaration=declaration,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping SOP '%s' in manifest: extraction error — %s",
                    entry.name, exc,
                )

        # @dialog → DialogDeclaration (extract class attributes, NO compilation)
        for entry in self._registry.list_by_type("dialog"):
            cls = entry.fn
            try:
                raw_slots: dict[str, Any] = getattr(cls, "slots", {})
                slots = {
                    slot_name: (
                        SlotDefinition(**slot_data)
                        if isinstance(slot_data, dict)
                        else slot_data  # already a SlotDefinition instance
                    )
                    for slot_name, slot_data in raw_slots.items()
                }
                on_complete = (
                    entry.metadata.get("on_complete_step")
                    or entry.metadata.get("on_complete_skill")
                    or getattr(cls, "on_complete_step", None)
                    or getattr(cls, "on_complete_skill", None)
                    or (entry.name + "_complete")
                )
                extra_meta = {
                    k: v for k, v in entry.metadata.items()
                    if k not in ("id", "on_complete_skill", "on_complete_step",
                                 "triggers", "cognitive_tools")
                }
                # Resolve triggers from decorator or class attribute
                raw_triggers = (
                    entry.metadata.get("triggers")
                    or getattr(cls, "triggers", None)
                )
                triggers_obj = None
                if raw_triggers is not None:
                    from aiknow_contracts.workflow import WorkflowTriggers
                    if isinstance(raw_triggers, dict):
                        triggers_obj = WorkflowTriggers(**raw_triggers)
                    else:
                        triggers_obj = raw_triggers  # already WorkflowTriggers

                # Resolve cognitive_tools
                cognitive_tools = (
                    entry.metadata.get("cognitive_tools")
                    or getattr(cls, "cognitive_tools", None)
                    or []
                )

                dialog_decl = DialogDeclaration(
                    type="dialog",
                    slots=slots,
                    on_complete_skill=on_complete,
                    on_complete_step=on_complete,
                    triggers=triggers_obj,
                    cognitive_tools=cognitive_tools,
                    metadata=extra_meta,
                )
                workflow_manifests.append(
                    WorkflowManifestContract(
                        workflow_id=entry.name,
                        version=entry.metadata.get("version", "1.0"),
                        declaration=dialog_decl,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping Dialog '%s' in manifest: extraction error — %s",
                    entry.name, exc,
                )

        # Build hook manifests
        hook_manifests = [
            HookManifestContract(event_type=entry.name)
            for entry in self._registry.list_by_type("hook")
        ]

        # Build tool manifests (Phase 2)
        from aiknow_contracts.extensions import ToolManifestContract
        tool_manifests = [
            ToolManifestContract(
                name=entry.metadata.get("name", entry.name),
                description=entry.metadata.get("description", {}),
                parameters_schema=self._tool_registry.get_schema(entry.name) or {},
                remote=entry.metadata.get("remote", False),
                required_permissions=entry.metadata.get("required_permissions", []),
                tags=entry.metadata.get("tags", []),
            )
            for entry in self._registry.list_by_type("tool")
        ]

        return AppManifest(
            app_name=self._app_name,
            version=self._app_version,
            tenant_id=self.tenant_id,
            skills=skill_manifests,
            workflows=workflow_manifests,
            hooks=hook_manifests,
            tools=tool_manifests,
            webhook_url=None,
        )

    async def _start_listener(self) -> None:
        """Start the WebhookListener in a background asyncio task.

        Non-fatal: errors during listener startup are logged but do not
        crash the app (it will run without remote skill invocations).
        """
        try:
            from aiknow.webhook_listener import WebhookListener
            self._listener = WebhookListener(
                registry=self._registry,
                host=self._webhook_host,
                port=self._webhook_port,
                hmac_secret=self._webhook_secret,
                advertise_host=self._webhook_advertise_host,
                tool_registry=self._tool_registry,  # Phase 2: expose /tools/{name}
            )
            await self._listener.start()
            logger.info(
                "WebhookListener started at http://%s:%d (tools: %d remote)",
                self._webhook_host,
                self._webhook_port,
                sum(
                    1 for e in self._registry.list_by_type("tool")
                    if e.metadata.get("remote", False)
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.critical(
                "WebhookListener FAILED to start on %s:%d: %s. "
                "ALL step invocations from the Platform will fail until this is resolved. "
                "Check port availability and permissions.",
                self._webhook_host, self._webhook_port, exc,
            )
            self._listener = None

    async def _upload_manifest(self) -> None:
        """Build and upload the AppManifest to the Platform.

        Injects the WebhookListener's base_url into the manifest so the
        Platform knows where to route skill invocations.

        Non-fatal: all errors are caught and logged as warnings.
        """
        if self._client is None:
            return
        try:
            manifest = self._build_manifest()

            # Inject webhook_url from the running listener (if available)
            if self._listener is not None:
                manifest = manifest.model_copy(
                    update={"webhook_url": self._listener.base_url}
                )

            await self._client.extensions.register_manifest(
                manifest.model_dump()
            )
            logger.info(
                "Manifest uploaded [tenant=%s]: %d step(s), %d workflow(s), %d hook(s), "
                "%d tool(s) (%d remote) webhook=%s",
                self.tenant_id,
                len(manifest.skills),
                len(manifest.workflows),
                len(manifest.hooks),
                len(manifest.tools),
                sum(1 for t in manifest.tools if t.remote),
                manifest.webhook_url,
            )
        except ImportError as exc:
            logger.warning(
                "Manifest upload skipped — aiknow_contracts not available: %s", exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Manifest upload failed (app continues in offline mode): %s", exc
            )

    async def stop(self) -> None:
        """Gracefully shut down the application."""
        if self._listener is not None:
            await self._listener.stop()
            self._listener = None
        if self._client:
            await self._client.close()
            self._client = None
        self._started = False
        logger.info("AiknowApp stopped [tenant=%s]", self.tenant_id)

    async def __aenter__(self) -> AiknowApp:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
