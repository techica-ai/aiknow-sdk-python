"""
Tests for Phase B: AgentRuntime, AppToolRegistry, ContextProviders, AiknowApp agent extensions.

Coverage:
    B1: AgentRuntime.register_agent(), list_agents(), _build_executor()
    B2: AgentRuntime.build_system_prompt() with ContextProviders
    B3: AppToolRegistry.register(), get(), get_schema(), list_all()
    B4: ContextProvider: ExecutionStateProvider, SessionInfoProvider
    B5: AiknowApp.register_agent(), register_tool(), get_agent_router()
    B6: @tool decorator attachments
    B7: @copilot_agent / @rag_agent decorator agent_type field
"""
from __future__ import annotations

import pytest

from aiknow.agents.base import AgentContext, AgentMeta, ToolMeta
from aiknow.agents.context_providers import (
    ContextProvider,
    ExecutionStateProvider,
    SessionInfoProvider,
    BUILTIN_PROVIDERS,
)
from aiknow.agents.tool_registry import AppToolRegistry
from aiknow.agents.runtime import AgentRuntime
from aiknow.extensions import tool, copilot_agent, rag_agent, autonomous_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_ctx() -> AgentContext:
    return AgentContext(tenant_id="acme", session_id="sess-001")


@pytest.fixture
def rich_ctx() -> AgentContext:
    return AgentContext(
        tenant_id="acme",
        session_id="sess-abc",
        execution_id="exec-001",
        workflow_id="refund_flow",
        current_state="verify_customer",
        execution_status="running",
        skill_outputs={"verify_customer": {"customer_id": "C001"}},
        user_context={"phone_number": "0987654321"},
        metadata={"caller_id": "0987", "channel": "phone"},
        locale="vi",
    )


@pytest.fixture
def tool_fn():
    @tool("get_session_info", description={"vi": "Lấy thông tin session", "en": "Get session info"})
    async def get_session_info(session_id: str = "", locale: str = "vi") -> dict:
        return {"session_id": session_id, "locale": locale}
    return get_session_info


@pytest.fixture
def copilot_cls():
    @copilot_agent(
        name="test_copilot",
        system_prompt="Bạn là AI Copilot hỗ trợ nội bộ.",
        builtin_tools=["kb_search"],
        tools=["get_session_info"],
        model="glm-5",
        max_steps=3,
    )
    class TestCopilot:
        pass
    return TestCopilot


# ---------------------------------------------------------------------------
# B6: @tool decorator
# ---------------------------------------------------------------------------

def test_tool_decorator_attaches_meta(tool_fn):
    """@tool attaches _tool_meta to the function."""
    assert hasattr(tool_fn, "_tool_meta")
    assert tool_fn._tool_meta["name"] == "get_session_info"
    assert "vi" in tool_fn._tool_meta["description"]


# ---------------------------------------------------------------------------
# B7: Agent decorators
# ---------------------------------------------------------------------------

def test_copilot_agent_decorator_sets_agent_type(copilot_cls):
    """@copilot_agent sets agent_type='copilot' in _agent_meta."""
    assert hasattr(copilot_cls, "_agent_meta")
    meta = copilot_cls._agent_meta
    assert meta["agent_type"] == "copilot"
    assert meta["name"] == "test_copilot"
    assert meta["model"] == "glm-5"
    assert "kb_search" in meta["builtin_tools"]
    assert "get_session_info" in meta["tools"]


def test_rag_agent_decorator_sets_agent_type():
    """@rag_agent sets agent_type='rag' and auto-includes kb_search."""
    @rag_agent(
        name="test_rag",
        knowledge_bases=["kb_policy"],
    )
    class TestRag:
        pass

    assert TestRag._agent_meta["agent_type"] == "rag"
    assert "kb_search" in TestRag._agent_meta["builtin_tools"]


def test_autonomous_agent_decorator():
    """@autonomous_agent sets agent_type='autonomous'."""
    @autonomous_agent(name="test_auto", max_steps=15)
    class TestAuto:
        pass

    assert TestAuto._agent_meta["agent_type"] == "autonomous"
    assert TestAuto._agent_meta["max_steps"] == 15


# ---------------------------------------------------------------------------
# B3: AppToolRegistry
# ---------------------------------------------------------------------------

def test_tool_registry_register_and_get(tool_fn):
    registry = AppToolRegistry()
    registry.register(tool_fn)
    assert registry.get("get_session_info") is tool_fn
    assert registry.size() == 1


def test_tool_registry_list_all(tool_fn):
    registry = AppToolRegistry()
    registry.register(tool_fn)
    all_tools = registry.list_all()
    assert len(all_tools) == 1
    assert isinstance(all_tools[0], ToolMeta)
    assert all_tools[0].name == "get_session_info"


def test_tool_registry_schema_generated(tool_fn):
    """AppToolRegistry generates OpenAI-compatible schema."""
    registry = AppToolRegistry()
    registry.register(tool_fn)
    schema = registry.get_schema("get_session_info")
    assert schema is not None
    assert schema["type"] == "function"
    fn_schema = schema["function"]
    assert fn_schema["name"] == "get_session_info"
    assert "session_id" in fn_schema["parameters"]["properties"]
    assert "locale" in fn_schema["parameters"]["properties"]
    # Both have defaults — neither should be required
    assert "session_id" not in fn_schema["parameters"].get("required", [])


def test_tool_registry_raises_on_non_tool():
    """register() raises TypeError if function is not decorated with @tool."""
    registry = AppToolRegistry()

    async def bare_fn(x: str) -> str:
        return x

    with pytest.raises(TypeError, match="@tool"):
        registry.register(bare_fn)


def test_tool_registry_list_schemas(tool_fn):
    registry = AppToolRegistry()
    registry.register(tool_fn)
    schemas = registry.list_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "get_session_info"


# ---------------------------------------------------------------------------
# B4: ContextProviders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execution_state_provider_empty_ctx(empty_ctx):
    """ExecutionStateProvider returns empty string when no active execution."""
    provider = ExecutionStateProvider()
    result = await provider.provide(empty_ctx)
    assert result == ""


@pytest.mark.asyncio
async def test_execution_state_provider_with_execution(rich_ctx):
    """ExecutionStateProvider returns structured Markdown with execution info."""
    provider = ExecutionStateProvider()
    result = await provider.provide(rich_ctx)
    assert "refund_flow" in result
    assert "verify_customer" in result
    assert "running" in result
    assert "C001" in result


@pytest.mark.asyncio
async def test_session_info_provider(rich_ctx):
    """SessionInfoProvider injects session metadata."""
    provider = SessionInfoProvider()
    result = await provider.provide(rich_ctx)
    assert "acme" in result
    assert "vi" in result
    assert "0987" in result  # caller_id from metadata
    assert "phone" in result  # channel


@pytest.mark.asyncio
async def test_session_info_provider_empty_ctx(empty_ctx):
    """SessionInfoProvider works without optional metadata."""
    provider = SessionInfoProvider()
    result = await provider.provide(empty_ctx)
    assert "acme" in result
    assert "vi" in result


def test_builtin_providers_registry():
    """BUILTIN_PROVIDERS dict has expected entries."""
    assert "execution_state" in BUILTIN_PROVIDERS
    assert "session_info" in BUILTIN_PROVIDERS
    assert BUILTIN_PROVIDERS["execution_state"] is ExecutionStateProvider


def test_custom_context_provider():
    """Custom ContextProvider subclass works correctly."""
    class TierProvider(ContextProvider):
        name = "customer_tier"

        async def provide(self, ctx: AgentContext) -> str:
            tier = ctx.metadata.get("tier", "standard")
            return f"Customer tier: {tier}"

    provider = TierProvider()
    assert provider.name == "customer_tier"


# ---------------------------------------------------------------------------
# B1: AgentRuntime registration
# ---------------------------------------------------------------------------

def test_agent_runtime_register_agent(copilot_cls):
    """AgentRuntime.register_agent() loads the agent and creates an executor."""
    runtime = AgentRuntime()
    runtime.register_agent(copilot_cls)
    agents = runtime.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "test_copilot"
    assert agents[0].agent_type == "copilot"


def test_agent_runtime_raises_on_undecorated():
    """register_agent() raises TypeError if class has no _agent_meta."""
    runtime = AgentRuntime()

    class PlainClass:
        pass

    with pytest.raises(TypeError, match="no _agent_meta"):
        runtime.register_agent(PlainClass)


def test_agent_runtime_get_agent(copilot_cls):
    runtime = AgentRuntime()
    runtime.register_agent(copilot_cls)
    entry = runtime.get_agent("test_copilot")
    assert entry is not None
    assert entry.meta.name == "test_copilot"
    assert entry.executor is not None


def test_agent_runtime_get_unknown_returns_none(copilot_cls):
    runtime = AgentRuntime()
    runtime.register_agent(copilot_cls)
    assert runtime.get_agent("nonexistent") is None


def test_agent_runtime_default_model_fallback(copilot_cls):
    """When agent model is unset, runtime falls back to default_model."""
    @copilot_agent(name="no_model_agent", system_prompt="test")
    class NoModelAgent:
        pass
    # model="" after decorator
    assert NoModelAgent._agent_meta["model"] == ""

    runtime = AgentRuntime(default_model="fallback-llm")
    runtime.register_agent(NoModelAgent)
    entry = runtime.get_agent("no_model_agent")
    assert entry.meta.model == "fallback-llm"


def test_agent_runtime_register_context_provider(copilot_cls):
    """register_context_provider() adds to runtime's custom providers."""
    runtime = AgentRuntime()

    class MyProvider(ContextProvider):
        name = "my_provider"
        async def provide(self, ctx: AgentContext) -> str:
            return "custom context"

    runtime.register_context_provider(MyProvider())
    # Should be in custom providers list
    assert any(p.name == "my_provider" for p in runtime._context_providers)


# ---------------------------------------------------------------------------
# B2: AgentRuntime.build_system_prompt()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_system_prompt_static_only():
    """build_system_prompt() returns static prompt when no execution context."""
    runtime = AgentRuntime()
    meta = AgentMeta(
        name="test",
        agent_type="copilot",
        system_prompt="Static instructions.",
        context_providers=[],
    )
    ctx = AgentContext(tenant_id="acme")
    result = await runtime.build_system_prompt(meta, ctx)
    assert "Static instructions." in result


@pytest.mark.asyncio
async def test_build_system_prompt_injects_execution_context(rich_ctx):
    """build_system_prompt() appends execution state when execution is active."""
    runtime = AgentRuntime()
    meta = AgentMeta(
        name="test",
        agent_type="copilot",
        system_prompt="Base prompt.",
        context_providers=["execution_state", "session_info"],
    )
    result = await runtime.build_system_prompt(meta, rich_ctx)
    assert "Base prompt." in result
    assert "refund_flow" in result
    assert "verify_customer" in result


@pytest.mark.asyncio
async def test_build_system_prompt_custom_provider(rich_ctx):
    """Custom ContextProvider output is appended to system prompt."""
    runtime = AgentRuntime()

    class TierProvider(ContextProvider):
        name = "tier_provider"
        async def provide(self, ctx: AgentContext) -> str:
            return "Customer is VIP"

    runtime.register_context_provider(TierProvider())
    meta = AgentMeta(
        name="test",
        agent_type="copilot",
        system_prompt="Base.",
        # Empty context_providers means all providers run (built-in + custom)
        context_providers=[],
    )
    result = await runtime.build_system_prompt(meta, rich_ctx)
    assert "Customer is VIP" in result


@pytest.mark.asyncio
async def test_build_system_prompt_filters_providers_by_meta(rich_ctx):
    """Providers NOT in agent_meta.context_providers are skipped."""
    runtime = AgentRuntime()

    class TierProvider(ContextProvider):
        name = "tier_provider"
        async def provide(self, ctx: AgentContext) -> str:
            return "Customer is VIP"

    runtime.register_context_provider(TierProvider())
    meta = AgentMeta(
        name="test",
        agent_type="copilot",
        system_prompt="Base.",
        context_providers=["session_info"],  # only session_info, not tier_provider
    )
    result = await runtime.build_system_prompt(meta, rich_ctx)
    # session_info should be in result
    assert "acme" in result
    # tier_provider should NOT be in result (filtered by context_providers list)
    assert "Customer is VIP" not in result


# ---------------------------------------------------------------------------
# B5: AiknowApp agent integration
# ---------------------------------------------------------------------------

def test_aiknow_app_register_agent(copilot_cls, tool_fn):
    """AiknowApp.register_agent() proxies to AgentRuntime."""
    from aiknow import AiknowApp
    app = AiknowApp(tenant_id="acme")
    app.register_tool(tool_fn)
    app.register_agent(copilot_cls)

    runtime = app._agent_runtime
    assert runtime is not None
    agents = runtime.list_agents()
    assert any(a.name == "test_copilot" for a in agents)


def test_aiknow_app_register_tool_populates_registry(tool_fn):
    """AiknowApp.register_tool() populates both LocalRegistry and AppToolRegistry."""
    from aiknow import AiknowApp
    app = AiknowApp(tenant_id="acme")
    app.register_tool(tool_fn)

    assert app._tool_registry.get("get_session_info") is tool_fn

    desc = app.describe()
    tool_names = [t["name"] for t in desc.get("tools", [])]
    assert "get_session_info" in tool_names


def test_aiknow_app_get_agent_router_builds_router(copilot_cls):
    """get_agent_router() returns a FastAPI router with correct routes."""
    pytest.importorskip("fastapi")
    from aiknow import AiknowApp
    app = AiknowApp(tenant_id="acme")
    app.register_agent(copilot_cls)

    router = app.get_agent_router()
    route_paths = [r.path for r in router.routes]
    # Router has prefix "/agents" so full path is "/agents/{name}"
    assert any("test_copilot" in p for p in route_paths), (
        f"No route containing 'test_copilot' found. Routes: {route_paths}"
    )


def test_aiknow_app_get_agent_router_no_agents():
    """get_agent_router() returns router even with no agents (just empty)."""
    pytest.importorskip("fastapi")
    from aiknow import AiknowApp
    app = AiknowApp(tenant_id="acme")
    # No agents registered — should still return a router (not crash)
    router = app.get_agent_router()
    assert router is not None
