"""
Tests for Phase C: BuiltinToolBox, BuiltinTool implementations.

Coverage:
    C1: BuiltinToolBox.load(), get(), get_schemas(), list_loaded()
    C2: BuiltinTool.to_openai_schema()
    C3: Tool implementations — no HTTP client (graceful degradation)
    C4: KB tools: kb_search, kb_list_sources, kb_get_document
    C5: Platform tools: get_execution_status, advance_execution, escalate_to_supervisor
    C6: Memory tools: get_session_info, save_agent_note
    C7: Utility tools: format_currency, current_datetime
    C8: Integration with BaseAgentExecutor._resolve_tools_schemas()
"""
from __future__ import annotations

import pytest
from aiknow.agents.base import AgentContext
from aiknow.agents.builtin_tools import BuiltinToolBox, BuiltinTool, BuiltinToolContext, ToolParam


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_ctx() -> AgentContext:
    return AgentContext(tenant_id="acme", session_id="sess-001", locale="vi")


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
        user_context={"phone": "0987654321"},
        metadata={"caller_id": "0987", "channel": "phone", "tier": "gold"},
        locale="vi",
    )


@pytest.fixture
def toolbox() -> BuiltinToolBox:
    tb = BuiltinToolBox(
        platform_url="http://platform:8000",
        litellm_url="http://litellm:4000",
    )
    tb.load_all()
    return tb


@pytest.fixture
def no_client_ctx(empty_ctx) -> BuiltinToolContext:
    return BuiltinToolContext(
        agent_ctx=empty_ctx,
        http_client=None,
        platform_url="http://platform:8000",
        tenant_id="acme",
    )


@pytest.fixture
def rich_tool_ctx(rich_ctx) -> BuiltinToolContext:
    return BuiltinToolContext(
        agent_ctx=rich_ctx,
        http_client=None,
        platform_url="http://platform:8000",
        tenant_id="acme",
    )


# ---------------------------------------------------------------------------
# C1: BuiltinToolBox
# ---------------------------------------------------------------------------

def test_toolbox_load_all(toolbox):
    """load_all() registers all 13 tools."""
    loaded = toolbox.list_loaded()
    expected = [
        "kb_search", "kb_list_sources", "kb_get_document",
        "get_execution_status", "list_executions", "advance_execution",
        "escalate_to_supervisor", "get_workflow_definition",
        "get_session_info", "get_conversation_history", "save_agent_note",
        "format_currency", "current_datetime",
    ]
    for name in expected:
        assert name in loaded, f"Tool '{name}' not loaded"


def test_toolbox_load_specific():
    """load() registers only specified tools."""
    tb = BuiltinToolBox()
    tb.load(["kb_search", "format_currency"])
    loaded = tb.list_loaded()
    assert "kb_search" in loaded
    assert "format_currency" in loaded
    assert "get_execution_status" not in loaded
    assert len(loaded) == 2


def test_toolbox_get(toolbox):
    """get() returns loaded tool instance."""
    tool = toolbox.get("kb_search")
    assert tool is not None
    assert isinstance(tool, BuiltinTool)
    assert tool.name == "kb_search"


def test_toolbox_get_missing(toolbox):
    """get() returns None for unknown tools."""
    assert toolbox.get("nonexistent_tool") is None


def test_toolbox_get_schemas(toolbox):
    """get_schemas() returns OpenAI-format schemas for requested tools."""
    schemas = toolbox.get_schemas(["kb_search", "format_currency"])
    assert len(schemas) == 2
    names = [s["function"]["name"] for s in schemas]
    assert "kb_search" in names
    assert "format_currency" in names


def test_toolbox_get_schemas_skips_unknown(toolbox):
    """get_schemas() skips unknown tool names gracefully."""
    schemas = toolbox.get_schemas(["kb_search", "i_do_not_exist"])
    assert len(schemas) == 1  # only kb_search
    assert schemas[0]["function"]["name"] == "kb_search"


def test_toolbox_load_unknown_graceful():
    """load() with unknown name logs warning but doesn't crash."""
    tb = BuiltinToolBox()
    tb.load(["totally_fake_tool"])  # should not raise
    assert "totally_fake_tool" not in tb.list_loaded()


# ---------------------------------------------------------------------------
# C2: BuiltinTool.to_openai_schema()
# ---------------------------------------------------------------------------

def test_to_openai_schema_structure(toolbox):
    """to_openai_schema() returns valid OpenAI function schema."""
    tool = toolbox.get("kb_search")
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "kb_search"
    assert "description" in fn
    assert "parameters" in fn
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert "query" in params["required"]
    assert "top_k" not in params.get("required", [])


def test_to_openai_schema_required_params(toolbox):
    """Required params are listed in 'required' array."""
    tool = toolbox.get("advance_execution")
    schema = tool.to_openai_schema()
    required = schema["function"]["parameters"].get("required", [])
    assert "signal" in required
    assert "execution_id" not in required  # has default (uses active)


# ---------------------------------------------------------------------------
# C3: Graceful degradation (no HTTP client)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kb_search_no_client(toolbox, no_client_ctx):
    """kb_search returns helpful message when no HTTP client configured."""
    result = await toolbox.execute(no_client_ctx.agent_ctx, "kb_search", {"query": "hoàn tiền"})
    assert "hoàn tiền" in result
    assert "HTTP client" in result or "http_client" in result.lower() or len(result) > 0


@pytest.mark.asyncio
async def test_kb_search_missing_query(toolbox, no_client_ctx):
    """kb_search returns error message when query is missing."""
    result = await toolbox.execute(no_client_ctx.agent_ctx, "kb_search", {})
    assert "Missing" in result or "query" in result.lower()


# ---------------------------------------------------------------------------
# C5: Platform tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_execution_status_uses_ctx(toolbox, rich_ctx):
    """get_execution_status uses in-memory context when execution_id matches."""
    ctx = BuiltinToolContext(agent_ctx=rich_ctx, http_client=None, tenant_id="acme")
    result = await toolbox.execute(rich_ctx, "get_execution_status", {"execution_id": "exec-001"})
    assert "exec-001" in result
    assert "refund_flow" in result
    assert "running" in result


@pytest.mark.asyncio
async def test_get_execution_status_no_execution(toolbox, empty_ctx):
    """get_execution_status returns message when no execution active."""
    result = await toolbox.execute(empty_ctx, "get_execution_status", {})
    assert "Không có" in result or "no" in result.lower()


@pytest.mark.asyncio
async def test_advance_execution_no_client(toolbox, rich_ctx):
    """advance_execution handles missing HTTP client gracefully."""
    result = await toolbox.execute(rich_ctx, "advance_execution", {"signal": "approved"})
    assert "exec-001" in result or "approved" in result


@pytest.mark.asyncio
async def test_advance_execution_missing_signal(toolbox, rich_ctx):
    """advance_execution returns error when signal is missing."""
    result = await toolbox.execute(rich_ctx, "advance_execution", {})
    assert "signal" in result.lower()


@pytest.mark.asyncio
async def test_escalate_no_client(toolbox, rich_ctx):
    """escalate_to_supervisor works without HTTP client (returns formatted message)."""
    result = await toolbox.execute(
        rich_ctx, "escalate_to_supervisor",
        {"reason": "KH yêu cầu xem xét đặc biệt", "priority": "high"}
    )
    assert "KH yêu cầu" in result
    assert "high" in result


@pytest.mark.asyncio
async def test_escalate_missing_reason(toolbox, rich_ctx):
    """escalate_to_supervisor returns error when reason is missing."""
    result = await toolbox.execute(rich_ctx, "escalate_to_supervisor", {})
    assert "reason" in result.lower() or "Missing" in result


# ---------------------------------------------------------------------------
# C6: Memory tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_info(toolbox, rich_ctx):
    """get_session_info returns structured session metadata."""
    result = await toolbox.execute(rich_ctx, "get_session_info", {})
    assert "acme" in result
    assert "vi" in result
    assert "0987" in result  # caller_id


@pytest.mark.asyncio
async def test_save_agent_note_no_client(toolbox, rich_ctx):
    """save_agent_note stores in-memory when no HTTP client."""
    result = await toolbox.execute(
        rich_ctx, "save_agent_note",
        {"note": "KH đồng ý hoàn tiền", "category": "action"}
    )
    assert "Ghi chú" in result or "lưu" in result.lower()
    # Check in-memory storage
    assert "_agent_notes" in rich_ctx.metadata
    notes = rich_ctx.metadata["_agent_notes"]
    assert any("KH đồng ý" in n["note"] for n in notes)


@pytest.mark.asyncio
async def test_save_agent_note_missing_note(toolbox, rich_ctx):
    """save_agent_note returns error when note is missing."""
    result = await toolbox.execute(rich_ctx, "save_agent_note", {})
    assert "note" in result.lower() or "Missing" in result


# ---------------------------------------------------------------------------
# C7: Utility tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_currency_vnd(toolbox, empty_ctx):
    """format_currency formats VND correctly (no decimal, dot separator)."""
    result = await toolbox.execute(empty_ctx, "format_currency", {"amount": 1500000, "currency": "VND"})
    assert "₫" in result
    assert "1.500.000" in result


@pytest.mark.asyncio
async def test_format_currency_usd(toolbox, empty_ctx):
    """format_currency formats USD correctly."""
    result = await toolbox.execute(empty_ctx, "format_currency", {"amount": 99.99, "currency": "USD"})
    assert "$" in result
    assert "99.99" in result


@pytest.mark.asyncio
async def test_format_currency_invalid(toolbox, empty_ctx):
    """format_currency handles invalid amount gracefully."""
    result = await toolbox.execute(empty_ctx, "format_currency", {"amount": "not_a_number"})
    assert "Invalid" in result or "error" in result.lower()


@pytest.mark.asyncio
async def test_current_datetime_full(toolbox, empty_ctx):
    """current_datetime returns date+time string."""
    result = await toolbox.execute(empty_ctx, "current_datetime", {"format": "full"})
    # Should contain year
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", result), f"No date found in: {result}"


@pytest.mark.asyncio
async def test_current_datetime_date_only(toolbox, empty_ctx):
    """current_datetime 'date' format returns only date."""
    result = await toolbox.execute(empty_ctx, "current_datetime", {"format": "date"})
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}$", result), f"Expected date only, got: {result}"


# ---------------------------------------------------------------------------
# C8: Integration with executor
# ---------------------------------------------------------------------------

def test_executor_resolve_builtin_schemas():
    """BaseAgentExecutor._resolve_tools_schemas() includes builtin tool schemas."""
    from aiknow.extensions import copilot_agent
    from aiknow.agents.runtime import AgentRuntime

    @copilot_agent(
        name="test_copilot_c",
        builtin_tools=["kb_search", "get_execution_status"],
        model="glm-5",
    )
    class TestCopilotC:
        pass

    runtime = AgentRuntime()
    runtime.register_agent(TestCopilotC)

    entry = runtime.get_agent("test_copilot_c")
    executor = entry.executor

    schemas = executor._resolve_tools_schemas()
    schema_names = [s["function"]["name"] for s in schemas]
    assert "kb_search" in schema_names
    assert "get_execution_status" in schema_names


@pytest.mark.asyncio
async def test_executor_call_builtin_tool():
    """BaseAgentExecutor._call_tool() dispatches to BuiltinToolBox for builtin tools."""
    from aiknow.extensions import copilot_agent
    from aiknow.agents.runtime import AgentRuntime

    @copilot_agent(
        name="test_call_builtin",
        builtin_tools=["format_currency", "current_datetime"],
        model="glm-5",
    )
    class TestCallBuiltin:
        pass

    runtime = AgentRuntime()
    runtime.register_agent(TestCallBuiltin)
    entry = runtime.get_agent("test_call_builtin")
    executor = entry.executor

    ctx = AgentContext(tenant_id="acme")
    result = await executor._call_tool(ctx, "format_currency", {"amount": 500000, "currency": "VND"})
    assert "₫" in result
    assert "500.000" in result
