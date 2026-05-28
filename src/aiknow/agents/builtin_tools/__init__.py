"""
agents/builtin_tools — SDK-provided built-in tools for agents.

Public API:
    BuiltinToolBox     — Central registry and executor
    BuiltinTool        — Abstract base class for all built-in tools
    BuiltinToolContext — Infra access object passed to each tool.execute()
    ToolParam          — Tool parameter schema descriptor

Built-in tools (13 tools, 4 categories):
    KB tools:
        kb_search              — Hybrid vector search
        kb_list_sources        — List KB sources
        kb_get_document        — Get full document content

    Platform tools:
        get_execution_status   — Get workflow execution status
        list_executions        — List recent executions
        advance_execution      — Advance waiting_human execution
        escalate_to_supervisor — Escalate case to supervisor
        get_workflow_definition — Get workflow state machine definition

    Memory tools:
        get_session_info       — Get session metadata
        get_conversation_history — Get recent conversation turns
        save_agent_note        — Save note to session

    Utility tools:
        format_currency        — Format amounts (VND/USD/EUR)
        current_datetime       — Get current date/time with timezone
"""
from aiknow.agents.builtin_tools._base import BuiltinTool, ToolParam
from aiknow.agents.builtin_tools._toolbox import BuiltinToolBox, BuiltinToolContext

__all__ = [
    "BuiltinToolBox",
    "BuiltinTool",
    "BuiltinToolContext",
    "ToolParam",
]
