"""Memory tools: get_session_info, get_conversation_history, save_agent_note."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, TYPE_CHECKING

from aiknow.agents.builtin_tools._base import BuiltinTool, ToolParam

if TYPE_CHECKING:
    from aiknow.agents.builtin_tools._toolbox import BuiltinToolContext


class GetSessionInfoTool(BuiltinTool):
    """Get current session metadata (caller_id, agent, channel, etc.)."""

    name = "get_session_info"
    description = {
        "vi": "Lấy thông tin phiên làm việc hiện tại (caller, kênh, agent)",
        "en": "Get current session metadata (caller, channel, agent info)",
    }
    parameters = []

    async def execute(self, ctx: "BuiltinToolContext", **kwargs: Any) -> str:
        agent_ctx = ctx.agent_ctx
        info: dict[str, Any] = {
            "session_id": agent_ctx.session_id,
            "tenant_id": agent_ctx.tenant_id,
            "locale": agent_ctx.locale,
            "execution_id": agent_ctx.execution_id,
            "sop_id": agent_ctx.sop_id,
            "execution_status": agent_ctx.execution_status,
            **agent_ctx.metadata,
        }
        # Format nicely
        lines = ["**Thông tin phiên làm việc:**\n"]
        for key, val in info.items():
            if val:
                label = key.replace("_", " ").title()
                lines.append(f"- **{label}**: {val}")
        return "\n".join(lines)


class GetConversationHistoryTool(BuiltinTool):
    """Get recent conversation history for the current session."""

    name = "get_conversation_history"
    description = {
        "vi": "Lấy lịch sử hội thoại trong phiên làm việc hiện tại",
        "en": "Get recent conversation history for the current session",
    }
    parameters = [
        ToolParam(
            name="last_n",
            type="integer",
            description={"en": "Number of recent turns to retrieve"},
            required=False,
            default=10,
        ),
    ]

    async def execute(self, ctx: "BuiltinToolContext", **kwargs: Any) -> str:
        last_n = int(kwargs.get("last_n", 10))
        session_id = ctx.agent_ctx.session_id

        if not session_id:
            return "Không có session hiện tại."

        if ctx.http_client is None:
            return "[get_conversation_history] HTTP client not configured."

        try:
            headers: dict[str, str] = {}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.get(
                f"{ctx.platform_url}/api/v1/sessions/{session_id}/history",
                params={"limit": last_n},
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            turns = data.get("turns", [])

            if not turns:
                return "Chưa có lịch sử hội thoại."

            lines = [f"**Lịch sử hội thoại** (gần nhất {len(turns)} turn):\n"]
            for turn in turns[-last_n:]:
                role = turn.get("role", "?").capitalize()
                content = turn.get("content", "")[:200]
                ts = turn.get("timestamp", "")
                lines.append(f"**{role}** ({ts}): {content}")

            return "\n".join(lines)

        except Exception as exc:  # noqa: BLE001
            return f"[get_conversation_history] Error: {exc}"


class SaveAgentNoteTool(BuiltinTool):
    """Save a structured note to the current session for future reference."""

    name = "save_agent_note"
    description = {
        "vi": "Lưu ghi chú vào phiên làm việc để tham chiếu sau",
        "en": "Save a note to the current session for future reference",
    }
    parameters = [
        ToolParam(
            name="note",
            type="string",
            description={"vi": "Nội dung ghi chú", "en": "Note content to save"},
            required=True,
        ),
        ToolParam(
            name="category",
            type="string",
            description={"en": "Note category (e.g. 'summary', 'action', 'warning')"},
            required=False,
            default="general",
        ),
    ]

    async def execute(self, ctx: "BuiltinToolContext", **kwargs: Any) -> str:
        note: str = kwargs.get("note", "")
        category: str = kwargs.get("category", "general")
        session_id = ctx.agent_ctx.session_id

        if not note:
            return "[save_agent_note] Missing required parameter: note"

        if ctx.http_client is None:
            # Store in agent_ctx.metadata as fallback
            notes = ctx.agent_ctx.metadata.setdefault("_agent_notes", [])
            notes.append({"category": category, "note": note, "ts": datetime.now().isoformat()})
            return f"✅ Ghi chú đã được lưu (in-memory): [{category}] {note[:80]}…"

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.post(
                f"{ctx.platform_url}/api/v1/sessions/{session_id}/notes",
                json={"note": note, "category": category},
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            return f"✅ Ghi chú đã được lưu: [{category}] {note[:80]}…"

        except Exception as exc:  # noqa: BLE001
            return f"[save_agent_note] Error: {exc}"
