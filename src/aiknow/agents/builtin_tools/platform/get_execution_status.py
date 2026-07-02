"""Platform tools: get_execution_status, list_executions, advance_execution,
escalate_to_supervisor, get_workflow_definition."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from aiknow.agents.builtin_tools._base import BuiltinTool, ToolParam

if TYPE_CHECKING:
    from aiknow.agents.builtin_tools._toolbox import BuiltinToolContext


class GetExecutionStatusTool(BuiltinTool):
    """Get current status and state of a workflow execution."""

    name = "get_execution_status"
    description = {
        "vi": "Lấy trạng thái hiện tại của luồng workflow đang thực thi",
        "en": "Get current status of an active workflow execution",
    }
    parameters = [
        ToolParam(
            name="execution_id",
            type="string",
            description={"en": "Execution ID to query (uses active execution if omitted)"},
            required=False,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        exec_id = kwargs.get("execution_id") or ctx.agent_ctx.execution_id

        if not exec_id:
            return "Không có luồng SOP nào đang hoạt động."

        # Use in-memory context first (fast path)
        agent_ctx = ctx.agent_ctx
        if agent_ctx.execution_id == exec_id:
            return json.dumps({
                "execution_id": exec_id,
                "workflow_id": agent_ctx.workflow_id,
                "current_state": agent_ctx.current_state,
                "status": agent_ctx.execution_status,
                "skill_outputs": agent_ctx.skill_outputs,
            }, ensure_ascii=False, indent=2)

        if ctx.http_client is None:
            return f"[get_execution_status] HTTP client not configured for exec: {exec_id}"

        try:
            headers: dict[str, str] = {}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"
            response = await ctx.http_client.get(
                f"executions/{exec_id}",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            return cast(str, response.text)
        except Exception as exc:  # noqa: BLE001
            return f"[get_execution_status] Error: {exc}"


class ListExecutionsTool(BuiltinTool):
    """List recent workflow executions for the current tenant."""

    name = "list_executions"
    description = {
        "vi": "Liệt kê các luồng workflow đã và đang thực thi",
        "en": "List recent workflow executions for this tenant",
    }
    parameters = [
        ToolParam(
            name="status",
            type="string",
            description={"en": "Filter by status (running/waiting_human/completed/failed)"},
            required=False,
            enum=["running", "waiting_human", "completed", "failed"],
        ),
        ToolParam(
            name="limit",
            type="integer",
            description={"en": "Max number of results"},
            required=False,
            default=10,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        status_filter = kwargs.get("status")
        limit = int(kwargs.get("limit", 10))

        if ctx.http_client is None:
            return "[list_executions] HTTP client not configured."

        try:
            params: dict[str, Any] = {
                "tenant_id": ctx.tenant_id,
                "limit": limit,
            }
            if status_filter:
                params["status"] = status_filter

            headers: dict[str, str] = {}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.get(
                f"executions",
                params=params,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])

            if not items:
                return "Không có luồng SOP nào phù hợp."

            lines = [f"**Danh sách executions** (tổng: {len(items)}):\n"]
            for item in items:
                exec_id = item.get("execution_id", "?")
                workflow_id = item.get("workflow_id", "?")
                item_status = item.get("status", "?")
                state = item.get("current_state", "?")
                lines.append(f"- `{exec_id}` — {workflow_id} | {item_status} | state: {state}")

            return "\n".join(lines)

        except Exception as exc:  # noqa: BLE001
            return f"[list_executions] Error: {exc}"


class AdvanceExecutionTool(BuiltinTool):
    """Advance a waiting_human workflow execution with a signal/outcome."""

    name = "advance_execution"
    description = {
        "vi": "Tiếp tục luồng workflow đang chờ xác nhận từ người dùng",
        "en": "Advance a waiting workflow execution with an approval signal",
    }
    parameters = [
        ToolParam(
            name="execution_id",
            type="string",
            description={"en": "Execution ID to advance (uses active if omitted)"},
            required=False,
        ),
        ToolParam(
            name="signal",
            type="string",
            description={"en": "Outcome signal (e.g. 'approved', 'rejected', 'confirmed')"},
            required=True,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        exec_id = kwargs.get("execution_id") or ctx.agent_ctx.execution_id
        signal: str = kwargs.get("signal", "")

        if not exec_id:
            return "Không có luồng workflow nào đang chờ."
        if not signal:
            return "[advance_execution] Missing required parameter: signal"

        if ctx.http_client is None:
            return (
                f"[advance_execution] HTTP client not configured. "
                f"Would advance {exec_id} with '{signal}'."
            )

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.post(
                f"executions/{exec_id}/advance",
                json={"signal": signal},
                headers=headers,
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            new_status = data.get("status", "unknown")
            new_state = data.get("current_state", "unknown")
            return (
                f"✅ Đã tiếp tục luồng `{exec_id}` với signal `{signal}`. "
                f"Trạng thái mới: **{new_status}** (state: {new_state})"
            )

        except Exception as exc:  # noqa: BLE001
            return f"[advance_execution] Error: {exc}"


class EscalateToSupervisorTool(BuiltinTool):
    """Escalate the current case to a human supervisor."""

    name = "escalate_to_supervisor"
    description = {
        "vi": "Leo thang ca làm việc lên supervisor hoặc bộ phận hỗ trợ",
        "en": "Escalate the current case to a human supervisor",
    }
    parameters = [
        ToolParam(
            name="reason",
            type="string",
            description={"vi": "Lý do leo thang", "en": "Reason for escalation"},
            required=True,
        ),
        ToolParam(
            name="priority",
            type="string",
            description={"en": "Escalation priority: low/medium/high/critical"},
            required=False,
            default="medium",
            enum=["low", "medium", "high", "critical"],
        ),
        ToolParam(
            name="notes",
            type="string",
            description={"en": "Additional notes for the supervisor"},
            required=False,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        reason: str = kwargs.get("reason", "")
        priority: str = kwargs.get("priority", "medium")
        notes: str = kwargs.get("notes", "")
        exec_id = ctx.agent_ctx.execution_id

        if not reason:
            return "[escalate_to_supervisor] Missing required parameter: reason"

        if ctx.http_client is None:
            return (
                f"⚠️ Leo thang ca làm việc (HTTP client không khả dụng).\n"
                f"- Lý do: {reason}\n"
                f"- Mức độ ưu tiên: {priority}\n"
                f"- Ghi chú: {notes}"
            )

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            payload: dict[str, Any] = {
                "tenant_id": ctx.tenant_id,
                "execution_id": exec_id,
                "reason": reason,
                "priority": priority,
            }
            if notes:
                payload["notes"] = notes

            response = await ctx.http_client.post(
                f"escalations",
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            ticket_id = response.json().get("ticket_id", "N/A")
            return (
                f"✅ Đã leo thang ca làm việc thành công.\n"
                f"- Ticket ID: `{ticket_id}`\n"
                f"- Mức độ ưu tiên: **{priority}**\n"
                f"- Lý do: {reason}"
            )

        except Exception as exc:  # noqa: BLE001
            return f"[escalate_to_supervisor] Error: {exc}"


class GetWorkflowDefinitionTool(BuiltinTool):
    """Get the definition and transitions of a workflow."""

    name = "get_workflow_definition"
    description = {
        "vi": "Lấy định nghĩa và các bước chuyển trạng thái của một workflow",
        "en": "Get the definition and state transitions of a workflow",
    }
    parameters = [
        ToolParam(
            name="workflow_id",
            type="string",
            description={"en": "Workflow ID to retrieve (uses active workflow if omitted)"},
            required=False,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        workflow_id = kwargs.get("workflow_id") or ctx.agent_ctx.workflow_id

        if not workflow_id:
            return "Không có workflow nào đang hoạt động."

        if ctx.http_client is None:
            return (
                f"[get_workflow_definition] HTTP client not configured "
                f"for workflow: {workflow_id}"
            )

        try:
            headers: dict[str, str] = {}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.get(
                f"workflow/definitions/{workflow_id}",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            nodes = data.get("nodes", {})
            lines = [f"**Workflow: {workflow_id}** (version: {data.get('version', '?')})\n"]
            for state_id, node in nodes.items():
                transitions = node.get("transitions", [])
                trans_str = ", ".join(
                    f"{t.get('outcome')} → {t.get('target_state')}"
                    for t in transitions
                ) or "(terminal)"
                lines.append(f"- **{state_id}**: {trans_str}")
            return "\n".join(lines)

        except Exception as exc:  # noqa: BLE001
            return f"[get_workflow_definition] Error: {exc}"
