"""
AsyncWorkflowsResource — SDK resource for managing workflow executions.

Wraps the Platform's /api/v1/workflow/* endpoints so App developers
can start, monitor, advance, and list workflow executions without writing
raw HTTP calls.

Usage::

    async with AsyncAIKnowClient(api_key="...") as client:
        # Start a refund flow
        exec_ = await client.workflows.start_execution(
            workflow_id="refund_flow",
            session_id="sess-abc",
            initial_inputs={"phone": "0912345678", "order_id": "ORD-001"},
        )

        # Poll status
        state = await client.workflows.get_execution(exec_["execution_id"])

        # Agent approves (after human-wait)
        updated = await client.workflows.advance(
            execution_id=exec_["execution_id"],
            signal="approved",
            payload={"agent_id": "agent-01", "note": "VIP customer"},
        )

        # List all waiting executions
        pending = await client.workflows.list_executions(status="waiting_human")
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from aiknow._http import raise_for_status, wrap_httpx_errors

logger = logging.getLogger(__name__)

WorkflowStatus = Literal["running", "waiting_human", "completed", "failed", "escalated"]


class ExecutionState:
    """Typed representation of a workflow execution snapshot.

    Returned by start_execution(), get_execution(), advance().

    Attributes:
        execution_id:      Unique execution identifier (UUID).
        workflow_id:       SOP definition ID (e.g. 'refund_flow').
        tenant_id:         Owning tenant.
        current_state:     Current FSM node ID.
        status:            One of: running, waiting_human, completed, failed, escalated.
        waiting_for_human: True when paused awaiting a human signal.
        skill_outputs:     Map of state_id → SkillResult.data for completed nodes.
        error_message:     Present only when status == 'failed'.
        available_signals: Valid signal strings for the current human-wait node.
                           Non-empty only when status == 'waiting_human'.
                           Example: ['approved', 'rejected', 'timeout'].
        raw:               Original response dict from Platform.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.execution_id: str = data["execution_id"]
        self.workflow_id: str = data["workflow_id"]
        self.tenant_id: str = data["tenant_id"]
        self.current_state: str = data["current_state"]
        self.status: WorkflowStatus = data["status"]
        self.waiting_for_human: bool = data.get("waiting_for_human", False)
        self.skill_outputs: dict[str, dict[str, Any]] = data.get("skill_outputs", {})
        self.error_message: str | None = data.get("error_message")
        self.metadata: dict[str, Any] = data.get("metadata", {})
        # Phase 2: available_signals populated by Platform when status='waiting_human'
        self.available_signals: list[str] = data.get("available_signals", [])
        self.raw: dict[str, Any] = data

    def __repr__(self) -> str:
        return (
            f"ExecutionState(id={self.execution_id!r}, "
            f"workflow={self.workflow_id!r}, "
            f"status={self.status!r}, "
            f"state={self.current_state!r})"
        )

    @property
    def is_terminal(self) -> bool:
        """True when the execution has reached a terminal state."""
        return self.status in ("completed", "failed", "escalated")

    @property
    def is_waiting(self) -> bool:
        """True when paused waiting for a human signal."""
        return self.status == "waiting_human"


class AsyncWorkflowsResource:
    """Async HTTP resource for managing workflow executions.

    Args:
        client: Shared httpx.AsyncClient (injected by AsyncAIKnowClient).
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def start_execution(
        self,
        workflow_id: str,
        initial_inputs: dict[str, Any] | None = None,
        session_id: str | None = None,
        traceparent: str | None = None,
    ) -> ExecutionState:
        """Start a new workflow execution.

        Loads the workflow definition from Platform, compiles it into a state
        machine, and runs until completion or a human-wait state.

        Args:
            workflow_id:    ID of the workflow to execute (must be registered).
            initial_inputs: Seed data forwarded to the first skill node.
                            Example: {"phone": "0912345678", "order_id": "ORD-001"}.
                            Available in each skill via AppSkillDeps.user_context.
            session_id:     Optional chat session to link this execution to.
            traceparent:    W3C traceparent header value (``00-<trace_id>-<span_id>-01``).
                            When provided, the workflow trace will be linked as a child
                            of the caller's conversation trace in the dashboard.

        Returns:
            ExecutionState snapshot. May already be completed for simple workflows
            that have no human-wait states.

        Raises:
            AIKnowAPIError:        404 if workflow not found; 422 on bad input.
            AIKnowConnectionError: Network-level failure.
        """
        payload: dict[str, Any] = {"workflow_id": workflow_id}
        if initial_inputs:
            payload["initial_inputs"] = initial_inputs
        if session_id:
            payload["session_id"] = session_id

        # Build extra headers for W3C trace context propagation
        extra_headers: dict[str, str] = {}
        if traceparent:
            extra_headers["traceparent"] = traceparent

        try:
            response = await self._client.post(
                "/workflow/executions",
                json=payload,
                headers=extra_headers or None,
            )
        except Exception as exc:
            wrap_httpx_errors("Workflows.start_execution", exc)
        raise_for_status("Workflows.start_execution", response)

        state = ExecutionState(response.json())
        logger.info(
            "Execution started: id=%s workflow=%s status=%s",
            state.execution_id, state.workflow_id, state.status,
        )
        return state

    async def get_execution(self, execution_id: str) -> ExecutionState:
        """Retrieve the current snapshot of a workflow execution.

        Args:
            execution_id: UUID of the execution to retrieve.

        Returns:
            Current ExecutionState.

        Raises:
            AIKnowAPIError: 404 if execution not found; 403 if access denied.
        """
        try:
            response = await self._client.get(f"/workflow/executions/{execution_id}")
        except Exception as exc:
            wrap_httpx_errors("Workflows.get_execution", exc)
        raise_for_status("Workflows.get_execution", response)
        return ExecutionState(response.json())

    async def advance(
        self,
        execution_id: str,
        signal: str,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionState:
        """Resume a paused (waiting_human) workflow execution.

        Sends a human signal to the Platform, which resumes the LangGraph
        FSM from the human-wait state.

        Args:
            execution_id: UUID of the waiting execution.
            signal:       Outcome string matching a transition from the
                          current human-wait state. Example: 'approved', 'rejected'.
            payload:      Optional extra context from the human approver.
                          Example: {"agent_id": "agent-01", "note": "VIP customer"}.

        Returns:
            Updated ExecutionState after resumption.

        Raises:
            AIKnowAPIError: 404 not found; 403 access denied;
                            422 if execution is not in waiting_human status.
        """
        body: dict[str, Any] = {"signal": signal}
        if payload:
            body["payload"] = payload

        try:
            response = await self._client.post(
                f"/workflow/executions/{execution_id}/advance",
                json=body,
            )
        except Exception as exc:
            wrap_httpx_errors("Workflows.advance", exc)
        raise_for_status("Workflows.advance", response)

        state = ExecutionState(response.json())
        logger.info(
            "Execution advanced: id=%s signal=%s status=%s",
            execution_id, signal, state.status,
        )
        return state

    async def list_executions(
        self,
        status: WorkflowStatus | None = None,
        workflow_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[ExecutionState]:
        """List workflow executions with optional filters.

        Useful for Agent Desktop to pull the pending escalation queue:
            pending = await client.workflows.list_executions(status="waiting_human")

        Args:
            status:      Filter by execution status. None = all statuses.
            workflow_id: Filter by workflow definition ID. None = all workflows.
            page:        Page number (1-indexed).
            page_size:   Results per page (max 100).

        Returns:
            List of ExecutionState snapshots, most recent first.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if workflow_id:
            params["workflow_id"] = workflow_id

        try:
            response = await self._client.get("/workflow/executions", params=params)
        except Exception as exc:
            wrap_httpx_errors("Workflows.list_executions", exc)
        raise_for_status("Workflows.list_executions", response)
        return [ExecutionState(item) for item in response.json()]
