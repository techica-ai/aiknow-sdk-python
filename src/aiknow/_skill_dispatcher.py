"""
SkillDispatcher — routes incoming skill invocation requests to the
correct @skill function in the local registry.

Responsibilities:
    1. Look up the @skill entry by name.
    2. Build AppSkillDeps from request context.
    3. Call the skill function (async).
    4. Normalize the return value to a SkillInvocationResponse.
    5. Catch and surface exceptions as error-status responses (never crash
       the WebhookListener server).

Design decisions:
    - Dispatcher is a plain class (no HTTP concern) — testable in isolation.
    - All exceptions from skill functions are caught and converted to
      error-status responses so the Platform always receives a valid JSON body.
    - AppSkillDeps is built from the request payload — the Platform is the
      authoritative source of tenant_id, session_id, locale, permissions.
"""
from __future__ import annotations

import logging
from typing import Any

from aiknow.extensions._local_registry import LocalExtensionRegistry
from aiknow.extensions.types import AppSkillDeps, SkillResult

logger = logging.getLogger(__name__)


class SkillInvocationResponse:
    """Serialisable response sent back to the Platform after skill execution.

    Attributes:
        status:        Outcome string (from SkillResult.status or 'error').
        data:          Structured data from SkillResult.data.
        next_state:    Optional hard state override from SkillResult.next_state.
        error_message: Set when skill raised an unhandled exception.
    """

    __slots__ = ("status", "data", "next_state", "error_message")

    def __init__(
        self,
        status: str,
        data: dict[str, Any] | None = None,
        next_state: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.status = status
        self.data: dict[str, Any] = data or {}
        self.next_state = next_state
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serialisable dict for HTTP response."""
        return {
            "status": self.status,
            "data": self.data,
            "next_state": self.next_state,
            "error_message": self.error_message,
        }


class SkillDispatcher:
    """Routes skill invocation payloads to the registered @skill function.

    Args:
        registry: The app's LocalExtensionRegistry (populated by @skill decorators).
    """

    def __init__(self, registry: LocalExtensionRegistry) -> None:
        self._registry = registry

    async def dispatch(
        self,
        skill_name: str,
        tenant_id: str,
        execution_id: str,
        params: dict[str, Any],
        session_id: str | None = None,
        locale: str = "vi",
        permissions: list[str] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> SkillInvocationResponse:
        """Find and invoke a registered skill, returning a serialisable response.

        Args:
            skill_name:    Skill to invoke (matches @skill name).
            tenant_id:     Tenant context (from Platform request).
            execution_id:  Workflow execution ID (for tracing).
            params:        Keyword arguments forwarded to the skill function.
            session_id:    Conversation session ID (optional).
            locale:        BCP-47 locale string.
            permissions:   RBAC permissions granted to this invocation.
            user_context:  Arbitrary seed data from start_execution.initial_inputs.
                           Accessible in skill via deps.user_context.

        Returns:
            SkillInvocationResponse — always (never raises, catches skill errors).
        """
        # Single lookup with "step" — registry.get() normalizes "skill" entries
        # automatically for backward compat (see LocalExtensionRegistry.get docstring)
        entry = self._registry.get(skill_name, "step")
        if entry is None:
            logger.warning(
                "SkillDispatcher: step '%s' not found in registry for tenant='%s'",
                skill_name, tenant_id,
            )
            return SkillInvocationResponse(
                status="error",
                error_message=(
                    f"Step '{skill_name}' is not registered in this app. "
                    "Ensure the @step decorator is applied and the step module is imported."
                ),
            )

        deps = AppSkillDeps(
            tenant_id=tenant_id,
            session_id=session_id,
            locale=locale,
            permissions=permissions or [],
            user_context=user_context or {},
            execution_id=execution_id,
        )

        logger.debug(
            "SkillDispatcher: invoking step='%s' execution_id='%s' params=%s",
            skill_name, execution_id, list(params.keys()),
        )

        try:
            result = await entry.fn(deps, **params)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "SkillDispatcher: step '%s' raised an unhandled exception: %s",
                skill_name, exc,
            )
            return SkillInvocationResponse(
                status="error",
                error_message=f"Step '{skill_name}' raised: {type(exc).__name__}: {exc}",
            )

        # Normalize return value: accept StepResult/SkillResult or plain str/dict
        return _normalize_result(skill_name, result)


def _normalize_result(
    skill_name: str,
    result: Any,
) -> SkillInvocationResponse:
    """Convert whatever the skill function returned into a SkillInvocationResponse.

    Accepted return types:
        - SkillResult (canonical — recommended)
        - str         → status=str, no data
        - dict        → must have 'status' key, optional 'data', 'next_state'

    Anything else is treated as an error.
    """
    if isinstance(result, SkillResult):
        return SkillInvocationResponse(
            status=result.status,
            data=result.data,
            next_state=result.next_state,
        )

    if isinstance(result, str):
        return SkillInvocationResponse(status=result)

    if isinstance(result, dict) and "status" in result:
        return SkillInvocationResponse(
            status=result["status"],
            data=result.get("data", {}),
            next_state=result.get("next_state"),
        )

    logger.warning(
        "SkillDispatcher: skill '%s' returned unexpected type '%s' — treating as error",
        skill_name, type(result).__name__,
    )
    return SkillInvocationResponse(
        status="error",
        error_message=(
            f"Skill '{skill_name}' returned unexpected type '{type(result).__name__}'. "
            "Skills must return SkillResult, str, or dict with 'status'."
        ),
    )
