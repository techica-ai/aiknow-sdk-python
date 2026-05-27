"""
aiknow.extensions — public extension API for AIKnow app developers.

Exports::

    from aiknow.extensions import step, sop, hook, parser, tool
    from aiknow.extensions import StepDeps, StepResult, HookEvent
    from aiknow.extensions import copilot_agent, rag_agent, autonomous_agent
    from aiknow.extensions import router_agent, supervisor_agent, conversation_agent

Deprecated (use step instead of skill, StepDeps instead of AppSkillDeps, etc.)::

    from aiknow.extensions import skill          # → use step
    from aiknow.extensions import AppSkillDeps   # → use StepDeps
    from aiknow.extensions import SkillResult    # → use StepResult
"""
from __future__ import annotations

from aiknow.extensions._decorators import (
    autonomous_agent,
    conversation_agent,
    copilot_agent,
    hook,
    parser,
    rag_agent,
    router_agent,
    skill,  # deprecated — kept for backward compat
    sop,
    step,
    supervisor_agent,
    tool,
)
from aiknow.extensions._local_registry import LocalExtensionRegistry
from aiknow.extensions.types import HookEvent, StepDeps, StepResult

# Deprecated aliases — import them lazily via types module __getattr__
# Direct re-export here for IDE autocomplete (types module warns at access time)
AppSkillDeps = StepDeps   # noqa: N816 — deprecated alias
SkillResult = StepResult  # noqa: N816 — deprecated alias

__all__ = [
    # Decorators — SOP
    "step",
    "sop",
    "hook",
    "parser",
    # Decorators — Agent Tools
    "tool",
    # Decorators — Agent Patterns
    "copilot_agent",
    "rag_agent",
    "autonomous_agent",
    "router_agent",
    "supervisor_agent",
    "conversation_agent",
    # Types
    "StepDeps",
    "StepResult",
    "HookEvent",
    # Registry (advanced usage)
    "LocalExtensionRegistry",
    # Deprecated
    "skill",
    "AppSkillDeps",
    "SkillResult",
]
