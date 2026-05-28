"""
Local Extension Registry — collects registered extensions within the current process.

This is the client-side registry. Before an AiknowApp starts, it collects
all @step/@sop/@dialog/@hook/@tool/@*_agent registrations here. On app.start(),
the registry contents are serialised and uploaded to the Platform.

Design:
- Module-level singleton (_default_registry) used by decorators
- AiknowApp also maintains its own instance for explicit registrations
- Thread-safe for read; append-only writes (no concurrent mutation needed)

Terminology:
- "step"  : SOP workflow step (replaces deprecated "skill" in SOP context)
- "sop"   : SOP finite-state-machine declaration (@sop)
- "dialog": Slot-filling dialog declaration (@dialog)
- "hook"  : lifecycle event handler
- "parser": custom document parser
- "tool"  : agent tool (callable by LLM agents)
- "agent" : AI agent declaration (@copilot_agent, @rag_agent, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExtType = Literal["skill", "step", "sop", "dialog", "hook", "parser", "tool", "agent"]


@dataclass
class ExtensionEntry:
    """One registered extension."""

    name: str
    """Unique name within its type (e.g. skill name, sop_id, event_type)."""

    ext_type: ExtType
    """Classification: skill | sop | hook | parser."""

    fn: Any
    """The callable (function or class) being registered."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Decorator-provided metadata: description, permissions, tags, etc."""


class LocalExtensionRegistry:
    """In-process registry for extensions declared via decorators.

    Usage::

        registry = LocalExtensionRegistry()

        @skill("fetch_customer")
        async def fetch_customer(deps, phone): ...

        registry.register(ExtensionEntry("fetch_customer", "skill", fetch_customer, {...}))
        skills = registry.list_by_type("skill")
    """

    def __init__(self) -> None:
        self._entries: list[ExtensionEntry] = []

    def register(self, entry: ExtensionEntry) -> None:
        """Add an extension entry. Duplicate names within same type are allowed
        (last registration wins at lookup time)."""
        self._entries.append(entry)

    def list_by_type(self, ext_type: ExtType) -> list[ExtensionEntry]:
        """Return all entries of the given type.

        Note: querying "step" also returns legacy "skill" entries for
        backward compatibility during the deprecation window.
        """
        if ext_type == "step":
            return [e for e in self._entries if e.ext_type in ("step", "skill")]
        return [e for e in self._entries if e.ext_type == ext_type]

    def get(self, name: str, ext_type: ExtType) -> ExtensionEntry | None:
        """Return the most recently registered entry matching name+type.

        When querying ``ext_type="step"``, also searches ``"skill"`` entries
        for backward compatibility — mirrors the same logic in ``list_by_type``.
        This ensures ``SkillDispatcher`` can use a single ``get(name, "step")``
        call without worrying about which decorator the App author used.
        """
        search_types = (ext_type,)
        if ext_type == "step":
            search_types = ("step", "skill")  # backward compat
        matches = [e for e in self._entries if e.name == name and e.ext_type in search_types]
        return matches[-1] if matches else None

    def all(self) -> list[ExtensionEntry]:
        return list(self._entries)

    def describe(self) -> dict[str, list[dict]]:
        """Return a dict suitable for debug display / manifest serialisation."""
        result: dict[str, list[dict]] = {
            "steps": [],
            "sops": [],
            "dialogs": [],
            "hooks": [],
            "parsers": [],
            "tools": [],
            "agents": [],
            # backward-compat key — same contents as "steps"
            "skills": [],
        }
        for e in self._entries:
            if e.ext_type == "skill":
                result["skills"].append({"name": e.name, **e.metadata})
                result["steps"].append({"name": e.name, **e.metadata})
            elif e.ext_type == "step":
                result["steps"].append({"name": e.name, **e.metadata})
                result["skills"].append({"name": e.name, **e.metadata})
            elif (e.ext_type + "s") in result:
                result[e.ext_type + "s"].append({"name": e.name, **e.metadata})
        return result


# Module-level default registry — used by decorators at import time.
_default_registry = LocalExtensionRegistry()
