"""
agents/routing.py — Pure data types for intent-based SOP routing.

These types form the contract between:
    - App layer: declares *what* intents exist (IntentSpec)
    - SDK layer: decides *how* to route (RouterAgentExecutor → RouteDecision)

Design principles:
    - IntentSpec is a pure data class. The SDK reads it but never interprets
      the business meaning of route_id, keywords, or llm_description.
    - RouteDecision is the opaque output. The app receives route_id as a string
      and decides what to do with it (e.g. call SOPTrigger).
    - No LLM calls, no regex compilation here — this file has zero side effects
      on import.

Usage (App layer)::

    from aiknow.agents.routing import IntentSpec, RouteDecision

    INTENT_SPECS = [
        IntentSpec(
            route_id="refund_flow",
            keywords=[r"refund", r"hoàn tiền"],
            llm_description="Customer wants a refund or return",
            slot_patterns={"order_id": r"ORD-\\d+"},
        ),
    ]

Usage (SDK layer — internal)::

    decision: RouteDecision = await executor.route(text)
    if decision.route_id:
        await sop_trigger.trigger(route=decision, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentSpec:
    """Declares one routable intent — pure data, no logic.

    The SDK uses this to:
        1. Compile keyword regex patterns (``keywords``) for fast Layer-1 scan.
        2. Build the LLM classification prompt from ``llm_description``
           and ``example_utterances`` for Layer-2 fallback.
        3. Extract slot values after a match using ``slot_patterns``.

    The App is responsible for providing meaningful values.
    The SDK never interprets the *meaning* of ``route_id`` or ``keywords``.

    Attributes:
        route_id:
            Opaque identifier for the intent. Typically a SOP ID
            (e.g. ``"refund_flow"``) or an agent name. The SDK treats
            this as a plain string and returns it unchanged in RouteDecision.
        keywords:
            List of regex pattern strings. The SDK compiles these with
            ``re.IGNORECASE``. First spec whose any pattern matches wins
            (Layer 1 — keyword scan).
        llm_description:
            Human-readable description used verbatim in the auto-generated
            LLM prompt. Written by the App in whatever language suits the
            LLM (typically English). Example: ``"Customer wants a refund"``.
        example_utterances:
            Optional sample user messages. Appended to the LLM prompt as
            few-shot examples to improve classification accuracy.
        slot_patterns:
            Dict mapping slot names to regex patterns. After a route is
            determined, the SDK runs these patterns against the input text
            and populates ``RouteDecision.slots``.
            Key   = slot name, passed as-is into SOP ``initial_inputs``.
            Value = regex pattern string (compiled by SDK, not by App).
        priority:
            Tie-breaker when multiple specs match the same keyword scan.
            Higher priority wins. Default 0.
    """

    route_id: str
    keywords: list[str]
    llm_description: str
    example_utterances: list[str] = field(default_factory=list)
    slot_patterns: dict[str, str] = field(default_factory=dict)
    priority: int = 0


@dataclass
class RouteDecision:
    """Result of RouterAgentExecutor.route().

    Returned by the SDK to the App after intent classification and slot
    extraction. The App uses ``route_id`` to decide which SOP to trigger.

    Attributes:
        route_id:
            The matched intent's ``route_id`` from IntentSpec, or ``None``
            if no intent was recognised with sufficient confidence.
        confidence:
            Classification confidence in [0.0, 1.0].
            Always 1.0 for keyword matches. LLM-derived for Layer-2 results.
        method:
            How the decision was made:
            - ``"keyword"`` — Layer 1 regex scan hit.
            - ``"llm"``     — Layer 2 LLM classification.
            - ``"none"``    — No match (route_id will be None).
        slots:
            Extracted slot values keyed by slot name (from IntentSpec.slot_patterns).
            These are passed directly into the SOP's ``initial_inputs``.
            Empty dict if no slots were extracted or route_id is None.
    """

    route_id: str | None
    confidence: float
    method: str  # "keyword" | "llm" | "none"
    slots: dict[str, Any] = field(default_factory=dict)
