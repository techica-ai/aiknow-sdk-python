"""
agents/executors/router.py — RouterAgentExecutor: 2-layer intent routing engine.

This executor is built by AgentRuntime when a ``@router_agent``-decorated class
is registered. It is *not* a BaseAgentExecutor subclass because it does not serve
HTTP/AG-UI endpoints — it is invoked in-process by ConversationManager.

Architecture:
    App declares intents via IntentSpec (what to route to).
    RouterAgentExecutor decides how to route (keyword scan → LLM fallback).
    The SDK never sees business-specific keyword strings or SOP names at
    definition time — all domain data arrives as List[IntentSpec] at runtime.

    Layer 1 — Keyword scan (O(n) compiled regex):
        Deterministic, zero-latency, zero-cost.
        Returns confidence=1.0 on first match.

    Layer 2 — LLM classification (via LiteLLM gateway):
        Invoked only when Layer 1 finds no match.
        Prompt is generated automatically from IntentSpec.llm_description.
        Model and gateway URL are resolved by AgentRuntime (not by App).

    Slot extraction:
        After a route is determined, slot_patterns from the matched spec
        are applied to extract named values (e.g. phone_number, order_id).

Infrastructure contract:
    llm_gateway_url — injected by AgentRuntime from LITELLM_BASE_URL env var.
    model           — from @router_agent(model=...) or AIKNOW_DEFAULT_MODEL env.
    The App never sets these; they are SDK-internal concerns.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from aiknow.agents.routing import IntentSpec, RouteDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — no business terms, no language-specific content
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_TEMPLATE = """\
You are an intent classifier. Given a customer message, classify it into \
EXACTLY ONE of the following intents or "none" if none applies:

{intent_list}
  - none: the message does not match any of the above intents

Rules:
- Output ONLY valid JSON, no markdown fences, no explanation.
- "confidence" must be a float between 0.0 and 1.0.
- Choose the single best matching intent.

Response format:
{{"intent": "<intent_id_or_none>", "confidence": <0.0-1.0>}}"""

_ROUTER_USER_TEMPLATE = "Customer message: {text}"


# ---------------------------------------------------------------------------
# Compiled spec (internal)
# ---------------------------------------------------------------------------

@dataclass
class _CompiledSpec:
    """Pre-compiled version of IntentSpec for fast matching."""
    spec: IntentSpec
    patterns: list[re.Pattern]
    slot_patterns: dict[str, re.Pattern]


# ---------------------------------------------------------------------------
# RouterAgentExecutor
# ---------------------------------------------------------------------------


class RouterAgentExecutor:
    """In-process intent routing engine — SDK-internal, App-agnostic.

    Built by AgentRuntime._build_executor() when agent_type=="router".
    Injected into ConversationManager by the App via
    AiknowApp.build_router_executor().

    Args:
        specs:                List[IntentSpec] provided by the App via
                              @router_agent(intents=...).
        model:                LLM model alias (resolved by AgentRuntime from
                              meta.model or AIKNOW_DEFAULT_MODEL env var).
        llm_gateway_url:      LiteLLM proxy base URL — injected by
                              AgentRuntime, never set by the App.
        confidence_threshold: Minimum LLM confidence to accept a result.
                              Results below this are treated as "none".
        llm_timeout:          Timeout in seconds for LLM calls.
    """

    def __init__(
        self,
        specs: list[IntentSpec],
        model: str,
        llm_gateway_url: str,
        confidence_threshold: float = 0.6,
        llm_timeout: float = 10.0,
    ) -> None:
        self._model = model
        self._llm_gateway_url = llm_gateway_url
        self._threshold = confidence_threshold
        self._llm_timeout = llm_timeout

        # Compile all specs once at startup
        self._compiled: list[_CompiledSpec] = self._compile_specs(specs)

        # Pre-build LLM prompt (static per executor lifetime)
        self._llm_system_prompt: str = self._build_llm_prompt()

        # Fast lookup: route_id → compiled spec
        self._spec_index: dict[str, _CompiledSpec] = {
            cs.spec.route_id: cs for cs in self._compiled
        }

        valid_ids = [cs.spec.route_id for cs in self._compiled]
        logger.info(
            "RouterAgentExecutor: initialised with %d intent(s): %s "
            "(model=%s, threshold=%.2f)",
            len(self._compiled), valid_ids, model, confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(self, text: str) -> RouteDecision:
        """Classify text and extract slots.

        Layer 1 (keyword) → Layer 2 (LLM fallback) → slot extraction.

        Args:
            text: Raw customer utterance.

        Returns:
            RouteDecision — always returned, never raises on LLM errors.
        """
        # Layer 1: keyword scan
        decision = self._keyword_scan(text)
        if decision is not None:
            decision.slots = self._extract_slots(text, decision.route_id)
            logger.debug(
                "RouterAgentExecutor: keyword hit route='%s' slots=%s",
                decision.route_id, list(decision.slots.keys()),
            )
            return decision

        logger.debug("RouterAgentExecutor: keyword miss, calling LLM for '%s...'", text[:60])

        # Layer 2: LLM classification
        try:
            decision = await asyncio.wait_for(
                self._llm_classify(text),
                timeout=self._llm_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "RouterAgentExecutor: LLM call timed out (%.1fs) — returning 'none'",
                self._llm_timeout,
            )
            return RouteDecision(route_id=None, confidence=0.0, method="none")

        if decision.route_id:
            decision.slots = self._extract_slots(text, decision.route_id)
            logger.info(
                "RouterAgentExecutor: LLM route='%s' confidence=%.2f slots=%s",
                decision.route_id, decision.confidence, list(decision.slots.keys()),
            )
        else:
            logger.info(
                "RouterAgentExecutor: LLM returned 'none' (confidence=%.2f)",
                decision.confidence,
            )

        return decision

    # ------------------------------------------------------------------
    # Layer 1 — keyword scan
    # ------------------------------------------------------------------

    def _keyword_scan(self, text: str) -> RouteDecision | None:
        """Scan compiled keyword patterns.

        Returns RouteDecision with method='keyword' on first match,
        or None if no pattern matches. Specs are checked in priority order
        (descending), then declaration order as tie-breaker.
        """
        # Sort by priority descending (compiled list already ordered, but be explicit)
        for cs in self._compiled:
            for pat in cs.patterns:
                if pat.search(text):
                    return RouteDecision(
                        route_id=cs.spec.route_id,
                        confidence=1.0,
                        method="keyword",
                    )
        return None

    # ------------------------------------------------------------------
    # Layer 2 — LLM classification
    # ------------------------------------------------------------------

    async def _llm_classify(self, text: str) -> RouteDecision:
        """Call LLM via LiteLLM gateway for intent classification.

        Returns RouteDecision with method='llm' or method='none'.
        Never raises — returns method='none' on any error.
        """
        try:
            import httpx
        except ImportError as exc:
            logger.error("RouterAgentExecutor: httpx not installed — LLM fallback disabled")
            raise RuntimeError("httpx is required for LLM classification") from exc

        messages = [
            {"role": "system", "content": self._llm_system_prompt},
            {"role": "user", "content": _ROUTER_USER_TEMPLATE.format(text=text)},
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,   # deterministic
            "max_tokens": 64,
        }

        try:
            async with httpx.AsyncClient(timeout=self._llm_timeout) as client:
                resp = await client.post(
                    f"{self._llm_gateway_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RouterAgentExecutor: LLM request failed: %s", exc)
            return RouteDecision(route_id=None, confidence=0.0, method="none")

        return self._parse_llm_response(resp.json())

    def _parse_llm_response(self, data: dict[str, Any]) -> RouteDecision:
        """Parse LiteLLM /v1/chat/completions response into RouteDecision."""
        try:
            raw = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "{}")
            )
            # Strip markdown fences if model adds them
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw.strip())

            parsed = json.loads(raw)
            intent = parsed.get("intent", "none")
            confidence = float(parsed.get("confidence", 0.0))

        except Exception as exc:  # noqa: BLE001
            logger.warning("RouterAgentExecutor: failed to parse LLM response: %s", exc)
            return RouteDecision(route_id=None, confidence=0.0, method="none")

        # Validate against known route_ids
        if intent == "none" or intent not in self._spec_index:
            return RouteDecision(route_id=None, confidence=confidence, method="llm")

        if confidence < self._threshold:
            logger.debug(
                "RouterAgentExecutor: LLM route='%s' confidence=%.2f below threshold %.2f",
                intent, confidence, self._threshold,
            )
            return RouteDecision(route_id=None, confidence=confidence, method="llm")

        return RouteDecision(route_id=intent, confidence=confidence, method="llm")

    # ------------------------------------------------------------------
    # Slot extraction
    # ------------------------------------------------------------------

    def _extract_slots(self, text: str, route_id: str | None) -> dict[str, Any]:
        """Extract slot values from text using the matched spec's slot_patterns."""
        if route_id is None:
            return {}
        cs = self._spec_index.get(route_id)
        if cs is None or not cs.slot_patterns:
            return {}

        slots: dict[str, Any] = {}
        for slot_name, pat in cs.slot_patterns.items():
            m = pat.search(text)
            if m:
                slots[slot_name] = m.group().lstrip("#")
        return slots

    # ------------------------------------------------------------------
    # LLM prompt builder
    # ------------------------------------------------------------------

    def _build_llm_prompt(self) -> str:
        """Build the system prompt from IntentSpec data.

        The template contains no business terms — all domain content
        comes from spec.route_id and spec.llm_description.
        """
        lines: list[str] = []
        for cs in self._compiled:
            spec = cs.spec
            line = f"  - {spec.route_id}: {spec.llm_description}"
            if spec.example_utterances:
                examples = "; ".join(f'"{u}"' for u in spec.example_utterances[:3])
                line += f" (e.g. {examples})"
            lines.append(line)

        intent_list = "\n".join(lines)
        return _ROUTER_SYSTEM_TEMPLATE.format(intent_list=intent_list)

    # ------------------------------------------------------------------
    # Compilation helpers
    # ------------------------------------------------------------------

    def _compile_specs(self, specs: list[IntentSpec]) -> list[_CompiledSpec]:
        """Pre-compile all regex patterns. Sorted by priority descending."""
        compiled: list[_CompiledSpec] = []
        for spec in sorted(specs, key=lambda s: s.priority, reverse=True):
            patterns = [re.compile(kw, re.IGNORECASE) for kw in spec.keywords]
            slot_pats = {
                name: re.compile(pat, re.IGNORECASE)
                for name, pat in spec.slot_patterns.items()
            }
            compiled.append(_CompiledSpec(spec=spec, patterns=patterns, slot_patterns=slot_pats))
        return compiled
