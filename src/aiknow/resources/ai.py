"""
AsyncAiResource — SDK resource for AI primitive operations.

Provides async methods for the 5 AI primitive endpoints:
    - extract()    → POST /api/v1/ai/extract
    - classify()   → POST /api/v1/ai/classify
    - evaluate()   → POST /api/v1/ai/evaluate
    - generate()   → POST /api/v1/ai/generate
    - similarity() → POST /api/v1/embedding/similarity

Design rules:
    - Zero business logic — HTTP client wrapper only.
    - Import ONLY from aiknow_contracts. Never from aiknow_core or aiknow_adapters.
    - exclude_none=True in model_dump() to avoid sending null/missing fields.
    - raw_prompt is a first-class field in the request body (NOT buried in context).
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .._http import raise_for_status

from aiknow_contracts.primitives import (
    ClassificationRequest,
    ClassificationResponse,
    EvaluationRequest,
    EvaluationResponse,
    ExtractionRequest,
    ExtractionResponse,
    GenerationRequest,
    GenerationResponse,
    SimilarityRequest,
    SimilarityResponse,
)

logger = logging.getLogger(__name__)


class AsyncAiResource:
    """SDK resource for AI primitive operations.

    Wraps the 5 LLM primitive endpoints exposed by the AIKnow platform.
    All methods are async and return validated Pydantic response DTOs.

    Usage::

        async with AsyncAIKnowClient(...) as client:
            result = await client.ai.extract(
                text="Invoice #1234, total: $500",
                schema={"invoice_number": "string", "total": "number"},
            )
            print(result.data)  # {"invoice_number": "1234", "total": 500}
    """

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        model: str | None = None,
        mode: str | None = None,
        raw_prompt: str | None = None,
    ) -> ExtractionResponse:
        """Extract structured data from text.

        Args:
            text: Input text to extract from.
            schema: JSON schema describing the structure to extract.
            context: Optional additional context dict.
            model: LLM model override (e.g. "gpt-4o").
            mode: Named prompt config (Phase 2). None = transparent mode.
            raw_prompt: Full raw system/user prompt override. Sent as a
                        separate field — never placed inside ``context``.

        Returns:
            ExtractionResponse with ``data`` dict matching the schema.
        """
        request = ExtractionRequest(
            text=text,
            schema=schema,
            context=context,
            model=model,
            mode=mode,
            raw_prompt=raw_prompt,
        )
        resp = await self._http.post(
            "/api/v1/ai/extract",
            json=request.model_dump(exclude_none=True, by_alias=True),
        )
        raise_for_status("ai", resp)
        return ExtractionResponse.model_validate(resp.json())

    async def classify(
        self,
        text: str,
        categories: list[str],
        *,
        context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ClassificationResponse:
        """Classify text into one of the provided categories.

        Args:
            text: Input text to classify.
            categories: List of candidate category labels.
            context: Optional additional context dict.
            model: LLM model override.

        Returns:
            ClassificationResponse with ``label``, ``labels`` (ranked), and ``scores``.
        """
        request = ClassificationRequest(
            text=text,
            categories=categories,
            context=context,
            model=model,
        )
        resp = await self._http.post(
            "/api/v1/ai/classify",
            json=request.model_dump(exclude_none=True, by_alias=True),
        )
        raise_for_status("ai", resp)
        return ClassificationResponse.model_validate(resp.json())

    async def evaluate(
        self,
        conditions: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        text: str | None = None,
        model: str | None = None,
    ) -> EvaluationResponse:
        """Evaluate conditions against context using hybrid rule+LLM strategy.

        Rule-based conditions (field/operator/value) are evaluated
        deterministically. Remaining conditions fall through to the LLM tier.

        Args:
            conditions: List of condition dicts with ``id``, ``operator``,
                        ``field``, ``value`` (or ``expression`` for LLM-only).
            context: Dict of field values used in rule-based evaluation.
            text: Optional text context for LLM-only conditions.
            model: LLM model override.

        Returns:
            EvaluationResponse with ``matched``, ``unmatched``, ``unknown`` lists.
        """
        request = EvaluationRequest(
            conditions=conditions,
            context=context,
            text=text,
            model=model,
        )
        resp = await self._http.post(
            "/api/v1/ai/evaluate",
            json=request.model_dump(exclude_none=True, by_alias=True),
        )
        raise_for_status("ai", resp)
        return EvaluationResponse.model_validate(resp.json())

    async def generate(
        self,
        template: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        instructions: str | None = None,
        model: str | None = None,
        mode: str | None = None,
        raw_prompt: str | None = None,
    ) -> GenerationResponse:
        """Generate text using a template or raw prompt.

        Args:
            template: Jinja2-style template string (rendered with ``context``).
            context: Template rendering context dict.
            instructions: Additional instructions appended to the prompt.
            model: LLM model override.
            mode: Named prompt config (Phase 2).
            raw_prompt: Full raw prompt override. Sent as a separate field —
                        never placed inside ``context``.

        Returns:
            GenerationResponse with ``text``, ``model``, and ``usage`` stats.
        """
        request = GenerationRequest(
            template=template,
            context=context,
            instructions=instructions,
            model=model,
            mode=mode,
            raw_prompt=raw_prompt,
        )
        resp = await self._http.post(
            "/api/v1/ai/generate",
            json=request.model_dump(exclude_none=True, by_alias=True),
        )
        raise_for_status("ai", resp)
        return GenerationResponse.model_validate(resp.json())

    async def similarity(
        self,
        text_a: str,
        text_b: str,
        *,
        model: str | None = None,
    ) -> SimilarityResponse:
        """Compute semantic similarity between two texts.

        Embeds both texts in a single batch call and returns cosine similarity.

        Args:
            text_a: First text.
            text_b: Second text.
            model: Embedding model override.

        Returns:
            SimilarityResponse with ``score`` in [0.0, 1.0] and ``model`` name.
        """
        request = SimilarityRequest(text_a=text_a, text_b=text_b, model=model)
        resp = await self._http.post(
            "/api/v1/embedding/similarity",
            json=request.model_dump(exclude_none=True, by_alias=True),
        )
        raise_for_status("ai", resp)
        return SimilarityResponse.model_validate(resp.json())
