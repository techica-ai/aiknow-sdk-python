"""
SpanBuilder — lightweight client-side span builder for AIKnow SDK clients.

Designed for external services (e.g. call-center-app) that need to record
their own spans and push them to the AIKnow Platform via ``push_trace()``.

Design goals:
    - Zero external dependencies (pure stdlib)
    - Thread-safe and asyncio-safe (no shared mutable state)
    - OTel W3C-compatible IDs (32-hex trace_id, 16-hex span_id)
    - Produces exactly the payload shape that ``POST /api/v1/observe/push`` expects

Usage::

    builder = SpanBuilder(trace_type="conversation", session_id=thread_id)
    trace_id = builder.trace_id  # use for W3C traceparent header

    with builder.span("intent.detect") as s:
        result = detect_intent(text)
        s.set("intent.workflow_id", result.workflow_id)
        s.set("intent.confidence", result.confidence)
        s.set("input.text_preview", text[:200])

    with builder.span("workflow.trigger") as s:
        s.set("workflow_id", workflow_id)
        execution = await trigger(...)
        s.set("execution_id", execution.execution_id)

    trace_payload, spans_payload = builder.flush(status="ok")
    await client.push_trace(trace_payload, spans_payload)
"""
from __future__ import annotations

import secrets
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _trace_id() -> str:
    """32-char lowercase hex (128-bit) — W3C OTel format."""
    return secrets.token_hex(16)


def _span_id() -> str:
    """16-char lowercase hex (64-bit) — W3C OTel format."""
    return secrets.token_hex(8)


def _now_us() -> int:
    """Current UTC time as Unix epoch microseconds."""
    return int(time.time() * 1_000_000)


# ---------------------------------------------------------------------------
# SpanHandle — returned by span() context manager
# ---------------------------------------------------------------------------

@dataclass
class SpanHandle:
    """In-flight span. Set attributes via :meth:`set`."""
    span_id: str
    name: str
    kind: str
    parent_span_id: str | None
    start_time_us: int
    _attrs: dict[str, Any] = field(default_factory=dict)
    _status: str = "ok"
    _status_description: str | None = None
    _end_time_us: int = 0

    def set(self, key: str, value: Any) -> None:
        """Add or update a span attribute."""
        self._attrs[key] = value

    def error(self, description: str) -> None:
        """Mark span as error."""
        self._status = "error"
        self._status_description = description[:500]

    # Internal: called by SpanBuilder.span() on __exit__
    def _close(self, status: str = "ok", description: str | None = None) -> None:
        self._end_time_us = _now_us()
        if status != "ok":
            self._status = status
            self._status_description = description

    def _to_dict(self, trace_id: str, tenant_id: str) -> dict[str, Any]:
        end = self._end_time_us or _now_us()
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "status": self._status,
            "status_description": self._status_description,
            "parent_span_id": self.parent_span_id,
            "start_time_us": self.start_time_us,
            "end_time_us": end,
            "duration_ms": round((end - self.start_time_us) / 1000.0, 3),
            "attributes": self._attrs or None,
        }


# ---------------------------------------------------------------------------
# SpanBuilder
# ---------------------------------------------------------------------------

class SpanBuilder:
    """Collects spans for a single trace; produces push-ready payloads.

    Args:
        trace_type:   Logical trace type shown in dashboard (e.g. "conversation").
        session_id:   Frontend/call session ID to correlate with other traces.
        service_name: Identifies the calling service (shown in trace detail).
        trace_id:     Pre-set trace ID. Generated if not provided.
        tenant_id:    Tenant context (forwarded to Platform). Optional here —
                      Platform enforces tenant from X-Tenant-Id header.
    """

    def __init__(
        self,
        trace_type: str = "conversation",
        session_id: str | None = None,
        service_name: str = "call-center-app",
        trace_id: str | None = None,
        tenant_id: str = "",
    ) -> None:
        self._trace_type = trace_type
        self._session_id = session_id
        self._service_name = service_name
        self._tenant_id = tenant_id
        self._trace_id = trace_id or _trace_id()
        self._started_at: str = datetime.now(UTC).isoformat()
        self._spans: list[SpanHandle] = []
        self._current_span_id: str | None = None  # simple stack (last-in-first-out)

    @property
    def trace_id(self) -> str:
        """The W3C-compatible trace ID for this builder."""
        return self._trace_id

    @property
    def traceparent(self) -> str:
        """W3C ``traceparent`` header value for this trace.

        Use this as the ``traceparent`` header when calling downstream services
        so their spans become children of this conversation trace.

        Format: ``00-<trace_id>-<dummy_parent_span>-01``
        """
        dummy_span = _span_id()
        return f"00-{self._trace_id}-{dummy_span}-01"

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = "INTERNAL",
    ) -> Generator[SpanHandle, None, None]:
        """Context manager that creates, yields, and closes a span.

        Example::

            with builder.span("intent.detect") as s:
                result = detect(text)
                s.set("intent.workflow_id", result.workflow_id)

        On exception the span is marked ``error`` and the exception re-raised.
        """
        handle = SpanHandle(
            span_id=_span_id(),
            name=name,
            kind=kind,
            parent_span_id=self._current_span_id,
            start_time_us=_now_us(),
        )
        prev = self._current_span_id
        self._current_span_id = handle.span_id

        try:
            yield handle
            handle._close(status="ok")
        except Exception as exc:
            handle._close(
                status="error",
                description=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
            raise
        finally:
            self._spans.append(handle)
            self._current_span_id = prev

    def flush(
        self,
        status: str = "ok",
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Produce the trace + spans payload ready for ``client.push_trace()``.

        Returns:
            (trace_dict, spans_list) — pass directly to :meth:`AsyncAIKnowClient.push_trace`.
        """
        finished_at = datetime.now(UTC).isoformat()
        trace_dict: dict[str, Any] = {
            "trace_id": self._trace_id,
            "trace_type": self._trace_type,
            "session_id": self._session_id,
            "service_name": self._service_name,
            "status": status,
            "started_at": self._started_at,
            "finished_at": finished_at,
            "error_type": error_type,
            "error_message": error_message,
        }
        spans_list = [s._to_dict(self._trace_id, self._tenant_id) for s in self._spans]
        return trace_dict, spans_list
