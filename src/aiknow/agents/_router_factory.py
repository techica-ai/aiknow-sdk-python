# NO `from __future__ import annotations` — intentional!
#
# This module must NOT use PEP 563 lazy annotations because it defines
# FastAPI route handler factories. FastAPI resolves type hints via
# typing.get_type_hints() at route-registration time, which requires
# annotations to be concrete class objects — not ForwardRef strings.
#
# Mixing `from __future__ import annotations` with dynamic handler creation
# inside class methods causes Pydantic to fail on ForwardRef resolution
# (it cannot find 'Request' or 'StreamingResponse' in the module globals
# because they were only imported locally).
"""
agents/_router_factory.py — FastAPI route handler factory.

Kept in a dedicated module to avoid PEP-563 (`from __future__ import
annotations`) converting type hints into ForwardRef strings that
FastAPI/Pydantic cannot resolve at schema-generation time.
"""
from starlette.requests import Request
from fastapi.responses import StreamingResponse


def make_agent_handler(executor):
    """Return a FastAPI-compatible async handler bound to *executor*.

    The returned coroutine function has concrete type annotations so
    FastAPI can inspect them without hitting ForwardRef resolution errors.

    Args:
        executor: A BaseAgentExecutor instance.

    Returns:
        An async callable ``(request: Request) -> StreamingResponse``.
    """
    async def _handle(request: Request) -> StreamingResponse:
        body = await request.json()
        return await executor.handle_agui_request(body)

    return _handle
