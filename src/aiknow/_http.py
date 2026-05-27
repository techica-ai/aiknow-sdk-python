"""
HTTP response helpers shared by all SDK resources.

Centralizes error checking so individual resource methods stay DRY.
"""
from __future__ import annotations

import httpx
from typing import NoReturn

from ._exceptions import (
    AIKnowAPIError,
    AIKnowConnectionError,
    AIKnowTimeoutError,
    AuthenticationError,
)


def raise_for_status(label: str, res: httpx.Response) -> None:
    """Raise an appropriate SDK exception if *res* is not a success.

    Maps HTTP status codes to typed exceptions:
    - 401 / 403  →  AuthenticationError
    - 4xx / 5xx  →  AIKnowAPIError
    """
    if res.is_success:
        return

    if res.status_code in (401, 403):
        raise AuthenticationError(
            f"{label}: authentication failed (HTTP {res.status_code})",
            status_code=res.status_code,
        )

    raise AIKnowAPIError(
        f"{label} failed (HTTP {res.status_code}): {res.text[:500]}",
        status_code=res.status_code,
    )


def wrap_httpx_errors(label: str, exc: Exception) -> NoReturn:
    """Re-raise httpx transport errors as typed SDK exceptions.

    Declared ``-> NoReturn`` because this function *always* raises —
    it never returns normally. This lets the type checker understand
    that callers (_get, _delete) don't have a missing-return path.

    Note: uses ``raise exc`` (not bare ``raise``) because this function
    is called inside the except clause but as a regular call, not via
    re-raise syntax — bare ``raise`` outside an active except frame
    would produce ``RuntimeError: No active exception to re-raise``.
    """
    if isinstance(exc, httpx.TimeoutException):
        raise AIKnowTimeoutError(
            f"{label}: request timed out — {exc}",
        ) from exc
    if isinstance(exc, httpx.RequestError):
        raise AIKnowConnectionError(
            f"{label}: connection error — {exc}",
        ) from exc
    raise AIKnowAPIError(f"{label}: unexpected error — {exc}") from exc
