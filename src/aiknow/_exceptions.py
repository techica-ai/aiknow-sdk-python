"""
AIKnow SDK Exception Hierarchy.

All SDK-raised exceptions inherit from `AIKnowAPIError` so consumers can
choose how broadly they want to catch:

    except aiknow.AIKnowAPIError:   # catch any SDK error
    except aiknow.AIKnowTimeoutError:  # catch only timeouts
"""
from __future__ import annotations

from typing import Any


class AIKnowAPIError(Exception):
    """Base class for all AIKNOW SDK errors.

    Attributes:
        status_code: HTTP status code, if the error came from an HTTP response.
        body:        Parsed response body, if available.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthenticationError(AIKnowAPIError):
    """Raised on HTTP 401/403 — invalid or missing API key."""


class AIKnowTimeoutError(AIKnowAPIError):
    """Raised when the request exceeds the configured timeout."""


class AIKnowConnectionError(AIKnowAPIError):
    """Raised when the SDK cannot connect to the AIKNOW API server
    (e.g., network unreachable, connection refused)."""


# ---------------------------------------------------------------------------
# Backward-compat alias (deprecated — use AIKnowTimeoutError)
# ---------------------------------------------------------------------------
TimeoutError = AIKnowTimeoutError  # noqa: A001 — intentional alias
