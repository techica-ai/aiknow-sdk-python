"""
HMAC signature verification utility for the SDK WebhookListener.

Used by the WebhookListener to verify that incoming requests from the
Platform are authentic and not tampered with.

Protocol:
    Platform sends: X-AIKnow-Signature: sha256=<hex_digest>
    App verifies:   HMAC-SHA256(secret, body) == hex_digest

Security properties:
    - Constant-time comparison (hmac.compare_digest) prevents timing attacks.
    - Raises SignatureVerificationError (not generic Exception) to allow
      callers to return 403 without leaking internal errors.
    - Signature format: "sha256=<64-char lowercase hex>"
"""
from __future__ import annotations

import hashlib
import hmac

_SIGNATURE_HEADER = "X-AIKnow-Signature"
_EXPECTED_PREFIX = "sha256="


class SignatureVerificationError(Exception):
    """Raised when a webhook signature is invalid or missing.

    HTTP callers should respond with 403 Forbidden.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid webhook signature: {reason}")
        self.reason = reason


def verify_signature(
    body: bytes,
    signature_header: str | None,
    secret: str,
) -> None:
    """Verify the HMAC-SHA256 signature on a webhook request body.

    Args:
        body:             Raw request body bytes (must be read before parsing JSON).
        signature_header: Value of the X-AIKnow-Signature header, or None.
        secret:           Shared HMAC secret (same value used by Platform).

    Raises:
        SignatureVerificationError: If signature is missing, malformed, or invalid.
    """
    if not signature_header:
        raise SignatureVerificationError(
            f"'{_SIGNATURE_HEADER}' header is missing. "
            "Ensure the Platform is configured with the same HMAC secret."
        )

    if not signature_header.startswith(_EXPECTED_PREFIX):
        raise SignatureVerificationError(
            f"Signature format invalid. Expected 'sha256=<digest>', "
            f"got '{signature_header[:30]}...'"
        )

    received_digest = signature_header[len(_EXPECTED_PREFIX):]

    expected_digest = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(received_digest, expected_digest):
        raise SignatureVerificationError(
            "Signature mismatch — body may have been tampered with."
        )
