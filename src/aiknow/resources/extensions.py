"""
ExtensionsResource — SDK resource for uploading AppManifest to Platform.

Called by AiknowApp._upload_manifest() during app.start().
Thin HTTP wrapper: serialises the manifest and POSTs it to the Platform.

Endpoint: POST /api/v1/extensions/manifest

Error handling:
    Non-fatal: manifest upload failure should never crash the app.
    Callers (AiknowApp) catch all exceptions and log warnings so the
    app continues running in offline mode.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from aiknow._http import raise_for_status, wrap_httpx_errors

logger = logging.getLogger(__name__)


class AsyncExtensionsResource:
    """Async HTTP resource for registering the AppManifest with the Platform.

    Args:
        client: Shared httpx.AsyncClient (injected by AsyncAIKnowClient).
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def register_manifest(self, manifest_dict: dict[str, Any]) -> None:
        """Upload the full AppManifest to the Platform.

        The Platform processes this by:
            - Upserting all workflow definitions into the workflow registry.
            - Registering skill endpoint URLs for remote invocation.
            - Setting up webhook subscriptions if webhook_url is provided.

        Args:
            manifest_dict: AppManifest serialised as a plain dict
                           (call ``manifest.model_dump()`` before passing).

        Raises:
            AIKnowAPIError:       Non-2xx HTTP response from Platform.
            AIKnowConnectionError: Network-level failure.
            AIKnowTimeoutError:   Request timed out.
        """
        try:
            response = await self._client.post(
                "/extensions/manifest",
                json=manifest_dict,
            )
            raise_for_status("ExtensionsResource.register_manifest", response)
            logger.debug(
                "Manifest uploaded: %d skills, %d workflows, %d hooks",
                len(manifest_dict.get("skills", [])),
                len(manifest_dict.get("workflows", [])),
                len(manifest_dict.get("hooks", [])),
            )
        except Exception as exc:
            if isinstance(exc, (httpx.RequestError, httpx.TimeoutException)):
                wrap_httpx_errors("ExtensionsResource.register_manifest", exc)
            raise
