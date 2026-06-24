"""
Tests for SDK retry / resilience behavior using httpx transport mocking.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from aiknow import AIKnowClient, AsyncAIKnowClient


class TestPingRobustness:
    """ping() should never raise — it must return False on failures."""

    def test_ping_returns_false_on_connection_error(self):
        with AIKnowClient(api_key="k") as client:
            with patch.object(client._client, "get", side_effect=httpx.ConnectError("refused")):
                assert client.ping() is False

    def test_ping_returns_false_on_timeout(self):
        with AIKnowClient(api_key="k") as client:
            with patch.object(client._client, "get", side_effect=httpx.TimeoutException("timeout")):
                assert client.ping() is False

    @pytest.mark.asyncio
    async def test_async_ping_returns_false_on_connection_error(self):
        async with AsyncAIKnowClient(api_key="k") as client:
            err = httpx.ConnectError("refused")
            with patch.object(
                client._client, "get", new_callable=AsyncMock, side_effect=err
            ):
                assert await client.ping() is False

    @pytest.mark.asyncio
    async def test_async_ping_returns_false_on_timeout(self):
        async with AsyncAIKnowClient(api_key="k") as client:
            err = httpx.TimeoutException("timeout")
            with patch.object(
                client._client, "get", new_callable=AsyncMock, side_effect=err
            ):
                assert await client.ping() is False
