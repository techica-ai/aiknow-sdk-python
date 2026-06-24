"""
Tests for AIKnowClient and AsyncAIKnowClient initialization.
"""
from __future__ import annotations

import pytest
from aiknow import AIKnowClient, AsyncAIKnowClient
from aiknow.resources.observe import AsyncObserveResource, ObserveResource


class TestClientInit:
    def test_sync_client_default_base_url(self):
        client = AIKnowClient(api_key="test-key")
        assert client._client.base_url is not None
        client.close()

    def test_sync_client_observe_none_without_admin_key(self):
        client = AIKnowClient(api_key="test-key")
        assert client.observe is None
        client.close()

    def test_sync_client_observe_set_with_admin_key(self):
        client = AIKnowClient(api_key="test-key", admin_key="admin-key")
        assert isinstance(client.observe, ObserveResource)
        client.close()

    def test_sync_client_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("AIKNOW_API_KEY", "env-api-key")
        monkeypatch.setenv("AIKNOW_ADMIN_KEY", "env-admin-key")
        monkeypatch.setenv("AIKNOW_BASE_URL", "http://test.host/api/v1")
        client = AIKnowClient()
        assert isinstance(client.observe, ObserveResource)
        client.close()

    def test_sync_client_context_manager(self):
        with AIKnowClient(api_key="test-key") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_async_client_observe_none_without_admin_key(self):
        client = AsyncAIKnowClient(api_key="test-key")
        assert client.observe is None
        await client.close()

    @pytest.mark.asyncio
    async def test_async_client_observe_set_with_admin_key(self):
        client = AsyncAIKnowClient(api_key="test-key", admin_key="admin-key")
        assert isinstance(client.observe, AsyncObserveResource)
        await client.close()

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self):
        async with AsyncAIKnowClient(api_key="test-key") as client:
            assert client is not None
