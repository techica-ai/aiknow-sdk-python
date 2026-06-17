"""
Tests for SDK pipeline atomic operations (parse, chunk, get_state, list_states).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from aiknow import AIKnowClient, AsyncAIKnowClient
from aiknow_core.common.models.knowledge import Chunk, ParsedDocument


class TestSDKPipelineOperations:
    def test_sync_parse(self) -> None:
        with AIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "state_token": "st_test_parse",
                "token_fingerprint": "abc123def456",
                "page_count": 1,
                "char_count": 11,
            }
            with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
                res = client.parse(source_id="src-123", parser="markitdown", persist=True)
                assert res["state_token"] == "st_test_parse"
                mock_post.assert_called_once_with(
                    "/pipeline/parse",
                    json={
                        "source_id": "src-123",
                        "parser": "markitdown",
                        "config": None,
                        "persist": True,
                    },
                )

    def test_sync_chunk(self) -> None:
        with AIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "state_token": "st_test_chunk",
                "token_fingerprint": "xyz789",
                "chunk_count": 1,
            }
            with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
                res = client.chunk(state_token="st_test_parse", persist=True)
                assert res["state_token"] == "st_test_chunk"
                mock_post.assert_called_once_with(
                    "/pipeline/chunk",
                    json={
                        "state_token": "st_test_parse",
                        "document": None,
                        "strategy": "recursive",
                        "chunk_size": None,
                        "overlap": None,
                        "dialog_config": None,
                        "persist": True,
                    },
                )

    def test_sync_get_state_parsed_doc(self) -> None:
        with AIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "type": "parsed_document",
                "data": {
                    "source_id": "src-123",
                    "original_name": "test.txt",
                    "elements": [{"element_type": "Paragraph", "content": "Hello"}],
                },
            }
            with patch.object(client._client, "get", return_value=mock_resp) as mock_get:
                doc = client.get_state("st_test_parse")
                assert isinstance(doc, ParsedDocument)
                assert doc.source_id == "src-123"
                mock_get.assert_called_once_with("/pipeline/state/st_test_parse")

    def test_sync_get_state_chunks(self) -> None:
        with AIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "type": "chunks",
                "data": [
                    {
                        "asset_id": "chunk-1",
                        "metadata": {
                            "source_id": "src-123",
                            "tenant_id": "t1",
                            "original_name": "test.txt",
                        },
                        "content": "Hello chunk",
                    }
                ],
            }
            with patch.object(client._client, "get", return_value=mock_resp):
                chunks = client.get_state("st_test_chunk")
                assert isinstance(chunks, list)
                assert len(chunks) == 1
                assert isinstance(chunks[0], Chunk)
                assert chunks[0].content == "Hello chunk"

    def test_sync_list_states(self) -> None:
        with AIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {
                    "state_token": "st_token",
                    "created_at": "2026-06-18T00:00:00Z",
                    "type": "chunks",
                    "expires_at": None,
                    "is_persistent": True,
                }
            ]
            with patch.object(client._client, "get", return_value=mock_resp) as mock_get:
                res = client.list_states()
                assert len(res) == 1
                assert res[0]["state_token"] == "st_token"
                mock_get.assert_called_once_with("/pipeline/states")

    @pytest.mark.asyncio
    async def test_async_parse(self) -> None:
        async with AsyncAIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "state_token": "st_test_parse",
                "token_fingerprint": "abc123def456",
                "page_count": 1,
                "char_count": 11,
            }
            patch_post = patch.object(
                client._client, "post", new_callable=AsyncMock, return_value=mock_resp
            )
            with patch_post as mock_post:
                res = await client.parse(source_id="src-123", parser="markitdown", persist=True)
                assert res["state_token"] == "st_test_parse"
                mock_post.assert_called_once_with(
                    "/pipeline/parse",
                    json={
                        "source_id": "src-123",
                        "parser": "markitdown",
                        "config": None,
                        "persist": True,
                    },
                )

    @pytest.mark.asyncio
    async def test_async_chunk(self) -> None:
        async with AsyncAIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "state_token": "st_test_chunk",
                "token_fingerprint": "xyz789",
                "chunk_count": 1,
            }
            patch_post = patch.object(
                client._client, "post", new_callable=AsyncMock, return_value=mock_resp
            )
            with patch_post as mock_post:
                res = await client.chunk(state_token="st_test_parse", persist=True)
                assert res["state_token"] == "st_test_chunk"
                mock_post.assert_called_once_with(
                    "/pipeline/chunk",
                    json={
                        "state_token": "st_test_parse",
                        "document": None,
                        "strategy": "recursive",
                        "chunk_size": None,
                        "overlap": None,
                        "dialog_config": None,
                        "persist": True,
                    },
                )

    @pytest.mark.asyncio
    async def test_async_get_state_parsed_doc(self) -> None:
        async with AsyncAIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "type": "parsed_document",
                "data": {
                    "source_id": "src-123",
                    "original_name": "test.txt",
                    "elements": [{"element_type": "Paragraph", "content": "Hello"}],
                },
            }
            patch_get = patch.object(
                client._client, "get", new_callable=AsyncMock, return_value=mock_resp
            )
            with patch_get as mock_get:
                doc = await client.get_state("st_test_parse")
                assert isinstance(doc, ParsedDocument)
                assert doc.source_id == "src-123"
                mock_get.assert_called_once_with("/pipeline/state/st_test_parse")

    @pytest.mark.asyncio
    async def test_async_list_states(self) -> None:
        async with AsyncAIKnowClient(api_key="test-key") as client:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {
                    "state_token": "st_token",
                    "created_at": "2026-06-18T00:00:00Z",
                    "type": "chunks",
                    "expires_at": None,
                    "is_persistent": True,
                }
            ]
            patch_get = patch.object(
                client._client, "get", new_callable=AsyncMock, return_value=mock_resp
            )
            with patch_get as mock_get:
                res = await client.list_states()
                assert len(res) == 1
                assert res[0]["state_token"] == "st_token"
                mock_get.assert_called_once_with("/pipeline/states")
