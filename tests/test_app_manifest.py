"""
Tests for AiknowApp manifest building and upload logic (Phase 2 — P2-F).

Tests cover:
    - _build_manifest(): correct mapping from registry → AppManifest
    - _upload_manifest(): calls client.extensions.register_manifest()
    - start() resilience: upload failure does NOT crash the app
    - start() skips upload when platform unreachable or no api_key
    - SOP compile error in _build_manifest(): skipped, not raised
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aiknow.app import AiknowApp
from aiknow.extensions._decorators import hook, skill, sop


# ---------------------------------------------------------------------------
# Helpers — build a minimal app with registered extensions
# ---------------------------------------------------------------------------

def _make_app(api_key: str = "test-key") -> AiknowApp:
    return AiknowApp(
        tenant_id="acme",
        api_key=api_key,
        base_url="http://localhost:8000",
        app_name="test-app",
        app_version="1.0",
    )


@skill("fetch_customer", description={"vi": "Lấy thông tin KH"}, tags=["crm"])
async def fetch_customer(deps, phone: str) -> str:
    return "ok"


@skill("process_refund", required_permissions=["payments:write"])
async def process_refund(deps, order_id: str) -> str:
    return "refunded"


@sop("refund_flow", initial_state="verify")
class RefundFlow:
    transitions = {
        "verify": {"ok": "__end__", "failed": "__end__"},
    }


@hook("workflow.completed")
async def on_done(event) -> None:
    pass


# ---------------------------------------------------------------------------
# _build_manifest() tests
# ---------------------------------------------------------------------------

def test_build_manifest_includes_skills():
    app = _make_app()
    app.register_skill(fetch_customer)
    app.register_skill(process_refund)

    manifest = app._build_manifest()

    assert len(manifest.skills) == 2
    skill_names = {s.name for s in manifest.skills}
    assert "fetch_customer" in skill_names
    assert "process_refund" in skill_names


def test_build_manifest_skill_metadata_preserved():
    app = _make_app()
    app.register_skill(fetch_customer)

    manifest = app._build_manifest()

    fc = next(s for s in manifest.skills if s.name == "fetch_customer")
    assert fc.description == {"vi": "Lấy thông tin KH"}
    assert "crm" in fc.tags


def test_build_manifest_includes_sops():
    app = _make_app()
    app.register_sop(RefundFlow)

    manifest = app._build_manifest()

    assert len(manifest.sops) == 1
    assert manifest.sops[0].sop_id == "refund_flow"


def test_build_manifest_sop_contains_definition():
    app = _make_app()
    app.register_sop(RefundFlow)

    manifest = app._build_manifest()

    defn = manifest.sops[0].definition
    assert "initial_state" in defn
    assert "nodes" in defn
    assert defn["initial_state"] == "verify"


def test_build_manifest_includes_hooks():
    app = _make_app()
    app.register_hook("workflow.completed", on_done)

    manifest = app._build_manifest()

    assert len(manifest.hooks) == 1
    assert manifest.hooks[0].event_type == "workflow.completed"


def test_build_manifest_tenant_id_correct():
    app = _make_app()
    manifest = app._build_manifest()
    assert manifest.tenant_id == "acme"


def test_build_manifest_empty_registry():
    app = _make_app()
    manifest = app._build_manifest()
    assert manifest.skills == []
    assert manifest.sops == []
    assert manifest.hooks == []


def test_build_manifest_sop_compile_error_skipped(caplog):
    """SOP classes that fail compilation must be skipped, not crash."""
    import logging

    @sop("broken_sop", initial_state="step1")
    class BrokenSOP:
        # transitions is missing entirely — will raise SOPCompileError
        pass

    app = _make_app()
    app.register_sop(BrokenSOP)

    with caplog.at_level(logging.WARNING, logger="aiknow.app"):
        manifest = app._build_manifest()

    assert manifest.sops == []
    assert any("broken_sop" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _upload_manifest() tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_manifest_calls_extensions_resource():
    """_upload_manifest() must call client.extensions.register_manifest() once."""
    app = _make_app()
    app.register_skill(fetch_customer)
    app.register_sop(RefundFlow)

    mock_client = MagicMock()
    mock_client.extensions = AsyncMock()
    mock_client.extensions.register_manifest = AsyncMock()
    app._client = mock_client

    await app._upload_manifest()

    mock_client.extensions.register_manifest.assert_called_once()
    call_arg = mock_client.extensions.register_manifest.call_args[0][0]
    assert call_arg["tenant_id"] == "acme"
    assert len(call_arg["skills"]) == 1
    assert len(call_arg["sops"]) == 1


@pytest.mark.asyncio
async def test_upload_manifest_non_fatal_on_network_error(caplog):
    """_upload_manifest() must log warning and NOT raise on failure."""
    import logging

    app = _make_app()

    mock_client = MagicMock()
    mock_client.extensions = AsyncMock()
    mock_client.extensions.register_manifest = AsyncMock(
        side_effect=ConnectionError("network down")
    )
    app._client = mock_client

    with caplog.at_level(logging.WARNING, logger="aiknow.app"):
        await app._upload_manifest()  # must NOT raise

    assert any("offline mode" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_upload_manifest_non_fatal_on_import_error(caplog):
    """_upload_manifest() must skip gracefully if aiknow_contracts not installed."""
    import logging

    app = _make_app()
    app._client = MagicMock()

    with patch.object(app, "_build_manifest", side_effect=ImportError("no contracts")):
        with caplog.at_level(logging.WARNING, logger="aiknow.app"):
            await app._upload_manifest()  # must NOT raise

    assert any("aiknow_contracts" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# start() integration tests (mocked ping + upload)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_uploads_manifest_when_platform_reachable():
    """start() must call _upload_manifest when ping succeeds and api_key set."""
    app = _make_app(api_key="ak_test")
    app.register_skill(fetch_customer)

    with (
        patch.object(app, "_upload_manifest", new=AsyncMock()) as mock_upload,
        patch("aiknow.app.AsyncAIKnowClient") as MockClient,
    ):
        mock_instance = AsyncMock()
        mock_instance.ping = AsyncMock(return_value=True)
        MockClient.return_value = mock_instance

        await app.start()

    mock_upload.assert_called_once()
    await app.stop()


@pytest.mark.asyncio
async def test_start_skips_upload_when_platform_unreachable():
    """start() must NOT call _upload_manifest when platform is offline."""
    app = _make_app(api_key="ak_test")

    with (
        patch.object(app, "_upload_manifest", new=AsyncMock()) as mock_upload,
        patch("aiknow.app.AsyncAIKnowClient") as MockClient,
    ):
        mock_instance = AsyncMock()
        mock_instance.ping = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        await app.start()

    mock_upload.assert_not_called()
    await app.stop()


@pytest.mark.asyncio
async def test_start_skips_upload_when_no_api_key():
    """start() must NOT call _upload_manifest when no api_key provided."""
    app = AiknowApp(tenant_id="acme", api_key=None, base_url="http://localhost:8000")

    with (
        patch.object(app, "_upload_manifest", new=AsyncMock()) as mock_upload,
        patch("aiknow.app.AsyncAIKnowClient") as MockClient,
    ):
        mock_instance = AsyncMock()
        mock_instance.ping = AsyncMock(return_value=True)
        MockClient.return_value = mock_instance

        await app.start()

    mock_upload.assert_not_called()
    await app.stop()


@pytest.mark.asyncio
async def test_start_continues_if_upload_fails():
    """start() must complete successfully even if manifest upload raises."""
    app = _make_app(api_key="ak_test")

    with (
        patch.object(app, "_upload_manifest", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch("aiknow.app.AsyncAIKnowClient") as MockClient,
    ):
        mock_instance = AsyncMock()
        mock_instance.ping = AsyncMock(return_value=True)
        MockClient.return_value = mock_instance

        # Must NOT raise even though _upload_manifest raises
        await app.start()

    assert app._started is True
    await app.stop()
