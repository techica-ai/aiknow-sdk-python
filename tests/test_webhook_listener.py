"""
Tests for P3-B: WebhookListener, SkillDispatcher, HMAC verification.

Strategy:
    - WebhookListener tested via httpx.ASGITransport (no real uvicorn).
    - SkillDispatcher tested in isolation (no HTTP).
    - HMAC verify_signature tested with valid/invalid inputs.
    - All tests are sync-free: pytest-asyncio AUTO mode.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest
import pytest_asyncio
from aiknow._hmac import SignatureVerificationError, verify_signature
from aiknow._skill_dispatcher import SkillDispatcher
from aiknow.extensions._local_registry import ExtensionEntry, LocalExtensionRegistry
from aiknow.extensions.types import AppSkillDeps, HookEvent, SkillResult
from aiknow.webhook_listener import WebhookListener

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _registry_with_skill(skill_name: str = "verify_customer") -> LocalExtensionRegistry:
    registry = LocalExtensionRegistry()

    async def _impl(deps: AppSkillDeps, **kwargs) -> SkillResult:
        return SkillResult(
            status="ok",
            message="Customer verified",
            data={"customer_id": "C001", **kwargs},
        )

    registry.register(
        ExtensionEntry(skill_name, "skill", _impl, {"name": skill_name})
    )
    return registry


def _make_signed_body(body_dict: dict, secret: str = "test-secret") -> tuple[bytes, str]:
    body_bytes = json.dumps(body_dict).encode()
    digest = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
    return body_bytes, f"sha256={digest}"


@pytest.fixture
def registry() -> LocalExtensionRegistry:
    return _registry_with_skill()


@pytest_asyncio.fixture
async def listener(registry) -> WebhookListener:
    """WebhookListener without uvicorn — tested via ASGITransport."""
    return WebhookListener(
        registry=registry,
        host="127.0.0.1",
        port=9000,
        hmac_secret="test-secret",
    )


@pytest_asyncio.fixture
async def client(listener) -> httpx.AsyncClient:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=listener.app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# HMAC verify_signature tests
# ---------------------------------------------------------------------------

def test_verify_signature_valid():
    body = b'{"skill_name": "verify_customer"}'
    sig = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    verify_signature(body, sig, "secret")  # must not raise


def test_verify_signature_missing_header():
    with pytest.raises(SignatureVerificationError, match="missing"):
        verify_signature(b"body", None, "secret")


def test_verify_signature_wrong_prefix():
    with pytest.raises(SignatureVerificationError, match="format invalid"):
        verify_signature(b"body", "md5=abc123", "secret")


def test_verify_signature_tampered_body():
    original = b'{"skill": "ok"}'
    tampered = b'{"skill": "HACKED"}'
    sig = "sha256=" + hmac.new(b"secret", original, hashlib.sha256).hexdigest()
    with pytest.raises(SignatureVerificationError, match="mismatch"):
        verify_signature(tampered, sig, "secret")


def test_verify_signature_wrong_secret():
    body = b"hello"
    sig = "sha256=" + hmac.new(b"right-secret", body, hashlib.sha256).hexdigest()
    with pytest.raises(SignatureVerificationError, match="mismatch"):
        verify_signature(body, sig, "wrong-secret")


# ---------------------------------------------------------------------------
# SkillDispatcher tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_returns_ok(registry):
    dispatcher = SkillDispatcher(registry)
    result = await dispatcher.dispatch(
        skill_name="verify_customer",
        tenant_id="acme",
        execution_id="exec-001",
        params={"phone_number": "0987"},
    )
    assert result.status == "ok"
    assert result.data["customer_id"] == "C001"
    assert result.data["phone_number"] == "0987"


@pytest.mark.asyncio
async def test_dispatcher_skill_not_found_returns_error():
    registry = LocalExtensionRegistry()  # empty
    dispatcher = SkillDispatcher(registry)
    result = await dispatcher.dispatch(
        skill_name="ghost_skill",
        tenant_id="acme",
        execution_id="exec-001",
        params={},
    )
    assert result.status == "error"
    assert "ghost_skill" in (result.error_message or "")


@pytest.mark.asyncio
async def test_dispatcher_catches_skill_exception():
    registry = LocalExtensionRegistry()

    async def _bad_skill(deps, **kwargs):
        raise RuntimeError("database exploded")

    registry.register(ExtensionEntry("bad_skill", "skill", _bad_skill, {}))
    dispatcher = SkillDispatcher(registry)
    result = await dispatcher.dispatch(
        skill_name="bad_skill",
        tenant_id="acme",
        execution_id="exec-001",
        params={},
    )
    assert result.status == "error"
    assert "RuntimeError" in (result.error_message or "")


@pytest.mark.asyncio
async def test_dispatcher_normalizes_str_return():
    registry = LocalExtensionRegistry()

    async def _str_skill(deps, **kwargs) -> str:
        return "verified"

    registry.register(ExtensionEntry("str_skill", "skill", _str_skill, {}))
    dispatcher = SkillDispatcher(registry)
    result = await dispatcher.dispatch("str_skill", "acme", "exec-001", {})
    assert result.status == "verified"


@pytest.mark.asyncio
async def test_dispatcher_normalizes_dict_return():
    registry = LocalExtensionRegistry()

    async def _dict_skill(deps, **kwargs) -> dict:
        return {"status": "approved", "data": {"amount": 100}}

    registry.register(ExtensionEntry("dict_skill", "skill", _dict_skill, {}))
    dispatcher = SkillDispatcher(registry)
    result = await dispatcher.dispatch("dict_skill", "acme", "exec-001", {})
    assert result.status == "approved"
    assert result.data["amount"] == 100


@pytest.mark.asyncio
async def test_dispatcher_injects_app_skill_deps():
    """AppSkillDeps must be passed correctly to the skill function."""
    received_deps: list[AppSkillDeps] = []
    registry = LocalExtensionRegistry()

    async def _capture_deps(deps: AppSkillDeps, **kwargs) -> SkillResult:
        received_deps.append(deps)
        return SkillResult(status="ok", message="ok")

    registry.register(ExtensionEntry("caps_skill", "skill", _capture_deps, {}))
    dispatcher = SkillDispatcher(registry)
    await dispatcher.dispatch(
        skill_name="caps_skill",
        tenant_id="acme",
        execution_id="exec-001",
        params={},
        session_id="sess-abc",
        locale="en",
        permissions=["crm:read"],
    )
    assert len(received_deps) == 1
    assert received_deps[0].tenant_id == "acme"
    assert received_deps[0].session_id == "sess-abc"
    assert received_deps[0].locale == "en"
    assert "crm:read" in received_deps[0].permissions


# ---------------------------------------------------------------------------
# WebhookListener HTTP endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(client):
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "skills" in data


@pytest.mark.asyncio
async def test_invoke_skill_success(client):
    payload = {
        "execution_id": "exec-001",
        "skill_name": "verify_customer",
        "tenant_id": "acme",
        "params": {"phone_number": "0987654321"},
    }
    body_bytes, sig = _make_signed_body(payload)
    res = await client.post(
        "/skills/verify_customer",
        content=body_bytes,
        headers={"Content-Type": "application/json", "X-AIKnow-Signature": sig},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["data"]["customer_id"] == "C001"


@pytest.mark.asyncio
async def test_invoke_skill_invalid_signature(client):
    payload = {
        "execution_id": "exec-001",
        "skill_name": "verify_customer",
        "tenant_id": "acme",
        "params": {},
    }
    body_bytes = json.dumps(payload).encode()
    res = await client.post(
        "/skills/verify_customer",
        content=body_bytes,
        headers={"Content-Type": "application/json", "X-AIKnow-Signature": "sha256=badhash"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_invoke_skill_missing_signature(client):
    payload = {
        "execution_id": "exec-001",
        "skill_name": "verify_customer",
        "tenant_id": "acme",
        "params": {},
    }
    body_bytes = json.dumps(payload).encode()
    res = await client.post(
        "/skills/verify_customer",
        content=body_bytes,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_invoke_skill_path_body_mismatch(client):
    """Skill name in path must match body skill_name."""
    payload = {
        "execution_id": "exec-001",
        "skill_name": "different_skill",  # mismatch!
        "tenant_id": "acme",
        "params": {},
    }
    body_bytes, sig = _make_signed_body(payload)
    res = await client.post(
        "/skills/verify_customer",
        content=body_bytes,
        headers={"Content-Type": "application/json", "X-AIKnow-Signature": sig},
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_invoke_skill_no_hmac_dev_mode():
    """Without hmac_secret, signature check is skipped (dev mode)."""
    registry = _registry_with_skill()
    listener = WebhookListener(registry=registry, hmac_secret=None)

    payload = {
        "execution_id": "exec-dev",
        "skill_name": "verify_customer",
        "tenant_id": "acme",
        "params": {},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=listener.app),
        base_url="http://test",
    ) as c:
        res = await c.post(
            "/skills/verify_customer",
            json=payload,  # no signature header
        )
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_hook_delivery_accepted(listener, registry):
    """POST /hooks/{event_type} must return 202 Accepted."""
    received: list[HookEvent] = []

    async def _hook_handler(event: HookEvent) -> None:
        received.append(event)

    registry.register(
        ExtensionEntry("workflow.completed", "hook", _hook_handler, {})
    )

    hook_payload = {
        "event_type": "workflow.completed",
        "workflow_id": "refund_flow",
        "execution_id": "exec-001",
        "tenant_id": "acme",
        "payload": {"result": "ok"},
    }
    body_bytes, sig = _make_signed_body(hook_payload)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=listener.app),
        base_url="http://test",
    ) as c:
        res = await c.post(
            "/hooks/workflow.completed",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-AIKnow-Signature": sig},
        )

    assert res.status_code == 202
    assert res.json()["accepted"] is True


@pytest.mark.asyncio
async def test_listener_base_url():
    registry = LocalExtensionRegistry()
    listener = WebhookListener(registry=registry, host="192.168.1.10", port=8080)
    assert listener.base_url == "http://192.168.1.10:8080"
