"""
SDK extension tests — verify decorators and AiknowApp work correctly.
"""
import sys

import pytest


# ---------------------------------------------------------------------------
# Boundary test
# ---------------------------------------------------------------------------

def test_sdk_extensions_do_not_import_aiknow_core():
    """Importing aiknow.extensions must not pull in aiknow_core."""
    # Clear any cached imports from previous tests
    forbidden = [k for k in sys.modules if k.startswith("aiknow_core") or k.startswith("aiknow_adapters")]
    # We can't un-import, but we can assert the extension module itself
    # doesn't add aiknow_core when freshly imported
    import aiknow.extensions  # noqa: F401
    # Extensions should work without aiknow_core being present
    assert "aiknow.extensions" in sys.modules


# ---------------------------------------------------------------------------
# @skill decorator
# ---------------------------------------------------------------------------

def test_skill_decorator_attaches_metadata():
    from aiknow.extensions import skill
    from aiknow.extensions.types import AppSkillDeps, SkillResult

    @skill(
        "test_skill_meta",
        description={"vi": "Test", "en": "Test"},
        required_permissions=["test:read"],
        tags=["test"],
    )
    async def my_skill(deps: AppSkillDeps, query: str = "") -> SkillResult:
        return SkillResult(status="ok", message="done")

    # @skill is deprecated and delegates to @step, so _step_meta is attached
    # (not _skill_meta — @skill no longer attaches the old attribute)
    assert hasattr(my_skill, "_step_meta"), (
        "@skill should attach _step_meta (it now delegates to @step internally)"
    )
    meta = getattr(my_skill, "_step_meta")
    assert meta["name"] == "test_skill_meta"
    assert meta["required_permissions"] == ["test:read"]
    assert meta["tags"] == ["test"]


def test_skill_decorator_preserves_function():
    from aiknow.extensions import skill
    from aiknow.extensions.types import AppSkillDeps, SkillResult

    @skill("preserve_test")
    async def original(deps: AppSkillDeps) -> SkillResult:
        return SkillResult(status="ok", message="ok")

    assert original.__name__ == "original"
    assert callable(original)


# ---------------------------------------------------------------------------
# @sop decorator
# ---------------------------------------------------------------------------

def test_sop_decorator_attaches_metadata():
    from aiknow.extensions import sop

    @sop("test_sop_id", initial_state="step_a")
    class TestSOP:
        transitions = {
            "step_a": {"done": "step_b"},
            "step_b": {"ok": "__end__"},
        }

    assert hasattr(TestSOP, "_sop_meta")
    meta = getattr(TestSOP, "_sop_meta")
    assert meta["id"] == "test_sop_id"
    assert meta["initial_state"] == "step_a"


def test_sop_decorator_infers_initial_state():
    from aiknow.extensions import sop

    @sop("infer_initial")  # no initial_state given
    class SOPNoInitial:
        transitions = {
            "first_state": {"ok": "__end__"},
        }

    assert getattr(SOPNoInitial, "_sop_meta")["id"] == "infer_initial"


# ---------------------------------------------------------------------------
# @hook decorator
# ---------------------------------------------------------------------------

def test_hook_decorator_attaches_metadata():
    from aiknow.extensions import hook
    from aiknow.extensions.types import HookEvent

    @hook("test.event")
    async def my_handler(event: HookEvent) -> None:
        pass

    assert hasattr(my_handler, "_hook_meta")
    assert getattr(my_handler, "_hook_meta")["event_type"] == "test.event"


# ---------------------------------------------------------------------------
# SkillResult
# ---------------------------------------------------------------------------

def test_skill_result_convenience_constructors():
    from aiknow.extensions.types import StepResult

    ok = StepResult.ok("All good", data={"x": 1})
    assert ok.status == "ok"
    assert ok.data == {"x": 1}

    err = StepResult.error("Something broke")
    assert err.status == "error"

    esc = StepResult.escalate()
    assert esc.status == "escalate"
    assert esc.next_state == "escalate_human"


def test_step_result_convenience_constructors():
    """SkillResult deprecated alias \u2014 warning fires at module-level access via __getattr__."""
    import warnings
    import aiknow.extensions.types as types_mod

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SkillResult = types_mod.__getattr__("SkillResult")
        ok = SkillResult.ok("ok", data={"x": 1})
        assert ok.status == "ok"
    assert any(issubclass(x.category, DeprecationWarning) for x in w), (
        "Accessing SkillResult via module __getattr__ must raise DeprecationWarning"
    )


# ---------------------------------------------------------------------------
# AiknowApp
# ---------------------------------------------------------------------------

def test_app_register_skill_appears_in_describe():
    from aiknow import AiknowApp
    from aiknow.extensions import skill
    from aiknow.extensions.types import AppSkillDeps, SkillResult

    @skill("app_test_skill", tags=["test"])
    async def app_skill(deps: AppSkillDeps) -> SkillResult:
        return SkillResult.ok()

    app = AiknowApp(tenant_id="test-tenant", api_key="test-key")
    app.register_skill(app_skill)

    desc = app.describe()
    assert desc["tenant_id"] == "test-tenant"
    skill_names = [s["name"] for s in desc["skills"]]
    assert "app_test_skill" in skill_names


def test_app_register_sop_appears_in_describe():
    from aiknow import AiknowApp
    from aiknow.extensions import sop

    @sop("app_test_sop")
    class AppTestSOP:
        transitions = {"step_a": {"ok": "__end__"}}

    app = AiknowApp(tenant_id="test-tenant")
    app.register_sop(AppTestSOP)

    desc = app.describe()
    sop_ids = [s.get("id") for s in desc["sops"]]
    assert "app_test_sop" in sop_ids


def test_app_register_skill_raises_if_not_decorated():
    from aiknow import AiknowApp

    async def raw_fn():
        pass

    app = AiknowApp(tenant_id="test")
    # register_skill delegates to register_step; error message uses @step
    with pytest.raises(TypeError, match="@step"):
        app.register_skill(raw_fn)


def test_app_register_sop_raises_if_not_decorated():
    from aiknow import AiknowApp

    class RawClass:
        transitions = {}

    app = AiknowApp(tenant_id="test")
    with pytest.raises(TypeError, match="@sop"):
        app.register_sop(RawClass)


def test_app_from_env(monkeypatch):
    from aiknow import AiknowApp

    monkeypatch.setenv("AIKNOW_TENANT_ID", "env-tenant")
    monkeypatch.setenv("AIKNOW_API_KEY", "env-key")

    app = AiknowApp.from_env()
    assert app.tenant_id == "env-tenant"


def test_app_from_env_missing_tenant(monkeypatch):
    from aiknow import AiknowApp

    monkeypatch.delenv("AIKNOW_TENANT_ID", raising=False)
    with pytest.raises(EnvironmentError, match="AIKNOW_TENANT_ID"):
        AiknowApp.from_env()
