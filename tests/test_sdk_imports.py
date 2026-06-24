"""
SDK boundary test — verify public import surface.
"""
import sys


def test_public_api_importable():
    from aiknow import (
        AiknowApp,
        AppSkillDeps,
        HookEvent,
        SkillResult,
        hook,
        parser,
        skill,
        sop,
    )

    assert AiknowApp is not None
    assert skill is not None
    assert sop is not None
    assert hook is not None
    assert parser is not None
    assert AppSkillDeps is not None
    assert SkillResult is not None
    assert HookEvent is not None


def test_aiknow_core_not_in_sdk_modules():
    """After importing aiknow (SDK only), aiknow_core must not be loaded."""
    import aiknow  # noqa: F401
    import aiknow.extensions  # noqa: F401
    # aiknow_core might be loaded by other test fixtures,
    # but aiknow.extensions itself must not import it
    assert "aiknow.extensions._decorators" in sys.modules
    # The key invariant: extensions module has no direct import of aiknow_core
    import inspect
    src = inspect.getsource(sys.modules["aiknow.extensions._decorators"])
    assert "aiknow_core" not in src
    assert "aiknow_adapters" not in src
