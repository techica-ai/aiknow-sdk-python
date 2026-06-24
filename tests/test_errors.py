"""
Tests for SDK exception hierarchy and HTTP error mapping.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from aiknow import (
    AIKnowAPIError,
    AIKnowConnectionError,
    AIKnowTimeoutError,
    AuthenticationError,
)
from aiknow._http import raise_for_status


def _mock_response(status_code: int, text: str = "") -> httpx.Response:
    """Create a minimal fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.is_success = status_code < 400
    return resp


class TestExceptionHierarchy:
    def test_auth_error_is_api_error(self):
        assert issubclass(AuthenticationError, AIKnowAPIError)

    def test_timeout_error_is_api_error(self):
        assert issubclass(AIKnowTimeoutError, AIKnowAPIError)

    def test_connection_error_is_api_error(self):
        assert issubclass(AIKnowConnectionError, AIKnowAPIError)

    def test_timeout_alias_is_same_class(self):
        from aiknow import TimeoutError as TErr
        assert TErr is AIKnowTimeoutError

    def test_api_error_stores_status_code(self):
        err = AIKnowAPIError("oops", status_code=500)
        assert err.status_code == 500

    def test_api_error_stores_body(self):
        body = {"detail": "bad"}
        err = AIKnowAPIError("oops", body=body)
        assert err.body == body


class TestRaiseForStatus:
    def test_success_does_not_raise(self):
        raise_for_status("Test", _mock_response(200))

    def test_401_raises_authentication_error(self):
        with pytest.raises(AuthenticationError):
            raise_for_status("Test", _mock_response(401))

    def test_403_raises_authentication_error(self):
        with pytest.raises(AuthenticationError):
            raise_for_status("Test", _mock_response(403))

    def test_404_raises_api_error(self):
        with pytest.raises(AIKnowAPIError):
            raise_for_status("Test", _mock_response(404))

    def test_500_raises_api_error(self):
        with pytest.raises(AIKnowAPIError):
            raise_for_status("Test", _mock_response(500))

    def test_error_message_includes_label(self):
        with pytest.raises(AIKnowAPIError, match="MyLabel"):
            raise_for_status("MyLabel", _mock_response(500, "internal error"))
