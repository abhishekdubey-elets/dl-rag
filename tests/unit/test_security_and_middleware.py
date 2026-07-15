"""API-key auth + rate-limit window logic unit tests."""

from __future__ import annotations

import pytest

from dl_rag.api import security
from dl_rag.config import Settings
from dl_rag.exceptions import AuthenticationError

from tests.conftest import FakeCache


class TestRequireApiKey:
    async def test_auth_disabled_allows_anonymous(self, monkeypatch):
        monkeypatch.setattr(
            security, "get_settings",
            lambda: Settings(require_auth=False, _env_file=None),
        )
        assert await security.require_api_key(None) == "anonymous"

    async def test_valid_key_accepted(self, monkeypatch):
        monkeypatch.setattr(
            security, "get_settings",
            lambda: Settings(require_auth=True, api_keys="k1,k2", _env_file=None),
        )
        assert await security.require_api_key("k2") == "k2"

    async def test_missing_or_bad_key_rejected(self, monkeypatch):
        monkeypatch.setattr(
            security, "get_settings",
            lambda: Settings(require_auth=True, api_keys="k1", _env_file=None),
        )
        with pytest.raises(AuthenticationError):
            await security.require_api_key(None)
        with pytest.raises(AuthenticationError):
            await security.require_api_key("wrong")


class TestRateWindow:
    async def test_incr_window_counts(self):
        cache = FakeCache()
        assert await cache.incr_window("k", 60) == 1
        assert await cache.incr_window("k", 60) == 2
        assert await cache.incr_window("other", 60) == 1
