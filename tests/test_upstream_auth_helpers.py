from __future__ import annotations

from headroom.proxy.helpers import ensure_upstream_auth


def test_ensure_upstream_auth_injects_openai_authorization_when_missing(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    headers: dict[str, str] = {}

    ensure_upstream_auth(headers, "openai")

    assert headers["Authorization"] == "Bearer sk-env-openai"


def test_ensure_upstream_auth_preserves_explicit_openai_authorization(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    headers = {"authorization": "Bearer client-key"}

    ensure_upstream_auth(headers, "openai")

    assert headers == {"authorization": "Bearer client-key"}


def test_ensure_upstream_auth_injects_anthropic_api_key_when_missing(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    headers: dict[str, str] = {}

    ensure_upstream_auth(headers, "anthropic")

    assert headers["x-api-key"] == "sk-ant-env"


def test_ensure_upstream_auth_preserves_explicit_anthropic_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    headers = {"X-API-Key": "client-anthropic-key"}

    ensure_upstream_auth(headers, "anthropic")

    assert headers == {"X-API-Key": "client-anthropic-key"}
