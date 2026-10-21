"""Regression tests for Anthropic connectivity probe routing (#3336).

Claude Code (and other Anthropic SDK clients) sends unauthenticated
``HEAD /api/hello`` or ``GET /api/hello`` requests to test upstream
connectivity. Because the probe carries no authorization headers,
it must not fall through to the default OpenAI target.

The proxy now answers the probe itself with ``200 OK`` instead of forwarding
it: the configured upstream (9Router) has no ``/api/hello`` and replied 404,
which made Claude Code report the proxy as unreachable.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.providers.proxy_targets import (
    is_anthropic_hello_path,
    select_passthrough_base_url,
)
from headroom.providers.route_specs import PROVIDER_PASSTHROUGH_ROUTES, is_local_probe_request
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def _mock_proxy(**legacy_targets: str) -> Any:
    class Runtime:
        @staticmethod
        def api_target(provider: str) -> str:
            return f"https://runtime.{provider}.test"

        @staticmethod
        def model_metadata_provider(headers: Any) -> str:
            return "anthropic" if headers.get("x-api-key") else "openai"

    return type("Proxy", (), {**legacy_targets, "provider_runtime": Runtime()})()


def test_is_anthropic_hello_path_matches_expected_patterns() -> None:
    assert is_anthropic_hello_path("/api/hello")
    assert is_anthropic_hello_path("api/hello")
    assert is_anthropic_hello_path("/api/hello/")
    assert is_anthropic_hello_path("api/hello/")

    assert not is_anthropic_hello_path(None)
    assert not is_anthropic_hello_path("")
    assert not is_anthropic_hello_path("/api/hello/subpath")
    assert not is_anthropic_hello_path("/v1/messages")
    assert not is_anthropic_hello_path("/api/other")


def test_select_passthrough_base_url_routes_hello_to_anthropic() -> None:
    proxy = _mock_proxy(
        ANTHROPIC_API_URL="https://anthropic.custom.test",
        OPENAI_API_URL="https://openai.custom.test",
    )

    # Keyless probe on /api/hello routes to Anthropic target.
    assert select_passthrough_base_url(proxy, {}, "/api/hello") == "https://anthropic.custom.test"
    assert select_passthrough_base_url(proxy, {}, "api/hello/") == "https://anthropic.custom.test"

    # Other keyless paths still fall through to OpenAI.
    assert select_passthrough_base_url(proxy, {}, "/other/endpoint") == "https://openai.custom.test"


def _hello_app(**overrides: Any) -> Any:
    return create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            anthropic_api_url="https://api.anthropic.test",
            openai_api_url="https://api.openai.test",
            **overrides,
        )
    )


def _spy_passthrough(monkeypatch: Any) -> list[tuple[str, str, str, str]]:
    calls: list[tuple[str, str, str, str]] = []

    async def fake_passthrough(
        self: Any, request: Any, base_url: str, sub_path: str = "", provider_name: str = ""
    ) -> JSONResponse:
        calls.append((request.method, request.url.path, base_url, provider_name))
        return JSONResponse({"forwarded": True}, status_code=404)

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", fake_passthrough)
    return calls


def test_local_probe_routes_declared() -> None:
    assert is_local_probe_request("GET", "/api/hello")
    assert is_local_probe_request("HEAD", "/api/hello")
    assert is_local_probe_request("head", "/api/hello/")
    assert not is_local_probe_request("POST", "/api/hello")
    assert not is_local_probe_request("GET", "/api/hello/subpath")
    assert all(r.path != "/api/hello" for r in PROVIDER_PASSTHROUGH_ROUTES)


def test_hello_probe_answered_locally_without_forwarding(monkeypatch: Any) -> None:
    calls = _spy_passthrough(monkeypatch)

    with TestClient(_hello_app()) as client:
        # Keyless HEAD request as sent by Claude Code's Bun connectivity probe.
        head_resp = client.head("/api/hello", headers={"user-agent": "Bun/1.1.20"})
        assert head_resp.status_code == 200

        # Keyless GET request.
        get_resp = client.get("/api/hello")
        assert get_resp.status_code == 200
        assert get_resp.json() == {"status": "ok", "service": "headroom-proxy"}

        assert client.get("/api/hello/").status_code == 200

    # Never relayed upstream (9Router has no /api/hello and answers 404).
    assert calls == []


def test_hello_probe_exempt_from_proxy_token_nonloopback(monkeypatch: Any) -> None:
    calls = _spy_passthrough(monkeypatch)
    nonloopback = ("203.0.113.5", 44444)

    app = _hello_app(proxy_token="s3cr3t-token")
    with TestClient(app, base_url="http://testserver", client=nonloopback) as client:
        assert client.get("/api/hello").status_code == 200
        assert client.head("/api/hello").status_code == 200
        # The exemption is limited to the probe: other methods/paths still need the token.
        assert client.post("/api/hello").status_code == 401
        assert client.get("/api/hello/subpath").status_code == 401
        assert client.get("/v1/models").status_code == 401

    assert calls == []
