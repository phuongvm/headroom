"""Tests for Reverse Proxy / Cloudflare Access dashboard and settings invariant."""

from __future__ import annotations

import pytest
from starlette.requests import Request
from headroom.proxy.server import (
    _request_can_view_dashboard_metadata,
)
from headroom.proxy.forwarded_headers import (
    load_trusted_dashboard_client_cidrs,
)


def _make_http_request(
    path: str = "/dashboard/settings",
    host: str = "headroom.ptdev.vip",
    client_ip: str = "172.20.0.1",
    headers: dict[str, str] | None = None,
) -> Request:
    raw_headers = [(b"host", host.encode("utf-8"))]
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("utf-8"), v.encode("utf-8")))

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": raw_headers,
        "client": (client_ip, 54321),
        "scheme": "https",
    }
    return Request(scope)


def test_cf_access_authenticated_viewer_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "test-token-123")

    req = _make_http_request(
        path="/dashboard/settings",
        host="headroom.ptdev.vip",
        client_ip="172.20.0.1",
        headers={
            "cf-access-authenticated-user-email": "vmphuongit@gmail.com",
            "cf-access-jwt-assertion": "valid.jwt.token",
        },
    )
    cidrs = load_trusted_dashboard_client_cidrs()
    assert _request_can_view_dashboard_metadata(req, cidrs) is True


def test_cf_access_from_untrusted_peer_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "test-token-123")

    # Untrusted external IP attempting to spoof CF Access header
    req = _make_http_request(
        path="/dashboard/settings",
        host="headroom.ptdev.vip",
        client_ip="203.0.113.5",
        headers={
            "cf-access-authenticated-user-email": "attacker@evil.com",
            "cf-access-jwt-assertion": "fake.jwt.token",
        },
    )
    cidrs = load_trusted_dashboard_client_cidrs()
    assert _request_can_view_dashboard_metadata(req, cidrs) is False


def test_untrusted_domain_without_ip_literal_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "test-token-123")

    req = _make_http_request(
        path="/dashboard/settings",
        host="evil.attacker.com",
        client_ip="172.20.0.1",
        headers={
            "cf-access-authenticated-user-email": "vmphuongit@gmail.com",
            "cf-access-jwt-assertion": "valid.jwt.token",
        },
    )
    cidrs = load_trusted_dashboard_client_cidrs()
    assert _request_can_view_dashboard_metadata(req, cidrs) is False


def test_cf_access_missing_jwt_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16")
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "test-token-123")

    # Only email provided without JWT assertion
    req = _make_http_request(
        path="/dashboard/settings",
        host="headroom.ptdev.vip",
        client_ip="172.20.0.1",
        headers={
            "cf-access-authenticated-user-email": "vmphuongit@gmail.com",
        },
    )
    cidrs = load_trusted_dashboard_client_cidrs()
    assert _request_can_view_dashboard_metadata(req, cidrs) is False
