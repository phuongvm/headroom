"""Stream outcome status: client disconnects are not upstream failures.

Since #3747 a stream that ends without its protocol terminal event
(``data: [DONE]``, ``message_stop``, ``response.completed``...) is booked as a
synthetic ``STREAM_FAILURE_STATUS`` (502) so truncated upstream streams stop
inflating the success counter. That rule could not tell *who* ended the
stream: a client that disconnects (or cancels its turn) halfway through a
perfectly healthy HTTP 200 upstream stream was booked exactly like an upstream
crash — ``record_failed(provider=...)`` and ``headroom_requests_failed_total``
— and on agent traffic, where clients abort and supersede turns constantly,
failures outnumbered completions on the dashboard.

These tests pin the split:

* clean completion                      -> 200, ``record_request``
* client disconnect on a healthy 200    -> 200, ``record_request``, never ``record_failed``
* genuine upstream failure (error event, transport error, 5xx, clean EOF
  without terminal event)               -> failure status, ``record_failed``
"""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.handlers.streaming import (
    STREAM_FAILURE_STATUS,
    StreamingMixin,
    _stream_outcome_status,
)
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.server import HeadroomProxy

CHAT_URL = "https://upstream.example/v1/chat/completions"
CHUNK_1 = b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
CHUNK_2 = b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
CHUNK_FINISH_STOP = b'data: {"id":"c2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
DONE = b"data: [DONE]\n\n"
ERROR_EVENT = b'data: {"error":{"type":"server_error","message":"upstream exploded"}}\n\n'


# ── helpers ──────────────────────────────────────────────────────────────


def _proxy(metrics=None):
    """A HeadroomProxy shell with the real SSE parser and the real outcome funnel."""
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    if metrics is None:
        metrics = MagicMock()
        metrics.record_request = AsyncMock(return_value=None)
        metrics.record_failed = AsyncMock(return_value=None)
        metrics.record_rate_limited = AsyncMock(return_value=None)
        metrics.record_cache_miss_attribution = AsyncMock(return_value=None)
    proxy.metrics = metrics
    proxy.cost_tracker = None
    proxy.logger = None
    proxy.memory_handler = None
    proxy.memory_manager = None
    config = MagicMock()
    config.memory_enabled = False
    config.ccr_inject_tool = False
    config.log_full_messages = False
    config.retry_max_attempts = 1
    config.retry_enabled = False
    config.retry_base_delay_ms = 0
    config.retry_max_delay_ms = 0
    proxy._config = config
    proxy.config = config
    return proxy


def _upstream(chunks, *, status_code=200, hang_after=False, raise_after=None):
    """Mock httpx streaming response.

    ``hang_after`` blocks forever after the last chunk (a slow, healthy
    upstream still generating); ``raise_after`` raises that exception after
    the last chunk (an upstream transport failure mid-stream).
    """
    response = AsyncMock()
    response.status_code = status_code
    response.headers = httpx.Headers({"content-type": "text/event-stream"})

    async def aiter_bytes():
        for chunk in chunks:
            yield chunk
        if raise_after is not None:
            raise raise_after
        if hang_after:
            await asyncio.Event().wait()

    response.aiter_bytes = aiter_bytes
    response.aclose = AsyncMock()
    response.aread = AsyncMock(return_value=b'{"error":{"message":"upstream 500"}}')
    return response


async def _open_stream(proxy, upstream, *, prefix_tracker=None):
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=upstream)
    return await proxy._stream_response(
        url=CHAT_URL,
        headers={"authorization": "Bearer sk-test"},
        body={
            "model": "gpt-test",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        provider="openai",
        model="gpt-test",
        request_id="test-stream-outcome",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
        prefix_tracker=prefix_tracker,
    )


def _capture_outcomes(proxy):
    """Spy on the funnel input while still running the real funnel."""
    outcomes: list[RequestOutcome] = []

    async def record(outcome):
        outcomes.append(outcome)
        await emit_request_outcome(proxy, outcome)

    proxy._record_request_outcome = record
    return outcomes


async def _drain(response):
    async for _ in response.body_iterator:
        pass


async def _consume_then_close(response, n_chunks):
    """Client reads ``n_chunks`` body chunks, then disconnects (generator closed)."""
    body = response.body_iterator
    received = []
    for _ in range(n_chunks):
        received.append(await body.__anext__())
    await body.aclose()  # GeneratorExit at the suspended yield
    return received


# ── unit: the status rule ────────────────────────────────────────────────


def test_status_rule_keeps_success_on_completion():
    assert _stream_outcome_status(200, True) == 200


def test_status_rule_client_disconnect_on_healthy_200_is_not_a_failure():
    assert _stream_outcome_status(200, False, client_disconnected=True) == 200


def test_status_rule_clean_eof_without_terminal_is_still_a_failure():
    # No client disconnect: the upstream simply stopped — #3747 behaviour kept.
    assert _stream_outcome_status(200, False) == STREAM_FAILURE_STATUS


def test_status_rule_error_event_wins_over_client_disconnect():
    assert (
        _stream_outcome_status(200, False, client_disconnected=True, stream_failed=True)
        == STREAM_FAILURE_STATUS
    )


def test_status_rule_upstream_5xx_is_never_masked_by_disconnect():
    assert _stream_outcome_status(500, False, client_disconnected=True) == STREAM_FAILURE_STATUS


# ── (a) clean complete stream ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clean_complete_stream_records_success():
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1, CHUNK_2, DONE]))

    await _drain(response)

    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200
    assert outcomes[0].client_disconnected is False
    proxy.metrics.record_request.assert_awaited_once()
    proxy.metrics.record_failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_with_finish_reason_stop_without_done_event_records_success():
    """Streams from gateways (e.g. 9Router proxying Claude) ending in finish_reason without [DONE]."""
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1, CHUNK_FINISH_STOP]))

    await _drain(response)

    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200
    assert outcomes[0].client_disconnected is False
    proxy.metrics.record_request.assert_awaited_once()
    proxy.metrics.record_failed.assert_not_awaited()


# ── (b) client-aborted stream ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_disconnect_midstream_is_not_counted_as_upstream_failure():
    """GeneratorExit path: client closes the body after the first chunk."""
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    upstream = _upstream([CHUNK_1, CHUNK_2, DONE])
    response = await _open_stream(proxy, upstream)

    received = await _consume_then_close(response, 1)

    assert received == [CHUNK_1]
    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200, "client abort must not synthesize a 502"
    assert outcomes[0].client_disconnected is True
    proxy.metrics.record_failed.assert_not_awaited()
    proxy.metrics.record_request.assert_awaited_once()
    upstream.aclose.assert_awaited()


@pytest.mark.asyncio
async def test_client_cancel_while_waiting_for_upstream_is_not_a_failure():
    """CancelledError path: the request task is cancelled mid-generation."""
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1], hang_after=True))
    first_chunk = asyncio.Event()

    async def client():
        async for _ in response.body_iterator:
            first_chunk.set()

    task = asyncio.create_task(client())
    await asyncio.wait_for(first_chunk.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200
    assert outcomes[0].client_disconnected is True
    proxy.metrics.record_failed.assert_not_awaited()
    proxy.metrics.record_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_asgi_disconnect_during_stream_is_not_a_failure():
    """Real Starlette lifecycle: the client sends http.disconnect mid-body."""
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1], hang_after=True))
    body_started = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            body_started.set()

    async def receive():
        await body_started.wait()
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": "POST", "headers": []}
    await asyncio.wait_for(response(scope, receive, send), timeout=5)

    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200
    proxy.metrics.record_failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_disconnect_does_not_increment_prometheus_failed_total():
    """End to end against the real PrometheusMetrics export."""
    metrics = PrometheusMetrics(stateless=True)
    proxy = _proxy(metrics)
    _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1, CHUNK_2, DONE]))

    await _consume_then_close(response, 1)

    assert metrics.requests_failed == 0
    assert sum(metrics.requests_failed_by_provider.values()) == 0
    assert metrics.requests_total == 1
    exported = await metrics.export()
    failed_lines = [
        line for line in exported.splitlines() if line.startswith("headroom_requests_failed_total")
    ]
    assert failed_lines, "failed counter must still be exported"
    assert all(line.rstrip().endswith(" 0") for line in failed_lines), failed_lines


@pytest.mark.asyncio
async def test_client_disconnect_does_not_advance_prefix_tracker():
    """A partial assistant message must never become the next turn's prefix."""
    proxy = _proxy()
    _capture_outcomes(proxy)
    prefix_tracker = MagicMock()
    response = await _open_stream(
        proxy, _upstream([CHUNK_1, CHUNK_2, DONE]), prefix_tracker=prefix_tracker
    )

    await _consume_then_close(response, 1)

    prefix_tracker.update_from_response.assert_not_called()


@pytest.mark.asyncio
async def test_client_disconnect_is_marked_on_perf_line(caplog):
    proxy = _proxy()
    _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1, CHUNK_2, DONE]))

    with caplog.at_level("INFO", logger="headroom.proxy"):
        await _consume_then_close(response, 1)

    perf = [r.getMessage() for r in caplog.records if " PERF " in r.getMessage()]
    assert perf and perf[-1].endswith("client_aborted=1"), perf


# ── (c) genuine upstream failures still count ────────────────────────────


@pytest.mark.asyncio
async def test_upstream_error_event_is_counted_as_failure():
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1, ERROR_EVENT, DONE]))

    await _drain(response)

    assert outcomes[0].status_code == STREAM_FAILURE_STATUS
    proxy.metrics.record_failed.assert_awaited_once()
    proxy.metrics.record_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_upstream_error_event_then_client_disconnect_is_still_a_failure():
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([ERROR_EVENT, CHUNK_2, DONE]))

    await _consume_then_close(response, 2)

    assert outcomes[0].status_code == STREAM_FAILURE_STATUS
    proxy.metrics.record_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_upstream_transport_error_midstream_is_counted_as_failure():
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    upstream = _upstream([CHUNK_1], raise_after=httpx.ReadError("connection reset by upstream"))
    response = await _open_stream(proxy, upstream)

    await _drain(response)

    assert outcomes[0].status_code == STREAM_FAILURE_STATUS
    assert outcomes[0].client_disconnected is False
    proxy.metrics.record_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_upstream_clean_eof_without_terminal_event_is_counted_as_failure():
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([CHUNK_1, CHUNK_2]))

    await _drain(response)

    assert outcomes[0].status_code == STREAM_FAILURE_STATUS
    proxy.metrics.record_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_upstream_5xx_status_is_counted_as_failure():
    proxy = _proxy()
    outcomes = _capture_outcomes(proxy)
    response = await _open_stream(proxy, _upstream([], status_code=500))

    assert response.status_code == 500
    assert outcomes[0].status_code == 500
    proxy.metrics.record_failed.assert_awaited_once()
    proxy.metrics.record_request.assert_not_awaited()


# ── backend-routed paths share the rule ──────────────────────────────────


class _Config:
    def __getattr__(self, name):
        return False


class _OpenAIBackend:
    name = "litellm"

    async def stream_openai_message(self, body, headers):
        yield CHUNK_1.decode()
        yield CHUNK_2.decode()
        yield DONE.decode()


class _AnthropicBackend:
    name = "bedrock"

    async def stream_message(self, body, headers):
        yield SimpleNamespace(
            event_type="message_start",
            raw_sse='event: message_start\ndata: {"type":"message_start","message":{}}\n\n',
            data={"type": "message_start", "message": {}},
        )
        yield SimpleNamespace(
            event_type="message_stop",
            raw_sse='event: message_stop\ndata: {"type":"message_stop"}\n\n',
            data={"type": "message_stop"},
        )


def _backend_handler(backend, outcomes):
    async def record(outcome):
        outcomes.append(outcome)

    handler = SimpleNamespace(
        anthropic_backend=backend, config=_Config(), _record_request_outcome=record
    )
    handler._parse_sse_usage_from_buffer = MethodType(
        StreamingMixin._parse_sse_usage_from_buffer, handler
    )
    handler._extract_anthropic_cache_ttl_metrics = lambda usage: (0, 0)
    return handler


@pytest.mark.asyncio
async def test_openai_backend_client_disconnect_is_not_a_failure():
    outcomes: list[RequestOutcome] = []
    handler = _backend_handler(_OpenAIBackend(), outcomes)
    response = await StreamingMixin._stream_openai_via_backend(
        handler, {"messages": []}, {}, "m", "rid", 0.0, 0, 0, 0, [], {}, 0.0
    )

    await _consume_then_close(response, 1)

    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200
    assert outcomes[0].client_disconnected is True


@pytest.mark.asyncio
async def test_bedrock_backend_client_disconnect_is_not_a_failure():
    outcomes: list[RequestOutcome] = []
    handler = _backend_handler(_AnthropicBackend(), outcomes)
    response = await StreamingMixin._stream_response_bedrock(
        handler, {"messages": []}, {}, "anthropic", "m", "rid", 0, 0, 0, [], {}, 0.0
    )

    # ping + message_start, then the client leaves before message_stop.
    await _consume_then_close(response, 2)

    assert len(outcomes) == 1
    assert outcomes[0].status_code == 200
    assert outcomes[0].client_disconnected is True


# ── funnel: a client-aborted outcome books as a request, never a failure ─


@pytest.mark.asyncio
async def test_funnel_books_client_aborted_outcome_as_request():
    metrics = MagicMock()
    metrics.record_request = AsyncMock(return_value=None)
    metrics.record_failed = AsyncMock(return_value=None)
    handler = SimpleNamespace(metrics=metrics, cost_tracker=None, logger=None)
    outcome = RequestOutcome(
        request_id="r",
        provider="openai",
        model="gpt-test",
        original_tokens=10,
        optimized_tokens=10,
        output_tokens=2,
        tokens_saved=0,
        attempted_input_tokens=10,
        status_code=200,
        client_disconnected=True,
    )

    await emit_request_outcome(handler, outcome)

    metrics.record_failed.assert_not_awaited()
    metrics.record_request.assert_awaited_once()
