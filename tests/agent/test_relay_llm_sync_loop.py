"""Regression coverage for Relay synchronous calls on active event loops (#148172)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm


@pytest.mark.asyncio
async def test_sync_execute_rejects_before_building_managed_coroutine(monkeypatch):
    calls: list[str] = []

    class Attempt:
        def run_managed(self, *_args, **_kwargs):
            calls.append("run_managed")

            async def result():
                return {"content": "unused"}

            return result()

        def resolve_failure(self, exc, _defer_logical_completion):
            raise exc

    monkeypatch.setattr(
        relay_llm._ManagedAttempt,
        "resolve",
        lambda *_args, **_kwargs: Attempt(),
    )

    with pytest.raises(
        RuntimeError,
        match="Synchronous Relay LLM execution cannot run on an event-loop thread",
    ):
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: {"content": "unused"},
            session_id="session-1",
            name="custom",
            model_name="test-model",
        )

    assert calls == []


@pytest.mark.asyncio
async def test_run_awaitable_closes_coroutine_before_rejecting_active_loop():
    async def pending():
        await asyncio.sleep(0)

    value = pending()

    with pytest.raises(
        RuntimeError,
        match="Synchronous Relay LLM execution cannot run on an event-loop thread",
    ):
        relay_llm._run_awaitable(value)

    assert value.cr_frame is None
