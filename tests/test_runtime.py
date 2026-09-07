from __future__ import annotations

import asyncio
import threading

from app import runtime


def test_cancellation_keeps_the_inference_slot_until_worker_finishes(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "GATE", asyncio.Semaphore(1))
        started, finish, next_started = threading.Event(), threading.Event(), threading.Event()
        def slow():
            started.set()
            finish.wait(5)
        first = asyncio.create_task(runtime.assess(slow))
        while not started.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        second = asyncio.create_task(runtime.assess(next_started.set))
        await asyncio.sleep(0.02)
        assert not next_started.is_set()
        finish.set()
        await asyncio.gather(first, second, return_exceptions=True)
        assert next_started.is_set()
    asyncio.run(scenario())
