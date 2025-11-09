from __future__ import annotations

import asyncio
from typing import List

from src.backend.utils import Bridge, msg_event, msg_status


class RecordingSocket:
    def __init__(self) -> None:
        self.messages: List[str] = []

    async def send(self, payload: str) -> None:
        await asyncio.sleep(0)
        self.messages.append(payload)


class FailingSocket:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, payload: str) -> None:  # pragma: no cover - behaviour tested via exception path
        self.calls += 1
        raise RuntimeError("boom")


def test_bridge_broadcasts_messages() -> None:
    async def _run() -> None:
        bridge = Bridge()
        client_a = RecordingSocket()
        client_b = RecordingSocket()
        bridge.register_ws(client_a)
        bridge.register_ws(client_b)

        await bridge.post(msg_status("ready", "booted"))
        await bridge.post(msg_event("started"))

        assert len(client_a.messages) == 2
        assert len(client_b.messages) == 2
        assert "\"ready\"" in client_a.messages[0]
        assert "started" in client_b.messages[1]

    asyncio.run(_run())


def test_bridge_deregisters_failing_clients() -> None:
    async def _run() -> None:
        bridge = Bridge()
        ok_client = RecordingSocket()
        bad_client = FailingSocket()
        bridge.register_ws(ok_client)
        bridge.register_ws(bad_client)

        await bridge.post(msg_event("message"))
        assert len(ok_client.messages) == 1

        # Second post should only hit the healthy client.
        await bridge.post(msg_event("again"))
        assert len(ok_client.messages) == 2
        assert bad_client.calls == 1

    asyncio.run(_run())
