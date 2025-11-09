"""Utility helpers shared by the runtime server and websocket bridge."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence, Set


__all__ = [
    "Bridge",
    "BridgeMessage",
    "WebSocketLike",
    "msg_event",
    "msg_image",
    "msg_speak",
    "msg_status",
    "msg_token",
    "to_data_url",
]


class WebSocketLike(Protocol):
    """Structural type for the websocket objects managed by :class:`Bridge`."""

    async def send(self, message: str) -> Any:  # pragma: no cover - typing hook
        ...


BridgeMessage = Mapping[str, Any]


@dataclass(slots=True)
class Bridge:
    """Simple async broadcaster with error handling and type hints."""

    _clients: Set[WebSocketLike] = field(default_factory=set)
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def register_ws(self, ws: WebSocketLike) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister_ws(self, ws: WebSocketLike) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def post(self, obj: BridgeMessage) -> None:
        payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        async with self._lock:
            clients: Sequence[WebSocketLike] = list(self._clients)

        if not clients:
            return

        results = await asyncio.gather(
            *[self._send_safe(client, payload) for client in clients],
            return_exceptions=True,
        )
        dead = [client for client, result in zip(clients, results) if isinstance(result, Exception)]
        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)

    async def _send_safe(self, ws: WebSocketLike, payload: str) -> None:
        await asyncio.wait_for(ws.send(payload), timeout=2)


def msg_status(value: str, detail: str = "") -> Dict[str, Any]:
    return {"type": "status", "value": value, "detail": detail}


def msg_event(message: str) -> Dict[str, Any]:
    return {"type": "event", "message": message}


def msg_image(data_url: str) -> Dict[str, Any]:
    return {"type": "image", "dataUrl": data_url}


def msg_speak(text: str) -> Dict[str, Any]:
    return {"type": "speak", "text": text}


def msg_token(count: int) -> Dict[str, Any]:
    return {"type": "token", "count": int(count)}


def to_data_url(img_bgr) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + b64
