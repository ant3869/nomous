# Title: Bridge + helpers
# Path: backend/utils.py
# Purpose: Single-event-loop broadcaster (queue) and message helpers.

import asyncio, base64, json
from typing import Any, Dict

class Bridge:
    def __init__(self):
        self._ws: list = []

    def register_ws(self, ws):
        if ws not in self._ws: self._ws.append(ws)

    def unregister_ws(self, ws):
        if ws in self._ws: self._ws.remove(ws)

    async def post(self, obj: Dict[str, Any]):
        dead = []
        payload = json.dumps(obj)
        for ws in self._ws:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister_ws(ws)

def msg_status(value: str, detail: str=""):
    return {"type":"status","value":value,"detail":detail}

def msg_event(message: str):
    return {"type":"event","message":message}

def msg_image(data_url: str):
    return {"type":"image","dataUrl": data_url}

def msg_speak(text: str):
    return {"type":"speak","text": text}

def msg_token(count: int):
    return {"type":"token","count": int(count)}

def to_data_url(img_bgr):
    import cv2, base64
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok: return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + b64
