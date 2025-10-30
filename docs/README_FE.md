# Nomous – Autonomy Dashboard (browser UI)

## Run
```bash
npm i
npm run dev
```

Optional `.env`:
```
VITE_WS_URL=ws://localhost:8765
```

## WebSocket protocol (expected by UI)

Backend → UI:
- `{"type":"status","value":"thinking"|"speaking"|"idle"|"noticing"|"learning"|"error","detail":"...optional..."}`
- `{"type":"token","count": 32}`
- `{"type":"image","dataUrl":"data:image/jpeg;base64,..."}`
- `{"type":"speak","text":"...utterance..."}`
- `{"type":"metrics","payload":{"onTopic":0..1,"brevity":0..1,"responsiveness":0..1,"nonsenseRate":0..1,"rewardTotal":number}}`
- `{"type":"memory","nodes":[...], "edges":[...]}`
- `{"type":"event","message":"...anything for console..."}`

UI → Backend:
- `{"type":"ping"}`
- `{"type":"reinforce","delta":1|-1}`
- `{"type":"toggle","what":"vision"|"tts","value":true|false}`
- `{"type":"param","key":"snapshot_debounce"|"motion_sensitivity","value": number}`
- `{"type":"audio","format":"webm","codec":"opus","pcm16":"<base64 blob>"}`  // 250ms chunks
