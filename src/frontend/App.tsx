import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Slider } from "./components/ui/slider";
import { Tabs, TabsContent, TabsTrigger } from "./components/ui/tabs";
import { TabsList } from "./components/ui/TabsList";
import { Switch } from "./components/ui/switch";
import { Progress } from "./components/ui/progress";
import { TooltipProvider } from "./components/ui/tooltip";
import { Separator } from "./components/ui/separator";
import { Activity, Brain, Camera, Cog, MessageSquare, Play, Radio, RefreshCw, Square, Mic, MicOff, Wifi, WifiOff, Volume2 } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip as RTooltip } from "recharts";

/** Nomous â€" Autonomy Dashboard (fixed) */
export type NomousStatus = "idle" | "thinking" | "speaking" | "noticing" | "learning" | "error";
interface BehaviorStats { onTopic: number; brevity: number; responsiveness: number; nonsenseRate: number; rewardTotal: number; }
interface TokenPoint { t: number; inTok: number; outTok: number }
interface MemoryEdge { id: string; from: string; to: string; weight: number }
interface MemoryNode { id: string; label: string; strength: number; kind: "stimulus" | "concept" | "event" | "self" }
interface WSMessage {
  type: string;
  value?: string;
  detail?: string;
  text?: string;
  dataUrl?: string;
  nodes?: MemoryNode[];
  edges?: MemoryEdge[];
  message?: string;
}
interface DashboardState {
  status: NomousStatus; statusDetail?: string; tokenWindow: TokenPoint[]; behavior: BehaviorStats;
  memory: { nodes: MemoryNode[]; edges: MemoryEdge[] }; lastEvent?: string; audioEnabled: boolean; visionEnabled: boolean;
  connected: boolean; url: string; micOn: boolean; vu: number; preview?: string; consoleLines: string[]; thoughtLines: string[];
}

function useNomousBridge() {
  const [state, setState] = useState<DashboardState>({
    status: "idle", statusDetail: "Disconnected",
    tokenWindow: Array.from({ length: 30 }, (_, i) => ({ t: i, inTok: 0, outTok: 0 })),
    behavior: { onTopic: 0, brevity: 0, responsiveness: 0, nonsenseRate: 0, rewardTotal: 0 },
    memory: { nodes: [{ id: "self", label: "Nomous", strength: 1, kind: "self" }], edges: [] },
    audioEnabled: true, visionEnabled: true, connected: false,
    url: typeof window !== "undefined" ? (localStorage.getItem("nomous.ws") || "ws://localhost:8765") : "ws://localhost:8765",
    micOn: false, vu: 0, consoleLines: [], thoughtLines: [],
  });
  const wsRef = useRef<WebSocket | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const hbRef = useRef<number | null>(null);
  const tCounter = useRef(0);

  const log = useCallback((line: string) => {
    // Filter out spam/duplicates
    const skipPatterns = [
      /^unknown message: \{"type":"pong"\}/,
      /^unknown message.*pong/i,
      /^speak.*→.*$/,  // Already shown in status
    ];
    
    if (skipPatterns.some(p => p.test(line))) {
      return;  // Skip noisy messages
    }
    
    setState(p => {
      // Check if last message is identical (prevent duplicates)
      if (p.consoleLines[0] === `[${new Date().toLocaleTimeString()}] ${line}`) {
        return p;
      }
      return { ...p, consoleLines: [`[${new Date().toLocaleTimeString()}] ${line}`, ...p.consoleLines.slice(0, 150)] };
    });
  }, []);

  const push = useCallback((obj: any) => {
    const ws = wsRef.current; if (!ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify(obj));
  }, []);

  const handleMessage = useCallback((ev: MessageEvent) => {
    try {
      const msg = JSON.parse(typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data));
      switch (msg.type) {
        case "status": setState(p => ({ ...p, status: msg.value, statusDetail: msg.detail ?? p.statusDetail })); break;
        case "token": {
          const inc = Number(msg.count || 0);
          tCounter.current += inc;
          setState(p => {
            const last = p.tokenWindow[p.tokenWindow.length - 1] ?? { t: 0, inTok: 0, outTok: 0 };
            const nextT = last.t + 1;
            return { ...p, tokenWindow: [...p.tokenWindow.slice(1), { t: nextT, inTok: inc, outTok: Math.max(0, Math.round(inc * 0.6)) }] };
          });
          break;
        }
        case "speak": log(`speak â†’ ${msg.text}`); setState(p => ({ ...p, status: "speaking", statusDetail: msg.text })); break;
        case "thought": 
          setState(p => ({ ...p, thoughtLines: [`[${new Date().toLocaleTimeString()}] ${msg.text}`, ...p.thoughtLines.slice(0, 200)] })); 
          break;
        case "image": setState(p => ({ ...p, preview: msg.dataUrl })); break;
        case "metrics": setState(p => ({ ...p, behavior: {
          onTopic: msg.payload.onTopic ?? p.behavior.onTopic,
          brevity: msg.payload.brevity ?? p.behavior.brevity,
          responsiveness: msg.payload.responsiveness ?? p.behavior.responsiveness,
          nonsenseRate: msg.payload.nonsenseRate ?? p.behavior.nonsenseRate,
          rewardTotal: msg.payload.rewardTotal ?? p.behavior.rewardTotal,
        } })); break;
        case "memory": setState(p => ({ ...p, memory: { nodes: msg.nodes ?? p.memory.nodes, edges: msg.edges ?? p.memory.edges } })); break;
        case "event": log(msg.message || "event"); setState(p => ({ ...p, lastEvent: msg.message })); break;
        default: log(`unknown message: ${ev.data?.toString()?.slice(0, 120)}`);
      }
    } catch (e: any) {
      log(`bad message: ${e?.message}`);
    }
  }, [log]);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === 1) return;
    try {
      const ws = new WebSocket(state.url);
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        localStorage.setItem("nomous.ws", state.url);
        setState(p => ({ ...p, connected: true, statusDetail: "Connected" }));
        log(`connected â†’ ${state.url}`);
        hbRef.current = window.setInterval(() => push({ type: "ping" }), 10000);
      };
      ws.onmessage = handleMessage;
      ws.onclose = () => {
        if (hbRef.current) window.clearInterval(hbRef.current);
        hbRef.current = null;
        setState(p => ({ ...p, connected: false, status: "idle", statusDetail: "Disconnected" }));
        log("disconnected");
      };
      ws.onerror = () => { log("ws error"); };
    } catch (e: any) { log(`connect error: ${e?.message}`); }
  }, [handleMessage, log, push, state.url]);

  const disconnect = useCallback(() => {
    recRef.current?.stop(); recRef.current = null;
    wsRef.current?.close(); wsRef.current = null;
  }, []);

  const setMic = useCallback(async (on: boolean) => {
    if (!on) { recRef.current?.stop(); recRef.current = null; setState(p => ({ ...p, micOn: false, vu: 0 })); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 }, video: false });
      const ctx = new AudioContext({ sampleRate: 16000 });
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048; src.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      let raf: number;
      const tick = () => {
        analyser.getByteTimeDomainData(data);
        let peak = 0; for (let i=0;i<data.length;i++){ const v=(data[i]-128)/128; peak=Math.max(peak, Math.abs(v)); }
        setState(p => ({ ...p, vu: Math.min(1, peak*2) }));
        raf = requestAnimationFrame(tick);
      };
      tick();

      const rec = new MediaRecorder(stream);
      recRef.current = rec;
      rec.ondataavailable = async (ev) => {
        if (!ev.data || ev.data.size === 0) return;
        const buf = await ev.data.arrayBuffer();
        const raw = new Uint8Array(buf);
        const b64 = btoa(String.fromCharCode(...raw));
        push({ type: "audio", rate: 16000, pcm16: b64 });
      };
      rec.start(250);
      setState(p => ({ ...p, micOn: true }));
    } catch (e: any) {
      setState(p => ({ ...p, micOn: false })); log(`mic error: ${e?.message}`);
    }
  }, [push, log]);

  return { state, setState, connect, disconnect, setMic, push, log };
}

const statusMap: Record<NomousStatus, { color: string; label: string }> = {
  idle: { color: "bg-zinc-400", label: "Idle" },
  thinking: { color: "bg-purple-500", label: "Thinking" },
  speaking: { color: "bg-emerald-500", label: "Speaking" },
  noticing: { color: "bg-amber-500", label: "Noticed" },
  learning: { color: "bg-cyan-500", label: "Learning" },
  error: { color: "bg-red-600", label: "Error" },
};

function Meter({ value, label }: { value: number; label: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-zinc-300"><span>{label}</span><span>{Math.round(value*100)}%</span></div>
      <Progress value={value*100} />
    </div>
  );
}

function MemoryGraph({ nodes, edges }: { nodes: MemoryNode[]; edges: MemoryEdge[] }) {
  const layout = React.useMemo(() => {
    const width = 560, height = 300;
    const buckets: Record<MemoryNode["kind"], MemoryNode[]> = { stimulus: [], concept: [], event: [], self: [] };
    nodes.forEach(n => buckets[n.kind].push(n));
    const xSlots: Record<MemoryNode["kind"], number> = { stimulus: 80, event: width/2, concept: width-80, self: width/2 };
    const yStep = (arr: MemoryNode[]) => arr.length>1 ? height/(arr.length+1) : height/2;
    const pos = new Map<string, {x:number,y:number}>();
    (Object.keys(buckets) as MemoryNode["kind"][]).forEach(k => {
      const arr = buckets[k];
      arr.forEach((n, i) => pos.set(n.id, { x: xSlots[k], y: (i+1)*yStep(arr) }));
    });
    return { width, height, pos };
  }, [nodes]);

  return (
    <svg width="100%" height="300" viewBox={`0 0 ${layout.width} ${layout.height}`} className="rounded-xl bg-zinc-900/60">
      {edges.map(e => {
        const a = layout.pos.get(e.from)!; const b = layout.pos.get(e.to)!;
        return <line key={e.id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="currentColor" strokeOpacity={0.35+e.weight*0.5} strokeWidth={1+e.weight*2} className="text-zinc-200"/>;
      })}
      {nodes.map(n => {
        const p = layout.pos.get(n.id)!;
        const fill = n.kind === "self" ? "fill-emerald-500" : n.kind === "stimulus" ? "fill-amber-500" : n.kind === "event" ? "fill-cyan-500" : "fill-purple-500";
        return (
          <g key={n.id}>
            <circle cx={p.x} cy={p.y} r={10 + n.strength*10} className={`${fill}`} opacity={0.95} />
            <text x={p.x} y={p.y-16- n.strength*6} textAnchor="middle" className="fill-zinc-50 text-xs">{n.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

export default function App(){
  const { state, setState, connect, disconnect, setMic, push, log } = useNomousBridge();
  const st = statusMap[state.status];
  const tokenTotal = state.tokenWindow.reduce((a, p) => a + p.inTok + p.outTok, 0);

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-950 to-black text-zinc-50 p-4 md:p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-3.5 h-3.5 rounded-full ${st.color} shadow-[0_0_20px_rgba(255,255,255,.15)]`} />
            <div>
              <div className="text-xs uppercase tracking-wide text-zinc-400/90">Status</div>
              <div className="font-semibold text-zinc-100">
                {st.label}{state.statusDetail ? (<><span> â€” </span><span className="text-zinc-300">{state.statusDetail}</span></>) : null}
              </div>
            </div>
            <Badge className="ml-2 bg-zinc-800/80 text-zinc-100">Tokens: {tokenTotal}</Badge>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden md:flex items-center gap-1 text-xs text-zinc-400 mr-2">
              {state.connected ? <Wifi className="w-4 h-4"/> : <WifiOff className="w-4 h-4"/>}
              <span>{state.url}</span>
            </div>
            {!state.connected ? (
              <Button onClick={connect} className="px-3 py-1 text-sm bg-emerald-600/90 hover:bg-emerald-500/90 text-white"><Play className="w-4 h-4 mr-1"/> Connect</Button>
            ) : (
              <Button variant="danger" onClick={disconnect} className="bg-red-600/90 text-white hover:bg-red-500/90"><Square className="w-4 h-4 mr-1"/> Disconnect</Button>
            )}
          </div>
        </div>

        <Card className="bg-zinc-900/60 backdrop-blur-sm border-zinc-800/70">
          <CardContent className="p-4">
            <div className="w-full">
              <Tabs defaultValue="overview">
              <TabsList className="flex flex-wrap gap-2 bg-zinc-950/60 border border-zinc-800/60">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="console">Console</TabsTrigger>
                <TabsTrigger value="behavior">Behavior</TabsTrigger>
                <TabsTrigger value="tokens">Tokens</TabsTrigger>
                <TabsTrigger value="runtime">Runtime</TabsTrigger>
                <TabsTrigger value="memory">Memory</TabsTrigger>
                <TabsTrigger value="thoughts">Thoughts</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="pt-4">
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                  <div className="xl:col-span-2 space-y-4">
                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-3 text-zinc-200">
                          <div className="flex items-center gap-2"><Camera className="w-4 h-4"/><span className="font-semibold">Live Vision Feed</span></div>
                          <div className="flex items-center gap-3 text-xs text-zinc-400">
                            <div className="flex items-center gap-1"><Radio className="w-3.5 h-3.5"/> dshow</div>
                            <div className="flex items-center gap-1"><Activity className="w-3.5 h-3.5"/> {state.status}</div>
                          </div>
                        </div>
                        <div className="aspect-video w-full rounded-xl overflow-hidden border border-zinc-800/70 bg-zinc-950 grid place-items-center">
                          {state.preview ? (
                            <img alt="preview" src={state.preview} className="w-full h-full object-cover"/>
                          ) : (
                            <div className="text-zinc-400 text-sm">Waiting for framesâ€¦</div>
                          )}
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-zinc-200 font-semibold"><Volume2 className="w-4 h-4"/> Voice</div>
                          <div className="flex items-center gap-2">
                            {!state.micOn ? (
                              <Button onClick={() => setMic(true)} className="px-3 py-1 text-sm bg-emerald-600/90 hover:bg-emerald-500/90 text-white"><Mic className="w-4 h-4 mr-1"/> Mic: OFF</Button>
                            ) : (
                              <Button variant="secondary" onClick={() => setMic(false)} className="px-3 py-1 text-sm bg-zinc-800/80 text-zinc-100 hover:bg-zinc-700/80"><MicOff className="w-4 h-4 mr-1"/> Mic: ON</Button>
                            )}
                          </div>
                        </div>
                        <div className="h-2 rounded bg-zinc-800 overflow-hidden">
                          <div className="h-full bg-emerald-500 transition-all" style={{ width: `${Math.round(state.vu*100)}%` }} />
                        </div>
                        <div className="text-xs text-zinc-400">Sends 16 kHz chunks to the runtime while ON.</div>
                      </CardContent>
                    </Card>

                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="p-4 space-y-3">
                        <div className="flex items-center gap-2 mb-1 text-zinc-200"><Brain className="w-4 h-4"/><span className="font-semibold">Behavior Snapshot</span></div>
                        <Meter value={state.behavior.onTopic} label="On-topic"/>
                        <Meter value={state.behavior.brevity} label="Brevity"/>
                        <Meter value={state.behavior.responsiveness} label="Responsiveness"/>
                        <Meter value={1 - state.behavior.nonsenseRate} label="Coherence"/>
                      </CardContent>
                    </Card>
                  </div>

                  <div className="space-y-4">
                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 mb-2 text-zinc-200"><Activity className="w-4 h-4"/><span className="font-semibold">Token Flow (last 30s)</span></div>
                        <div className="h-40">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={state.tokenWindow} margin={{ left: 0, right: 0, top: 0, bottom: 0 }}>
                              <defs>
                                <linearGradient id="gIn" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.9}/>
                                  <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.08}/>
                                </linearGradient>
                                <linearGradient id="gOut" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.9}/>
                                  <stop offset="95%" stopColor="#34d399" stopOpacity={0.08}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.12} />
                              <XAxis dataKey="t" hide />
                              <YAxis hide />
                              <RTooltip contentStyle={{ background: "#0b0b0c", border: "1px solid #2a2a2e", color: "#e5e7eb" }} />
                              <Area type="monotone" dataKey="inTok" stroke="#a78bfa" fillOpacity={1} fill="url(#gIn)" />
                              <Area type="monotone" dataKey="outTok" stroke="#34d399" fillOpacity={1} fill="url(#gOut)" />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="console" className="pt-4">
                <Card className="bg-zinc-900/70 border-zinc-800/60">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-2 text-zinc-200"><MessageSquare className="w-4 h-4"/><span className="font-semibold">Event Console</span></div>
                    <div className="h-80 overflow-auto rounded-md bg-black/60 p-3 font-mono text-xs leading-relaxed">
                      {state.consoleLines.map((l, i) => (
                        <div key={i} className="text-zinc-200">{l}</div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="behavior" className="pt-4">
                <div className="grid sm:grid-cols-2 gap-4">
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="p-4 space-y-3 text-zinc-200">
                      <div className="flex items-center gap-2 mb-1"><Brain className="w-4 h-4"/><span className="font-semibold">Behavior Metrics</span></div>
                      <Meter value={state.behavior.onTopic} label="On-topic"/>
                      <Meter value={state.behavior.brevity} label="Brevity"/>
                      <Meter value={state.behavior.responsiveness} label="Responsiveness"/>
                      <Meter value={1 - state.behavior.nonsenseRate} label="Coherence"/>
                      <Separator className="my-1"/>
                      <div className="text-sm">Reward total: <span className={state.behavior.rewardTotal>=0?"text-emerald-400":"text-red-400"}>{state.behavior.rewardTotal.toFixed(1)}</span></div>
                      <div className="grid grid-cols-2 gap-2 pt-1">
                        <Button variant="secondary" className="px-3 py-1 text-sm bg-zinc-800/80 text-zinc-100 hover:bg-zinc-700/80" onClick={()=>push({ type: "reinforce", delta: +1 })}>Reward +1</Button>
                        <Button variant="secondary" className="px-3 py-1 text-sm bg-zinc-800/80 text-zinc-100 hover:bg-zinc-700/80" onClick={()=>push({ type: "reinforce", delta: -1 })}>Penalty -1</Button>
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="p-4 text-zinc-200">
                      <div className="font-semibold mb-2">Guidance</div>
                      <p className="text-sm text-zinc-300">Use reward/penalty to nudge style: concise, on-topic, timely. This UI mirrors reinforcement numbers from the runtime.</p>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="tokens" className="pt-4">
                <Card className="bg-zinc-900/70 border-zinc-800/60">
                  <CardContent className="p-4 text-zinc-200">
                    <div className="flex items-center gap-2 mb-2"><Activity className="w-4 h-4"/><span className="font-semibold">Token Flow (last 30s)</span></div>
                    <div className="h-60">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={state.tokenWindow} margin={{ left: 0, right: 0, top: 0, bottom: 0 }}>
                          <defs>
                            <linearGradient id="gIn2" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.9}/>
                              <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.08}/>
                            </linearGradient>
                            <linearGradient id="gOut2" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#34d399" stopOpacity={0.9}/>
                              <stop offset="95%" stopColor="#34d399" stopOpacity={0.08}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.12} />
                          <XAxis dataKey="t" hide />
                          <YAxis hide />
                          <RTooltip contentStyle={{ background: "#0b0b0c", border: "1px solid #2a2a2e", color: "#e5e7eb" }} />
                          <Area type="monotone" dataKey="inTok" stroke="#a78bfa" fillOpacity={1} fill="url(#gIn2)" />
                          <Area type="monotone" dataKey="outTok" stroke="#34d399" fillOpacity={1} fill="url(#gOut2)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="runtime" className="pt-4">
                <Card className="bg-zinc-900/70 border-zinc-800/60">
                  <CardContent className="p-4 space-y-3 text-zinc-200">
                    <div className="flex items-center gap-2 mb-1"><Cog className="w-4 h-4"/><span className="font-semibold">Runtime Controls</span></div>
                    <div className="flex items-center justify-between text-sm">
                      <span>Vision</span>
                      <Switch checked={state.visionEnabled} onCheckedChange={(v)=>{ setState({ ...state, visionEnabled: v }); push({ type: "toggle", what: "vision", value: v }); }} />
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span>Audio (TTS)</span>
                      <Switch checked={state.audioEnabled} onCheckedChange={(v)=>{ setState({ ...state, audioEnabled: v }); push({ type: "toggle", what: "tts", value: v }); }} />
                    </div>
                    <Separator/>
                    <div className="text-xs text-zinc-300">Debounce snapshots (sec)</div>
                    <Slider defaultValue={[4]} min={1} max={10} step={0.5} onValueChange={(v)=>push({ type: "param", key: "snapshot_debounce", value: v[0] })} />
                    <div className="text-xs text-zinc-300">Motion sensitivity</div>
                    <Slider defaultValue={[30]} min={5} max={80} step={1} onValueChange={(v)=>push({ type: "param", key: "motion_sensitivity", value: v[0] })} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="memory" className="pt-4">
                <Card className="bg-zinc-900/70 border-zinc-800/60">
                  <CardContent className="p-4 text-zinc-200">
                    <div className="flex items-center gap-2 mb-2"><Brain className="w-4 h-4"/><span className="font-semibold">Memory Graph</span></div>
                    <MemoryGraph nodes={state.memory.nodes} edges={state.memory.edges} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="thoughts" className="pt-4">
                <Card className="bg-zinc-900/70 border-zinc-800/60">
                  <CardContent className="p-4 text-zinc-200">
                    <div className="flex items-center gap-2 mb-2"><MessageSquare className="w-4 h-4"/><span className="font-semibold">Thought Trace</span></div>
                    <div className="h-60 overflow-auto rounded-md bg-black/60 p-3 font-mono text-xs leading-relaxed text-zinc-100/95">
                      {state.thoughtLines.length > 0 ? (
                        state.thoughtLines.map((l,i)=>(
                          <div key={i} className="mb-1 text-purple-300">{l}</div>
                        ))
                      ) : (
                        <div className="text-zinc-500 text-center mt-8">Waiting for thoughts...</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

                <TabsContent value="settings" className="pt-4">
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="p-4 space-y-3 text-zinc-200">
                      <div className="font-semibold">Bridge</div>
                      <div className="flex items-center gap-2">
                        <input className="w-full px-2 py-1 rounded bg-zinc-950 border border-zinc-800 text-sm" value={state.url} onChange={(e)=>setState(p=>({ ...p, url: e.target.value }))}/>
                        {!state.connected ? (
                          <Button onClick={connect} className="px-3 py-1 text-sm"><Play className="w-4 h-4 mr-1"/>Connect</Button>
                        ) : (
                          <Button variant="danger" onClick={disconnect} className="px-3 py-1 text-sm"><Square className="w-4 h-4 mr-1"/>Disconnect</Button>
                        )}
                      </div>
                      <div className="text-xs text-zinc-400">The UI speaks JSON frames over WebSocket. Run the Python bridge on ws://localhost:8765.</div>
                    </CardContent>
                  </Card>
                </TabsContent>
                </Tabs>
              </div>
          </CardContent>
        </Card>

        <div className="text-[10px] text-zinc-400/80">Nomous Autonomy UI â€¢ WebSocket JSON from Python. Colors: purple=thinking, amber=noticed, emerald=speaking, cyan=learning, red=error. Mic sends 16kHz chunks.</div>
      </div>
    </TooltipProvider>
  );
}
