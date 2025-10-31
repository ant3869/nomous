import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Slider } from "./components/ui/slider";
import { Tabs, TabsContent, TabsTrigger } from "./components/ui/tabs";
import { TabsList } from "./components/ui/TabsList";
import { FilePathInput } from "./components/FilePathInput";
import { Switch } from "./components/ui/switch";
import { Progress } from "./components/ui/progress";
import { TooltipProvider } from "./components/ui/tooltip";
import { Separator } from "./components/ui/separator";
import { Activity, Brain, Camera, Cog, MessageSquare, Play, Radio, RefreshCw, Square, Mic, MicOff, Wifi, WifiOff, Volume2, Flag, Database, Clock } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip as RTooltip } from "recharts";
import type { MemoryEdge, MemoryNode } from "./types/memory";
import { buildMemoryNodeDetail, computeMemoryInsights } from "./utils/memory";
import type { MemoryNodeDetail, MemoryInsightEntry } from "./utils/memory";

/** Nomous â€" Autonomy Dashboard (fixed) */
export type NomousStatus = "idle" | "thinking" | "speaking" | "noticing" | "learning" | "error";
interface BehaviorStats { onTopic: number; brevity: number; responsiveness: number; nonsenseRate: number; rewardTotal: number; }
interface TokenPoint { t: number; inTok: number; outTok: number }
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
interface ControlSettings {
  cameraEnabled: boolean;
  microphoneEnabled: boolean;
  ttsEnabled: boolean;
  speakerEnabled: boolean;
  sttEnabled: boolean;
  ttsVoice: string;
  micSensitivity: number;
  masterVolume: number;
  cameraResolution: string;
  cameraExposure: number;
  cameraBrightness: number;
  llmModelPath: string;
  visionModelPath: string;
  audioModelPath: string;
  sttModelPath: string;
  llmTemperature: number;
  llmMaxTokens: number;
  modelStrategy: "speed" | "balanced" | "accuracy" | "custom";
}

type ModelPathKey = "llmModelPath" | "visionModelPath" | "audioModelPath" | "sttModelPath";
type PresetStrategy = Exclude<ControlSettings["modelStrategy"], "custom">;

interface DashboardState {
  status: NomousStatus; statusDetail?: string; tokenWindow: TokenPoint[]; behavior: BehaviorStats;
  memory: { nodes: MemoryNode[]; edges: MemoryEdge[] }; lastEvent?: string; audioEnabled: boolean; visionEnabled: boolean;
  connected: boolean; url: string; micOn: boolean; vu: number; preview?: string; consoleLines: string[]; thoughtLines: string[];
  speechLines: string[]; systemLines: string[]; settings: ControlSettings;
}

const TARGET_SAMPLE_RATE = 16000;

interface MicChain {
  ctx: AudioContext;
  stream: MediaStream;
  source: MediaStreamAudioSourceNode;
  analyser: AnalyserNode;
  processor: ScriptProcessorNode;
  gain: GainNode;
  raf?: number;
  residual?: Float32Array;
}

function concatFloat32(a: Float32Array | undefined, b: Float32Array): Float32Array {
  if (!a || a.length === 0) {
    return b.slice();
  }
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function floatTo16BitPCM(samples: Float32Array): Uint8Array {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return new Uint8Array(buffer);
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const slice = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

function encodeAudioChunk(chain: MicChain, chunk: Float32Array, sampleRate: number): string | null {
  if (chunk.length === 0) {
    return null;
  }

  const combined = concatFloat32(chain.residual, chunk);
  const ratio = sampleRate / TARGET_SAMPLE_RATE;

  if (ratio <= 0) {
    chain.residual = new Float32Array(0);
    return null;
  }

  if (ratio < 1) {
    const upRatio = TARGET_SAMPLE_RATE / sampleRate;
    const newLength = Math.floor(combined.length * upRatio);
    if (newLength <= 0) {
      chain.residual = combined;
      return null;
    }
    const upsampled = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const idx = i / upRatio;
      const low = Math.floor(idx);
      const high = Math.min(low + 1, combined.length - 1);
      const weight = idx - low;
      const sample = combined[low] + (combined[high] - combined[low]) * weight;
      upsampled[i] = sample;
    }
    chain.residual = new Float32Array(0);
    return bytesToBase64(floatTo16BitPCM(upsampled));
  }

  const newLength = Math.floor(combined.length / ratio);
  if (newLength <= 0) {
    chain.residual = combined;
    return null;
  }

  const downsampled = new Float32Array(newLength);
  let offset = 0;
  for (let i = 0; i < newLength; i++) {
    const nextOffset = Math.floor((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < combined.length; j++) {
      sum += combined[j];
      count++;
    }
    downsampled[i] = count > 0 ? sum / count : 0;
    offset = nextOffset;
  }

  chain.residual = combined.slice(offset);
  return bytesToBase64(floatTo16BitPCM(downsampled));
}

function useNomousBridge() {
  const defaultSettings: ControlSettings = {
    cameraEnabled: true,
    microphoneEnabled: false,
    ttsEnabled: true,
    speakerEnabled: true,
    sttEnabled: true,
    ttsVoice: "piper/en_US-amy-medium",
    micSensitivity: 60,
    masterVolume: 70,
    cameraResolution: "1280x720",
    cameraExposure: 45,
    cameraBrightness: 55,
    llmModelPath: "/models/llm/main.gguf",
    visionModelPath: "/models/vision/runtime.bin",
    audioModelPath: "/models/audio/piper.onnx",
    sttModelPath: "/models/stt/whisper-small",
    llmTemperature: 0.8,
    llmMaxTokens: 4096,
    modelStrategy: "balanced",
  };

  const [state, setState] = useState<DashboardState>({
    status: "idle", statusDetail: "Disconnected",
    tokenWindow: Array.from({ length: 30 }, (_, i) => ({ t: i, inTok: 0, outTok: 0 })),
    behavior: { onTopic: 0, brevity: 0, responsiveness: 0, nonsenseRate: 0, rewardTotal: 0 },
    memory: { nodes: [{ id: "self", label: "Nomous", strength: 1, kind: "self" }], edges: [] },
    audioEnabled: true, visionEnabled: true, connected: false,
    url: typeof window !== "undefined" ? (localStorage.getItem("nomous.ws") || "ws://localhost:8765") : "ws://localhost:8765",
    micOn: false, vu: 0, consoleLines: [], thoughtLines: [], speechLines: [], systemLines: [],
    settings: defaultSettings,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicChain | null>(null);
  const hbRef = useRef<number | null>(null);
  const tCounter = useRef(0);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = localStorage.getItem("nomous.settings");
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      setState(p => ({ ...p, settings: { ...defaultSettings, ...parsed } }));
    } catch {
      // ignore invalid settings payloads
    }
  }, []);

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
    
    const stamped = `[${new Date().toLocaleTimeString()}] ${line}`;
    setState(p => {
      // Check if last message is identical (prevent duplicates)
      if (p.consoleLines[0] === stamped) {
        return p;
      }
      return { ...p, consoleLines: [stamped, ...p.consoleLines.slice(0, 150)] };
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
        case "status": {
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          setState(p => ({
            ...p,
            status: msg.value,
            statusDetail: msg.detail ?? p.statusDetail,
            systemLines: msg.detail
              ? [`${stamp} ${String(msg.value ?? "status").toUpperCase()} → ${msg.detail}`, ...p.systemLines.slice(0, 200)]
              : p.systemLines,
          }));
          break;
        }
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
        case "speak": {
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          log(`speak â†’ ${msg.text}`);
          setState(p => ({
            ...p,
            status: "speaking",
            statusDetail: msg.text,
            speechLines: msg.text ? [`${stamp} ${msg.text}`, ...p.speechLines.slice(0, 200)] : p.speechLines,
          }));
          break;
        }
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
        case "event": {
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          const payload = msg.message;
          log(payload || "event");
          setState(p => ({
            ...p,
            lastEvent: payload,
            systemLines: payload ? [`${stamp} EVENT → ${payload}`, ...p.systemLines.slice(0, 200)] : p.systemLines,
          }));
          break;
        }
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

  const stopMic = useCallback(() => {
    const chain = micRef.current;
    if (!chain) {
      return;
    }
    micRef.current = null;
    if (chain.raf) {
      cancelAnimationFrame(chain.raf);
    }
    try {
      chain.processor.disconnect();
      chain.analyser.disconnect();
      chain.gain.disconnect();
      chain.source.disconnect();
      chain.processor.onaudioprocess = null;
    } catch {
      // ignore cleanup errors
    }
    try {
      chain.stream.getTracks().forEach(track => track.stop());
    } catch {
      // ignore
    }
    chain.ctx.close().catch(() => {});
    setState(p => ({ ...p, micOn: false, vu: 0 }));
  }, [setState]);

  const setMic = useCallback(async (on: boolean) => {
    if (!on) {
      stopMic();
      return;
    }

    if (typeof window === "undefined" || typeof navigator === "undefined") {
      log("mic error: unavailable in this environment");
      return;
    }

    if (micRef.current) {
      return;
    }

    const AudioContextClass = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext | undefined;
    if (!AudioContextClass) {
      log("mic error: AudioContext unsupported");
      return;
    }

    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: TARGET_SAMPLE_RATE, echoCancellation: true, noiseSuppression: true },
        video: false,
      });

      ctx = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE }) as AudioContext;
      await ctx.resume();
      const inputSampleRate = ctx.sampleRate;

      if (typeof ctx.createScriptProcessor !== "function") {
        throw new Error("ScriptProcessorNode not supported in this browser");
      }

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const processor = ctx.createScriptProcessor(4096, 1, 1);
      source.connect(processor);

      const gain = ctx.createGain();
      gain.gain.value = 0;
      processor.connect(gain);
      gain.connect(ctx.destination);

      const chain: MicChain = { ctx, stream, source, analyser, processor, gain, residual: new Float32Array(0) };
      micRef.current = chain;

      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        if (micRef.current !== chain) {
          return;
        }
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128;
          peak = Math.max(peak, Math.abs(v));
        }
        setState(p => ({ ...p, vu: Math.min(1, peak * 2) }));
        chain.raf = requestAnimationFrame(tick);
      };
      tick();

      processor.onaudioprocess = (event) => {
        if (micRef.current !== chain) {
          return;
        }
        const channelData = event.inputBuffer.getChannelData(0);
        const encoded = encodeAudioChunk(chain, channelData, inputSampleRate);
        if (encoded) {
          push({ type: "audio", rate: TARGET_SAMPLE_RATE, pcm16: encoded });
        }
      };

      setState(p => ({ ...p, micOn: true }));
    } catch (e: any) {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (ctx) {
        ctx.close().catch(() => {});
      }
      micRef.current = null;
      setState(p => ({ ...p, micOn: false, settings: { ...p.settings, microphoneEnabled: false }, vu: 0 }));
      log(`mic error: ${e?.message ?? e}`);
    }
  }, [log, push, setState, stopMic]);

  const disconnect = useCallback(() => {
    stopMic();
    wsRef.current?.close(); wsRef.current = null;
  }, [stopMic]);

  const updateSettings = useCallback((patch: Partial<ControlSettings>) => {
    setState(prev => {
      const nextSettings = { ...prev.settings, ...patch };
      if (typeof window !== "undefined") {
        localStorage.setItem("nomous.settings", JSON.stringify(nextSettings));
      }
      return { ...prev, settings: nextSettings };
    });
  }, []);

  return { state, setState, connect, disconnect, setMic, push, log, updateSettings };
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

interface MemoryGraphProps {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
  selectedNodeId?: string | null;
  onSelect?: (id: string) => void;
}

function MemoryGraph({ nodes, edges, selectedNodeId, onSelect }: MemoryGraphProps) {
  const layout = React.useMemo(() => {
    const width = 560;
    const height = 320;
    const buckets: Record<MemoryNode["kind"], MemoryNode[]> = { stimulus: [], concept: [], event: [], self: [] };
    nodes.forEach(node => buckets[node.kind].push(node));
    const xSlots: Record<MemoryNode["kind"], number> = { stimulus: 80, event: width / 2, concept: width - 80, self: width / 2 };
    const yStep = (arr: MemoryNode[]) => (arr.length > 1 ? height / (arr.length + 1) : height / 2);
    const pos = new Map<string, { x: number; y: number }>();
    (Object.keys(buckets) as MemoryNode["kind"][]).forEach(kind => {
      const arr = buckets[kind];
      arr.forEach((node, index) => pos.set(node.id, { x: xSlots[kind], y: (index + 1) * yStep(arr) }));
    });
    return { width, height, pos };
  }, [nodes]);

  const connectedNodes = React.useMemo(() => {
    if (!selectedNodeId) {
      return new Set<string>();
    }
    const set = new Set<string>([selectedNodeId]);
    edges.forEach(edge => {
      if (edge.from === selectedNodeId || edge.to === selectedNodeId) {
        set.add(edge.from);
        set.add(edge.to);
      }
    });
    return set;
  }, [selectedNodeId, edges]);

  const activeEdges = React.useMemo(() => {
    if (!selectedNodeId) {
      return new Set(edges.map(edge => edge.id));
    }
    return new Set(
      edges
        .filter(edge => edge.from === selectedNodeId || edge.to === selectedNodeId)
        .map(edge => edge.id),
    );
  }, [selectedNodeId, edges]);

  const handleSelect = useCallback(
    (id: string) => {
      onSelect?.(id);
    },
    [onSelect],
  );

  const handleKey = useCallback(
    (event: React.KeyboardEvent<SVGGElement>, id: string) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onSelect?.(id);
      }
    },
    [onSelect],
  );

  return (
    <svg
      width="100%"
      height="320"
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      className="rounded-xl bg-zinc-900/60"
      role="list"
      aria-label="Nomous memory graph"
    >
      {edges.map(edge => {
        const from = layout.pos.get(edge.from);
        const to = layout.pos.get(edge.to);
        if (!from || !to) {
          return null;
        }
        const isActive = activeEdges.has(edge.id);
        const strokeOpacity = isActive ? 0.45 + edge.weight * 0.45 : 0.15;
        const strokeWidth = isActive ? 1.4 + edge.weight * 2.2 : 0.8 + edge.weight;
        return (
          <line
            key={edge.id}
            x1={from.x}
            y1={from.y}
            x2={to.x}
            y2={to.y}
            stroke="currentColor"
            strokeOpacity={strokeOpacity}
            strokeWidth={strokeWidth}
            className={isActive ? "text-emerald-300" : "text-zinc-600"}
          />
        );
      })}
      {nodes.map(node => {
        const position = layout.pos.get(node.id);
        if (!position) {
          return null;
        }
        const fill =
          node.kind === "self"
            ? "fill-emerald-500"
            : node.kind === "stimulus"
            ? "fill-amber-500"
            : node.kind === "event"
            ? "fill-cyan-500"
            : "fill-purple-500";
        const isSelected = selectedNodeId === node.id;
        const isConnected = connectedNodes.has(node.id);
        const radius = 10 + node.strength * 12 + (isSelected ? 4 : 0);
        const stroke = isSelected ? "#34d399" : isConnected ? "#818cf8" : "#1f1f23";
        const strokeWidth = isSelected ? 3 : isConnected ? 2 : 1;
        return (
          <g
            key={node.id}
            tabIndex={0}
            role="button"
            aria-pressed={isSelected}
            onClick={() => handleSelect(node.id)}
            onKeyDown={event => handleKey(event, node.id)}
            className="cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/70"
          >
            <circle cx={position.x} cy={position.y} r={radius} className={fill} opacity={0.95} stroke={stroke} strokeWidth={strokeWidth} />
            <text x={position.x} y={position.y - (18 + node.strength * 6)} textAnchor="middle" className="fill-zinc-50 text-xs drop-shadow">
              {node.label}
            </text>
            <title>{node.label}</title>
          </g>
        );
      })}
    </svg>
  );
}

type StreamCardProps = {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  description?: string;
  lines: string[];
  emptyLabel: string;
  accentClassName: string;
  height?: string;
};

function StreamCard({ title, icon: Icon, description, lines, emptyLabel, accentClassName, height }: StreamCardProps) {
  const bodyHeight = height ?? "h-56";
  return (
    <Card className="bg-zinc-900/70 border-zinc-800/60">
      <CardContent className="p-4 text-zinc-200 space-y-3">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4" />
          <span className="font-semibold">{title}</span>
        </div>
        {description ? <p className="text-xs text-zinc-400 leading-relaxed">{description}</p> : null}
        <div
          className={`overflow-auto rounded-md bg-black/60 p-3 font-mono text-xs leading-relaxed text-zinc-100/95 whitespace-pre-wrap ${bodyHeight}`}
        >
          {lines.length > 0 ? (
            lines.map((l, i) => (
              <div key={i} className={`mb-1 last:mb-0 ${accentClassName}`}>
                {l}
              </div>
            ))
          ) : (
            <div className="text-zinc-500 text-center mt-8">{emptyLabel}</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function MemorySummaryStat({ label, value, helper }: { label: string; value: string; helper?: string }) {
  return (
    <div className="rounded-xl border border-zinc-800/60 bg-black/40 px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">{label}</div>
      <div className="text-lg font-semibold text-zinc-100">{value}</div>
      {helper ? <div className="text-xs text-zinc-400">{helper}</div> : null}
    </div>
  );
}

interface MemoryListProps {
  title: string;
  icon: React.ReactNode;
  entries: MemoryInsightEntry[];
  emptyLabel: string;
  selectedId?: string | null;
  onSelect: (id: string) => void;
}

function MemoryList({ title, icon, entries, emptyLabel, selectedId, onSelect }: MemoryListProps) {
  return (
    <div className="rounded-xl border border-zinc-800/60 bg-black/40 p-3">
      <div className="flex items-center gap-2 mb-3 text-zinc-200">
        {icon}
        <span className="font-semibold text-sm">{title}</span>
      </div>
      <div className="space-y-2">
        {entries.length === 0 ? (
          <div className="text-xs text-zinc-500">{emptyLabel}</div>
        ) : (
          entries.map(entry => {
            const formattedTimestamp = formatTimestamp(entry.timestamp);
            return (
              <button
                key={entry.id}
                type="button"
                onClick={() => onSelect(entry.id)}
                className={`w-full rounded-lg border px-3 py-2 text-left transition ${
                  entry.id === selectedId
                    ? "border-emerald-500/60 bg-emerald-500/10"
                    : "border-zinc-800/60 bg-zinc-950/40 hover:border-zinc-700/60"
                }`}
              >
                <div className="flex items-center justify-between gap-2 text-sm text-zinc-100">
                  <span className="truncate font-medium">{entry.label}</span>
                  <Badge className="bg-zinc-800/70 text-zinc-200 border border-zinc-700/60">{entry.connections} links</Badge>
                </div>
                <div className="mt-1 flex items-center justify-between text-[11px] text-zinc-400">
                  <span className="capitalize">{entry.kind}</span>
                  <span>Strength {entry.strength.toFixed(2)}</span>
                </div>
                {entry.source ? (
                  <div className="mt-1 text-[11px] text-emerald-300">Source: {entry.source}</div>
                ) : null}
                {formattedTimestamp ? (
                  <div className="text-[11px] text-zinc-500">Updated {formattedTimestamp}</div>
                ) : null}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

function formatTimestamp(value?: string) {
  if (!value) return undefined;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return undefined;
  return date.toLocaleString();
}

function MemoryDetailCard({ detail }: { detail: MemoryNodeDetail | null }) {
  if (!detail) {
    return (
      <Card className="h-full bg-zinc-900/70 border-dashed border-zinc-800/60">
        <CardContent className="flex h-full items-center justify-center p-4 text-center text-sm text-zinc-400">
          Select a memory node from the graph or insight lists to inspect its learned behaviour, context, and related connections.
        </CardContent>
      </Card>
    );
  }

  const formattedTimestamp = formatTimestamp(detail.timestamp);

  return (
    <Card className="h-full bg-zinc-900/70 border-zinc-800/60">
      <CardContent className="h-full space-y-4 p-4 text-sm text-zinc-200">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Selected Memory</div>
            <h3 className="mt-1 text-xl font-semibold text-zinc-100">{detail.label}</h3>
          </div>
          <Badge className="bg-zinc-800/80 text-zinc-100 border border-zinc-700/60 capitalize">{detail.kind}</Badge>
        </div>

        {detail.description ? (
          <p className="leading-relaxed text-zinc-300">{detail.description}</p>
        ) : (
          <p className="italic text-zinc-500">No narrative description provided by the runtime.</p>
        )}

        <div className="grid grid-cols-2 gap-3 text-xs">
          <MemorySummaryStat label="Strength" value={detail.strength.toFixed(2)} helper="Association weight" />
          <MemorySummaryStat label="Connections" value={String(detail.connections)} helper="Linked nodes" />
          {detail.confidence !== undefined ? (
            <MemorySummaryStat label="Confidence" value={`${Math.round(detail.confidence * 100)}%`} helper="Runtime certainty" />
          ) : null}
          {formattedTimestamp ? (
            <MemorySummaryStat label="Updated" value={formattedTimestamp} helper="Last reinforcement" />
          ) : null}
          {detail.source ? (
            <MemorySummaryStat label="Source" value={detail.source} helper="Input channel" />
          ) : null}
        </div>

        <div className="space-y-2">
          <div className="text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500">Tags</div>
          {detail.tags.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {detail.tags.map(tag => (
                <Badge key={tag} className="bg-purple-500/10 text-purple-200 border border-purple-500/30">{tag}</Badge>
              ))}
            </div>
          ) : (
            <div className="text-xs text-zinc-500">No tags provided.</div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500">
            <Clock className="h-3.5 w-3.5" />
            <span>Related Associations</span>
          </div>
          {detail.related.length > 0 ? (
            <ul className="space-y-2">
              {detail.related.map(relation => (
                <li
                  key={`${detail.id}-${relation.id}-${relation.direction}`}
                  className="rounded-lg border border-zinc-800/60 bg-zinc-950/40 px-3 py-2 text-xs text-zinc-300"
                >
                  <div className="flex items-center justify-between gap-2 text-sm text-zinc-100">
                    <span className="font-medium">{relation.label}</span>
                    <span className="text-[11px] text-zinc-400">{relation.direction === "outbound" ? "influences" : "influenced by"}</span>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-zinc-400">
                    <span>Weight {relation.weight.toFixed(2)}</span>
                    <span>Strength {relation.strength.toFixed(2)}</span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-zinc-500">No related associations recorded yet.</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface ControlCenterProps {
  open: boolean;
  onClose: () => void;
  state: DashboardState;
  connect: () => void;
  disconnect: () => void;
  setMic: (on: boolean) => void;
  push: (data: any) => void;
  updateSettings: (patch: Partial<ControlSettings>) => void;
  setState: React.Dispatch<React.SetStateAction<DashboardState>>;
  beginModelSwitch: (label: string) => void;
}

function ControlCenter({ open, onClose, state, connect, disconnect, setMic, push, updateSettings, setState, beginModelSwitch }: ControlCenterProps) {
  const voices = useMemo(() => [
    "piper/en_US-amy-medium",
    "piper/en_US-kathleen-low",
    "piper/en_GB-sarah-medium",
    "piper/ja_JP-kokoro-high",
    "piper/es_ES-mls_10246-low"
  ], []);

  const { llmModelPath, visionModelPath, audioModelPath, sttModelPath, modelStrategy } = state.settings;

  const performancePresets = useMemo<Array<{
    id: PresetStrategy;
    label: string;
    description: string;
    conditions: string[];
    models: { llm: string; vision: string; audio: string; stt: string };
  }>>(
    () => [
      {
        id: "speed",
        label: "Low-Latency",
        description: "Favor responsiveness for live, reactive loops.",
        conditions: ["Latency < 150ms", "Short prompts"],
        models: {
          llm: "/models/llm/main-q4_0.gguf",
          vision: "/models/vision/runtime-lite.bin",
          audio: "/models/audio/piper-fast.onnx",
          stt: "/models/stt/whisper-small",
        },
      },
      {
        id: "balanced",
        label: "Balanced",
        description: "Blend quality and speed for everyday operations.",
        conditions: ["Adaptive context", "Standard workloads"],
        models: {
          llm: "/models/llm/main.gguf",
          vision: "/models/vision/runtime.bin",
          audio: "/models/audio/piper.onnx",
          stt: "/models/stt/whisper-small",
        },
      },
      {
        id: "accuracy",
        label: "High Accuracy",
        description: "Maximize reasoning depth and recognition fidelity.",
        conditions: ["Long-form tasks", "Detailed perception"],
        models: {
          llm: "/models/llm/main-q6_k.gguf",
          vision: "/models/vision/runtime-highres.bin",
          audio: "/models/audio/piper-high.onnx",
          stt: "/models/stt/whisper-large-v3",
        },
      },
    ],
    []
  );

  const applyPreset = useCallback(
    (presetId: PresetStrategy) => {
      const preset = performancePresets.find(item => item.id === presetId);
      if (!preset) return;

      const changes: { key: string; value: string }[] = [];
      if (llmModelPath !== preset.models.llm) {
        changes.push({ key: "llm_model_path", value: preset.models.llm });
      }
      if (visionModelPath !== preset.models.vision) {
        changes.push({ key: "vision_model_path", value: preset.models.vision });
      }
      if (audioModelPath !== preset.models.audio) {
        changes.push({ key: "audio_model_path", value: preset.models.audio });
      }
      if (sttModelPath !== preset.models.stt) {
        changes.push({ key: "stt_model_path", value: preset.models.stt });
      }

      const strategyChanged = modelStrategy !== preset.id;
      if (!strategyChanged && changes.length === 0) {
        return;
      }

      updateSettings({
        modelStrategy: preset.id,
        llmModelPath: preset.models.llm,
        visionModelPath: preset.models.vision,
        audioModelPath: preset.models.audio,
        sttModelPath: preset.models.stt,
      });

      if (changes.length > 0) {
        beginModelSwitch(`Applying ${preset.label} preset`);
        changes.forEach(change => {
          push({ type: "param", key: change.key, value: change.value });
        });
      }
    },
    [audioModelPath, beginModelSwitch, llmModelPath, modelStrategy, performancePresets, push, sttModelPath, updateSettings, visionModelPath]
  );

  const createModelCommitHandler = useCallback(
    (key: ModelPathKey, paramKey: string, label: string) => (next: string) => {
      const trimmed = next.trim();
      const currentValue =
        key === "llmModelPath"
          ? llmModelPath
          : key === "visionModelPath"
            ? visionModelPath
            : key === "audioModelPath"
              ? audioModelPath
              : sttModelPath;

      const pathChanged = trimmed !== currentValue;
      const patch: Partial<ControlSettings> = { [key]: trimmed } as Partial<ControlSettings>;
      if (pathChanged && modelStrategy !== "custom") {
        patch.modelStrategy = "custom";
      }
      updateSettings(patch);

      if (pathChanged && trimmed) {
        beginModelSwitch(label);
        push({ type: "param", key: paramKey, value: trimmed });
      }
    },
    [audioModelPath, beginModelSwitch, llmModelPath, modelStrategy, push, sttModelPath, updateSettings, visionModelPath]
  );

  if (!open) return null;

  const handleToggle = (key: keyof ControlSettings, value: boolean, action?: () => void) => {
    updateSettings({ [key]: value } as Partial<ControlSettings>);
    action?.();
  };

  const sliderKey = (label: string, value: number) => `${label}-${value}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4 py-6">
      <div className="relative w-full max-w-5xl max-h-[90vh] overflow-hidden rounded-3xl border border-zinc-800/70 bg-zinc-950/95 shadow-[0_30px_120px_rgba(0,0,0,0.45)]">
        <div className="flex items-start justify-between border-b border-zinc-800/60 px-6 py-4">
          <div>
            <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Control Center</div>
            <h2 className="text-2xl font-semibold text-zinc-100">Runtime, Devices &amp; LLM Settings</h2>
            <p className="text-sm text-zinc-400">Configure every input, output, and model path for Nomous from a single glassmorphic panel.</p>
          </div>
          <Button variant="secondary" className="bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-100" onClick={onClose}>Close</Button>
        </div>
        <div className="overflow-y-auto px-6 py-6 space-y-6">
          <section>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-lg font-semibold text-zinc-100">Bridge Connection</h3>
                <p className="text-xs text-zinc-400">Manage the WebSocket runtime bridge and endpoint.</p>
              </div>
              <div className="flex items-center gap-2">
                {!state.connected ? (
                  <Button className="bg-emerald-600/90 hover:bg-emerald-500/90 text-white" onClick={connect}><Play className="w-4 h-4 mr-1"/>Connect</Button>
                ) : (
                  <Button variant="danger" className="bg-red-600/90 hover:bg-red-500/90 text-white" onClick={disconnect}><Square className="w-4 h-4 mr-1"/>Disconnect</Button>
                )}
              </div>
            </div>
            <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
              <label className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-wide text-zinc-400">Runtime URL</span>
                <input value={state.url} onChange={(e)=>setState(p=>({ ...p, url: e.target.value }))} className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm focus:border-emerald-500/80 focus:outline-none focus:ring-0 text-zinc-100"/>
              </label>
              <Card className="bg-zinc-900/60 border-zinc-800/60">
                <CardContent className="p-4 text-xs text-zinc-400 space-y-1">
                  <div className="font-semibold text-zinc-200 text-sm">Status</div>
                  <div className="flex items-center gap-2 text-zinc-300"><span className={`w-2.5 h-2.5 rounded-full ${state.connected ? "bg-emerald-500" : "bg-red-500"}`}></span>{state.connected ? "Connected" : "Disconnected"}</div>
                  <div className="text-zinc-500">The UI will remember this endpoint and auto-reconnect.</div>
                </CardContent>
              </Card>
            </div>
          </section>

          <Separator className="bg-zinc-800/60" />

          <section className="grid gap-4 lg:grid-cols-2">
            <Card className="bg-zinc-900/60 border-zinc-800/60">
              <CardContent className="space-y-4 p-4">
                <div>
                  <h3 className="text-lg font-semibold text-zinc-100">Device Routing</h3>
                  <p className="text-xs text-zinc-400">Toggle hardware inputs &amp; outputs used by the autonomy stack.</p>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium text-zinc-200 flex items-center gap-2"><Camera className="w-4 h-4"/>Camera</div>
                      <p className="text-xs text-zinc-400">Enable or disable vision streaming to the runtime.</p>
                    </div>
                    <Switch checked={state.settings.cameraEnabled} onCheckedChange={(value)=>{
                      handleToggle("cameraEnabled", value, () => {
                        setState(p => ({ ...p, visionEnabled: value }));
                        push({ type: "toggle", what: "vision", value });
                      });
                    }}/>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium text-zinc-200 flex items-center gap-2"><Mic className="w-4 h-4"/>Microphone Capture</div>
                      <p className="text-xs text-zinc-400">Stream live audio into STT and conversational buffers.</p>
                    </div>
                    <Switch checked={state.settings.microphoneEnabled || state.micOn} onCheckedChange={(value)=>{
                      handleToggle("microphoneEnabled", value, () => setMic(value));
                    }}/>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium text-zinc-200 flex items-center gap-2"><Volume2 className="w-4 h-4"/>Speaker Output</div>
                      <p className="text-xs text-zinc-400">Route TTS audio to speakers and remote peers.</p>
                    </div>
                    <Switch checked={state.settings.speakerEnabled} onCheckedChange={(value)=>{
                      handleToggle("speakerEnabled", value, () => push({ type: "toggle", what: "speaker", value }));
                    }}/>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium text-zinc-200 flex items-center gap-2"><Radio className="w-4 h-4"/>Text-to-Speech</div>
                      <p className="text-xs text-zinc-400">Controls synthetic voice playback in the runtime.</p>
                    </div>
                    <Switch checked={state.settings.ttsEnabled && state.audioEnabled} onCheckedChange={(value)=>{
                      handleToggle("ttsEnabled", value, () => {
                        setState(p => ({ ...p, audioEnabled: value }));
                        push({ type: "toggle", what: "tts", value });
                      });
                    }}/>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium text-zinc-200 flex items-center gap-2"><Brain className="w-4 h-4"/>Speech-to-Text</div>
                      <p className="text-xs text-zinc-400">Enable transcription for microphone and streamed audio.</p>
                    </div>
                    <Switch checked={state.settings.sttEnabled} onCheckedChange={(value)=>{
                      handleToggle("sttEnabled", value, () => push({ type: "toggle", what: "stt", value }));
                    }}/>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/60 border-zinc-800/60">
              <CardContent className="space-y-4 p-4">
                <div>
                  <h3 className="text-lg font-semibold text-zinc-100">Audio Pipeline</h3>
                  <p className="text-xs text-zinc-400">Tune voice characteristics, sensitivity, and levels.</p>
                </div>
                <div className="space-y-4">
                  <label className="flex flex-col gap-2">
                    <span className="text-xs uppercase tracking-wide text-zinc-400">Piper Voice</span>
                    <select value={state.settings.ttsVoice} onChange={(e)=>{
                      updateSettings({ ttsVoice: e.target.value });
                      push({ type: "param", key: "tts_voice", value: e.target.value });
                    }} className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/80 focus:outline-none">
                      {voices.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </label>
                  <div>
                    <div className="flex items-center justify-between text-xs text-zinc-400"><span>Mic Sensitivity</span><span>{state.settings.micSensitivity}%</span></div>
                    <Slider key={sliderKey("mic", state.settings.micSensitivity)} defaultValue={[state.settings.micSensitivity]} min={0} max={100} step={5} onValueChange={(v)=>{
                      updateSettings({ micSensitivity: v[0] });
                      push({ type: "param", key: "mic_sensitivity", value: v[0] });
                    }}/>
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-xs text-zinc-400"><span>Output Volume</span><span>{state.settings.masterVolume}%</span></div>
                    <Slider key={sliderKey("vol", state.settings.masterVolume)} defaultValue={[state.settings.masterVolume]} min={0} max={100} step={5} onValueChange={(v)=>{
                      updateSettings({ masterVolume: v[0] });
                      push({ type: "param", key: "master_volume", value: v[0] });
                    }}/>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          <Separator className="bg-zinc-800/60" />

          <section className="grid gap-4 lg:grid-cols-2">
            <Card className="bg-zinc-900/60 border-zinc-800/60">
              <CardContent className="space-y-4 p-4">
                <div>
                  <h3 className="text-lg font-semibold text-zinc-100">Camera Configuration</h3>
                  <p className="text-xs text-zinc-400">Frame rate, exposure, and clarity tuning for the perception stack.</p>
                </div>
                <div className="space-y-4">
                  <label className="flex flex-col gap-2">
                    <span className="text-xs uppercase tracking-wide text-zinc-400">Resolution</span>
                    <select value={state.settings.cameraResolution} onChange={(e)=>{
                      updateSettings({ cameraResolution: e.target.value });
                      push({ type: "param", key: "camera_resolution", value: e.target.value });
                    }} className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/80 focus:outline-none">
                      {['1920x1080','1280x720','1024x576','640x480'].map(res => <option key={res} value={res}>{res}</option>)}
                    </select>
                  </label>
                  <div>
                    <div className="flex items-center justify-between text-xs text-zinc-400"><span>Exposure</span><span>{state.settings.cameraExposure}%</span></div>
                    <Slider key={sliderKey("exposure", state.settings.cameraExposure)} defaultValue={[state.settings.cameraExposure]} min={0} max={100} step={5} onValueChange={(v)=>{
                      updateSettings({ cameraExposure: v[0] });
                      push({ type: "param", key: "camera_exposure", value: v[0] });
                    }}/>
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-xs text-zinc-400"><span>Brightness</span><span>{state.settings.cameraBrightness}%</span></div>
                    <Slider key={sliderKey("brightness", state.settings.cameraBrightness)} defaultValue={[state.settings.cameraBrightness]} min={0} max={100} step={5} onValueChange={(v)=>{
                      updateSettings({ cameraBrightness: v[0] });
                      push({ type: "param", key: "camera_brightness", value: v[0] });
                    }}/>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/60 border-zinc-800/60">
              <CardContent className="space-y-4 p-4">
                <div>
                  <h3 className="text-lg font-semibold text-zinc-100">LLM Runtime</h3>
                  <p className="text-xs text-zinc-400">All controls that shape cognition stay together for clarity.</p>
                </div>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <span className="text-xs uppercase tracking-wide text-zinc-400">Conditional Presets</span>
                    <div className="grid gap-2 md:grid-cols-3">
                      {performancePresets.map(preset => {
                        const active = state.settings.modelStrategy === preset.id;
                        const presetClasses = active
                          ? "border-emerald-500/60 bg-emerald-500/10"
                          : "border-zinc-800/70 bg-zinc-950/40 hover:border-emerald-500/40";
                        const buttonClasses = "rounded-xl border px-3 py-3 text-left transition " + presetClasses;
                        return (
                          <button
                            key={preset.id}
                            type="button"
                            onClick={() => applyPreset(preset.id)}
                            className={buttonClasses}
                          >
                            <div className="flex items-center justify-between text-sm font-semibold text-zinc-100">
                              {preset.label}
                              {active && (
                                <Badge className="bg-emerald-500/20 text-emerald-100 border border-emerald-400/30">Active</Badge>
                              )}
                            </div>
                            <p className="mt-1 text-xs text-zinc-400">{preset.description}</p>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {preset.conditions.map(condition => (
                                <Badge
                                  key={condition}
                                  className="bg-zinc-900/70 text-zinc-300 border border-zinc-800/70"
                                >
                                  {condition}
                                </Badge>
                              ))}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                    {state.settings.modelStrategy === "custom" && (
                      <p className="text-xs text-amber-300/80">
                        Using custom model paths. Presets will overwrite your manual configuration.
                      </p>
                    )}
                  </div>
                  <div className="space-y-3">
                    <label className="flex flex-col gap-2">
                      <span className="text-xs uppercase tracking-wide text-zinc-400">Conversation Model</span>
                      <FilePathInput
                        value={state.settings.llmModelPath}
                        onChange={val => updateSettings({ llmModelPath: val })}
                        onCommit={createModelCommitHandler("llmModelPath", "llm_model_path", "Switching conversation model")}
                        accept={[".gguf"]}
                      />
                    </label>
                    <label className="flex flex-col gap-2">
                      <span className="text-xs uppercase tracking-wide text-zinc-400">Vision Model</span>
                      <FilePathInput
                        value={state.settings.visionModelPath}
                        onChange={val => updateSettings({ visionModelPath: val })}
                        onCommit={createModelCommitHandler("visionModelPath", "vision_model_path", "Switching vision model")}
                        accept={[".bin", ".onnx"]}
                      />
                    </label>
                    <label className="flex flex-col gap-2">
                      <span className="text-xs uppercase tracking-wide text-zinc-400">Audio Model</span>
                      <FilePathInput
                        value={state.settings.audioModelPath}
                        onChange={val => updateSettings({ audioModelPath: val })}
                        onCommit={createModelCommitHandler("audioModelPath", "audio_model_path", "Switching audio model")}
                        accept={[".onnx", ".bin"]}
                      />
                    </label>
                    <label className="flex flex-col gap-2">
                      <span className="text-xs uppercase tracking-wide text-zinc-400">STT Model</span>
                      <FilePathInput
                        value={state.settings.sttModelPath}
                        onChange={val => updateSettings({ sttModelPath: val })}
                        onCommit={createModelCommitHandler("sttModelPath", "stt_model_path", "Switching speech-to-text model")}
                        allowDirectories
                      />
                    </label>
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs text-zinc-400">
                    <span>Temperature</span>
                    <span>{state.settings.llmTemperature.toFixed(2)}</span>
                  </div>
                  <Slider
                    key={sliderKey("temp", Math.round(state.settings.llmTemperature * 100))}
                    defaultValue={[Math.round(state.settings.llmTemperature * 100)]}
                    min={0}
                    max={120}
                    step={5}
                    onValueChange={(v)=>{
                      const val = Math.round((v[0] / 100) * 100) / 100;
                      updateSettings({ llmTemperature: val });
                      push({ type: "param", key: "llm_temperature", value: val });
                    }}
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs text-zinc-400">
                    <span>Max Tokens</span>
                    <span>{state.settings.llmMaxTokens}</span>
                  </div>
                  <Slider
                    key={sliderKey("maxtok", state.settings.llmMaxTokens)}
                    defaultValue={[state.settings.llmMaxTokens]}
                    min={512}
                    max={8192}
                    step={256}
                    onValueChange={(v)=>{
                      updateSettings({ llmMaxTokens: v[0] });
                      push({ type: "param", key: "llm_max_tokens", value: v[0] });
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </section>

          <Separator className="bg-zinc-800/60" />

          <section>
            <Card className="bg-zinc-900/60 border-zinc-800/60">
              <CardContent className="p-4">
                <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-zinc-100">Live Diagnostics</h3>
                    <p className="text-xs text-zinc-400">Adjustments stream immediately to the runtime and persist between sessions.</p>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs text-zinc-300">
                    <Badge className="bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">LLM</Badge>
                    <Badge className="bg-purple-500/10 text-purple-300 border border-purple-500/20">Audio</Badge>
                    <Badge className="bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">Vision</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}

export default function App(){
  const { state, setState, connect, disconnect, setMic, push, log, updateSettings } = useNomousBridge();
  const st = statusMap[state.status];
  const tokenTotal = state.tokenWindow.reduce((a, p) => a + p.inTok + p.outTok, 0);
  const [controlCenterOpen, setControlCenterOpen] = useState(false);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const memoryInsights = useMemo(() => computeMemoryInsights(state.memory.nodes, state.memory.edges), [state.memory.nodes, state.memory.edges]);
  const [modelSwitching, setModelSwitching] = useState<{ label: string; progress: number } | null>(null);
  const modelSwitchStartRef = useRef<number>(0);
  const hideModelSwitchRef = useRef<number | null>(null);

  const beginModelSwitch = useCallback((label: string) => {
    modelSwitchStartRef.current = Date.now();
    if (hideModelSwitchRef.current !== null) {
      window.clearTimeout(hideModelSwitchRef.current);
      hideModelSwitchRef.current = null;
    }
    setModelSwitching({ label, progress: 5 });
  }, []);

  const completeModelSwitch = useCallback(() => {
    setModelSwitching(prev => {
      if (!prev) return prev;
      if (prev.progress >= 100) {
        return prev;
      }
      return { ...prev, progress: 100 };
    });
    if (hideModelSwitchRef.current !== null) {
      window.clearTimeout(hideModelSwitchRef.current);
    }
    hideModelSwitchRef.current = window.setTimeout(() => {
      setModelSwitching(null);
      hideModelSwitchRef.current = null;
    }, 220);
  }, []);

  const isSwitching = modelSwitching !== null;

  useEffect(() => {
    if (!isSwitching) return;
    let frame = 0;
    const step = () => {
      const elapsed = Date.now() - modelSwitchStartRef.current;
      const progress = Math.min(95, Math.round((elapsed / 2000) * 100));
      setModelSwitching(prev => {
        if (!prev) return prev;
        if (progress <= prev.progress) {
          return prev;
        }
        return { ...prev, progress };
      });
      if (progress < 95) {
        frame = requestAnimationFrame(step);
      }
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [isSwitching]);

  useEffect(() => {
    if (!isSwitching) return;
    const timeout = window.setTimeout(() => {
      completeModelSwitch();
    }, 3200);
    return () => window.clearTimeout(timeout);
  }, [completeModelSwitch, isSwitching]);

  useEffect(() => {
    if (!isSwitching) return;
    const elapsed = Date.now() - modelSwitchStartRef.current;
    if (elapsed < 600) return;
    if (state.status !== "learning") {
      completeModelSwitch();
    }
  }, [completeModelSwitch, isSwitching, state.status]);

  useEffect(() => {
    return () => {
      if (hideModelSwitchRef.current !== null) {
        window.clearTimeout(hideModelSwitchRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (selectedMemoryId && !memoryInsights.nodeById.has(selectedMemoryId)) {
      setSelectedMemoryId(null);
    }
  }, [selectedMemoryId, memoryInsights]);

  useEffect(() => {
    if (selectedMemoryId || state.memory.nodes.length === 0) {
      return;
    }
    const defaultSelection = memoryInsights.milestones[0]?.id ?? state.memory.nodes[0]?.id ?? null;
    if (defaultSelection) {
      setSelectedMemoryId(defaultSelection);
    }
  }, [selectedMemoryId, memoryInsights, state.memory.nodes.length]);

  const selectedMemoryNode = selectedMemoryId ? memoryInsights.nodeById.get(selectedMemoryId) : undefined;
  const selectedMemoryDetail = useMemo<MemoryNodeDetail | null>(() => {
    if (!selectedMemoryNode) {
      return null;
    }
    return buildMemoryNodeDetail(selectedMemoryNode, state.memory.edges, memoryInsights.connectionIndex, memoryInsights.nodeById);
  }, [selectedMemoryNode, state.memory.edges, memoryInsights]);

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gradient-to-br from-black via-zinc-950 to-zinc-900 text-zinc-50 p-4 md:p-6 space-y-4">
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
            <Button variant="secondary" onClick={()=>setControlCenterOpen(true)} className="hidden sm:inline-flex bg-zinc-900/80 hover:bg-zinc-800/80 text-zinc-100 border border-zinc-700/60">
              <Cog className="w-4 h-4 mr-2"/> Control Center
            </Button>
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
                <TabsTrigger value="memory">Memory</TabsTrigger>
                <TabsTrigger value="thoughts">Thoughts</TabsTrigger>
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
                              <Button onClick={() => { setMic(true); updateSettings({ microphoneEnabled: true }); }} className="px-3 py-1 text-sm bg-emerald-600/90 hover:bg-emerald-500/90 text-white"><Mic className="w-4 h-4 mr-1"/> Mic: OFF</Button>
                            ) : (
                              <Button variant="secondary" onClick={() => { setMic(false); updateSettings({ microphoneEnabled: false }); }} className="px-3 py-1 text-sm bg-zinc-800/80 text-zinc-100 hover:bg-zinc-700/80"><MicOff className="w-4 h-4 mr-1"/> Mic: ON</Button>
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

              <TabsContent value="memory" className="pt-4">
                <div className="grid gap-4 xl:grid-cols-[1.7fr_1fr]">
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="space-y-4 p-4 text-zinc-200">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <Brain className="h-4 w-4" />
                          <span className="font-semibold">Memory Intelligence</span>
                        </div>
                        <Badge className="bg-zinc-800/80 text-zinc-100 border border-zinc-700/60">
                          Nodes {memoryInsights.summary.totalNodes}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                        <MemorySummaryStat label="Memories" value={`${memoryInsights.summary.totalNodes}`} helper="Graph nodes" />
                        <MemorySummaryStat label="Associations" value={`${memoryInsights.summary.totalEdges}`} helper="Synapses" />
                        <MemorySummaryStat label="Avg Strength" value={memoryInsights.summary.averageStrength.toFixed(2)} helper="Mean weight" />
                        <MemorySummaryStat label="Density" value={`${(memoryInsights.summary.density * 100).toFixed(1)}%`} helper="Connection coverage" />
                      </div>

                      <MemoryGraph
                        nodes={state.memory.nodes}
                        edges={state.memory.edges}
                        selectedNodeId={selectedMemoryId}
                        onSelect={id => setSelectedMemoryId(id)}
                      />

                      <div className="grid gap-3 md:grid-cols-2">
                        <MemoryList
                          title="Major Milestones"
                          icon={<Flag className="h-4 w-4 text-emerald-300" />}
                          entries={memoryInsights.milestones}
                          emptyLabel="No milestone memories reported yet."
                          selectedId={selectedMemoryId}
                          onSelect={id => setSelectedMemoryId(id)}
                        />
                        <MemoryList
                          title="Data Entry Points"
                          icon={<Database className="h-4 w-4 text-cyan-300" />}
                          entries={memoryInsights.dataEntries}
                          emptyLabel="No sensory inputs stored yet."
                          selectedId={selectedMemoryId}
                          onSelect={id => setSelectedMemoryId(id)}
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <MemoryDetailCard detail={selectedMemoryDetail} />
                </div>
              </TabsContent>

              <TabsContent value="thoughts" className="pt-4">
                <div className="grid gap-4 lg:grid-cols-3">
                  <div className="lg:col-span-2">
                    <StreamCard
                      title="Cognitive Stream"
                      icon={MessageSquare}
                      description="Live reasoning trace as the model explores options and intermediate thoughts."
                      lines={state.thoughtLines}
                      emptyLabel="Waiting for thoughts..."
                      accentClassName="text-purple-300"
                      height="h-72"
                    />
                  </div>
                  <div className="space-y-4 lg:col-span-1">
                    <StreamCard
                      title="Speech Commitments"
                      icon={Volume2}
                      description="Finalized responses that were sent to speech synthesis."
                      lines={state.speechLines}
                      emptyLabel="No speech prepared yet."
                      accentClassName="text-emerald-300"
                      height="h-36"
                    />
                    <StreamCard
                      title="System & Prompt Signals"
                      icon={Radio}
                      description="System prompts, status transitions, and other generated directives."
                      lines={state.systemLines}
                      emptyLabel="No system activity captured yet."
                      accentClassName="text-sky-300"
                      height="h-36"
                    />
                  </div>
                </div>
              </TabsContent>
                </Tabs>
              </div>
          </CardContent>
        </Card>

        <div className="text-[10px] text-zinc-400/80">Nomous Autonomy UI • WebSocket JSON from Python. Colors: purple=thinking, amber=noticed, emerald=speaking, cyan=learning, red=error. Mic sends 16kHz chunks.</div>
        <Button onClick={()=>setControlCenterOpen(true)} className="sm:hidden fixed bottom-6 right-6 rounded-full px-5 py-3 text-sm shadow-xl bg-emerald-600/90 hover:bg-emerald-500/90 text-white">
          <Cog className="w-4 h-4 mr-2"/> Controls
        </Button>
        <ControlCenter
          open={controlCenterOpen}
          onClose={()=>setControlCenterOpen(false)}
          state={state}
          connect={connect}
          disconnect={disconnect}
          setMic={setMic}
          push={push}
          updateSettings={updateSettings}
          setState={setState}
          beginModelSwitch={beginModelSwitch}
        />
        {modelSwitching && (
          <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 backdrop-blur-sm px-4">
            <div className="w-full max-w-md rounded-2xl border border-emerald-500/30 bg-zinc-950/90 p-6 shadow-[0_24px_100px_rgba(16,185,129,0.25)]">
              <div className="mb-4 flex items-center gap-3 text-emerald-200">
                <RefreshCw className="h-5 w-5 animate-spin" />
                <div>
                  <div className="text-sm font-semibold text-emerald-100">{modelSwitching.label}</div>
                  <div className="text-xs text-zinc-400">Loading models and refreshing the runtime…</div>
                </div>
              </div>
              <Progress value={Math.min(modelSwitching.progress, 100)} />
            </div>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}
