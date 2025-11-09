import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Slider } from "./components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { FilePathInput } from "./components/FilePathInput";
import { Switch } from "./components/ui/switch";
import { Progress } from "./components/ui/progress";
import { TooltipProvider } from "./components/ui/tooltip";
import { Separator } from "./components/ui/separator";
import { Activity, Brain, Camera, Cog, MessageSquare, Play, Radio, RefreshCw, Square, Mic, MicOff, Wifi, WifiOff, Volume2, Flag, Database, Clock, Sparkles, Gauge, Send, Wrench } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip as RTooltip } from "recharts";
import type { MemoryEdge, MemoryNode } from "./types/memory";
import type { SystemMetricsPayload } from "./types/system";
import type { BehaviorStats } from "./types/behavior";
import { EMPTY_BEHAVIOR_STATS } from "./types/behavior";
import { BehaviorInsights } from "./components/BehaviorInsights";
import { buildMemoryNodeDetail, computeMemoryInsights } from "./utils/memory";
import type { MemoryNodeDetail, MemoryInsightEntry } from "./utils/memory";
import { normaliseVoiceFilename, readJson, writeJson } from "./utils/storage";
import { ToolActivity, ToolStats } from "./components/ToolActivity";
import { SystemUsageCard } from "./components/SystemUsageCard";

/** Nomous – Autonomy Dashboard (fixed) */
export type NomousStatus = "idle" | "thinking" | "speaking" | "noticing" | "learning" | "error";
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
  progress?: number;
  label?: string;
  target?: string;
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

interface ToolResult {
  tool: string;
  result: any;
  timestamp: number;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  timestamp: number;
}
type ModelPathKey = "llmModelPath" | "visionModelPath" | "audioModelPath" | "sttModelPath";
type PresetStrategy = Exclude<ControlSettings["modelStrategy"], "custom">;

interface DashboardState {
  status: NomousStatus; statusDetail?: string; tokenWindow: TokenPoint[]; behavior: BehaviorStats;
  memory: { nodes: MemoryNode[]; edges: MemoryEdge[] }; lastEvent?: string; audioEnabled: boolean; visionEnabled: boolean;
  connected: boolean; url: string; micOn: boolean; vu: number; preview?: string; consoleLines: string[]; thoughtLines: string[];
  speechLines: string[]; systemLines: string[]; chatMessages: ChatMessage[]; toolActivity: ToolResult[]; systemMetrics: SystemMetricsPayload | null;
  settings: ControlSettings; loadingOverlay: LoadingOverlay | null;
}

interface LoadingOverlay {
  label: string;
  progress: number;
  detail?: string;
}

const TARGET_SAMPLE_RATE = 16000;
const MAX_CHAT_HISTORY = 200;
const STORAGE_SETTINGS_KEY = "nomous.settings";
const STORAGE_WS_KEY = "nomous.ws";

function clamp01(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function createEmptyBehaviorStats(): BehaviorStats {
  return {
    summary: { ...EMPTY_BEHAVIOR_STATS.summary },
    signals: { ...EMPTY_BEHAVIOR_STATS.signals },
    history: [],
    rewardTotal: 0,
    anomalies: [],
    lastSample: undefined,
  };
}

function mergeBehaviorStats(prev: BehaviorStats, payload: any): BehaviorStats {
  if (!payload || typeof payload !== "object") {
    return prev;
  }

  const next: BehaviorStats = {
    summary: { ...prev.summary },
    signals: { ...prev.signals },
    history: [...prev.history],
    rewardTotal: prev.rewardTotal,
    anomalies: [...prev.anomalies],
    lastSample: prev.lastSample,
  };

  const summarySource = payload.summary ?? payload;
  if (summarySource && typeof summarySource === "object") {
    if ("onTopic" in summarySource || "on_topic" in summarySource) {
      next.summary.onTopic = clamp01(summarySource.onTopic ?? summarySource.on_topic);
    }
    if ("responsiveness" in summarySource) {
      next.summary.responsiveness = clamp01(summarySource.responsiveness);
    }
    if ("brevity" in summarySource) {
      next.summary.brevity = clamp01(summarySource.brevity);
    }
    if ("coherence" in summarySource) {
      next.summary.coherence = clamp01(summarySource.coherence);
    }
    if ("sentiment" in summarySource) {
      next.summary.sentiment = clamp01(summarySource.sentiment);
    }
    if ("lexicalRichness" in summarySource || "lexical" in summarySource) {
      next.summary.lexicalRichness = clamp01(summarySource.lexicalRichness ?? summarySource.lexical);
    }
    if ("safety" in summarySource) {
      next.summary.safety = clamp01(summarySource.safety);
    }
    if ("stability" in summarySource) {
      next.summary.stability = clamp01(summarySource.stability);
    }
  }

  const signalsSource = payload.signals ?? payload;
  if (signalsSource && typeof signalsSource === "object") {
    if (typeof signalsSource.latencyMs === "number") {
      next.signals.latencyMs = signalsSource.latencyMs;
    }
    if (typeof signalsSource.tokensIn === "number") {
      next.signals.tokensIn = signalsSource.tokensIn;
    }
    if (typeof signalsSource.tokensOut === "number") {
      next.signals.tokensOut = signalsSource.tokensOut;
    }
    if (typeof signalsSource.conversationPace === "number") {
      next.signals.conversationPace = signalsSource.conversationPace;
    }
    if (typeof signalsSource.avgResponseLength === "number") {
      next.signals.avgResponseLength = signalsSource.avgResponseLength;
    }
  }

  if (Array.isArray(payload.history)) {
    next.history = payload.history
      .map((point: any) => ({
        timestamp: typeof point.timestamp === "number" ? point.timestamp : Date.now(),
        onTopic: clamp01(point.onTopic ?? point.on_topic),
        sentiment: clamp01(point.sentiment ?? 0.5),
        coherence: clamp01(point.coherence ?? 0),
        safety: clamp01(point.safety ?? 1),
      }))
      .slice(-120);
  }

  if (typeof payload.rewardTotal === "number" && Number.isFinite(payload.rewardTotal)) {
    next.rewardTotal = payload.rewardTotal;
  }

  if (Array.isArray(payload.anomalies)) {
    next.anomalies = payload.anomalies
      .map((item: any) => ({
        label: String(item.label ?? ""),
        severity: (item.severity ?? "info") as any,
        detail: String(item.detail ?? ""),
      }))
      .filter((item: any) => item.label.length > 0)
      .slice(-12);
  }

  if (payload.lastSample && typeof payload.lastSample === "object") {
    const last = payload.lastSample;
    next.lastSample = {
      userText: typeof last.userText === "string" ? last.userText : prev.lastSample?.userText,
      assistantText: typeof last.assistantText === "string" ? last.assistantText : prev.lastSample?.assistantText,
      latencyMs: typeof last.latencyMs === "number" ? last.latencyMs : prev.lastSample?.latencyMs ?? 0,
      tokensIn: typeof last.tokensIn === "number" ? last.tokensIn : prev.lastSample?.tokensIn ?? 0,
      tokensOut: typeof last.tokensOut === "number" ? last.tokensOut : prev.lastSample?.tokensOut ?? 0,
      scores: {
        onTopic: clamp01(last.scores?.onTopic),
        responsiveness: clamp01(last.scores?.responsiveness),
        brevity: clamp01(last.scores?.brevity),
        coherence: clamp01(last.scores?.coherence),
        sentiment: clamp01(last.scores?.sentiment ?? 0.5),
        lexicalRichness: clamp01(last.scores?.lexicalRichness),
        safety: clamp01(last.scores?.safety ?? 1),
        toxicity: clamp01(last.scores?.toxicity ?? 0),
      },
    };
  }

  return next;
}

function describeLatency(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) {
    return "<100 ms";
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(1)} s`;
  }
  return `${Math.round(ms)} ms`;
}

const DEFAULT_CONTROL_SETTINGS: ControlSettings = {
  cameraEnabled: true,
  microphoneEnabled: false,
  ttsEnabled: true,
  speakerEnabled: true,
  sttEnabled: true,
  ttsVoice: "en_US-libritts-high.onnx",
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

const MODEL_STRATEGIES: ControlSettings["modelStrategy"][] = [
  "speed",
  "balanced",
  "accuracy",
  "custom",
];

function coerceBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function coerceNumber(value: unknown, fallback: number, options?: { min?: number; max?: number }): number {
  let numeric: number | undefined;
  if (typeof value === "number" && Number.isFinite(value)) {
    numeric = value;
  } else if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      numeric = parsed;
    }
  }
  if (numeric === undefined) {
    numeric = fallback;
  }
  if (options?.min !== undefined) {
    numeric = Math.max(options.min, numeric);
  }
  if (options?.max !== undefined) {
    numeric = Math.min(options.max, numeric);
  }
  return numeric;
}

function coerceString(value: unknown, fallback: string, allowEmpty = true): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  if (!allowEmpty && trimmed.length === 0) {
    return fallback;
  }
  return trimmed;
}

function coerceStrategy(value: unknown, fallback: ControlSettings["modelStrategy"]): ControlSettings["modelStrategy"] {
  if (typeof value === "string" && MODEL_STRATEGIES.includes(value as ControlSettings["modelStrategy"])) {
    return value as ControlSettings["modelStrategy"];
  }
  return fallback;
}

function mergeSettings(base: ControlSettings, patch: Partial<ControlSettings>): ControlSettings {
  const next = { ...base };

  if ("cameraEnabled" in patch) next.cameraEnabled = coerceBoolean(patch.cameraEnabled, next.cameraEnabled);
  if ("microphoneEnabled" in patch) next.microphoneEnabled = coerceBoolean(patch.microphoneEnabled, next.microphoneEnabled);
  if ("ttsEnabled" in patch) next.ttsEnabled = coerceBoolean(patch.ttsEnabled, next.ttsEnabled);
  if ("speakerEnabled" in patch) next.speakerEnabled = coerceBoolean(patch.speakerEnabled, next.speakerEnabled);
  if ("sttEnabled" in patch) next.sttEnabled = coerceBoolean(patch.sttEnabled, next.sttEnabled);
  if ("ttsVoice" in patch && patch.ttsVoice !== undefined) {
    const voice = coerceString(patch.ttsVoice, next.ttsVoice, false);
    next.ttsVoice = normaliseVoiceFilename(voice);
  }
  if ("micSensitivity" in patch) {
    next.micSensitivity = coerceNumber(patch.micSensitivity, next.micSensitivity, { min: 0, max: 100 });
  }
  if ("masterVolume" in patch) {
    next.masterVolume = coerceNumber(patch.masterVolume, next.masterVolume, { min: 0, max: 100 });
  }
  if ("cameraResolution" in patch && patch.cameraResolution !== undefined) {
    next.cameraResolution = coerceString(patch.cameraResolution, next.cameraResolution);
  }
  if ("cameraExposure" in patch) {
    next.cameraExposure = coerceNumber(patch.cameraExposure, next.cameraExposure, { min: 0, max: 100 });
  }
  if ("cameraBrightness" in patch) {
    next.cameraBrightness = coerceNumber(patch.cameraBrightness, next.cameraBrightness, { min: 0, max: 100 });
  }
  if ("llmModelPath" in patch && patch.llmModelPath !== undefined) {
    next.llmModelPath = coerceString(patch.llmModelPath, next.llmModelPath);
  }
  if ("visionModelPath" in patch && patch.visionModelPath !== undefined) {
    next.visionModelPath = coerceString(patch.visionModelPath, next.visionModelPath);
  }
  if ("audioModelPath" in patch && patch.audioModelPath !== undefined) {
    next.audioModelPath = coerceString(patch.audioModelPath, next.audioModelPath);
  }
  if ("sttModelPath" in patch && patch.sttModelPath !== undefined) {
    next.sttModelPath = coerceString(patch.sttModelPath, next.sttModelPath);
  }
  if ("llmTemperature" in patch) {
    next.llmTemperature = coerceNumber(patch.llmTemperature, next.llmTemperature, { min: 0.0, max: 2.0 });
  }
  if ("llmMaxTokens" in patch) {
    next.llmMaxTokens = Math.round(coerceNumber(patch.llmMaxTokens, next.llmMaxTokens, { min: 64 }));
  }
  if ("modelStrategy" in patch) {
    next.modelStrategy = coerceStrategy(patch.modelStrategy, next.modelStrategy);
  }

  return next;
}

function resolveStoredSettings(): ControlSettings {
  const stored = readJson<Partial<ControlSettings>>(STORAGE_SETTINGS_KEY);
  if (!stored || typeof stored !== "object") {
    return { ...DEFAULT_CONTROL_SETTINGS };
  }
  return mergeSettings({ ...DEFAULT_CONTROL_SETTINGS }, stored);
}

function resolveInitialUrl(): string {
  if (typeof window === "undefined") {
    return "ws://localhost:8765";
  }
  const stored = window.localStorage.getItem(STORAGE_WS_KEY);
  return stored && stored.trim().length > 0 ? stored : "ws://localhost:8765";
}

function createInitialState(): DashboardState {
  const settings = resolveStoredSettings();
  return {
    status: "idle",
    statusDetail: "Disconnected",
    tokenWindow: Array.from({ length: 30 }, (_, i) => ({ t: i, inTok: 0, outTok: 0 })),
    behavior: createEmptyBehaviorStats(),
    memory: { nodes: [{ id: "self", label: "Nomous", strength: 1, kind: "self" }], edges: [] },
    audioEnabled: settings.ttsEnabled,
    visionEnabled: settings.cameraEnabled,
    connected: false,
    url: resolveInitialUrl(),
    micOn: settings.microphoneEnabled,
    vu: 0,
    preview: undefined,
    consoleLines: [],
    thoughtLines: [],
    speechLines: [],
    systemLines: [],
    chatMessages: [],
    toolActivity: [],
    systemMetrics: null,
    settings,
    loadingOverlay: null,
  };
}

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
  const [state, setState] = useState<DashboardState>(() => createInitialState());
  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicChain | null>(null);
  const hbRef = useRef<number | null>(null);
  const tCounter = useRef(0);
  const [chatInput, setChatInput] = useState("");
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = chatScrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [state.chatMessages]);

  useEffect(() => {
    writeJson(STORAGE_SETTINGS_KEY, state.settings);
  }, [state.settings]);

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

  const sendChatMessage = useCallback((input: string) => {
    const trimmed = input.trim();
    if (!trimmed) {
      return;
    }

    const ws = wsRef.current;
    if (!ws || ws.readyState !== 1) {
      setState(p => ({
        ...p,
        chatMessages: [...p.chatMessages, createChatMessage("system", "Unable to send message: not connected to runtime.")].slice(-MAX_CHAT_HISTORY),
      }));
      log("chat send failed: not connected");
      return;
    }

    const message = createChatMessage("user", trimmed);
    setState(p => ({
      ...p,
      chatMessages: [...p.chatMessages, message].slice(-MAX_CHAT_HISTORY),
    }));
    push({ type: "text", value: trimmed });
    setChatInput("");
  }, [log, push]);

  const handleChatSubmit = useCallback((event?: React.FormEvent<HTMLFormElement>) => {
    if (event) {
      event.preventDefault();
    }
    sendChatMessage(chatInput);
  }, [chatInput, sendChatMessage]);

  const handleChatKeyDown = useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChatMessage(chatInput);
    }
  }, [chatInput, sendChatMessage]);

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
              ? [`${stamp} ${String(msg.value ?? "status").toUpperCase()} → ${msg.detail}`, ...p.systemLines.slice(0, MAX_CHAT_HISTORY)]
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
            speechLines: msg.text ? mergeSpeechLines(p.speechLines, stamp, msg.text) : p.speechLines,
            chatMessages: msg.text ? mergeAssistantChatMessages(p.chatMessages, msg.text) : p.chatMessages,
          }));
          break;
        }
        case "thought": 
          setState(p => ({ ...p, thoughtLines: [`[${new Date().toLocaleTimeString()}] ${msg.text}`, ...p.thoughtLines.slice(0, MAX_CHAT_HISTORY)] })); 
          break;
        case "image": setState(p => ({ ...p, preview: msg.dataUrl })); break;
        case "metrics":
          setState(p => ({
            ...p,
            behavior: mergeBehaviorStats(p.behavior, msg.payload ?? {}),
          }));
          break;
        case "system_metrics": {
          if (msg.payload) {
            const payload = msg.payload as SystemMetricsPayload;
            setState(p => ({ ...p, systemMetrics: payload }));
          }
          break;
        }
        case "memory": setState(p => ({ ...p, memory: { nodes: msg.nodes ?? p.memory.nodes, edges: msg.edges ?? p.memory.edges } })); break;
        case "tool_result": {
          const toolResult: ToolResult = {
            tool: msg.tool || 'unknown',
            result: msg.result || {},
            timestamp: Date.now()
          };
          setState(p => ({
            ...p,
            toolActivity: [...p.toolActivity, toolResult].slice(-100), // Keep last 100
            systemLines: [`[${new Date().toLocaleTimeString()}] 🛠️ Tool: ${msg.tool}`, ...p.systemLines.slice(0, MAX_CHAT_HISTORY)]
          }));
          break;
        }
        case "event": {
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          const payload = msg.message;
          log(payload || "event");
          setState(p => ({
            ...p,
            lastEvent: payload,
            systemLines: payload ? [`${stamp} EVENT → ${payload}`, ...p.systemLines.slice(0, MAX_CHAT_HISTORY)] : p.systemLines,
          }));
          break;
        }
        case "load_progress": {
          const rawValue = Number.isFinite(msg.progress as number) ? Number(msg.progress) : 0;
          const normalized = Math.max(0, Math.min(100, Math.round(rawValue)));
          const overlay = normalized >= 100
            ? null
            : {
                label: msg.label ?? "Loading models…",
                progress: normalized,
                detail: msg.detail,
              };
          setState(p => ({ ...p, loadingOverlay: overlay }));
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
        if (typeof window !== "undefined") {
          window.localStorage.setItem(STORAGE_WS_KEY, state.url);
        }
        setState(p => ({ ...p, connected: true, statusDetail: "Connected" }));
        log(`connected â†’ ${state.url}`);
        hbRef.current = window.setInterval(() => push({ type: "ping" }), 10000);
      };
      ws.onmessage = handleMessage;
      ws.onclose = () => {
        if (hbRef.current) window.clearInterval(hbRef.current);
        hbRef.current = null;
        setState(p => ({ ...p, connected: false, status: "idle", statusDetail: "Disconnected", loadingOverlay: null }));
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
      setState(p => ({ ...p, micOn: false, settings: mergeSettings(p.settings, { microphoneEnabled: false }), vu: 0 }));
      log(`mic error: ${e?.message ?? e}`);
    }
  }, [log, push, setState, stopMic]);

  const disconnect = useCallback(() => {
    stopMic();
    wsRef.current?.close(); wsRef.current = null;
  }, [stopMic]);

  const updateSettings = useCallback((patch: Partial<ControlSettings>) => {
    setState(prev => {
      const nextSettings = mergeSettings(prev.settings, patch);
      return {
        ...prev,
        settings: nextSettings,
        audioEnabled: "ttsEnabled" in patch ? nextSettings.ttsEnabled : prev.audioEnabled,
        visionEnabled: "cameraEnabled" in patch ? nextSettings.cameraEnabled : prev.visionEnabled,
        micOn: "microphoneEnabled" in patch ? nextSettings.microphoneEnabled : prev.micOn,
      };
    });
  }, []);

  return { 
    state, 
    setState, 
    connect, 
    disconnect, 
    setMic, 
    push, 
    log, 
    updateSettings,
    chatInput,
    setChatInput,
    chatScrollRef,
    handleChatSubmit,
    handleChatKeyDown,
  };
}

const statusMap: Record<NomousStatus, { color: string; label: string }> = {
  idle: { color: "bg-zinc-400", label: "Idle" },
  thinking: { color: "bg-purple-500", label: "Thinking" },
  speaking: { color: "bg-emerald-500", label: "Speaking" },
  noticing: { color: "bg-amber-500", label: "Noticed" },
  learning: { color: "bg-cyan-500", label: "Learning" },
  error: { color: "bg-red-600", label: "Error" },
};

function QuickStatCard({
  icon: Icon,
  label,
  value,
  helper,
  status,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  helper?: string;
  status?: "positive" | "negative" | "neutral";
}) {
  const accent =
    status === "positive"
      ? "text-emerald-300"
      : status === "negative"
        ? "text-red-300"
        : "text-zinc-300";

  return (
    <div className="relative overflow-hidden rounded-2xl border border-zinc-800/60 bg-zinc-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition hover:border-emerald-500/40">
      <div className="absolute -right-6 top-6 h-24 w-24 rounded-full bg-emerald-500/10 blur-3xl" aria-hidden />
      <div className="flex items-center gap-3 text-sm text-zinc-300">
        <div className="grid h-10 w-10 place-items-center rounded-xl border border-zinc-800/80 bg-zinc-900/70">
          <Icon className="h-5 w-5 text-emerald-300" />
        </div>
        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">{label}</div>
          <div className={`text-lg font-semibold ${accent}`}>{value}</div>
          {helper ? <div className="text-[11px] text-zinc-500">{helper}</div> : null}
        </div>
      </div>
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

/**
 * Generates a unique ID for chat messages.
 * Uses crypto.randomUUID() if available, otherwise falls back to a random string.
 */
function generateUniqueId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback: generate a random string (not cryptographically secure)
  return (
    Math.random().toString(36).slice(2) +
    Math.random().toString(36).slice(2)
  ).slice(0, 16);
}

function createChatMessage(role: ChatMessage["role"], text: string): ChatMessage {
  const ts = Date.now();
  return {
    id: `${role}-${ts}-${generateUniqueId()}`,
    role,
    text,
    timestamp: ts,
  };
}

function mergeAssistantChatMessages(messages: ChatMessage[], nextText: string): ChatMessage[] {
  if (!nextText) {
    return messages;
  }

  if (messages.length > 0) {
    const lastIndex = messages.length - 1;
    const lastMessage = messages[lastIndex];
    if (lastMessage.role === "assistant") {
      if (lastMessage.text === nextText) {
        return messages;
      }

      if (nextText.startsWith(lastMessage.text) || lastMessage.text.startsWith(nextText)) {
        const updated = { ...lastMessage, text: nextText, timestamp: Date.now() };
        const nextMessages = messages.slice();
        nextMessages[lastIndex] = updated;
        return nextMessages;
      }
    }
  }

  const appended = [...messages, createChatMessage("assistant", nextText)];
  if (appended.length > MAX_CHAT_HISTORY) {
    return appended.slice(appended.length - MAX_CHAT_HISTORY);
  }
  return appended;
}

function extractSpeechLineText(line: string): string {
  const separatorIndex = line.indexOf("] ");
  if (separatorIndex === -1) {
    return line;
  }
  return line.slice(separatorIndex + 2);
}

function mergeSpeechLines(lines: string[], stamp: string, nextText: string): string[] {
  if (!nextText) {
    return lines;
  }

  const newEntry = `${stamp} ${nextText}`;
  if (lines.length === 0) {
    return [newEntry];
  }

  const [first, ...rest] = lines;
  const limitedRest = rest.slice(0, Math.max(0, MAX_CHAT_HISTORY - 1));
  const firstText = extractSpeechLineText(first);

  if (firstText === nextText) {
    if (first === newEntry) {
      return lines;
    }
    return [newEntry, ...limitedRest];
  }

  if (nextText.startsWith(firstText) || firstText.startsWith(nextText)) {
    return [newEntry, ...limitedRest];
  }

  const trimmedRest = lines.slice(0, Math.max(0, MAX_CHAT_HISTORY - 1));
  return [newEntry, ...trimmedRest];
}

function MemoryDetailCard({ detail }: { detail: MemoryNodeDetail | null }) {
  if (!detail) {
    return (
      <Card className="h-full bg-zinc-900/70 border-dashed border-zinc-800/60">
        <CardContent className="flex h-full items-center justify-center p-4 text-center text-sm text-zinc-400">
          Select a memory node from the graph or insight lists to inspect its learned behavior, context, and related connections.
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

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const alignment = isUser ? "items-end" : "items-start";
  const label = isUser ? "You" : isAssistant ? "Nomous" : "System";
  const bubbleClass = isUser
    ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-100"
    : isAssistant
      ? "bg-zinc-800/70 border-zinc-700/60 text-zinc-100"
      : "bg-amber-500/10 border-amber-400/40 text-amber-100";

  return (
    <div className={`flex flex-col ${alignment} gap-1`}>
      <div className="chat-label text-zinc-500">
        {label} • {new Date(message.timestamp).toLocaleTimeString()}
      </div>
      <div className={`max-w-xl rounded-2xl border px-3 py-2 text-sm leading-relaxed shadow-sm ${bubbleClass}`}>
        {message.text}
      </div>
    </div>
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
    { value: "en_US-amy-medium.onnx", label: "Amy · en-US (Medium)" },
    { value: "en_US-kristin-medium.onnx", label: "Kristin · en-US (Medium)" },
    { value: "en_US-libritts_r-medium.onnx", label: "LibriTTS-R · en-US (Medium)" },
    { value: "en_US-libritts-high.onnx", label: "LibriTTS · en-US (High)" },
    { value: "en_US-ryan-high.onnx", label: "Ryan · en-US (High)" }
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
                      {voices.map(v => (
                        <option key={v.value} value={v.value}>{v.label}</option>
                      ))}
                      {!voices.some(v => v.value === state.settings.ttsVoice) && state.settings.ttsVoice ? (
                        <option value={state.settings.ttsVoice}>{state.settings.ttsVoice}</option>
                      ) : null}
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
  const { 
    state, 
    setState, 
    connect, 
    disconnect, 
    setMic, 
    push, 
    log, 
    updateSettings,
    chatInput,
    setChatInput,
    chatScrollRef,
    handleChatSubmit,
    handleChatKeyDown,
  } = useNomousBridge();
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

  const avgInTokens = state.tokenWindow.reduce((acc, point) => acc + point.inTok, 0) / Math.max(1, state.tokenWindow.length);
  const avgOutTokens = state.tokenWindow.reduce((acc, point) => acc + point.outTok, 0) / Math.max(1, state.tokenWindow.length);
  const behaviorSummary = state.behavior.summary;
  const behaviorSignals = state.behavior.signals;
  const rewardTotal = state.behavior.rewardTotal;
  const focusPercent = Math.round(behaviorSummary.onTopic * 100);
  const focusStatus: "positive" | "neutral" | "negative" = focusPercent >= 70 ? "positive" : focusPercent < 40 ? "negative" : "neutral";
  const latencyLabel = describeLatency(behaviorSignals.latencyMs);
  const latencyStatus: "positive" | "neutral" | "negative" = behaviorSignals.latencyMs <= 1800 ? "positive" : behaviorSignals.latencyMs > 4000 ? "negative" : "neutral";
  const expressionStatus: "positive" | "neutral" | "negative" = behaviorSummary.coherence > 0.65 && behaviorSummary.lexicalRichness > 0.55
    ? "positive"
    : behaviorSummary.coherence < 0.4
      ? "negative"
      : "neutral";
  const focusAccent = focusStatus === "positive" ? "text-emerald-300" : focusStatus === "negative" ? "text-red-300" : "text-zinc-200";
  const safetyAccent = behaviorSummary.safety > 0.8 ? "text-emerald-300" : behaviorSummary.safety < 0.55 ? "text-red-300" : "text-zinc-200";
  const coherenceAccent = behaviorSummary.coherence > 0.65 ? "text-emerald-300" : behaviorSummary.coherence < 0.4 ? "text-red-300" : "text-zinc-200";
  const micSensitivity = state.settings.micSensitivity;
  const micStatusLabel = state.micOn ? "Capturing" : state.settings.microphoneEnabled ? "Armed" : "Muted";
  const micStatus: "positive" | "negative" | "neutral" = state.micOn
    ? "positive"
    : state.settings.microphoneEnabled
      ? "neutral"
      : "negative";
  const visionActive = state.settings.cameraEnabled && state.visionEnabled;

  return (
    <TooltipProvider>
      <div className="relative min-h-screen overflow-hidden bg-[#040406] text-zinc-50">
        <div className="pointer-events-none absolute inset-0 -z-20 bg-[radial-gradient(circle_at_top,_rgba(124,58,237,0.22),_transparent_65%)]" aria-hidden />
        <div className="pointer-events-none absolute inset-0 -z-10" aria-hidden>
          <div className="absolute -top-24 right-10 h-72 w-72 rounded-full bg-emerald-500/15 blur-3xl" />
          <div className="absolute bottom-[-6rem] left-1/3 h-96 w-96 -translate-x-1/2 rounded-full bg-cyan-500/10 blur-[140px]" />
        </div>
        <div className="pointer-events-none absolute inset-0 -z-[15] bg-[linear-gradient(0deg,rgba(39,39,42,0.35)_1px,transparent_1px),linear-gradient(90deg,rgba(39,39,42,0.35)_1px,transparent_1px)] bg-[size:48px_48px] opacity-60" aria-hidden />

        <div className="relative z-10 mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 pb-12 pt-10 md:px-8">
          <header className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-6">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <div className="grid h-14 w-14 place-items-center rounded-2xl border border-zinc-800/70 bg-zinc-950/80 shadow-[0_25px_60px_rgba(10,10,15,0.65)]">
                    <Brain className="h-7 w-7 text-emerald-300" />
                  </div>
                  <div className="absolute -bottom-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full border border-zinc-800/70 bg-zinc-950/90">
                    <Sparkles className="h-3 w-3 text-emerald-300" />
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-xs uppercase tracking-[0.5em] text-zinc-500">Nomous Autonomy</p>
                  <h1 className="text-3xl font-semibold text-zinc-100 md:text-[2.1rem]">Immersive Control Deck</h1>
                  <p className="max-w-xl text-sm text-zinc-400">
                    Visualize cognition, steer devices, and orchestrate every runtime decision with a purpose-built dark interface.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div className="hidden md:flex max-w-xs items-center gap-2 truncate rounded-full border border-zinc-800/70 bg-zinc-950/70 px-3 py-1.5 text-xs text-zinc-400">
                  {state.connected ? <Wifi className="h-4 w-4 text-emerald-300" /> : <WifiOff className="h-4 w-4 text-red-400" />}
                  <span className="truncate">{state.url}</span>
                </div>
                <Button
                  variant="secondary"
                  onClick={() => setControlCenterOpen(true)}
                  className="hidden sm:inline-flex items-center gap-2 rounded-full border border-zinc-700/60 bg-zinc-900/80 px-4 py-2 text-sm text-zinc-100 hover:bg-zinc-800/80"
                >
                  <Cog className="h-4 w-4" /> Control Center
                </Button>
                {!state.connected ? (
                  <Button onClick={connect} className="flex items-center gap-2 rounded-full bg-emerald-600/90 px-4 py-2 text-sm text-white shadow-lg transition hover:bg-emerald-500/90">
                    <Play className="h-4 w-4" /> Connect
                  </Button>
                ) : (
                  <Button
                    variant="danger"
                    onClick={disconnect}
                    className="flex items-center gap-2 rounded-full bg-red-600/90 px-4 py-2 text-sm text-white shadow-lg transition hover:bg-red-500/90"
                  >
                    <Square className="h-4 w-4" /> Disconnect
                  </Button>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-4 rounded-2xl border border-zinc-800/70 bg-zinc-950/70 px-5 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] md:flex-row md:items-center md:justify-between">
              <div className="flex items-center gap-3">
                <div className={`h-3 w-3 rounded-full ${st.color} shadow-[0_0_20px_rgba(34,197,94,0.35)]`} />
                <div>
                  <div className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Runtime Status</div>
                  <div className="text-lg font-semibold text-zinc-100">
                    {st.label}
                    {state.statusDetail ? <span className="ml-2 text-sm text-zinc-300">{state.statusDetail}</span> : null}
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
                <Badge className="bg-zinc-900/80 text-zinc-200 border border-zinc-700/60">Tokens window • {tokenTotal}</Badge>
                <Badge className="bg-zinc-900/80 text-zinc-200 border border-zinc-700/60">Mic {state.micOn ? "Active" : "Muted"}</Badge>
                <Badge className="bg-zinc-900/80 text-zinc-200 border border-zinc-700/60">Vision {state.visionEnabled ? "Online" : "Paused"}</Badge>
              </div>
            </div>
          </header>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <QuickStatCard
              icon={Brain}
              label="Cognitive Focus"
              value={`${focusPercent}%`}
              helper="Topical alignment (rolling window)"
              status={focusStatus}
            />
            <QuickStatCard
              icon={Gauge}
              label="Response Latency"
              value={latencyLabel}
              helper={`In ${Math.round(avgInTokens)} • Out ${Math.round(behaviorSignals.avgResponseLength)} tokens`}
              status={latencyStatus}
            />
            <QuickStatCard
              icon={Sparkles}
              label="Expression Quality"
              value={`${Math.round(behaviorSummary.coherence * 100)}% / ${Math.round(behaviorSummary.lexicalRichness * 100)}%`}
              helper="Coherence • Lexical richness"
              status={expressionStatus}
            />
            <QuickStatCard
              icon={Flag}
              label="Reward Signal"
              value={rewardTotal.toFixed(1)}
              helper={`Pace ${behaviorSignals.conversationPace.toFixed(2)}/min`}
              status={rewardTotal >= 0 ? "positive" : "negative"}
            />
          </div>

        <Card className="bg-zinc-900/60 backdrop-blur-sm border-zinc-800/70">
          <CardContent className="p-4">
            <div className="w-full">
              <Tabs defaultValue="overview">
              <TabsList className="flex flex-wrap gap-2 bg-zinc-950/60 border border-zinc-800/60">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="console">Console</TabsTrigger>
                <TabsTrigger value="chat">Chat</TabsTrigger>
                <TabsTrigger value="behavior">Behavior</TabsTrigger>
                <TabsTrigger value="tokens">Tokens</TabsTrigger>
                <TabsTrigger value="memory">Memory</TabsTrigger>
                <TabsTrigger value="tools">Tools</TabsTrigger>
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
                      <CardContent className="p-4 space-y-4">
                        <div className="flex items-center gap-2 text-zinc-200 font-semibold"><Brain className="w-4 h-4"/> Cognitive Telemetry</div>
                        <div className="space-y-3 text-xs text-zinc-300">
                          <div>
                            <div className="flex items-center justify-between"><span>Focus</span><span className={focusAccent}>{focusPercent}%</span></div>
                            <Progress value={behaviorSummary.onTopic * 100} />
                          </div>
                          <div>
                            <div className="flex items-center justify-between"><span>Safety</span><span className={safetyAccent}>{Math.round(behaviorSummary.safety * 100)}%</span></div>
                            <Progress value={behaviorSummary.safety * 100} />
                          </div>
                          <div>
                            <div className="flex items-center justify-between"><span>Coherence</span><span className={coherenceAccent}>{Math.round(behaviorSummary.coherence * 100)}%</span></div>
                            <Progress value={behaviorSummary.coherence * 100} />
                          </div>
                          <div className="flex items-center justify-between"><span>Latency</span><span className="text-zinc-200">{latencyLabel}</span></div>
                        </div>
                        <div className="text-xs text-zinc-400">Reward total <span className={rewardTotal>=0?"text-emerald-300":"text-red-300"}>{rewardTotal.toFixed(1)}</span></div>
                      </CardContent>
                    </Card>
                  </div>

                  <div className="space-y-4">
                    <SystemUsageCard metrics={state.systemMetrics} />
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

              <TabsContent value="chat" className="pt-4">
                <Card className="bg-zinc-900/70 border-zinc-800/60">
                  <CardContent className="flex h-[26rem] flex-col gap-4 p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-zinc-200">
                        <MessageSquare className="h-4 w-4" />
                        <span className="font-semibold">Manual Chat</span>
                      </div>
                      <Badge className={`border ${state.connected ? "bg-emerald-500/20 text-emerald-100 border-emerald-500/40" : "bg-red-500/20 text-red-200 border-red-500/40"}`}>
                        {state.connected ? "Connected" : "Offline"}
                      </Badge>
                    </div>

                    <div className="flex-1 overflow-y-auto rounded-lg border border-zinc-800/60 bg-black/50 p-4" ref={chatScrollRef}>
                      <div className="flex flex-col gap-3">
                        {state.chatMessages.length === 0 ? (
                          <div className="text-sm text-zinc-400">
                            Type a message below to talk to the model without relying on the microphone or camera. Conversations stay in this session only.
                          </div>
                        ) : (
                          state.chatMessages.map(message => (
                            <ChatBubble key={message.id} message={message} />
                          ))
                        )}
                      </div>
                    </div>

                    <form onSubmit={handleChatSubmit} className="space-y-2">
                      <label className="text-xs uppercase tracking-[0.3em] text-zinc-500">Message</label>
                      <div className="flex flex-col gap-2 sm:flex-row">
                        <textarea
                          value={chatInput}
                          onChange={event => setChatInput(event.target.value)}
                          onKeyDown={handleChatKeyDown}
                          rows={2}
                          placeholder="Type your instruction or question..."
                          className="min-h-[64px] flex-1 rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-emerald-500/80 focus:outline-none focus:ring-0"
                        />
                        <Button
                          type="submit"
                          disabled={!chatInput.trim()}
                          className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600/90 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500/90 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
                        >
                          <Send className="h-4 w-4" />
                          Send
                        </Button>
                      </div>
                      <div className="text-[11px] text-zinc-500">Press Enter to send. Use Shift + Enter for a new line.</div>
                    </form>
                  </CardContent>
                </Card>
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

              <TabsContent value="behavior" className="pt-4 space-y-4">
                <BehaviorInsights stats={state.behavior} />
                <div className="grid gap-4 lg:grid-cols-2">
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="p-4 space-y-4 text-zinc-200">
                      <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.3em] text-zinc-400"><Brain className="w-4 h-4"/> Manual Steering</div>
                      <div className="text-sm text-zinc-300">Reward balance <span className={rewardTotal>=0?"text-emerald-300":"text-red-300"}>{rewardTotal.toFixed(1)}</span> • Latest sentiment {Math.round(behaviorSummary.sentiment * 100)}%</div>
                      <div className="grid grid-cols-2 gap-2">
                        <Button variant="secondary" className="px-3 py-2 text-sm bg-emerald-600/80 hover:bg-emerald-500/80 text-white" onClick={()=>push({ type: "reinforce", delta: +1 })}>Reward +1</Button>
                        <Button variant="secondary" className="px-3 py-2 text-sm bg-red-600/70 hover:bg-red-500/70 text-white" onClick={()=>push({ type: "reinforce", delta: -1 })}>Penalty -1</Button>
                      </div>
                      <div className="text-xs text-zinc-400">Latency {latencyLabel} • Pace {behaviorSignals.conversationPace.toFixed(2)}/min • Safety {Math.round(behaviorSummary.safety * 100)}%</div>
                    </CardContent>
                  </Card>
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="p-4 text-zinc-200 space-y-3">
                      <div className="font-semibold uppercase tracking-[0.3em] text-xs text-zinc-400">How to interpret</div>
                      <p className="text-sm text-zinc-300">Focus gauges topical alignment; safety tracks toxicity risk; expression blends coherence with lexical variety. Use reinforcement to encourage behaviors you want to see amplified.</p>
                      <p className="text-xs text-zinc-400">These scores are computed from the last {state.behavior.history.length} exchanges and refresh in real time as the model speaks.</p>
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

              <TabsContent value="tools" className="pt-4">
                <div className="grid gap-4 lg:grid-cols-3">
                  <div className="lg:col-span-2">
                    <ToolActivity tools={state.toolActivity} maxDisplay={20} />
                  </div>
                  <div className="space-y-4 lg:col-span-1">
                    <ToolStats tools={state.toolActivity} />
                    <Card className="bg-slate-900/50 border-slate-700">
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Wrench className="h-4 w-4 text-slate-400" />
                          <span className="text-sm font-medium text-slate-200">About Tools</span>
                        </div>
                        <p className="text-xs text-slate-400">
                          The LLM can use 9 built-in tools to enhance its capabilities:
                          memory search, observations, self-evaluation, pattern recognition,
                          sentiment analysis, and more.
                        </p>
                      </CardContent>
                    </Card>
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
        {state.loadingOverlay && (
          <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 backdrop-blur-sm px-4">
            <div className="w-full max-w-md rounded-2xl border border-emerald-500/30 bg-zinc-950/90 p-6 shadow-[0_24px_100px_rgba(16,185,129,0.25)]">
              <div className="mb-4 flex items-center gap-3 text-emerald-200">
                <RefreshCw className="h-5 w-5 animate-spin" />
                <div>
                  <div className="text-sm font-semibold text-emerald-100">{state.loadingOverlay.label}</div>
                  {state.loadingOverlay.detail ? (
                    <div className="text-xs text-zinc-400">{state.loadingOverlay.detail}</div>
                  ) : (
                    <div className="text-xs text-zinc-400">Preparing weights and buffers…</div>
                  )}
                </div>
              </div>
              <Progress value={Math.min(state.loadingOverlay.progress, 100)} />
            </div>
          </div>
        )}
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
    </div>
  </TooltipProvider>
  );
}
