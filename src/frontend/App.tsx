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
import { Input } from "./components/ui/input";
import { Activity, AlertTriangle, AudioWaveform, Brain, Camera, Cog, Cpu, MessageSquare, Play, Radio, RefreshCw, Square, Mic, MicOff, Speaker, Wifi, WifiOff, Volume2, Flag, Database, Clock, Sparkles, Gauge, Send, Wrench, Network, CirclePlus, Trash2 } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip as RTooltip } from "recharts";
import type { MemoryEdge, MemoryNode, MemoryNodeKind, Entity, EntityType, TimelineEvent, SemanticSearchResult } from "./types/memory";
import type { SystemMetricsPayload } from "./types/system";
import type { BehaviorStats } from "./types/behavior";
import { EMPTY_BEHAVIOR_STATS } from "./types/behavior";
import { BehaviorInsights } from "./components/BehaviorInsights";
import { buildMemoryNodeDetail, computeMemoryInsights } from "./utils/memory";
import type { MemoryNodeDetail, MemoryInsightEntry } from "./utils/memory";
import { normaliseVoiceFilename, readJson, writeJson } from "./utils/storage";
import { ToolActivity, ToolStats } from "./components/ToolActivity";
import type { ToolResult } from "./components/ToolActivity";
import { SystemUsageCard } from "./components/SystemUsageCard";
import { EntityBrowser } from "./components/EntityBrowser";
import { LearningTimeline } from "./components/LearningTimeline";
import { SemanticSearch } from "./components/SemanticSearch";
import { TimelinePerspective } from "./components/TimelinePerspective";
import { GenerationProgress } from "./components/GenerationProgress";

const DEFAULT_SYSTEM_PROMPT =
  "You are Nomous, an autonomous multimodal AI orchestrator. Support the operator with concise, confident guidance, coordinate sensors and tools, and narrate decisions with a collaborative tone.";

const DEFAULT_THINKING_PROMPT =
  "Think in small, verifiable steps. Reference available tools and memories, note uncertainties, and decide on an action plan before committing to a response. Keep thoughts structured and actionable.";

type ModelCatalogType = "conversation" | "vision" | "coder" | "reasoning" | "audio" | "other";

interface ModelCatalogEntry {
  name: string;
  path: string;
  sizeBytes: number;
  sizeLabel: string;
  type: ModelCatalogType;
}

const MODEL_TYPE_STYLES: Record<ModelCatalogType, string> = {
  conversation: "bg-emerald-500/10 text-emerald-200 border border-emerald-500/30",
  vision: "bg-cyan-500/10 text-cyan-200 border border-cyan-500/30",
  coder: "bg-sky-500/10 text-sky-200 border border-sky-500/30",
  reasoning: "bg-purple-500/10 text-purple-200 border border-purple-500/30",
  audio: "bg-amber-500/10 text-amber-200 border border-amber-500/30",
  other: "bg-zinc-800/60 text-zinc-200 border border-zinc-700/60",
};

const MODEL_TYPE_LABEL: Record<ModelCatalogType, string> = {
  conversation: "Conversation",
  vision: "Vision",
  coder: "Coder",
  reasoning: "Reasoning",
  audio: "Audio",
  other: "General",
};

/** Nomous – Autonomy Dashboard (fixed) */
export type NomousStatus = "idle" | "thinking" | "speaking" | "noticing" | "learning" | "error";
interface TokenPoint { t: number; inTok: number; outTok: number }
type STTPhase = "partial" | "final" | "forwarded";

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
  models?: any[];
  directory?: string;
  error?: string | null;
  system_prompt?: string;
  thinking_prompt?: string;
  payload?: any;
  entities?: Entity[];
  events?: TimelineEvent[];
  results?: SemanticSearchResult[];
  count?: number;
  tool?: string;
  display_name?: string;
  category?: string;
  description?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown> | unknown;
  success?: boolean;
  summary?: string;
  warnings?: string[];
  timestamp?: number;
  duration_ms?: number;
  durationMs?: number;
}

interface STTEventMessage extends WSMessage {
  type: "stt";
  phase?: STTPhase;
  forwarded?: boolean | "true";
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
  systemPrompt: string;
  thinkingPrompt: string;
  modelDirectory: string;
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
  speechLines: string[]; systemLines: string[]; sttLines: string[]; chatMessages: ChatMessage[]; toolActivity: ToolResult[]; systemMetrics: SystemMetricsPayload | null;
  settings: ControlSettings; loadingOverlay: LoadingOverlay | null; modelCatalog: ModelCatalogEntry[]; modelCatalogError: string | null;
  promptReloadRequired: boolean; generationProgress: { active: boolean; text: string; tokens: number } | null;
}

interface LoadingOverlay {
  label: string;
  progress: number;
  detail?: string;
}

interface ThoughtAccumulator {
  prefix: string | null;
  stamp: string;
  content: string;
  updatedAt: number;
}

interface SttFinalRecord {
  text: string;
  stamp: string;
}

const TARGET_SAMPLE_RATE = 16000;
const MAX_CHAT_HISTORY = 200;
const THOUGHT_ACCUMULATOR_WINDOW_MS = 4500;
const STATUS_KEYWORDS = new Set([
  "idle",
  "thinking",
  "ready",
  "error",
  "connected",
  "disconnected",
  "listening",
  "speaking",
  "noticing",
  "learning",
]);
const STORAGE_SETTINGS_KEY = "nomous.settings";
const STORAGE_WS_KEY = "nomous.ws";
const ENV_DEFAULT_WS_URL = typeof import.meta.env.VITE_DEFAULT_WS_URL === "string"
  ? import.meta.env.VITE_DEFAULT_WS_URL.trim()
  : "";
const FALLBACK_WS_URL = ENV_DEFAULT_WS_URL.length > 0 ? ENV_DEFAULT_WS_URL : "ws://localhost:8765";

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

function removeListeningLines(lines: string[]): string[] {
  return lines.filter(line => !line.includes("Listening:"));
}

function limitSttHistory(lines: string[]): string[] {
  return lines.slice(0, Math.max(0, MAX_CHAT_HISTORY - 1));
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
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  thinkingPrompt: DEFAULT_THINKING_PROMPT,
  modelDirectory: "",
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

function normaliseModelType(value: unknown): ModelCatalogType {
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    if (lowered.includes("vision") || lowered === "vl") return "vision";
    if (lowered.includes("coder") || lowered.includes("code")) return "coder";
    if (lowered.includes("reason") || lowered.includes("think")) return "reasoning";
    if (lowered.includes("audio") || lowered.includes("voice") || lowered.includes("speech")) return "audio";
    if (lowered.includes("chat") || lowered.includes("assistant") || lowered.includes("conversation")) return "conversation";
  }
  return "other";
}

function formatBytesCompact(size: number): string {
  if (!Number.isFinite(size) || size <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function extractModelName(path: string): string {
  if (typeof path !== "string" || path.trim().length === 0) {
    return "Unknown";
  }
  const segments = path.split(/[/\\]/).filter(Boolean);
  if (segments.length === 0) {
    return path.trim();
  }
  return segments[segments.length - 1];
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
  if ("systemPrompt" in patch && patch.systemPrompt !== undefined) {
    next.systemPrompt = coerceString(patch.systemPrompt, next.systemPrompt);
  }
  if ("thinkingPrompt" in patch && patch.thinkingPrompt !== undefined) {
    next.thinkingPrompt = coerceString(patch.thinkingPrompt, next.thinkingPrompt);
  }
  if ("modelDirectory" in patch && patch.modelDirectory !== undefined) {
    next.modelDirectory = coerceString(patch.modelDirectory, next.modelDirectory);
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

function computeDefaultWsUrl(): string {
  if (typeof window === "undefined") {
    return FALLBACK_WS_URL;
  }
  // Use window.location.host only in development mode, otherwise fallback to documented backend address
  const isDev =
    (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.MODE === "development") ||
    (typeof process !== "undefined" && process.env && process.env.NODE_ENV === "development");
  if (isDev) {
    const { protocol, host } = window.location;
    if (typeof host === "string" && host.trim().length > 0) {
      const wsProtocol = protocol === "https:" ? "wss:" : "ws:";
      return `${wsProtocol}//${host}/ws`;
    }
  }
  // In production, fallback to documented backend address
  return FALLBACK_WS_URL;
}

function resolveInitialUrl(): string {
  if (typeof window === "undefined") {
    return FALLBACK_WS_URL;
  }
  const stored = window.localStorage.getItem(STORAGE_WS_KEY);
  if (stored && stored.trim().length > 0) {
    return stored;
  }
  if (ENV_DEFAULT_WS_URL.length > 0) {
    return ENV_DEFAULT_WS_URL;
  }
  return computeDefaultWsUrl();
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
    sttLines: [],
    chatMessages: [],
    toolActivity: [],
    systemMetrics: null,
    settings,
    loadingOverlay: null,
    modelCatalog: [],
    modelCatalogError: null,
    promptReloadRequired: false,
    generationProgress: null,
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
  temporarilyMuted?: boolean;
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
  const thoughtAccumulatorRef = useRef<ThoughtAccumulator | null>(null);
  const sttFinalRef = useRef<SttFinalRecord | null>(null);

  // Enhanced memory system state
  const [entities, setEntities] = useState<Entity[]>([]);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [searchResults, setSearchResults] = useState<SemanticSearchResult[]>([]);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);

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

  const updateMemoryNode = useCallback((id: string, patch: Record<string, unknown>) => {
    if (!id || !patch || typeof patch !== "object") return;
    push({ type: "memory_update", id, patch });
  }, [push]);

  const deleteMemoryNode = useCallback((id: string) => {
    if (!id) return;
    push({ type: "memory_delete", id });
  }, [push]);

  const createMemoryNode = useCallback((draft: ManualMemoryNodeDraft) => {
    if (!draft || typeof draft !== "object") return;
    push({ type: "memory_create", node: draft });
  }, [push]);

  const createMemoryLink = useCallback(
    (fromId: string, toId: string, weight: number, relationship?: string, context?: Record<string, unknown>) => {
      if (!fromId || !toId) return;
      push({
        type: "memory_link",
        from: fromId,
        to: toId,
        weight,
        relationship,
        context,
      });
    },
    [push],
  );

  const deleteMemoryEdge = useCallback((edgeId: string) => {
    if (!edgeId) return;
    push({ type: "memory_unlink", id: edgeId });
  }, [push]);

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

  // Enhanced memory system WebSocket helpers
  const fetchEntities = useCallback((entityType?: string) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) {
      console.log("[Memory] Requesting entities from backend...");
      ws.send(JSON.stringify({ type: "get_entities", entity_type: entityType, limit: 100 }));
    } else {
      console.log("[Memory] Cannot fetch entities - WebSocket not ready:", ws?.readyState);
    }
  }, []);

  const fetchTimeline = useCallback((entityId?: string) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) {
      console.log("[Memory] Requesting timeline from backend...");
      ws.send(JSON.stringify({ type: "get_timeline", entity_id: entityId, limit: 50 }));
    }
  }, []);

  const performSearch = useCallback((query: string, options: { limit: number; threshold: number; entityType?: string }) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) {
      setIsSearching(true);
      ws.send(JSON.stringify({ type: "semantic_search", query, ...options }));
    }
  }, []);

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
      const msg = JSON.parse(
        typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data),
      ) as WSMessage;
      switch (msg.type) {
        case "status": {
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          const value = typeof msg.value === "string" ? msg.value : String(msg.value ?? "status");
          const rawDetail = typeof msg.detail === "string" ? msg.detail : "";

          setState(p => {
            let speechLines = p.speechLines;
            let chatMessages = p.chatMessages;
            let statusDetail = rawDetail || p.statusDetail;
            let speechText: string | null = null;

            if (value.toLowerCase() === "speaking") {
              speechText = extractSpeechText(rawDetail || msg);
              if (speechText) {
                speechLines = mergeSpeechLines(p.speechLines, stamp, speechText);
                chatMessages = mergeAssistantChatMessages(p.chatMessages, speechText);
                statusDetail = speechText;
              }
            }

            const systemLines = statusDetail
              ? [`${stamp} ${value.toUpperCase()} → ${statusDetail}`, ...p.systemLines.slice(0, MAX_CHAT_HISTORY)]
              : p.systemLines;

            // Ensure status is a valid NomousStatus
            const validStatuses: NomousStatus[] = ["idle", "thinking", "speaking", "noticing", "learning", "error"];
            const normalizedStatus: NomousStatus = validStatuses.includes(value as NomousStatus) 
              ? (value as NomousStatus) 
              : "idle";

            return {
              ...p,
              status: normalizedStatus,
              statusDetail,
              systemLines,
              speechLines,
              chatMessages,
            };
          });
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
        case "speak":
        case "speech":
        case "speech_commitment":
        case "speech_final": {
          const speechText = extractSpeechText(msg);
          if (!speechText) {
            break;
          }
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          log(`speak → ${speechText}`);
          
          // Temporarily mute microphone to prevent feedback during TTS
          const mic = micRef.current;
          if (mic) {
            mic.temporarilyMuted = true;
            log("🔇 Microphone temporarily muted during TTS playback");
            
            // Estimate TTS duration (rough approximation: 150 words per minute)
            const wordCount = speechText.split(/\s+/).length;
            const durationMs = Math.max(2000, (wordCount / 150) * 60 * 1000 + 500);
            
            setTimeout(() => {
              if (micRef.current === mic) {
                mic.temporarilyMuted = false;
                log("🔊 Microphone unmuted after TTS playback");
              }
            }, durationMs);
          }
          
          setState(p => ({
            ...p,
            status: "speaking",
            statusDetail: speechText,
            speechLines: mergeSpeechLines(p.speechLines, stamp, speechText),
            chatMessages: mergeAssistantChatMessages(p.chatMessages, speechText),
            generationProgress: null,
          }));
          break;
        }
        case "thought": {
          const rawThought = typeof msg.text === "string" ? msg.text : "";
          const normalizedThought = normalizeStreamText(rawThought);
          if (!normalizedThought) {
            break;
          }

          const { prefix, detail } = splitThoughtMessage(normalizedThought);
          if (!detail && !prefix) {
            break;
          }

          // Extract token count for progress tracking
          const tokenMatch = detail.match(/(\d+)\s+tokens?/i);
          const tokens = tokenMatch ? parseInt(tokenMatch[1], 10) : 0;
          const isProcessing = detail.toLowerCase().includes("processing");

          // Check if this is a token processing message (should show in progress bar, not thought stream)
          const isTokenProcessingMessage = isProcessing && tokens > 0;

          const now = Date.now();
          setState(p => {
            // Update generation progress for token processing messages
            const generationProgress = isTokenProcessingMessage
              ? { active: true, text: detail, tokens }
              : p.generationProgress;

            // Skip adding token processing messages to the thought stream
            // They will be shown as a progress bar instead
            if (isTokenProcessingMessage) {
              return { ...p, generationProgress };
            }

            const prevAccumulator = thoughtAccumulatorRef.current;
            let nextLines = p.thoughtLines;

            if (
              prevAccumulator &&
              prevAccumulator.prefix === prefix &&
              now - prevAccumulator.updatedAt < THOUGHT_ACCUMULATOR_WINDOW_MS
            ) {
              const combinedDetail = combineThoughtDetails(prevAccumulator.content, detail);
              const updatedEntry = `${prevAccumulator.stamp} ${formatThoughtDisplay(prefix, combinedDetail)}`;
              const [, ...rest] = nextLines;
              const limitedRest = rest.slice(0, Math.max(0, MAX_CHAT_HISTORY - 1));
              nextLines = [updatedEntry, ...limitedRest];
              thoughtAccumulatorRef.current = { ...prevAccumulator, content: combinedDetail, updatedAt: now };
            } else {
              const stamp = `[${new Date().toLocaleTimeString()}]`;
              const entry = `${stamp} ${formatThoughtDisplay(prefix, detail)}`;
              nextLines = [entry, ...nextLines.slice(0, Math.max(0, MAX_CHAT_HISTORY - 1))];
              thoughtAccumulatorRef.current = { prefix, stamp, content: detail, updatedAt: now };
            }

            return { ...p, thoughtLines: nextLines, generationProgress };
          });
          break;
        }
        case "stt": {
          const sttMsg = msg as STTEventMessage;
          const rawText = typeof sttMsg.text === "string" ? sttMsg.text : "";
          const normalizedText = normalizeStreamText(rawText);
          if (!normalizedText) {
            break;
          }

          const phaseRaw = typeof sttMsg.phase === "string" ? sttMsg.phase.toLowerCase() : "";
          const phase: STTPhase =
            phaseRaw === "final" || phaseRaw === "forwarded" ? (phaseRaw as STTPhase) : "partial";
          // Default to "partial" for unrecognized phases or when the backend omits the phase field, so that transient captions still render.
          const forwarded = sttMsg.forwarded === true || sttMsg.forwarded === "true";

          setState(p => {
            const filtered = removeListeningLines(p.sttLines);
            const trimmedHistory = limitSttHistory(filtered);
            const stamp = `[${new Date().toLocaleTimeString()}]`;
            let nextLines;

            if (phase === "partial") {
              const entry = `${stamp} Listening: ${normalizedText}`;
              nextLines = [entry, ...trimmedHistory];
            } else if (phase === "final") {
              const suffix = forwarded ? " → queued for response" : "";
              const entry = `${stamp} Captured: ${normalizedText}${suffix}`;
              nextLines = [entry, ...trimmedHistory];
              sttFinalRef.current = { text: normalizedText, stamp };
            } else {
              // phase === "forwarded" - Add user message to chat
              const reference = sttFinalRef.current;
              if (reference && reference.text === normalizedText) {
                const [, ...rest] = filtered;
                const entry = `${reference.stamp} Dispatched: ${normalizedText} → reasoning core`;
                nextLines = [entry, ...limitSttHistory(rest)];
              } else {
                const entry = `${stamp} Dispatched: ${normalizedText} → reasoning core`;
                // A forwarded event can arrive before we capture the matching final payload; fall back to the filtered log.
                nextLines = [entry, ...trimmedHistory];
              }
              sttFinalRef.current = null;

              // Add user message to chat when speech is forwarded to LLM
              const userMessage = createChatMessage("user", normalizedText);
              return {
                ...p,
                sttLines: nextLines,
                chatMessages: [...p.chatMessages, userMessage].slice(-MAX_CHAT_HISTORY),
              };
            }

            return { ...p, sttLines: nextLines };
          });
          break;
        }
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
        case "entities": {
          console.log("[Memory] Received entities:", msg.entities?.length || 0, msg.entities);
          setEntities(msg.entities || []);
          break;
        }
        case "timeline": {
          console.log("[Memory] Received timeline events:", msg.events?.length || 0);
          setTimeline(msg.events || []);
          break;
        }
        case "search_results": {
          setSearchResults(msg.results || []);
          setIsSearching(false);
          break;
        }
        case "tool_result": {
          const timestamp = typeof msg.timestamp === "number" ? msg.timestamp : Date.now();
          const rawDuration = typeof msg.duration_ms === "number" ? msg.duration_ms : Number(msg.durationMs ?? 0);
          const durationMs = Number.isFinite(rawDuration) ? Number(rawDuration) : 0;
          const toolId = typeof msg.tool === "string" ? msg.tool : "unknown";
          const resultPayload =
            msg.result && typeof msg.result === "object"
              ? (msg.result as Record<string, unknown>)
              : { value: msg.result };
          const success = typeof msg.success === "boolean" ? msg.success : !("error" in resultPayload);

          const toolResult: ToolResult = {
            tool: toolId,
            displayName:
              typeof msg.display_name === "string"
                ? msg.display_name
                : toolId.replace(/_/g, " ").replace(/\b\w/g, s => s.toUpperCase()),
            category: typeof msg.category === "string" ? msg.category : "general",
            description: typeof msg.description === "string" ? msg.description : "",
            args:
              msg.args && typeof msg.args === "object" && !Array.isArray(msg.args)
                ? (msg.args as Record<string, unknown>)
                : {},
            result: resultPayload,
            success,
            summary: typeof msg.summary === "string" ? msg.summary : "",
            warnings: Array.isArray(msg.warnings) ? msg.warnings.map(w => String(w)) : [],
            timestamp,
            durationMs,
          };

          const summarySnippet = toolResult.summary ? toolResult.summary.slice(0, 160) : "";
          const statusEmoji = toolResult.success ? "✅" : "⚠️";
          const displayName = toolResult.displayName || toolResult.tool;
          const toolLine = summarySnippet
            ? `${statusEmoji} Tool ${displayName} — ${summarySnippet}`
            : `${statusEmoji} Tool ${displayName}`;

          setState(p => ({
            ...p,
            toolActivity: [...p.toolActivity, toolResult].slice(-100),
            systemLines: [`[${new Date().toLocaleTimeString()}] ${toolLine}`, ...p.systemLines.slice(0, MAX_CHAT_HISTORY)]
          }));
          break;
        }
        case "event": {
          const stamp = `[${new Date().toLocaleTimeString()}]`;
          const payload = msg.message;
          log(payload || "event");
          const clearsPromptWarning = typeof payload === "string" && /LLM model/i.test(payload || "");
          setState(p => ({
            ...p,
            lastEvent: payload,
            promptReloadRequired: clearsPromptWarning ? false : p.promptReloadRequired,
            systemLines: payload ? [`${stamp} EVENT → ${payload}`, ...p.systemLines.slice(0, MAX_CHAT_HISTORY)] : p.systemLines,
          }));
          break;
        }
        case "prompt_state": {
          setState(p => ({
            ...p,
            settings: mergeSettings(p.settings, {
              systemPrompt: typeof msg.system_prompt === "string" ? msg.system_prompt : p.settings.systemPrompt,
              thinkingPrompt: typeof msg.thinking_prompt === "string" ? msg.thinking_prompt : p.settings.thinkingPrompt,
            }),
            promptReloadRequired: false,
          }));
          break;
        }
        case "model_catalog": {
          const rawModels = Array.isArray(msg.models) ? msg.models : [];
          const entries: ModelCatalogEntry[] = rawModels.map((item: any) => {
            const sizeBytes = Number(item?.size_bytes ?? item?.sizeBytes ?? 0);
            const derivedLabel = typeof item?.size_label === "string" ? item.size_label : formatBytesCompact(sizeBytes);
            return {
              name: typeof item?.name === "string" ? item.name : "unknown.gguf",
              path: typeof item?.path === "string" ? item.path : "",
              sizeBytes: Number.isFinite(sizeBytes) ? sizeBytes : 0,
              sizeLabel: derivedLabel,
              type: normaliseModelType(item?.type),
            };
          });
          const directory = typeof msg.directory === "string" ? msg.directory : undefined;
          const error = typeof msg.error === "string" ? msg.error : null;
          setState(p => ({
            ...p,
            settings: directory ? mergeSettings(p.settings, { modelDirectory: directory }) : p.settings,
            modelCatalog: entries,
            modelCatalogError: error,
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
  }, [handleMessage, log, push, state.url, fetchEntities, fetchTimeline]);

  // Fetch initial entity and timeline data when connected
  useEffect(() => {
    if (state.connected && wsRef.current && wsRef.current.readyState === 1) {
      fetchEntities();
      fetchTimeline();
    }
  }, [state.connected, fetchEntities, fetchTimeline]);

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
      log("🎤 Microphone disabled");
      stopMic();
      return;
    }

    log("🎤 Microphone activation requested...");

    if (typeof window === "undefined" || typeof navigator === "undefined") {
      log("mic error: unavailable in this environment");
      return;
    }

    if (micRef.current) {
      log("🎤 Microphone already active");
      return;
    }

    const AudioContextClass = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext | undefined;
    if (!AudioContextClass) {
      log("❌ mic error: AudioContext unsupported in this browser");
      return;
    }

    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;

    try {
      log("🎤 Requesting microphone permission...");
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: TARGET_SAMPLE_RATE, echoCancellation: true, noiseSuppression: true },
        video: false,
      });
      log("✅ Microphone permission granted");

      log("🔊 Creating audio context...");
      ctx = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE }) as AudioContext;
      await ctx.resume();
      const inputSampleRate = ctx.sampleRate;
      log(`🔊 Audio context created (sample rate: ${inputSampleRate}Hz)`);

      if (typeof ctx.createScriptProcessor !== "function") {
        throw new Error("ScriptProcessorNode not supported in this browser");
      }

      log("🎙️ Setting up audio processing pipeline...");
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

      let chunkCount = 0;
      processor.onaudioprocess = (event) => {
        if (micRef.current !== chain) {
          return;
        }
        // Don't process audio if temporarily muted (TTS is speaking)
        if (chain.temporarilyMuted) {
          return;
        }
        const channelData = event.inputBuffer.getChannelData(0);
        const encoded = encodeAudioChunk(chain, channelData, inputSampleRate);
        if (encoded) {
          chunkCount++;
          if (chunkCount === 1) {
            log("🎵 Audio streaming started - speaking into microphone...");
          }
          push({ type: "audio", rate: TARGET_SAMPLE_RATE, pcm16: encoded });
        }
      };

      setState(p => ({ ...p, micOn: true }));
      log("✅ Microphone is now ACTIVE and listening");
    } catch (e: any) {
      const errorMsg = e?.message ?? String(e);
      if (errorMsg.includes("Permission denied") || errorMsg.includes("NotAllowedError")) {
        log("❌ Microphone permission denied - please allow microphone access in your browser");
      } else if (errorMsg.includes("NotFoundError") || errorMsg.includes("not found")) {
        log("❌ No microphone found - please connect a microphone device");
      } else if (errorMsg.includes("NotReadableError")) {
        log("❌ Microphone is busy or being used by another application");
      } else {
        log(`❌ Microphone error: ${errorMsg}`);
      }
      
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (ctx) {
        ctx.close().catch(() => {});
      }
      micRef.current = null;
      setState(p => ({ ...p, micOn: false, settings: mergeSettings(p.settings, { microphoneEnabled: false }), vu: 0 }));
    }
  }, [log, push, setState, stopMic]);

  const disconnect = useCallback(() => {
    stopMic();
    wsRef.current?.close(); wsRef.current = null;
  }, [stopMic]);

  // Auto-start microphone when microphoneEnabled is true and connected
  useEffect(() => {
    if (state.connected && state.settings.microphoneEnabled && !state.micOn) {
      log("🎤 Auto-starting microphone (enabled in settings)...");
      setMic(true);
    } else if (!state.settings.microphoneEnabled && state.micOn) {
      log("🎤 Auto-stopping microphone (disabled in settings)...");
      setMic(false);
    }
  }, [state.connected, state.settings.microphoneEnabled, state.micOn, log, setMic]);

  const updateSettings = useCallback((patch: Partial<ControlSettings>) => {
    setState(prev => {
      const nextSettings = mergeSettings(prev.settings, patch);
      const promptChanged =
        ("systemPrompt" in patch && patch.systemPrompt !== undefined && patch.systemPrompt !== prev.settings.systemPrompt) ||
        ("thinkingPrompt" in patch && patch.thinkingPrompt !== undefined && patch.thinkingPrompt !== prev.settings.thinkingPrompt);
      return {
        ...prev,
        settings: nextSettings,
        audioEnabled: "ttsEnabled" in patch ? nextSettings.ttsEnabled : prev.audioEnabled,
        visionEnabled: "cameraEnabled" in patch ? nextSettings.cameraEnabled : prev.visionEnabled,
        micOn: "microphoneEnabled" in patch ? nextSettings.microphoneEnabled : prev.micOn,
        promptReloadRequired: promptChanged ? true : prev.promptReloadRequired,
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
    updateMemoryNode,
    deleteMemoryNode,
    createMemoryNode,
    createMemoryLink,
    deleteMemoryEdge,
    // Enhanced memory system
    entities,
    setEntities,
    timeline,
    setTimeline,
    searchResults,
    setSearchResults,
    selectedEntityId,
    setSelectedEntityId,
    isSearching,
    setIsSearching,
    fetchEntities,
    fetchTimeline,
    performSearch,
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

function TokenStat({
  label,
  value,
  helper,
  tone = "neutral",
}: {
  label: string;
  value: string;
  helper: string;
  tone?: "neutral" | "positive" | "caution";
}) {
  const accent =
    tone === "positive"
      ? "text-emerald-300"
      : tone === "caution"
        ? "text-amber-300"
        : "text-zinc-100";

  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-950/70 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${accent}`}>{value}</div>
      <div className="text-[11px] text-zinc-500">{helper}</div>
    </div>
  );
}

interface MemoryGraphProps {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
  selectedNodeId?: string | null;
  onSelect?: (id: string) => void;
  zoom?: number;
}

function MemoryGraph({ nodes, edges, selectedNodeId, onSelect, zoom = 1 }: MemoryGraphProps) {
  const layout = React.useMemo(() => {
    const order: MemoryNode["kind"][] = ["stimulus", "event", "self", "behavior", "concept"];
    const buckets: Record<MemoryNode["kind"], MemoryNode[]> = {
      stimulus: [],
      concept: [],
      event: [],
      self: [],
      behavior: [],
    };
    nodes.forEach(node => buckets[node.kind].push(node));
    const maxNodes = Math.max(1, ...order.map(kind => buckets[kind].length));
    const columnSpacing = 160;
    const paddingX = 120;
    const columnWidth = 140;
    const verticalSpacing = Math.max(64, Math.min(110, Math.floor(520 / Math.max(1, Math.log2(maxNodes + 1)))));
    const height = Math.max(360, 120 + (maxNodes - 1) * verticalSpacing);
    const width = paddingX * 2 + columnSpacing * (order.length - 1);
    const xSlots: Record<MemoryNode["kind"], number> = {
      stimulus: paddingX,
      event: paddingX + columnSpacing,
      self: paddingX + columnSpacing * 2,
      behavior: paddingX + columnSpacing * 3,
      concept: paddingX + columnSpacing * 4,
    };
    const pos = new Map<string, { x: number; y: number }>();
    order.forEach(kind => {
      const arr = buckets[kind]
        .slice()
        .sort((a, b) => {
          const importanceDiff = (b.importance ?? 0) - (a.importance ?? 0);
          if (Math.abs(importanceDiff) > 0.0001) return importanceDiff;
          return (b.strength ?? 0) - (a.strength ?? 0);
        });
      arr.forEach((node, index) => {
        const y = 80 + index * verticalSpacing;
        pos.set(node.id, { x: xSlots[kind], y });
      });
    });
    return { width, height, pos, order, columnWidth, buckets, xSlots };
  }, [nodes]);

  const zoomTransform = React.useMemo(() => {
    if (Math.abs(zoom - 1) < 0.001) {
      return undefined;
    }
    const cx = layout.width / 2;
    const cy = layout.height / 2;
    return `translate(${cx} ${cy}) scale(${zoom}) translate(-${cx} -${cy})`;
  }, [layout.height, layout.width, zoom]);

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

  const columnPalette: Record<MemoryNode["kind"], { background: string; border: string; label: string; text: string }> = {
    stimulus: { background: "rgba(251, 191, 36, 0.08)", border: "rgba(251, 191, 36, 0.28)", label: "Stimuli", text: "#fbbf24" },
    event: { background: "rgba(34, 211, 238, 0.08)", border: "rgba(34, 211, 238, 0.26)", label: "Events", text: "#22d3ee" },
    self: { background: "rgba(16, 185, 129, 0.08)", border: "rgba(16, 185, 129, 0.32)", label: "Identity", text: "#34d399" },
    behavior: { background: "rgba(45, 212, 191, 0.08)", border: "rgba(45, 212, 191, 0.26)", label: "Behaviors", text: "#2dd4bf" },
    concept: { background: "rgba(167, 139, 250, 0.08)", border: "rgba(167, 139, 250, 0.26)", label: "Concepts", text: "#a78bfa" },
  };

  const nodeFillClass: Record<MemoryNode["kind"], string> = {
    stimulus: "fill-amber-500",
    concept: "fill-purple-500",
    event: "fill-cyan-500",
    self: "fill-emerald-500",
    behavior: "fill-teal-400",
  };

  // Get occurrence count for entity-linked nodes
  const getOccurrenceCount = (node: MemoryNode): number | undefined => {
    // If node has metadata with occurrence_count or entity_occurrence
    if (node.metadata) {
      if (typeof node.metadata.occurrence_count === 'number') {
        return node.metadata.occurrence_count;
      }
      if (typeof node.metadata.entity_occurrence === 'number') {
        return node.metadata.entity_occurrence;
      }
    }
    return undefined;
  };

  return (
    <div className="mt-4">
      <div className="max-h-[420px] overflow-auto rounded-xl border border-zinc-800/60 bg-zinc-950/40">
        <svg
          width={layout.width}
          height={layout.height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="block"
          role="list"
          aria-label="Nomous memory graph"
          style={{ minWidth: `${layout.width}px`, minHeight: `${layout.height}px` }}
        >
          <g transform={zoomTransform ?? undefined}>
            {layout.order.map(kind => {
              const palette = columnPalette[kind];
              const x = layout.xSlots[kind] - layout.columnWidth / 2;
              return (
                <g key={`column-${kind}`}>
                  <rect
                    x={x}
                    y={40}
                    width={layout.columnWidth}
                    height={layout.height - 80}
                    rx={28}
                    fill={palette.background}
                    stroke={palette.border}
                    strokeWidth={1.5}
                  />
                  <text
                    x={layout.xSlots[kind]}
                    y={62}
                    textAnchor="middle"
                    fontSize={12}
                    fill={palette.text}
                    letterSpacing="0.25em"
                  >
                    {palette.label.toUpperCase()}
                  </text>
                </g>
              );
            })}

            {edges.map(edge => {
              const from = layout.pos.get(edge.from);
              const to = layout.pos.get(edge.to);
              if (!from || !to) {
                return null;
              }
              const isActive = activeEdges.has(edge.id);
              const strokeOpacity = isActive ? 0.45 + edge.weight * 0.35 : 0.12;
              const strokeWidth = isActive ? 1.2 + edge.weight * 2 : 0.6 + edge.weight * 0.8;
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
              const isSelected = selectedNodeId === node.id;
              const isConnected = connectedNodes.has(node.id);
              const importanceBoost = typeof node.importance === "number" ? node.importance * 10 : 0;
              const baseRadius = 12 + node.strength * 9 + importanceBoost;
              const radius = Math.min(baseRadius, 34) + (isSelected ? 4 : 0);
              const stroke = isSelected ? "#34d399" : isConnected ? "#818cf8" : "#1f1f23";
              const strokeWidth = isSelected ? 3.2 : isConnected ? 2.2 : 1.2;
              const labelY = position.y - (radius + 12);
              const occurrenceCount = getOccurrenceCount(node);
              
              // Check if node has entity_type metadata for custom styling
              const entityType = node.metadata?.entity_type as EntityType | undefined;
              let fillClass = nodeFillClass[node.kind];
              
              // Override fill color if entity_type is specified
              if (entityType === "person") {
                fillClass = "fill-emerald-400";
              } else if (entityType === "place") {
                fillClass = "fill-cyan-400";
              } else if (entityType === "object") {
                fillClass = "fill-amber-400";
              } else if (entityType === "preference") {
                fillClass = "fill-purple-400";
              }
              
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
                  <circle
                    cx={position.x}
                    cy={position.y}
                    r={radius}
                    className={fillClass}
                    opacity={0.95}
                    stroke={stroke}
                    strokeWidth={strokeWidth}
                  />
                  
                  {/* Occurrence count badge */}
                  {occurrenceCount !== undefined && occurrenceCount > 1 && (
                    <>
                      <circle
                        cx={position.x + radius * 0.6}
                        cy={position.y - radius * 0.6}
                        r={Math.min(10, radius * 0.35)}
                        className="fill-zinc-900"
                        stroke="#34d399"
                        strokeWidth={1.5}
                      />
                      <text
                        x={position.x + radius * 0.6}
                        y={position.y - radius * 0.6}
                        textAnchor="middle"
                        dominantBaseline="central"
                        className="fill-emerald-400 text-[9px] font-semibold"
                      >
                        {occurrenceCount > 99 ? "99+" : occurrenceCount}
                      </text>
                    </>
                  )}
                  
                  <text
                    x={position.x}
                    y={labelY}
                    textAnchor="middle"
                    className="fill-zinc-50 text-[11px] drop-shadow"
                    fontWeight={500}
                  >
                    {node.label}
                  </text>
                  <title>{node.meaning || node.label}</title>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
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
            const categoryLabel = entry.category || entry.kind;
            const importanceLabel = typeof entry.importance === "number" ? `${Math.round(entry.importance * 100)}%` : null;
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
                  <span className="capitalize">{categoryLabel}</span>
                  <span>Strength {entry.strength.toFixed(2)}</span>
                </div>
                {importanceLabel ? (
                  <div className="text-[11px] text-zinc-500">Priority {importanceLabel}</div>
                ) : null}
                {entry.meaning ? (
                  <div className="mt-1 text-[11px] text-zinc-400 overflow-hidden text-ellipsis whitespace-nowrap">{entry.meaning}</div>
                ) : null}
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

interface ManualMemoryNodeDraft {
  id?: string;
  label: string;
  kind: MemoryNodeKind;
  description?: string;
  meaning?: string;
  tags?: string[];
  category?: string;
  source?: string;
  valence?: string;
  importance?: number;
  milestone?: boolean;
  strength?: number;
  confidence?: number;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

interface ManualMemoryControlsProps {
  onCreate: (draft: ManualMemoryNodeDraft) => void;
  onDelete: (id: string) => void;
}

function ManualMemoryControls({ onCreate, onDelete }: ManualMemoryControlsProps) {
  const [form, setForm] = useState({
    id: "",
    label: "",
    kind: "concept" as MemoryNodeKind,
    description: "",
    meaning: "",
    tags: "",
    category: "",
    source: "manual_entry",
    valence: "",
    importance: 0.5,
    strength: 1,
    milestone: false,
    timestamp: "",
    confidence: 0.75,
    note: "",
  });
  const [deleteId, setDeleteId] = useState("");
  const [status, setStatus] = useState<{ tone: "success" | "warn"; message: string } | null>(null);

  const handleFieldChange = useCallback((key: keyof typeof form, value: any) => {
    setForm(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleCreate = useCallback(() => {
    if (!form.label.trim()) {
      setStatus({ tone: "warn", message: "Provide a label before recording a manual memory." });
      return;
    }

    const tags = form.tags
      .split(",")
      .map(tag => tag.trim())
      .filter(Boolean);

    const payload: ManualMemoryNodeDraft = {
      id: form.id.trim() ? form.id.trim() : undefined,
      label: form.label.trim(),
      kind: form.kind,
      description: form.description.trim() || undefined,
      meaning: form.meaning.trim() || undefined,
      tags: tags.length ? tags : undefined,
      category: form.category.trim() || undefined,
      source: form.source.trim() || undefined,
      valence: form.valence.trim() || undefined,
      importance: form.importance,
      strength: form.strength,
      milestone: form.milestone,
      confidence: form.confidence,
      metadata: form.note.trim() ? { note: form.note.trim(), created_via: "ui_manual" } : undefined,
    };

    if (form.timestamp.trim()) {
      const parsed = new Date(form.timestamp);
      if (!Number.isNaN(parsed.getTime())) {
        payload.timestamp = parsed.toISOString();
      }
    }

    onCreate(payload);
    setStatus({ tone: "success", message: `Manual memory "${form.label.trim()}" queued for creation.` });
    setForm(prev => ({
      ...prev,
      id: "",
      label: "",
      description: "",
      meaning: "",
      tags: "",
      valence: "",
      note: "",
    }));
  }, [form, onCreate]);

  const handleManualDelete = useCallback(() => {
    const trimmed = deleteId.trim();
    if (!trimmed) {
      setStatus({ tone: "warn", message: "Enter a node ID to delete manually." });
      return;
    }
    onDelete(trimmed);
    setDeleteId("");
    setStatus({ tone: "success", message: `Delete requested for ${trimmed}.` });
  }, [deleteId, onDelete]);

  return (
    <Card className="border-zinc-800/60 bg-black/40">
      <CardContent className="space-y-4 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
          <CirclePlus className="h-4 w-4 text-emerald-400" />
          Manual Memory Controls
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Label</span>
            <Input
              value={form.label}
              onChange={event => handleFieldChange("label", event.target.value)}
              placeholder="Memory summary"
              className="bg-zinc-950/70"
            />
          </label>
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Memory type</span>
            <select
              value={form.kind}
              onChange={event => handleFieldChange("kind", event.target.value as MemoryNodeKind)}
              className="w-full rounded-md border border-zinc-800 bg-zinc-950/70 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
            >
              <option value="stimulus">Stimulus</option>
              <option value="concept">Concept</option>
              <option value="event">Event</option>
              <option value="behavior">Behavior</option>
              <option value="self">Identity</option>
            </select>
          </label>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Custom node ID (optional)</span>
            <Input
              value={form.id}
              onChange={event => handleFieldChange("id", event.target.value)}
              placeholder="node:manual:xyz"
              className="bg-zinc-950/70"
            />
          </label>
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Source</span>
            <Input
              value={form.source}
              onChange={event => handleFieldChange("source", event.target.value)}
              placeholder="manual_entry"
              className="bg-zinc-950/70"
            />
          </label>
        </div>

        <div className="space-y-2">
          <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Description</span>
          <textarea
            value={form.description}
            onChange={event => handleFieldChange("description", event.target.value)}
            rows={2}
            placeholder="How should the runtime remember this?"
            className="w-full rounded-md border border-zinc-800 bg-zinc-950/70 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
          />
        </div>

        <div className="space-y-2">
          <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Meaning / insight</span>
          <textarea
            value={form.meaning}
            onChange={event => handleFieldChange("meaning", event.target.value)}
            rows={2}
            placeholder="Semantic takeaway..."
            className="w-full rounded-md border border-zinc-800 bg-zinc-950/70 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Tags</span>
            <Input
              value={form.tags}
              onChange={event => handleFieldChange("tags", event.target.value)}
              placeholder="context, location"
              className="bg-zinc-950/70"
            />
          </label>
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Category</span>
            <Input
              value={form.category}
              onChange={event => handleFieldChange("category", event.target.value)}
              placeholder="safety, user, workflow..."
              className="bg-zinc-950/70"
            />
          </label>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Valence</span>
            <Input
              value={form.valence}
              onChange={event => handleFieldChange("valence", event.target.value)}
              placeholder="positive, neutral..."
              className="bg-zinc-950/70"
            />
          </label>
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Timestamp</span>
            <Input
              type="datetime-local"
              value={form.timestamp}
              onChange={event => handleFieldChange("timestamp", event.target.value)}
              className="bg-zinc-950/70"
            />
          </label>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Importance</span>
              <span className="font-mono text-emerald-300">{Math.round(form.importance * 100)}%</span>
            </div>
            <Slider
              defaultValue={[Math.round(form.importance * 100)]}
              onValueChange={value => handleFieldChange("importance", value[0] / 100)}
              min={0}
              max={100}
              step={5}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Strength</span>
              <span className="font-mono text-cyan-300">{form.strength.toFixed(1)}</span>
            </div>
            <Slider
              defaultValue={[Math.round(form.strength * 10)]}
              onValueChange={value => handleFieldChange("strength", value[0] / 10)}
              min={2}
              max={50}
              step={1}
            />
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm text-zinc-300">
            <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Confidence</span>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={form.confidence}
              onChange={event => handleFieldChange("confidence", Math.max(0, Math.min(1, Number(event.target.value))))}
              className="bg-zinc-950/70"
            />
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800/60 bg-black/30 px-3 py-2 text-sm text-zinc-200">
            <div>
              <div className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Milestone</div>
              <div className="text-xs text-zinc-400">Surface prominently in graph</div>
            </div>
            <Switch checked={form.milestone} onCheckedChange={value => handleFieldChange("milestone", value)} />
          </label>
        </div>

        <div className="space-y-1">
          <span className="text-[11px] uppercase tracking-[0.3em] text-zinc-500">Context note</span>
          <textarea
            value={form.note}
            onChange={event => handleFieldChange("note", event.target.value)}
            rows={2}
            placeholder="Any JSON-safe note will be attached to metadata."
            className="w-full rounded-md border border-zinc-800 bg-zinc-950/70 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleCreate} disabled={!form.label.trim()} className="bg-emerald-600/90 text-white hover:bg-emerald-500/90 disabled:opacity-50">
            <CirclePlus className="mr-2 h-4 w-4" />
            Record memory
          </Button>
          <Button
            type="button"
            variant="secondary"
            onClick={() => setForm(prev => ({ ...prev, label: "", description: "", meaning: "", tags: "", valence: "", note: "", id: "" }))}
            className="text-xs text-zinc-300"
          >
            Clear fields
          </Button>
        </div>

        <div className="rounded-lg border border-zinc-800/60 bg-zinc-950/50 p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.3em] text-zinc-500">
            <Trash2 className="h-3.5 w-3.5 text-red-400" />
            Manual delete
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              value={deleteId}
              onChange={event => setDeleteId(event.target.value)}
              placeholder="memory node id"
              className="bg-zinc-950/70"
            />
            <Button type="button" variant="secondary" onClick={handleManualDelete} className="bg-red-600/80 text-white hover:bg-red-500/80">
              Delete node
            </Button>
          </div>
        </div>

        {status ? (
          <div
            className={`rounded-md border px-3 py-2 text-xs ${
              status.tone === "success"
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                : "border-amber-500/40 bg-amber-500/10 text-amber-100"
            }`}
          >
            {status.message}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

type StreamView = "thoughts" | "speech" | "signals";

interface ConversationStreamPanelProps {
  thoughtLines: string[];
  speechLines: string[];
  systemLines: string[];
}

function ConversationStreamPanel({ thoughtLines, speechLines, systemLines }: ConversationStreamPanelProps) {
  const [activeView, setActiveView] = useState<StreamView>("thoughts");

  const streams = [
    {
      id: "thoughts" as const,
      title: "Cognitive Stream",
      description: "Trace intermediate reasoning, decisions, and planning notes in real time.",
      icon: Brain,
      lines: thoughtLines,
      empty: "Waiting for thoughts...",
      accent: "text-purple-300",
      shortLabel: "Think",
      height: "h-72",
    },
    {
      id: "speech" as const,
      title: "Speech Commitments",
      description: "Monitor which responses were finalized and handed to speech synthesis.",
      icon: Volume2,
      lines: speechLines,
      empty: "No speech prepared yet.",
      accent: "text-emerald-300",
      shortLabel: "Speak",
      height: "h-60",
    },
    {
      id: "signals" as const,
      title: "System & Prompt Signals",
      description: "System prompts, status transitions, and any runtime directives from the core.",
      icon: Radio,
      lines: systemLines,
      empty: "No system activity captured yet.",
      accent: "text-sky-300",
      shortLabel: "Signals",
      height: "h-60",
    },
  ];

  const active = streams.find(stream => stream.id === activeView) ?? streams[0];
  const ActiveIcon = active.icon;

  return (
    <Card className="bg-zinc-900/70 border-zinc-800/60">
      <CardContent className="space-y-4 p-4 text-zinc-200">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <ActiveIcon className="h-4 w-4" />
              <span className="font-semibold">{active.title}</span>
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">{active.description}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {streams.map(stream => (
              <button
                type="button"
                key={stream.id}
                onClick={() => setActiveView(stream.id)}
                className={`rounded-full border px-3 py-1 text-xs transition ${
                  activeView === stream.id
                    ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-100"
                    : "border-zinc-700/60 bg-black/30 text-zinc-400 hover:border-zinc-600/80"
                }`}
              >
                {stream.shortLabel}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800/60 bg-black/60">
          <div className={`${active.height} overflow-y-auto p-4 font-mono text-xs leading-relaxed text-zinc-100/95 whitespace-pre-wrap`}>
            {active.lines.length > 0 ? (
              active.lines.map((line, index) => (
                <div key={index} className={`mb-1 last:mb-0 ${active.accent}`}>
                  {line}
                </div>
              ))
            ) : (
              <div className="mt-12 text-center text-zinc-500">{active.empty}</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface SttMonitorCardProps {
  lines: string[];
  micOn: boolean;
  micStatus: "positive" | "neutral" | "negative";
  micStatusLabel: string;
  micSensitivity: number;
  sttEnabled: boolean;
  vu: number;
  setMic: (on: boolean) => void;
  updateSettings: (patch: Partial<ControlSettings>) => void;
}

function SttMonitorCard({ lines, micOn, micStatus, micStatusLabel, micSensitivity, sttEnabled, vu, setMic, updateSettings }: SttMonitorCardProps) {
  const toggleMic = () => {
    const next = !micOn;
    updateSettings({ microphoneEnabled: next });
    setMic(next);
  };

  const vuPercent = Math.round(Math.max(0, Math.min(1, vu)) * 100);

  const getAccentForLine = (line: string) => {
    if (line.includes("Listening:")) return "text-amber-300";
    if (line.includes("Captured:")) return "text-emerald-300";
    if (line.includes("Dispatched:")) return "text-sky-300";
    return "text-zinc-200";
  };

  const micStatusAccent =
    micStatus === "positive" ? "text-emerald-300" : micStatus === "negative" ? "text-red-300" : "text-zinc-300";

  return (
    <Card className="bg-zinc-900/70 border-zinc-800/60">
      <CardContent className="space-y-4 p-4 text-zinc-200">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AudioWaveform className="h-4 w-4" />
            <span className="font-semibold">Speech Input Monitor</span>
          </div>
          <button
            type="button"
            onClick={toggleMic}
            className={`inline-flex items-center gap-2 rounded-lg px-3 py-1 text-sm transition ${
              micOn
                ? "bg-zinc-800/80 text-zinc-100 hover:bg-zinc-700/80"
                : "bg-emerald-600/90 text-white hover:bg-emerald-500/90"
            }`}
          >
            {micOn ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            {micOn ? "Stop Listening" : "Enable Mic"}
          </button>
        </div>

        <div className="grid gap-3 text-xs text-zinc-400 sm:grid-cols-3">
          <div>
            Mic • <span className={`font-medium ${micStatusAccent}`}>{micStatusLabel}</span>
          </div>
          <div>
            STT • <span className={sttEnabled ? "text-emerald-300" : "text-zinc-500"}>{sttEnabled ? "Enabled" : "Disabled"}</span>
          </div>
          <div>Sensitivity • {micSensitivity}%</div>
        </div>

        <div className="flex items-center gap-3">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
            <div
              className={`h-full transition-all ${
                micStatus === "positive" ? "bg-emerald-500" : micStatus === "negative" ? "bg-red-500" : "bg-sky-500/70"
              }`}
              style={{ width: `${vuPercent}%` }}
            />
          </div>
          <span className="w-16 text-right text-[11px] text-zinc-500">{vuPercent}%</span>
        </div>

        <div className="rounded-xl border border-zinc-800/60 bg-black/60">
          {!sttEnabled && (
            <div className="border-b border-zinc-800/60 bg-amber-500/10 px-4 py-2 text-xs text-amber-200">
              ⚠️ STT is disabled. Enable it in the Devices tab to transcribe speech.
            </div>
          )}
          {!micOn && sttEnabled && (
            <div className="border-b border-zinc-800/60 bg-sky-500/10 px-4 py-2 text-xs text-sky-200">
              🎤 Click "Enable Mic" above to start capturing audio.
            </div>
          )}
          <div className="h-48 overflow-y-auto p-4 font-mono text-xs leading-relaxed text-zinc-100/95 whitespace-pre-wrap">
            {lines.length > 0 ? (
              lines.map((line, index) => (
                <div key={index} className={`mb-1 last:mb-0 ${getAccentForLine(line)}`}>
                  {line}
                </div>
              ))
            ) : (
              <div className="mt-10 text-center text-zinc-500">No speech captured yet.</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface ConversationQuickSettingsProps {
  settings: ControlSettings;
  micOn: boolean;
  setMic: (on: boolean) => void;
  updateSettings: (patch: Partial<ControlSettings>) => void;
  push: (message: any) => void;
}

function ConversationQuickSettings({ settings, micOn, setMic, updateSettings, push }: ConversationQuickSettingsProps) {
  const handleMicChange = (value: boolean) => {
    updateSettings({ microphoneEnabled: value });
    setMic(value);
  };

  const handleSttChange = (value: boolean) => {
    updateSettings({ sttEnabled: value });
    push({ type: "toggle", what: "stt", value });
  };

  const handleTtsChange = (value: boolean) => {
    updateSettings({ ttsEnabled: value });
    push({ type: "toggle", what: "tts", value });
  };

  const handleSpeakerChange = (value: boolean) => {
    updateSettings({ speakerEnabled: value });
    push({ type: "toggle", what: "speaker", value });
  };

  return (
    <Card className="bg-zinc-900/70 border-zinc-800/60">
      <CardContent className="space-y-4 p-4 text-zinc-200">
        <div className="flex items-center gap-2">
          <Wrench className="h-4 w-4" />
          <span className="font-semibold">Conversation Controls</span>
        </div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          Toggle realtime capture and response paths without leaving the conversation view.
        </p>
        <div className="space-y-3">
          <QuickToggleRow
            icon={Mic}
            label="Microphone capture"
            helper="Stream audio into the runtime for transcription."
            checked={micOn}
            onCheckedChange={handleMicChange}
          />
          <QuickToggleRow
            icon={AudioWaveform}
            label="Speech-to-text"
            helper="Transcribe live voice input before routing to the model."
            checked={settings.sttEnabled}
            onCheckedChange={handleSttChange}
          />
          <QuickToggleRow
            icon={Volume2}
            label="Voice replies"
            helper="Allow the assistant to speak its responses aloud."
            checked={settings.ttsEnabled}
            onCheckedChange={handleTtsChange}
          />
          <QuickToggleRow
            icon={Speaker}
            label="Speaker output"
            helper="Send synthesized audio to the connected speakers."
            checked={settings.speakerEnabled}
            onCheckedChange={handleSpeakerChange}
          />
        </div>
      </CardContent>
    </Card>
  );
}

interface QuickToggleRowProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  helper?: string;
  checked: boolean;
  onCheckedChange: (value: boolean) => void;
}

function QuickToggleRow({ icon: Icon, label, helper, checked, onCheckedChange }: QuickToggleRowProps) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-800/60 bg-black/40 px-3 py-2">
      <div className="flex items-center gap-3">
        <div className="grid h-9 w-9 place-items-center rounded-lg border border-zinc-800/60 bg-zinc-900/80 text-zinc-300">
          <Icon className="h-4 w-4" />
        </div>
        <div>
          <div className="text-sm text-zinc-200">{label}</div>
          {helper ? <div className="text-[11px] text-zinc-500">{helper}</div> : null}
        </div>
      </div>
      <Switch checked={checked} onCheckedChange={onCheckedChange} aria-label={label} />
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

const MAX_SPEECH_PAYLOAD_DEPTH = 5;

/**
 * Normalizes a candidate value that may contain speech text into a single trimmed string.
 *
 * @param candidate - The raw candidate value that may represent speech text.
 * @param options - Normalization options.
 * @param options.skipStatusWords - When true, ignore status keywords such as "thinking".
 * @returns A normalized speech string or null if the candidate cannot be converted.
 *
 * @example
 * normalizeSpeechCandidate([" Hello ", "world  "]); // "Hello world"
 *
 * @example
 * normalizeSpeechCandidate("thinking", { skipStatusWords: true }); // null
 */
function normalizeSpeechCandidate(candidate: unknown, { skipStatusWords = false } = {}): string | null {
  if (typeof candidate === "string") {
    const trimmed = candidate.trim();
    if (!trimmed) {
      return null;
    }
    if (skipStatusWords && STATUS_KEYWORDS.has(trimmed.toLowerCase())) {
      return null;
    }
    return trimmed;
  }

  if (Array.isArray(candidate)) {
    const joined = candidate
      // Non-string array values are treated as empty strings intentionally.
      .map(part => (typeof part === "string" ? part.trim() : ""))
      .filter(Boolean)
      .join(" ")
      .trim();
    return joined || null;
  }

  return null;
}

/**
 * Extracts human-readable speech text from a nested payload structure.
 *
 * @param payload - The payload to inspect for speech text.
 * @param depth - Current recursion depth used to cap traversal.
 * @param visited - A set tracking visited objects to prevent circular references.
 * @returns The extracted speech string or null when unavailable.
 *
 * @example
 * extractSpeechText({ text: "Hello" }); // "Hello"
 *
 * @example
 * extractSpeechText({ payload: { message: ["Hi", "there"] } }); // "Hi there"
 */
function extractSpeechText(payload: unknown, depth = 0, visited?: Set<unknown>): string | null {
  if (depth >= MAX_SPEECH_PAYLOAD_DEPTH) {
    return null;
  }

  const direct = normalizeSpeechCandidate(payload);
  if (direct) {
    return direct;
  }

  if (payload && typeof payload === "object") {
    const visitSet = visited ?? new Set<unknown>();
    if (visitSet.has(payload)) {
      return null;
    }
    visitSet.add(payload);

    const obj = payload as Record<string, unknown>;
    const fromFields = normalizeSpeechCandidate(obj.text)
      ?? normalizeSpeechCandidate(obj.detail)
      ?? normalizeSpeechCandidate(obj.message)
      ?? normalizeSpeechCandidate(obj.content)
      ?? normalizeSpeechCandidate(obj.segments);
    if (fromFields) {
      return fromFields;
    }

    const valueCandidate = normalizeSpeechCandidate(obj.value, { skipStatusWords: true });
    if (valueCandidate) {
      return valueCandidate;
    }

    const nestedKeys: string[] = ["payload", "data"];
    for (const key of nestedKeys) {
      if (key in obj && obj[key] && obj[key] !== payload) {
        const nested = extractSpeechText(obj[key], depth + 1, visitSet);
        if (nested) {
          return nested;
        }
      }
    }
  }

  return null;
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

function normalizeStreamText(text: string): string {
  if (!text) {
    return "";
  }
  return text.replace(/\s+/g, " ").replace(/\s([,.;!?])/g, "$1").trim();
}

function splitThoughtMessage(text: string): { prefix: string | null; detail: string } {
  const colonIndex = text.indexOf(":");
  if (colonIndex === -1) {
    return { prefix: null, detail: text };
  }
  const prefix = text.slice(0, colonIndex).trim();
  const detail = text.slice(colonIndex + 1).trim();
  return { prefix: prefix.length > 0 ? prefix : null, detail };
}

function combineThoughtDetails(existing: string, incoming: string): string {
  const parts = [existing, incoming].map(part => part.trim()).filter(Boolean);
  if (parts.length === 0) {
    return "";
  }
  return normalizeStreamText(parts.join(" "));
}

function formatThoughtDisplay(prefix: string | null, detail: string): string {
  const normalizedDetail = normalizeStreamText(detail);
  if (prefix && normalizedDetail) {
    return `${prefix}: ${normalizedDetail}`;
  }
  if (prefix) {
    return prefix;
  }
  return normalizedDetail;
}

interface MemoryDetailCardProps {
  detail: MemoryNodeDetail | null;
  onUpdate: (id: string, patch: Record<string, unknown>) => void;
  onDelete: (id: string) => void;
  onCreateLink: (
    fromId: string,
    toId: string,
    weight: number,
    relationship?: string,
    context?: Record<string, unknown>,
  ) => void;
}

function MemoryDetailCard({ detail, onUpdate, onDelete, onCreateLink }: MemoryDetailCardProps) {
  const [form, setForm] = useState(() => ({
    label: detail?.label ?? "",
    description: detail?.description ?? "",
    meaning: detail?.meaning ?? "",
    tags: detail ? detail.tags.join(", ") : "",
    category: detail?.category ?? "",
    importance: detail?.importance ?? 0,
    valence: detail?.valence ?? "",
    milestone: Boolean(detail?.milestone),
  }));
  const [linkForm, setLinkForm] = useState({ target: "", weight: 1, relationship: "association", note: "" });

  useEffect(() => {
    setForm({
      label: detail?.label ?? "",
      description: detail?.description ?? "",
      meaning: detail?.meaning ?? "",
      tags: detail ? detail.tags.join(", ") : "",
      category: detail?.category ?? "",
      importance: detail?.importance ?? 0,
      valence: detail?.valence ?? "",
      milestone: Boolean(detail?.milestone),
    });
    setLinkForm({ target: "", weight: 1, relationship: "association", note: "" });
  }, [detail?.id]);

  const handleFormChange = useCallback((key: keyof typeof form, value: any) => {
    setForm(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleSave = useCallback(() => {
    if (!detail) return;
    const patch: Record<string, unknown> = {};
    if (form.label.trim() && form.label.trim() !== detail.label) patch.label = form.label.trim();
    if ((form.description || "").trim() !== (detail.description || "")) patch.description = form.description.trim();
    if ((form.meaning || "").trim() !== (detail.meaning || "")) patch.meaning = form.meaning.trim();
    const normalisedTags = form.tags
      .split(",")
      .map(tag => tag.trim())
      .filter(tag => tag.length > 0);
    if (normalisedTags.join(",") !== detail.tags.join(",")) patch.tags = normalisedTags;
    if ((form.category || "") !== (detail.category || "")) patch.category = form.category || null;
    const normalisedImportance = Math.max(0, Math.min(1, Number(form.importance)));
    if (Number.isFinite(normalisedImportance) && Math.abs(normalisedImportance - (detail.importance ?? 0)) > 0.01) {
      patch.importance = normalisedImportance;
    }
    if ((form.valence || "") !== (detail.valence || "")) patch.valence = form.valence || null;
    if (form.milestone !== Boolean(detail.milestone)) patch.milestone = form.milestone;
    if (Object.keys(patch).length === 0) {
      return;
    }
    onUpdate(detail.id, patch);
  }, [detail, form, onUpdate]);

  const isDirty = useMemo(() => {
    if (!detail) return false;
    if (form.label.trim() !== detail.label) return true;
    if ((form.description || "").trim() !== (detail.description || "")) return true;
    if ((form.meaning || "").trim() !== (detail.meaning || "")) return true;
    if (form.category !== (detail.category || "")) return true;
    if (Math.abs((form.importance ?? 0) - (detail.importance ?? 0)) > 0.01) return true;
    if ((form.valence || "") !== (detail.valence || "")) return true;
    if (form.milestone !== Boolean(detail.milestone)) return true;
    const currentTags = form.tags
      .split(",")
      .map(tag => tag.trim())
      .filter(Boolean)
      .join(",");
    if (currentTags !== detail.tags.join(",")) return true;
    return false;
  }, [detail, form]);

  const handleDelete = useCallback(() => {
    if (!detail) return;
    onDelete(detail.id);
  }, [detail, onDelete]);

  const handleLinkCreate = useCallback(() => {
    if (!detail) return;
    const target = linkForm.target.trim();
    if (!target) return;
    const context = linkForm.note ? { note: linkForm.note } : undefined;
    onCreateLink(detail.id, target, Math.max(0.1, Math.min(5, Number(linkForm.weight) || 1)), linkForm.relationship || undefined, context);
    setLinkForm(prev => ({ ...prev, target: "", note: "" }));
  }, [detail, linkForm, onCreateLink]);

  if (!detail) {
    return (
      <Card className="h-full bg-zinc-900/70 border-dashed border-zinc-800/60">
        <CardContent className="flex h-full items-center justify-center p-4 text-center text-sm text-zinc-400">
          Select a memory node from the graph or insight lists to inspect and edit its learned behavior and associations.
        </CardContent>
      </Card>
    );
  }

  const formattedTimestamp = formatTimestamp(detail.timestamp);
  const formattedCreated = formatTimestamp(detail.createdAt);
  const formattedUpdated = formatTimestamp(detail.updatedAt);
  const formattedAccessed = formatTimestamp(detail.lastAccessed);
  const metadata = (detail.metadata ?? {}) as Record<string, unknown>;
  const cues = Array.isArray((metadata as any).cues) ? ((metadata as any).cues as string[]) : [];
  const expectation = typeof (metadata as any).expectation === "string" ? String((metadata as any).expectation) : undefined;
  const persona = (metadata as any).persona ? String((metadata as any).persona) : undefined;

  return (
    <Card className="h-full bg-zinc-900/70 border-zinc-800/60">
      <CardContent className="h-full space-y-4 p-4 text-sm text-zinc-200">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Selected Memory</div>
            <h3 className="mt-1 text-xl font-semibold text-zinc-100">{detail.label}</h3>
            {detail.meaning ? <div className="mt-1 text-xs text-zinc-400">{detail.meaning}</div> : null}
          </div>
          <div className="flex flex-col items-end gap-2">
            <Badge className="bg-zinc-800/80 text-zinc-100 border border-zinc-700/60 capitalize">{detail.kind}</Badge>
            <div className="flex items-center gap-2 text-[11px] text-zinc-400">
              <span>Milestone</span>
              <Switch checked={form.milestone} onCheckedChange={value => handleFormChange("milestone", value)} />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 text-xs">
          <MemorySummaryStat label="Strength" value={detail.strength.toFixed(2)} helper="Association weight" />
          <MemorySummaryStat label="Connections" value={String(detail.connections)} helper="Linked nodes" />
          {detail.confidence !== undefined ? (
            <MemorySummaryStat label="Confidence" value={`${Math.round(detail.confidence * 100)}%`} helper="Runtime certainty" />
          ) : null}
          {formattedTimestamp ? (
            <MemorySummaryStat label="Observed" value={formattedTimestamp} helper="Latest reinforcement" />
          ) : null}
          {formattedCreated ? (
            <MemorySummaryStat label="Created" value={formattedCreated} helper="First recorded" />
          ) : null}
          {formattedUpdated ? (
            <MemorySummaryStat label="Updated" value={formattedUpdated} helper="Metadata refresh" />
          ) : null}
          {formattedAccessed ? (
            <MemorySummaryStat label="Accessed" value={formattedAccessed} helper="Last access" />
          ) : null}
          {detail.source ? (
            <MemorySummaryStat label="Source" value={detail.source} helper="Input channel" />
          ) : null}
        </div>

        <div className="space-y-3 text-xs">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Label</span>
            <input
              value={form.label}
              onChange={event => handleFormChange("label", event.target.value)}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Description</span>
            <textarea
              value={form.description}
              onChange={event => handleFormChange("description", event.target.value)}
              rows={3}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Meaning / Interpretation</span>
            <textarea
              value={form.meaning}
              onChange={event => handleFormChange("meaning", event.target.value)}
              rows={2}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
            />
          </label>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Tags</span>
              <input
                value={form.tags}
                onChange={event => handleFormChange("tags", event.target.value)}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Category</span>
              <input
                value={form.category}
                onChange={event => handleFormChange("category", event.target.value)}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Importance</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={form.importance}
                onChange={event => handleFormChange("importance", Number(event.target.value))}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Valence</span>
              <input
                value={form.valence}
                onChange={event => handleFormChange("valence", event.target.value)}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
              />
            </label>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Button onClick={handleSave} disabled={!isDirty} className="bg-emerald-600/90 text-white hover:bg-emerald-500/90 disabled:opacity-50">
            Save changes
          </Button>
          <Button variant="secondary" onClick={handleDelete} className="bg-red-600/80 text-white hover:bg-red-500/80">
            Delete memory
          </Button>
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
                    <span className="text-[11px] text-zinc-400">
                      {relation.relationship ? relation.relationship.replace(/_/g, " ") : relation.direction === "outbound" ? "influences" : "influenced by"}
                    </span>
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

        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Add manual link</div>
          <div className="grid gap-2 sm:grid-cols-4">
            <input
              value={linkForm.target}
              onChange={event => setLinkForm(prev => ({ ...prev, target: event.target.value }))}
              placeholder="Target node ID"
              className="rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none sm:col-span-2"
            />
            <input
              type="number"
              min={0.1}
              max={5}
              step={0.1}
              value={linkForm.weight}
              onChange={event => setLinkForm(prev => ({ ...prev, weight: Number(event.target.value) }))}
              className="rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
              placeholder="Weight"
            />
            <input
              value={linkForm.relationship}
              onChange={event => setLinkForm(prev => ({ ...prev, relationship: event.target.value }))}
              placeholder="Relationship"
              className="rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
            />
          </div>
          <textarea
            value={linkForm.note}
            onChange={event => setLinkForm(prev => ({ ...prev, note: event.target.value }))}
            rows={2}
            placeholder="Optional context / note"
            className="w-full rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
          />
          <div className="flex justify-end">
            <Button onClick={handleLinkCreate} className="bg-cyan-600/80 text-white hover:bg-cyan-500/80" disabled={!linkForm.target.trim()}>
              Link memory
            </Button>
          </div>
        </div>

        {detail.metadata ? (
          <div className="space-y-2">
            <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Behavioral context</div>
            {cues.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {cues.map(cue => (
                  <Badge key={cue} className="bg-teal-500/10 text-teal-200 border border-teal-500/30">{cue}</Badge>
                ))}
              </div>
            ) : null}
            {expectation ? (
              <div className="rounded-lg border border-zinc-800/60 bg-black/40 p-3 text-xs text-zinc-300">
                <span className="font-semibold text-zinc-200">Expectation:</span> {expectation}
              </div>
            ) : null}
            {persona ? (
              <div className="text-xs text-zinc-400">Persona alignment: {persona}</div>
            ) : null}
          </div>
        ) : null}
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

  const {
    llmModelPath,
    visionModelPath,
    audioModelPath,
    sttModelPath,
    modelStrategy,
    systemPrompt,
    thinkingPrompt,
    modelDirectory,
  } = state.settings;

  const [modelScanPending, setModelScanPending] = useState(false);
  const [modelScanError, setModelScanError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("runtime");

  useEffect(() => {
    if (state.modelCatalogError) {
      setModelScanError(state.modelCatalogError);
    } else {
      setModelScanError(null);
    }
    setModelScanPending(false);
  }, [state.modelCatalog, state.modelCatalogError]);

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

  const handleModelDirectoryCommit = useCallback(
    (raw: string) => {
      const trimmed = raw.trim();
      updateSettings({ modelDirectory: trimmed });
      if (!trimmed) {
        setModelScanPending(false);
        setModelScanError(null);
        setState(prev => ({ ...prev, modelCatalog: [] }));
        return;
      }
      setModelScanPending(true);
      setModelScanError(null);
      push({ type: "scan_models", directory: trimmed });
    },
    [push, setState, updateSettings]
  );

  const handleModelRescan = useCallback(() => {
    const trimmed = modelDirectory.trim();
    if (!trimmed) {
      setModelScanError("Select a model directory first.");
      return;
    }
    setModelScanPending(true);
    setModelScanError(null);
    push({ type: "scan_models", directory: trimmed });
  }, [modelDirectory, push]);

  const handleModelSelect = useCallback(
    (entry: ModelCatalogEntry) => {
      const base = modelDirectory.trim();
      const resolved = entry.path && entry.path.length > 0
        ? entry.path
        : base
          ? `${base.replace(/[\\/]+$/, "")}/${entry.name}`
          : entry.name;
      if (!resolved) {
        setModelScanError("Unable to resolve model path. Update the directory path first.");
        return;
      }
      updateSettings({ llmModelPath: resolved, modelStrategy: "custom" });
      beginModelSwitch(`Loading ${entry.name}`);
      push({ type: "param", key: "llm_model_path", value: resolved });
    },
    [beginModelSwitch, modelDirectory, push, updateSettings]
  );

  const handlePromptCommit = useCallback(
    (key: "system_prompt" | "thinking_prompt", value: string) => {
      push({ type: "param", key, value: value.trim() });
    },
    [push]
  );

  const catalogEntries = state.modelCatalog;
  const selectedModelPath = state.settings.llmModelPath;

  if (!open) return null;

  const handleToggle = (key: keyof ControlSettings, value: boolean, action?: () => void) => {
    updateSettings({ [key]: value } as Partial<ControlSettings>);
    action?.();
  };

  const sliderKey = (label: string, value: number) => `${label}-${value}`;

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4 py-6">
      <div className="relative w-full max-w-5xl max-h-[90vh] overflow-hidden rounded-3xl border border-zinc-800/70 bg-zinc-950/95 shadow-[0_30px_120px_rgba(0,0,0,0.45)] flex flex-col">
        <div className="flex items-start justify-between border-b border-zinc-800/60 px-6 py-4">
          <div>
            <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Control Center</div>
            <h2 className="text-2xl font-semibold text-zinc-100">Runtime, Devices &amp; LLM Settings</h2>
            <p className="text-sm text-zinc-400">Configure every input, output, and model path for Nomous from a single glassmorphic panel.</p>
          </div>
          <Button variant="secondary" className="bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-100" onClick={onClose}>Close</Button>
        </div>
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          defaultValue="runtime"
          className="flex h-full flex-col px-6 pb-6 pt-4"
        >
          <TabsList className="w-full justify-start flex-shrink-0 mb-4 gap-2">
            <TabsTrigger value="runtime" className="rounded-full border border-zinc-800/60 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-300 data-[state=active]:border-emerald-500/40 data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-200">Runtime</TabsTrigger>
            <TabsTrigger value="prompts" className="rounded-full border border-zinc-800/60 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-300 data-[state=active]:border-emerald-500/40 data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-200">Prompts</TabsTrigger>
            <TabsTrigger value="devices" className="rounded-full border border-zinc-800/60 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-300 data-[state=active]:border-emerald-500/40 data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-200">Devices</TabsTrigger>
            <TabsTrigger value="audio" className="rounded-full border border-zinc-800/60 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-300 data-[state=active]:border-emerald-500/40 data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-200">Audio</TabsTrigger>
            <TabsTrigger value="vision" className="rounded-full border border-zinc-800/60 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-300 data-[state=active]:border-emerald-500/40 data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-200">Vision</TabsTrigger>
            <TabsTrigger value="models" className="rounded-full border border-zinc-800/60 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-300 data-[state=active]:border-emerald-500/40 data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-200">Models</TabsTrigger>
          </TabsList>
          <div className="flex-1 overflow-hidden min-h-0">
            <TabsContent value="runtime" className="h-full overflow-y-auto">
              <div className="space-y-6 pb-8 pr-1">
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
            </TabsContent>

            <TabsContent value="prompts" className="h-full overflow-y-auto">
              <div className="space-y-6 pb-8 pr-1">
                <section>
                  <Card className="bg-zinc-900/60 border-zinc-800/60">
                    <CardContent className="space-y-4 p-4">
                      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-zinc-100">System Prompts</h3>
                          <p className="text-xs text-zinc-400">Define the persona and internal reasoning guidance used by the language model.</p>
                        </div>
                        <div className="text-[11px] uppercase tracking-[0.3em] text-emerald-300/80">Live runtime sync</div>
                      </div>
                      <div className="grid gap-4 lg:grid-cols-2">
                        <label className="flex flex-col gap-2">
                          <span className="text-xs uppercase tracking-wide text-zinc-400">System Persona Prompt</span>
                          <textarea
                            value={systemPrompt}
                            onChange={event => updateSettings({ systemPrompt: event.target.value })}
                            onBlur={event => handlePromptCommit("system_prompt", event.target.value)}
                            placeholder="Set the global behavior for Nomous..."
                            className="min-h-[140px] w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/80 focus:outline-none focus:ring-0"
                          />
                        </label>
                        <label className="flex flex-col gap-2">
                          <span className="text-xs uppercase tracking-wide text-zinc-400">Thinking Prompt</span>
                          <textarea
                            value={thinkingPrompt}
                            onChange={event => updateSettings({ thinkingPrompt: event.target.value })}
                            onBlur={event => handlePromptCommit("thinking_prompt", event.target.value)}
                            placeholder="Guide internal reasoning, tool usage, and reflection..."
                            className="min-h-[140px] w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/80 focus:outline-none focus:ring-0"
                          />
                        </label>
                      </div>
                      <p className="text-xs text-zinc-500">
                        Updates are saved locally and streamed to the runtime. Reload the active language model to bake the new prompts into its context.
                      </p>
                    </CardContent>
                  </Card>
                </section>
              </div>
            </TabsContent>
            <TabsContent value="devices" className="h-full overflow-y-auto">
              <div className="space-y-6 pb-8 pr-1">
                <section>
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
                </section>
              </div>
            </TabsContent>
            <TabsContent value="audio" className="h-full overflow-y-auto">
              <div className="space-y-6 pb-8 pr-1">
                <section>
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
              </div>
            </TabsContent>
            <TabsContent value="vision" className="h-full overflow-y-auto">
              <div className="space-y-6 pb-8 pr-1">
                <section>
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
                            {["1920x1080","1280x720","1024x576","640x480"].map(res => <option key={res} value={res}>{res}</option>)}
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
                </section>
              </div>
            </TabsContent>
            <TabsContent value="models" className="h-full overflow-y-auto">
              <div className="space-y-6 pb-8 pr-1">
                <section>
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
                          <div className="flex flex-col gap-2 md:flex-row md:items-center md:gap-3">
                            <span className="flex-1 text-xs uppercase tracking-wide text-zinc-400">Model Directory</span>
                            <div className="flex items-center gap-2">
                              <Button
                                type="button"
                                variant="secondary"
                                onClick={handleModelRescan}
                                disabled={modelScanPending || !modelDirectory.trim()}
                                className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-60"
                              >
                                {modelScanPending ? "Scanning…" : "Rescan"}
                              </Button>
                            </div>
                          </div>
                          <FilePathInput
                            value={modelDirectory}
                            onChange={val => updateSettings({ modelDirectory: val })}
                            onCommit={handleModelDirectoryCommit}
                            allowDirectories
                            placeholder="/models/llm"
                          />
                          {modelScanError ? (
                            <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
                              {modelScanError}
                            </div>
                          ) : null}
                          <div className="space-y-2">
                            {modelScanPending ? (
                              <div className="text-xs text-zinc-400">Scanning directory for .gguf models…</div>
                            ) : catalogEntries.length === 0 ? (
                              <div className="rounded-lg border border-zinc-800/60 bg-zinc-950/60 px-3 py-3 text-xs text-zinc-500">
                                No .gguf models discovered yet. Select a directory and rescan to populate the catalog.
                              </div>
                            ) : (
                              catalogEntries.map(entry => {
                                const base = modelDirectory.trim();
                                const resolved = entry.path && entry.path.length > 0
                                  ? entry.path
                                  : base
                                    ? `${base.replace(/[\\/]+$/, "")}/${entry.name}`
                                    : entry.name;
                                const normalizedResolved = resolved.replace(/\\+/g, "/");
                                const normalizedSelected = (selectedModelPath || "").replace(/\\+/g, "/");
                                const isSelected = normalizedResolved === normalizedSelected;
                                const buttonClass = isSelected
                                  ? "border-emerald-500/60 bg-emerald-500/10"
                                  : "border-zinc-800/70 bg-zinc-950/40 hover:border-emerald-500/30";
                                return (
                                  <button
                                    key={entry.path || entry.name}
                                    type="button"
                                    onClick={() => handleModelSelect(entry)}
                                    className={`w-full rounded-xl border px-3 py-3 text-left text-sm transition ${buttonClass}`}
                                  >
                                    <div className="flex items-start justify-between gap-3">
                                      <div className="space-y-1">
                                        <div className="text-sm font-semibold text-zinc-100 truncate">{entry.name}</div>
                                        <div className="max-w-[240px] truncate text-[11px] text-zinc-500">{entry.path || resolved}</div>
                                      </div>
                                      <div className="space-y-1 text-right">
                                        <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${MODEL_TYPE_STYLES[entry.type]}`}>
                                          {MODEL_TYPE_LABEL[entry.type]}
                                        </span>
                                        <div className="text-[11px] text-zinc-400">{entry.sizeLabel}</div>
                                      </div>
                                    </div>
                                  </button>
                                );
                              })
                            )}
                          </div>
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
                      </div>
                    </CardContent>
                  </Card>
                </section>
              </div>
            </TabsContent>
          </div>
        </Tabs>
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
    updateMemoryNode,
    deleteMemoryNode,
    createMemoryNode,
    createMemoryLink,
    deleteMemoryEdge,
    entities,
    setEntities,
    timeline,
    setTimeline,
    searchResults,
    setSearchResults,
    selectedEntityId,
    setSelectedEntityId,
    isSearching,
    setIsSearching,
    fetchEntities,
    fetchTimeline,
    performSearch,
  } = useNomousBridge();
  const st = statusMap[state.status];
  const tokenTotal = state.tokenWindow.reduce((a, p) => a + p.inTok + p.outTok, 0);
  const totalInboundTokens = state.tokenWindow.reduce((acc, point) => acc + point.inTok, 0);
  const totalOutboundTokens = state.tokenWindow.reduce((acc, point) => acc + point.outTok, 0);
  const peakInboundTokens = state.tokenWindow.reduce((max, point) => Math.max(max, point.inTok), 0);
  const peakOutboundTokens = state.tokenWindow.reduce((max, point) => Math.max(max, point.outTok), 0);
  const windowSamples = Math.max(1, state.tokenWindow.length);
  const inboundPerSecond = totalInboundTokens / windowSamples;
  const outboundPerSecond = totalOutboundTokens / windowSamples;
  const outboundPerMinute = outboundPerSecond * 60;
  const netTokenFlow = totalOutboundTokens - totalInboundTokens;
  const maxTokenBudget = state.settings.llmMaxTokens || 0;
  const latestTokenSample = state.tokenWindow[state.tokenWindow.length - 1] ?? { inTok: 0, outTok: 0 };
  const recentTokenWarning = useMemo(
    () => state.systemLines.find(line => /max token/i.test(line)),
    [state.systemLines]
  );
  const netFlowTone: "neutral" | "positive" | "caution" = Math.abs(netTokenFlow) < 1
    ? "neutral"
    : netTokenFlow > 0
      ? "positive"
      : "caution";
  const netFlowLabel = `${netTokenFlow >= 0 ? "+" : ""}${Math.round(netTokenFlow)}`;
  const utilizationPercent = maxTokenBudget > 0 ? Math.min(100, (totalOutboundTokens / maxTokenBudget) * 100) : 0;
  const utilizationTone: "neutral" | "positive" | "caution" = utilizationPercent > 70
    ? "caution"
    : utilizationPercent < 40
      ? "positive"
      : "neutral";
  const [controlCenterOpen, setControlCenterOpen] = useState(false);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const memoryInsights = useMemo(() => computeMemoryInsights(state.memory.nodes, state.memory.edges), [state.memory.nodes, state.memory.edges]);
  const [memoryKindFilter, setMemoryKindFilter] = useState<Set<MemoryNodeKind>>(() => new Set<MemoryNodeKind>(["stimulus", "concept", "event", "behavior", "self"]));
  const [memoryZoom, setMemoryZoom] = useState(1);
  
  const memoryKindOptions = useMemo(
    () => [
      { kind: "stimulus" as MemoryNodeKind, label: "Stimuli", className: "border-amber-500/60 bg-amber-500/10 text-amber-200" },
      { kind: "concept" as MemoryNodeKind, label: "Concepts", className: "border-purple-500/60 bg-purple-500/10 text-purple-200" },
      { kind: "event" as MemoryNodeKind, label: "Events", className: "border-cyan-500/60 bg-cyan-500/10 text-cyan-200" },
      { kind: "behavior" as MemoryNodeKind, label: "Behaviors", className: "border-teal-500/60 bg-teal-500/10 text-teal-200" },
      { kind: "self" as MemoryNodeKind, label: "Identity", className: "border-emerald-500/60 bg-emerald-500/10 text-emerald-200" },
    ],
    [],
  );
  const toggleMemoryKind = useCallback((kind: MemoryNodeKind) => {
    setMemoryKindFilter(prev => {
      const next = new Set(prev);
      if (next.has(kind)) {
        next.delete(kind);
      } else {
        next.add(kind);
      }
      return next;
    });
  }, []);
  const filteredMemoryNodes = useMemo(
    () => state.memory.nodes.filter(node => memoryKindFilter.has(node.kind)),
    [state.memory.nodes, memoryKindFilter],
  );
  const filteredMemoryEdges = useMemo(() => {
    const allowed = new Set(filteredMemoryNodes.map(node => node.id));
    return state.memory.edges.filter(edge => allowed.has(edge.from) && allowed.has(edge.to));
  }, [state.memory.edges, filteredMemoryNodes]);
  const [modelSwitching, setModelSwitching] = useState<{ label: string; progress: number } | null>(null);
  const modelSwitchStartRef = useRef<number>(0);
  const hideModelSwitchRef = useRef<number | null>(null);
  const activeModelName = extractModelName(state.settings.llmModelPath);

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
    if (!selectedMemoryId) {
      return;
    }
    const node = memoryInsights.nodeById.get(selectedMemoryId);
    if (!node || !memoryKindFilter.has(node.kind)) {
      setSelectedMemoryId(null);
    }
  }, [selectedMemoryId, memoryInsights, memoryKindFilter]);

  useEffect(() => {
    if (!state.connected) {
      return;
    }
    fetchTimeline(selectedEntityId ?? undefined);
  }, [selectedEntityId, state.connected, fetchTimeline]);

  useEffect(() => {
    if (selectedMemoryId || filteredMemoryNodes.length === 0) {
      return;
    }
    const firstVisibleMilestone = memoryInsights.milestones.find(entry => {
      const node = memoryInsights.nodeById.get(entry.id);
      return node ? memoryKindFilter.has(node.kind) : false;
    });
    const defaultSelection = firstVisibleMilestone?.id ?? filteredMemoryNodes[0]?.id ?? null;
    if (defaultSelection) {
      setSelectedMemoryId(defaultSelection);
    }
  }, [selectedMemoryId, filteredMemoryNodes, memoryInsights, memoryKindFilter]);

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
      <div className="relative min-h-screen overflow-x-hidden bg-[#040406] text-zinc-50">
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

            {state.promptReloadRequired ? (
              <div className="flex items-start gap-3 rounded-2xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-amber-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                <div className="mt-1 shrink-0">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div className="space-y-1">
                  <div className="text-sm font-semibold tracking-wide">Prompts updated — reload required</div>
                  <p className="text-xs text-amber-100/80">
                    The system or thinking prompt changed. Reload the active language model to apply the new guidance.
                  </p>
                </div>
              </div>
            ) : null}

            <div className="grid gap-4 rounded-2xl border border-zinc-800/70 bg-zinc-950/70 px-5 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] md:grid-cols-[1.1fr_auto]">
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
              <div className="flex flex-col items-start gap-3 text-xs text-zinc-400 md:items-end">
                <div className="flex items-center gap-3 rounded-xl border border-zinc-800/70 bg-zinc-950/80 px-3 py-2 text-left text-zinc-100 md:text-right">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-emerald-500/40 bg-emerald-500/10">
                    <Cpu className="h-5 w-5 text-emerald-300" />
                  </div>
                  <div className="space-y-0.5 md:items-end">
                    <div className="text-[11px] uppercase tracking-[0.3em] text-emerald-300/90 md:text-right">Active Model</div>
                    <div className="max-w-[220px] truncate text-sm font-semibold text-zinc-100 md:text-right">{activeModelName}</div>
                    <div className="max-w-[220px] truncate text-[11px] text-zinc-500 md:text-right">
                      {state.settings.modelDirectory ? state.settings.modelDirectory : "Directory not set"}
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-3 md:justify-end">
                  <Badge className="bg-zinc-900/80 text-zinc-200 border border-zinc-700/60">Tokens window • {tokenTotal}</Badge>
                  <Badge className="bg-zinc-900/80 text-zinc-200 border border-zinc-700/60">Mic {state.micOn ? "Active" : "Muted"}</Badge>
                  <Badge className="bg-zinc-900/80 text-zinc-200 border border-zinc-700/60">Vision {state.visionEnabled ? "Online" : "Paused"}</Badge>
                </div>
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
                <TabsTrigger value="conversation">Conversation</TabsTrigger>
                <TabsTrigger value="behavior">Behavior</TabsTrigger>
                <TabsTrigger value="tokens">Tokens</TabsTrigger>
                <TabsTrigger value="tools">Tools</TabsTrigger>
                <TabsTrigger value="memory">Memory</TabsTrigger>
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

              <TabsContent value="conversation" className="pt-4">
                <div className="grid gap-4 xl:grid-cols-[7fr_5fr]">
                  <div className="space-y-4">
                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="flex h-[32rem] flex-col gap-4 p-4">
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div className="flex items-center gap-3 text-zinc-200">
                            <div className="grid h-10 w-10 place-items-center rounded-xl border border-zinc-800/70 bg-black/50">
                              <MessageSquare className="h-4 w-4" />
                            </div>
                            <div>
                              <div className="text-sm font-semibold">Manual Chat</div>
                              <div className="text-xs text-zinc-500">Send precise directives without using voice.</div>
                            </div>
                          </div>
                          <Badge className={`border ${state.connected ? "bg-emerald-500/20 text-emerald-100 border-emerald-500/40" : "bg-red-500/20 text-red-200 border-red-500/40"}`}>
                            {state.connected ? "Connected" : "Offline"}
                          </Badge>
                        </div>

                        <div className="flex-1 overflow-hidden rounded-xl border border-zinc-800/60 bg-black/50">
                          <div className="h-full overflow-y-auto p-4 scroll-smooth" ref={chatScrollRef}>
                            <div className="flex flex-col gap-3">
                              {state.chatMessages.length === 0 ? (
                                <div className="text-sm text-zinc-400">
                                  Draft instructions or follow-ups here when you want full control over the wording.
                                </div>
                              ) : (
                                state.chatMessages.map(message => (
                                  <ChatBubble key={message.id} message={message} />
                                ))
                              )}
                              {state.generationProgress?.active && (
                                <GenerationProgress
                                  text={state.generationProgress.text}
                                  tokens={state.generationProgress.tokens}
                                />
                              )}
                            </div>
                          </div>
                        </div>

                        <form onSubmit={handleChatSubmit} className="space-y-2">
                          <label className="text-xs uppercase tracking-[0.3em] text-zinc-500">Message</label>
                          <div className="flex flex-col gap-2 sm:flex-row">
                            <textarea
                              value={chatInput}
                              onChange={event => setChatInput(event.target.value)}
                              onKeyDown={handleChatKeyDown}
                              rows={3}
                              placeholder="Type your instruction or question..."
                              className="min-h-[96px] flex-1 rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-emerald-500/80 focus:outline-none focus:ring-0"
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

                    <SttMonitorCard
                      lines={state.sttLines}
                      micOn={state.micOn}
                      micStatus={micStatus}
                      micStatusLabel={micStatusLabel}
                      micSensitivity={micSensitivity}
                      sttEnabled={state.settings.sttEnabled}
                      vu={state.vu}
                      setMic={setMic}
                      updateSettings={updateSettings}
                    />
                  </div>

                  <div className="space-y-4">
                    <ConversationStreamPanel
                      thoughtLines={state.thoughtLines}
                      speechLines={state.speechLines}
                      systemLines={state.systemLines}
                    />
                    <ConversationQuickSettings
                      settings={state.settings}
                      micOn={state.micOn}
                      setMic={setMic}
                      updateSettings={updateSettings}
                      push={push}
                    />
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

              <TabsContent value="tokens" className="pt-4 space-y-4">
                <div className="grid gap-4 xl:grid-cols-[2fr_1fr]">
                  <Card className="bg-zinc-900/70 border-zinc-800/60">
                    <CardContent className="p-4 text-zinc-200">
                      <div className="mb-3 flex items-center justify-between">
                        <div className="flex items-center gap-2"><Activity className="w-4 h-4"/><span className="font-semibold">Token Flow (last 30s)</span></div>
                        <Badge className="bg-zinc-900/70 text-zinc-200 border border-zinc-700/60">{Math.round(outboundPerMinute)} tok/min</Badge>
                      </div>
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

                  <div className="space-y-4">
                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="space-y-4 p-4 text-zinc-200">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-semibold">Usage Snapshot</span>
                          <Badge className="bg-zinc-900/70 text-zinc-200 border border-zinc-700/60">Window · 30s</Badge>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <TokenStat label="Inbound" value={totalInboundTokens.toLocaleString()} helper={`≈ ${inboundPerSecond.toFixed(1)} tok/s`} />
                          <TokenStat label="Outbound" value={totalOutboundTokens.toLocaleString()} helper={`≈ ${outboundPerSecond.toFixed(1)} tok/s`} tone="positive" />
                          <TokenStat label="Latest burst" value={`${latestTokenSample.inTok} / ${latestTokenSample.outTok}`} helper="In / Out tokens (last sample)" />
                          <TokenStat label="Peak load" value={`${peakInboundTokens} / ${peakOutboundTokens}`} helper="Highest in / out sample" />
                          <TokenStat label="Net flow" value={netFlowLabel} helper="Out - In across the window" tone={netFlowTone} />
                          <TokenStat label="Context budget" value={maxTokenBudget ? `${Math.round(utilizationPercent)}%` : "n/a"} helper={maxTokenBudget ? `${maxTokenBudget.toLocaleString()} token cap` : "Set max tokens to track saturation"} tone={utilizationTone} />
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-zinc-900/70 border-zinc-800/60">
                      <CardContent className="space-y-2 p-4 text-sm text-zinc-300">
                        <div className="flex items-center gap-2 font-semibold text-zinc-100"><AlertTriangle className="h-4 w-4 text-amber-300" /> Token Health</div>
                        <p className="text-xs text-zinc-400">
                          Track the rolling context budget to avoid truncation. Keep outbound flow under 70% of the configured max tokens or force a prompt refresh before long explanations.
                        </p>
                        {recentTokenWarning ? (
                          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-100">
                            {recentTokenWarning}
                          </div>
                        ) : (
                          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-100">
                            No max-token warnings detected this session.
                          </div>
                        )}
                      </CardContent>
                    </Card>
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

              <TabsContent value="memory" className="pt-4">
                <div className="space-y-4">
                  {/* Simplified two-column layout */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                    <div className="space-y-4">
                      <SemanticSearch
                        onSearch={performSearch}
                        results={searchResults}
                        isSearching={isSearching}
                        onSelectResult={(nodeId) => {
                          setSelectedMemoryId(nodeId);
                          const entity = entities.find(e => e.id === nodeId);
                          if (entity) {
                            setSelectedEntityId(nodeId);
                          }
                        }}
                        selectedResultId={selectedMemoryId}
                      />
                      <EntityBrowser
                        entities={entities}
                        selectedEntityId={selectedEntityId}
                        onSelect={(entityId) => {
                          setSelectedEntityId(entityId);
                          setSelectedMemoryId(entityId);
                        }}
                        onEdit={(entity) => {
                          console.log("Edit entity:", entity);
                        }}
                        onDelete={(entityId) => {
                          console.log("Delete entity:", entityId);
                        }}
                        onSearch={(query) => {
                          if (query.trim()) {
                            const filtered = entities.filter(e => 
                              e.name.toLowerCase().includes(query.toLowerCase()) ||
                              e.description?.toLowerCase().includes(query.toLowerCase())
                            );
                            console.log("Filtered entities:", filtered.length);
                          }
                        }}
                      />
                    </div>
                    
                    <div className="space-y-4">
                      <LearningTimeline
                        events={timeline}
                        selectedEntityId={selectedEntityId}
                        onSelectEntity={(entityId) => {
                          setSelectedEntityId(entityId);
                          if (entityId) {
                            setSelectedMemoryId(entityId);
                          }
                        }}
                        limit={50}
                      />
                      <TimelinePerspective events={timeline} />
                    </div>
                  </div>

                  <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
                    <Card className="border-zinc-800/60 bg-black/40">
                      <CardContent className="space-y-4 p-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                            <map className="h-4 w-4 text-cyan-400" />
                            Memory Graph Workbench
                          </div>
                          <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.3em] text-zinc-500">
                            Filters
                            <div className="flex flex-wrap gap-2">
                              {memoryKindOptions.map(option => {
                                const active = memoryKindFilter.has(option.kind);
                                return (
                                  <button
                                    key={option.kind}
                                    type="button"
                                    onClick={() => toggleMemoryKind(option.kind)}
                                    className={`rounded-full border px-3 py-1 text-[11px] transition ${
                                      active
                                        ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-200"
                                        : "border-zinc-800/60 bg-black/30 text-zinc-400 hover:border-zinc-700/60"
                                    }`}
                                  >
                                    {option.label}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
                          <div className="flex items-center gap-2">
                            <span className="uppercase tracking-[0.3em]">Zoom</span>
                            <span className="font-mono text-zinc-100">{Math.round(memoryZoom * 100)}%</span>
                          </div>
                          <div className="w-full max-w-xs">
                            <Slider
                              defaultValue={[Math.round(memoryZoom * 100)]}
                              onValueChange={value => setMemoryZoom(value[0] / 100)}
                              min={60}
                              max={160}
                              step={5}
                            />
                          </div>
                          <Button type="button" variant="secondary" onClick={() => setMemoryZoom(1)} className="text-xs text-zinc-300">
                            Reset
                          </Button>
                        </div>

                        <div className="rounded-xl border border-zinc-800/60 bg-zinc-950/40">
                          {filteredMemoryNodes.length === 0 ? (
                            <div className="p-6 text-center text-sm text-zinc-500">No memory nodes match the current filters.</div>
                          ) : (
                            <div className="overflow-x-auto">
                              <MemoryGraph
                                nodes={filteredMemoryNodes}
                                edges={filteredMemoryEdges}
                                selectedNodeId={selectedMemoryId}
                                onSelect={setSelectedMemoryId}
                                zoom={memoryZoom}
                              />
                            </div>
                          )}
                        </div>
                        <p className="text-[11px] text-zinc-500">
                          Click nodes to inspect associations. Zoom adjusts column spacing so you can focus on dense concepts without hiding existing context.
                        </p>
                      </CardContent>
                    </Card>
                    <MemoryDetailCard
                      detail={selectedMemoryDetail}
                      onUpdate={updateMemoryNode}
                      onDelete={deleteMemoryNode}
                      onCreateLink={createMemoryLink}
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
