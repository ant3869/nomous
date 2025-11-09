export type BehaviorSeverity = "info" | "warn" | "critical";

export interface BehaviorSummary {
  onTopic: number;
  responsiveness: number;
  brevity: number;
  coherence: number;
  sentiment: number;
  lexicalRichness: number;
  safety: number;
  stability: number;
}

export interface BehaviorSignals {
  latencyMs: number;
  tokensIn: number;
  tokensOut: number;
  conversationPace: number;
  avgResponseLength: number;
}

export interface BehaviorHistoryPoint {
  timestamp: number;
  onTopic: number;
  sentiment: number;
  coherence: number;
  safety: number;
}

export interface BehaviorAnomaly {
  label: string;
  severity: BehaviorSeverity;
  detail: string;
}

export interface BehaviorSnapshot {
  userText?: string;
  assistantText?: string;
  latencyMs: number;
  tokensIn: number;
  tokensOut: number;
  scores: {
    onTopic: number;
    responsiveness: number;
    brevity: number;
    coherence: number;
    sentiment: number;
    lexicalRichness: number;
    safety: number;
    toxicity?: number;
  };
}

export interface BehaviorStats {
  summary: BehaviorSummary;
  signals: BehaviorSignals;
  history: BehaviorHistoryPoint[];
  rewardTotal: number;
  anomalies: BehaviorAnomaly[];
  lastSample?: BehaviorSnapshot;
}

export const EMPTY_BEHAVIOR_STATS: BehaviorStats = {
  summary: {
    onTopic: 0,
    responsiveness: 0,
    brevity: 0,
    coherence: 0,
    sentiment: 0.5,
    lexicalRichness: 0,
    safety: 1,
    stability: 1,
  },
  signals: {
    latencyMs: 0,
    tokensIn: 0,
    tokensOut: 0,
    conversationPace: 0,
    avgResponseLength: 0,
  },
  history: [],
  rewardTotal: 0,
  anomalies: [],
  lastSample: undefined,
};
