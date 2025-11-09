import React from "react";
import { Card, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import {
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  Legend,
} from "recharts";
import type { BehaviorStats, BehaviorHistoryPoint } from "../types/behavior";

interface BehaviorInsightsProps {
  stats: BehaviorStats;
}

const severityStyles: Record<string, string> = {
  info: "bg-zinc-900/70 border-zinc-800/70 text-zinc-200",
  warn: "bg-amber-500/15 border-amber-500/40 text-amber-200",
  critical: "bg-red-500/15 border-red-500/40 text-red-200",
};

const metricLabels: Record<string, string> = {
  onTopic: "On-topic",
  responsiveness: "Responsiveness",
  brevity: "Brevity",
  coherence: "Coherence",
  sentiment: "Sentiment",
  lexicalRichness: "Expression",
  safety: "Safety",
  stability: "Stability",
};

function formatPercent(value: number, digits = 0): string {
  const safe = Number.isFinite(value) ? value : 0;
  return `${(safe * 100).toFixed(digits)}%`;
}

function toRadarDataset(summary: BehaviorStats["summary"]) {
  return Object.entries(summary).map(([key, value]) => ({
    metric: metricLabels[key] ?? key,
    score: Math.max(0, Math.min(1, value)) * 100,
  }));
}

function buildTrend(history: BehaviorHistoryPoint[]) {
  return history.map(point => ({
    t: point.timestamp,
    onTopic: Math.round(point.onTopic * 100),
    coherence: Math.round(point.coherence * 100),
    safety: Math.round(point.safety * 100),
    sentiment: Math.round(point.sentiment * 100),
  }));
}

function formatLatency(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) {
    return "<100 ms";
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(1)} s`;
  }
  return `${Math.round(ms)} ms`;
}

function InsightBadge({ label, value, helper, tone }: { label: string; value: string; helper: string; tone: "positive" | "neutral" | "negative" }) {
  const accent = tone === "positive" ? "text-emerald-300" : tone === "negative" ? "text-red-300" : "text-zinc-300";
  return (
    <div className="rounded-2xl border border-zinc-800/70 bg-zinc-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <div className="text-[10px] uppercase tracking-[0.35em] text-zinc-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold ${accent}`}>{value}</div>
      <div className="text-xs text-zinc-500">{helper}</div>
    </div>
  );
}

export function BehaviorInsights({ stats }: BehaviorInsightsProps) {
  const radarData = React.useMemo(() => toRadarDataset(stats.summary), [stats.summary]);
  const trendData = React.useMemo(() => buildTrend(stats.history.slice(-60)), [stats.history]);
  const latency = formatLatency(stats.signals.latencyMs);
  const pace = Number.isFinite(stats.signals.conversationPace) ? stats.signals.conversationPace : 0;
  const avgLength = Number.isFinite(stats.signals.avgResponseLength) ? stats.signals.avgResponseLength : 0;
  const anomalies = stats.anomalies.slice(-4);
  const snapshot = stats.lastSample;

  const responseTone: "positive" | "neutral" | "negative" = stats.summary.onTopic > 0.65
    ? "positive"
    : stats.summary.onTopic < 0.4
      ? "negative"
      : "neutral";
  const safetyTone: "positive" | "neutral" | "negative" = stats.summary.safety > 0.8
    ? "positive"
    : stats.summary.safety < 0.55
      ? "negative"
      : "neutral";
  const coherenceTone: "positive" | "neutral" | "negative" = stats.summary.coherence > 0.65
    ? "positive"
    : stats.summary.coherence < 0.4
      ? "negative"
      : "neutral";

  return (
    <div className="grid gap-4 lg:grid-cols-[1.4fr,1fr] xl:grid-cols-[1.6fr,1fr]">
      <Card className="border-zinc-800/70 bg-zinc-950/70">
        <CardContent className="p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
            <div className="w-full lg:w-1/2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-[0.35em] text-zinc-400">Model Posture</h3>
                <Badge className="bg-emerald-500/20 text-emerald-200 border border-emerald-500/40">
                  {formatPercent(stats.summary.stability)} stable
                </Badge>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} outerRadius="80%">
                    <PolarGrid stroke="#27272a" strokeOpacity={0.6} />
                    <PolarAngleAxis dataKey="metric" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                    <PolarRadiusAxis tick={false} axisLine={false} />
                    <Radar
                      name="Score"
                      dataKey="score"
                      stroke="#10b981"
                      strokeOpacity={0.8}
                      fill="#10b981"
                      fillOpacity={0.25}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="flex w-full flex-col gap-3 lg:w-1/2">
              <InsightBadge
                label="Cognitive Focus"
                value={formatPercent(stats.summary.onTopic)}
                helper="Topical alignment (rolling window)"
                tone={responseTone}
              />
              <InsightBadge
                label="Conversational Safety"
                value={formatPercent(stats.summary.safety)}
                helper="Toxicity shielding"
                tone={safetyTone}
              />
              <InsightBadge
                label="Expression Quality"
                value={`${formatPercent(stats.summary.coherence)} / ${formatPercent(stats.summary.lexicalRichness)}`}
                helper="Coherence • Lexical richness"
                tone={coherenceTone}
              />
              <div className="grid grid-cols-2 gap-3 text-xs text-zinc-400">
                <div className="rounded-xl border border-zinc-800/70 bg-zinc-900/40 p-3">
                  <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Latency</div>
                  <div className="mt-1 text-lg font-semibold text-zinc-200">{latency}</div>
                  <div>In tokens: {stats.signals.tokensIn}</div>
                </div>
                <div className="rounded-xl border border-zinc-800/70 bg-zinc-900/40 p-3">
                  <div className="text-[10px] uppercase tracking-[0.3em] text-zinc-500">Response</div>
                  <div className="mt-1 text-lg font-semibold text-zinc-200">{Math.round(avgLength)} tokens</div>
                  <div>Out tokens: {stats.signals.tokensOut}</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="border-zinc-800/70 bg-zinc-950/70">
        <CardContent className="flex h-full flex-col gap-4 p-6">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-[0.35em] text-zinc-400">Interaction Pulse</h3>
            <div className="mt-1 text-xs text-zinc-500">{pace.toFixed(2)} turns/min • Reward {stats.rewardTotal.toFixed(2)}</div>
          </div>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData} margin={{ top: 10, left: -12, right: 12 }}>
                <CartesianGrid stroke="#27272a" strokeDasharray="4 8" />
                <XAxis dataKey="t" tickFormatter={() => ""} stroke="#52525b" />
                <YAxis domain={[0, 100]} stroke="#52525b" tick={{ fontSize: 10 }} />
                <RTooltip
                  cursor={{ stroke: "#3f3f46" }}
                  contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", color: "#e4e4e7" }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="onTopic" stroke="#34d399" strokeWidth={2} dot={false} name="On-topic" />
                <Line type="monotone" dataKey="coherence" stroke="#a855f7" strokeWidth={2} dot={false} name="Coherence" />
                <Line type="monotone" dataKey="safety" stroke="#f97316" strokeWidth={2} dot={false} name="Safety" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {anomalies.length > 0 ? (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-[0.35em] text-zinc-400">Anomalies</h4>
              <div className="space-y-2">
                {anomalies.map((anomaly, idx) => (
                  <div
                    key={`${anomaly.label}-${idx}`}
                    className={`flex items-start gap-2 rounded-xl border p-3 text-sm ${severityStyles[anomaly.severity] ?? severityStyles.info}`}
                  >
                    <span className="mt-1 inline-flex h-2.5 w-2.5 flex-none rounded-full bg-current opacity-80" />
                    <div>
                      <div className="font-semibold">{anomaly.label}</div>
                      <div className="text-xs text-zinc-400">{anomaly.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-200">
              No anomalies detected in the latest exchange.
            </div>
          )}
          {snapshot && (
            <div className="space-y-3">
              <h4 className="text-xs font-semibold uppercase tracking-[0.35em] text-zinc-400">Latest Exchange</h4>
              {snapshot.userText ? (
                <div className="rounded-lg border border-zinc-800/70 bg-zinc-900/40 p-3 text-xs text-zinc-300">
                  <span className="font-semibold text-zinc-100">User:</span> {snapshot.userText}
                </div>
              ) : null}
              {snapshot.assistantText ? (
                <div className="rounded-lg border border-zinc-800/70 bg-zinc-900/40 p-3 text-xs text-emerald-200">
                  <span className="font-semibold text-emerald-200">Model:</span> {snapshot.assistantText}
                </div>
              ) : null}
              <div className="grid grid-cols-2 gap-2 text-[11px] text-zinc-500">
                <div>Latency: {formatLatency(snapshot.latencyMs)}</div>
                <div>Responsiveness: {formatPercent(snapshot.scores.responsiveness)}</div>
                <div>Brevity: {formatPercent(snapshot.scores.brevity)}</div>
                <div>Sentiment: {formatPercent(snapshot.scores.sentiment)}</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="lg:col-span-2 border-zinc-800/70 bg-zinc-950/70">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-[0.35em] text-zinc-400">Sentiment Stream</h3>
            <Badge className="bg-blue-500/20 text-blue-200 border border-blue-500/40">
              {trendData.length} points
            </Badge>
          </div>
          <div className="mt-3 h-40">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData} margin={{ top: 10, left: 0, right: 0 }}>
                <defs>
                  <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.45} />
                    <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#27272a" strokeDasharray="3 6" />
                <XAxis dataKey="t" tickFormatter={() => ""} stroke="#52525b" />
                <YAxis domain={[0, 100]} stroke="#52525b" tick={{ fontSize: 10 }} />
                <RTooltip
                  cursor={{ stroke: "#38bdf8", strokeOpacity: 0.2 }}
                  contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", color: "#e4e4e7" }}
                />
                <Area type="monotone" dataKey="sentiment" stroke="#38bdf8" fill="url(#sentimentGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
