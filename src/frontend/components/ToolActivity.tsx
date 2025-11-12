import React from 'react';
import {
  Wrench,
  Search,
  Eye,
  Brain,
  Target,
  TrendingUp,
  MessageCircle,
  ClipboardList,
  ListChecks,
  BarChart3,
  Clock,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight
} from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';

export interface ToolResult {
  tool: string;
  displayName: string;
  category: string;
  description: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
  success: boolean;
  summary: string;
  warnings: string[];
  timestamp: number;
  durationMs: number;
}

interface ToolActivityProps {
  tools: ToolResult[];
  maxDisplay?: number;
}

const TOOL_ICONS: Record<string, React.ReactNode> = {
  search_memory: <Search className="h-4 w-4" />,
  recall_recent_context: <Brain className="h-4 w-4" />,
  summarize_recent_context: <ClipboardList className="h-4 w-4" />,
  record_observation: <Eye className="h-4 w-4" />,
  evaluate_interaction: <Target className="h-4 w-4" />,
  identify_pattern: <TrendingUp className="h-4 w-4" />,
  track_milestone: <Target className="h-4 w-4" />,
  get_current_capabilities: <Brain className="h-4 w-4" />,
  analyze_sentiment: <MessageCircle className="h-4 w-4" />,
  check_appropriate_response: <MessageCircle className="h-4 w-4" />,
  list_available_tools: <ListChecks className="h-4 w-4" />,
  get_tool_usage_stats: <BarChart3 className="h-4 w-4" />
};

const CATEGORY_STYLES: Record<
  string,
  { label: string; accent: string; badge: string; icon: React.ReactNode }
> = {
  memory: {
    label: 'Memory',
    accent: 'border-sky-500/40 bg-sky-500/10 text-sky-200',
    badge: 'border-sky-500/40 text-sky-200 bg-sky-500/10',
    icon: <Brain className="h-4 w-4" />
  },
  observation: {
    label: 'Observation',
    accent: 'border-purple-500/40 bg-purple-500/10 text-purple-200',
    badge: 'border-purple-500/40 text-purple-200 bg-purple-500/10',
    icon: <Eye className="h-4 w-4" />
  },
  learning: {
    label: 'Learning',
    accent: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
    badge: 'border-emerald-500/40 text-emerald-200 bg-emerald-500/10',
    icon: <TrendingUp className="h-4 w-4" />
  },
  social: {
    label: 'Social',
    accent: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
    badge: 'border-amber-500/40 text-amber-200 bg-amber-500/10',
    icon: <MessageCircle className="h-4 w-4" />
  },
  analytics: {
    label: 'Analytics',
    accent: 'border-indigo-500/40 bg-indigo-500/10 text-indigo-200',
    badge: 'border-indigo-500/40 text-indigo-200 bg-indigo-500/10',
    icon: <BarChart3 className="h-4 w-4" />
  },
  general: {
    label: 'General',
    accent: 'border-zinc-700 bg-zinc-800/60 text-zinc-200',
    badge: 'border-zinc-700 text-zinc-200 bg-zinc-800/60',
    icon: <Wrench className="h-4 w-4" />
  }
};

const STATUS_BADGES = {
  success: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/40',
  failure: 'bg-rose-500/15 text-rose-200 border-rose-500/40'
};

function formatRelativeTime(timestamp: number): string {
  if (!Number.isFinite(timestamp)) {
    return 'just now';
  }
  const delta = Math.max(0, Date.now() - timestamp);
  if (delta < 1000) return 'just now';
  const seconds = Math.round(delta / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

function formatDuration(durationMs: number): string {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return '<1 ms';
  }
  if (durationMs >= 1000) {
    return `${(durationMs / 1000).toFixed(2)} s`;
  }
  return `${durationMs.toFixed(1)} ms`;
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map(item => formatValue(item)).join(', ');
  }
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch (err) {
      return String(value);
    }
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  return String(value);
}

const DEFAULT_CATEGORY_STYLE = CATEGORY_STYLES.general;

export function ToolActivity({ tools, maxDisplay = 10 }: ToolActivityProps) {
  const [expandedId, setExpandedId] = React.useState<string | null>(null);
  const displayTools = tools.slice(-maxDisplay).reverse();

  if (displayTools.length === 0) {
    return (
      <Card className="border-zinc-800/60 bg-zinc-900/70">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm text-zinc-200">
            <Wrench className="h-4 w-4" />
            Tool Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="py-4 text-center text-sm text-zinc-500">
            No tools used yet
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-zinc-800/60 bg-zinc-900/70">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm text-zinc-200">
          <Wrench className="h-4 w-4" />
          Tool Activity
          <Badge variant="outline" className="ml-auto border-emerald-500/40 text-emerald-200">
            {tools.length} total
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {displayTools.map(toolUse => {
          const categoryStyle = CATEGORY_STYLES[toolUse.category] ?? DEFAULT_CATEGORY_STYLE;
          const icon = TOOL_ICONS[toolUse.tool] ?? categoryStyle.icon;
          const statusClass = toolUse.success ? STATUS_BADGES.success : STATUS_BADGES.failure;
          const statusIcon = toolUse.success ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />;
          const id = `${toolUse.tool}-${toolUse.timestamp}`;
          const expanded = expandedId === id;
          const displayName = toolUse.displayName || toolUse.tool;

          return (
            <div
              key={id}
              className={`rounded-lg border border-zinc-800/60 bg-zinc-950/60 p-3 transition-colors hover:border-emerald-500/40`}
            >
              <div className="flex items-start gap-3">
                <div className={`rounded border ${categoryStyle.accent} p-1.5`}>{icon}</div>
                <div className="min-w-0 flex-1 space-y-2">
                  <div className="flex flex-wrap items-start gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-zinc-100">{displayName}</span>
                        <Badge variant="outline" className={`text-xs ${categoryStyle.badge}`}>
                          {categoryStyle.label}
                        </Badge>
                        <Badge variant="outline" className={`flex items-center gap-1 text-xs ${statusClass}`}>
                          {statusIcon}
                          {toolUse.success ? 'Success' : 'Issue'}
                        </Badge>
                      </div>
                      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{toolUse.tool}</div>
                    </div>
                    <div className="ml-auto flex flex-col items-end gap-1 text-right text-[11px] text-zinc-500">
                      <span>{formatRelativeTime(toolUse.timestamp)}</span>
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatDuration(toolUse.durationMs)}
                      </span>
                    </div>
                  </div>

                  <div className="text-xs leading-relaxed text-zinc-300">
                    {toolUse.summary || 'No summary returned.'}
                  </div>

                  {toolUse.warnings.length > 0 && (
                    <div className="flex items-center gap-2 text-xs text-amber-300">
                      <AlertTriangle className="h-3.5 w-3.5" />
                      <span>{toolUse.warnings.join(' • ')}</span>
                    </div>
                  )}

                  <div className="flex items-center gap-2 pt-1">
                    <button
                      type="button"
                      onClick={() => setExpandedId(prev => (prev === id ? null : id))}
                      className="flex items-center gap-1 text-xs font-medium text-emerald-300 transition hover:text-emerald-200"
                    >
                      {expanded ? (
                        <>
                          <ChevronDown className="h-3 w-3" /> Hide details
                        </>
                      ) : (
                        <>
                          <ChevronRight className="h-3 w-3" /> View details
                        </>
                      )}
                    </button>
                  </div>

                  {expanded && (
                    <div className="space-y-3 rounded-md border border-zinc-800/60 bg-zinc-900/60 p-3 text-xs">
                      <div>
                        <div className="text-xs font-semibold uppercase text-zinc-400">Arguments</div>
                        {Object.keys(toolUse.args || {}).length > 0 ? (
                          <dl className="mt-1 grid gap-2 sm:grid-cols-2">
                            {Object.entries(toolUse.args).map(([key, value]) => (
                              <div key={key} className="rounded bg-zinc-950/80 p-2">
                                <dt className="text-[10px] uppercase tracking-wide text-zinc-500">{key}</dt>
                                <dd className="text-xs text-zinc-200">{formatValue(value)}</dd>
                              </div>
                            ))}
                          </dl>
                        ) : (
                          <div className="mt-1 text-zinc-500">No arguments provided.</div>
                        )}
                      </div>

                      <div>
                        <div className="text-xs font-semibold uppercase text-zinc-400">Result</div>
                        <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-black/40 p-2 text-[11px] text-zinc-300">
{JSON.stringify(toolUse.result, null, 2)}
                        </pre>
                      </div>

                      <div>
                        <div className="text-xs font-semibold uppercase text-zinc-400">Tool Description</div>
                        <p className="mt-1 text-[11px] text-zinc-400">{toolUse.description}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

export function ToolStats({ tools }: { tools: ToolResult[] }) {
  const stats = React.useMemo(() => {
    const total = tools.length;
    const successCount = tools.filter(t => t.success).length;
    const successRate = total ? successCount / total : 0;
    const averageDuration = total
      ? tools.reduce((acc, tool) => acc + (Number.isFinite(tool.durationMs) ? tool.durationMs : 0), 0) / total
      : 0;
    const uniqueTools = new Set(tools.map(t => t.tool)).size;
    const categoryMap = new Map<string, number>();
    tools.forEach(t => {
      categoryMap.set(t.category, (categoryMap.get(t.category) || 0) + 1);
    });
    const categoryBreakdown = Array.from(categoryMap.entries())
      .map(([category, count]) => ({
        category,
        count,
        label: CATEGORY_STYLES[category]?.label || category,
        percentage: total ? Math.round((count / total) * 100) : 0
      }))
      .sort((a, b) => b.count - a.count);

    const toolUsageMap = new Map<string, { count: number; displayName: string }>();
    tools.forEach(t => {
      const existing = toolUsageMap.get(t.tool) || { count: 0, displayName: t.displayName || t.tool };
      existing.count += 1;
      existing.displayName = t.displayName || existing.displayName;
      toolUsageMap.set(t.tool, existing);
    });

    const topToolEntry = Array.from(toolUsageMap.entries())
      .map(([tool, meta]) => ({ tool, ...meta }))
      .sort((a, b) => b.count - a.count)[0];

    const recentFailures = tools
      .filter(t => !t.success)
      .slice(-3)
      .reverse()
      .map(item => ({
        id: `${item.tool}-${item.timestamp}`,
        summary: item.summary || 'Tool run failed',
        timestamp: item.timestamp,
        displayName: item.displayName || item.tool
      }));

    return {
      total,
      uniqueTools,
      successRate,
      averageDuration,
      categoryBreakdown,
      topTool: topToolEntry || null,
      recentFailures
    };
  }, [tools]);

  if (tools.length === 0) {
    return (
      <Card className="border-zinc-800/60 bg-zinc-900/70">
        <CardContent className="py-6 text-center text-sm text-zinc-500">
          Tool stats will appear once tools are used.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <Card className="border-zinc-800/60 bg-zinc-900/70">
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-zinc-100">{stats.total}</div>
            <div className="text-xs text-zinc-500">Total tool executions</div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800/60 bg-zinc-900/70">
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-zinc-100">{stats.uniqueTools}</div>
            <div className="text-xs text-zinc-500">Unique tools used</div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800/60 bg-zinc-900/70 col-span-2">
          <CardContent className="pt-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Success rate</span>
              <span className="text-sm text-emerald-300">{Math.round(stats.successRate * 100)}%</span>
            </div>
            <Progress value={Math.round(stats.successRate * 100)} />
          </CardContent>
        </Card>

        <Card className="border-zinc-800/60 bg-zinc-900/70 col-span-2">
          <CardContent className="pt-4">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>Average duration</span>
              <span className="text-sm text-zinc-200">{formatDuration(stats.averageDuration)}</span>
            </div>
            {stats.topTool && (
              <div className="mt-3 rounded border border-zinc-800/60 bg-zinc-950/50 p-2 text-xs text-zinc-400">
                <div className="text-[11px] uppercase tracking-wide text-zinc-500">Most used</div>
                <div className="flex items-center justify-between text-sm text-zinc-200">
                  <span>{stats.topTool.displayName}</span>
                  <span>{stats.topTool.count}×</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="border-zinc-800/60 bg-zinc-900/70">
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-semibold text-zinc-300">Category breakdown</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {stats.categoryBreakdown.map(category => {
            const style = CATEGORY_STYLES[category.category] ?? DEFAULT_CATEGORY_STYLE;
            return (
              <div key={category.category} className="space-y-1">
                <div className="flex items-center justify-between text-xs text-zinc-400">
                  <span className="flex items-center gap-2">
                    <span className={`flex h-5 w-5 items-center justify-center rounded-full border ${style.accent}`}>
                      {style.icon}
                    </span>
                    {category.label}
                  </span>
                  <span className="text-zinc-300">{category.count} ({category.percentage}%)</span>
                </div>
                <Progress value={category.percentage} />
              </div>
            );
          })}
        </CardContent>
      </Card>

      {stats.recentFailures.length > 0 && (
        <Card className="border-rose-500/40 bg-rose-500/10">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-xs font-semibold text-rose-100">
              <AlertTriangle className="h-4 w-4" /> Recent issues
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {stats.recentFailures.map(item => (
              <div key={item.id} className="rounded border border-rose-500/40 bg-rose-500/10 p-2 text-[11px] text-rose-100">
                <div className="flex items-center justify-between font-medium">
                  <span>{item.displayName}</span>
                  <span>{formatRelativeTime(item.timestamp)}</span>
                </div>
                <div className="mt-1 text-rose-100/80">{item.summary}</div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

