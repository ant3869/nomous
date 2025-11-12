import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Wrench, Search, Eye, Brain, Target, TrendingUp, MessageCircle } from 'lucide-react';

interface ToolResult {
  tool: string;
  result: any;
  timestamp: number;
}

interface ToolActivityProps {
  tools: ToolResult[];
  maxDisplay?: number;
}

const TOOL_ICONS: Record<string, React.ReactNode> = {
  search_memory: <Search className="h-4 w-4" />,
  recall_recent_context: <Brain className="h-4 w-4" />,
  record_observation: <Eye className="h-4 w-4" />,
  evaluate_interaction: <Target className="h-4 w-4" />,
  identify_pattern: <TrendingUp className="h-4 w-4" />,
  track_milestone: <Target className="h-4 w-4" />,
  get_current_capabilities: <Brain className="h-4 w-4" />,
  analyze_sentiment: <MessageCircle className="h-4 w-4" />,
  check_appropriate_response: <MessageCircle className="h-4 w-4" />
};

const TOOL_COLORS: Record<string, string> = {
  memory: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  observation: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  learning: 'bg-green-500/20 text-green-300 border-green-500/30',
  social: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  general: 'bg-gray-500/20 text-gray-300 border-gray-500/30'
};

function getToolCategory(toolName: string): string {
  if (toolName.includes('memory') || toolName.includes('recall')) return 'memory';
  if (toolName.includes('observation') || toolName.includes('record')) return 'observation';
  if (toolName.includes('evaluate') || toolName.includes('pattern') || toolName.includes('milestone') || toolName.includes('capabilities')) return 'learning';
  if (toolName.includes('sentiment') || toolName.includes('appropriate')) return 'social';
  return 'general';
}

function formatToolResult(tool: string, result: any): string {
  if (!result) return 'No result';
  
  if (result.error) {
    return `Error: ${result.error}`;
  }
  
  if (result.success === false) {
    return result.message || 'Failed';
  }
  
  // Handle different tool result formats
  switch (tool) {
    case 'search_memory':
      return `Found ${result.found || 0} results`;
    case 'recall_recent_context':
      return `Recalled ${result.count || 0} items`;
    case 'record_observation':
      return result.message || 'Observation recorded';
    case 'evaluate_interaction':
      return `Quality: ${result.quality_score || 'N/A'}/10`;
    case 'identify_pattern':
      const PATTERN_PREVIEW_LENGTH = 30;
      return `Pattern: ${result.pattern?.substring(0, PATTERN_PREVIEW_LENGTH) || 'recorded'}...`;
    case 'track_milestone':
      const MILESTONE_PREVIEW_LENGTH = 30;
      return `${result.milestone?.substring(0, MILESTONE_PREVIEW_LENGTH) || 'Milestone'}...`;
    case 'get_current_capabilities':
      return `${result.tools_available || 0} tools, ${result.milestones_achieved || 0} milestones`;
    case 'analyze_sentiment':
      return `${result.sentiment || 'neutral'} (${Math.round((result.confidence || 0) * 100)}%)`;
    case 'check_appropriate_response':
      return result.appropriate ? '✓ Appropriate' : '⚠ Check needed';
    default:
      if (typeof result === 'string') return result;
      if (result.message) return result.message;
      return 'Completed';
  }
}

export function ToolActivity({ tools, maxDisplay = 10 }: ToolActivityProps) {
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
      <CardContent className="space-y-2">
        {displayTools.map((toolUse, idx) => {
          const category = getToolCategory(toolUse.tool);
          const colorClass = TOOL_COLORS[category] || TOOL_COLORS.general;
          const icon = TOOL_ICONS[toolUse.tool] || <Wrench className="h-4 w-4" />;
          const resultText = formatToolResult(toolUse.tool, toolUse.result);
          const timeAgo = Math.floor((Date.now() - toolUse.timestamp) / 1000);
          
          return (
            <div
              key={`${toolUse.tool}-${toolUse.timestamp}-${idx}`}
              className="flex items-start gap-2 rounded-lg border border-zinc-800/60 bg-zinc-950/50 p-2 transition-colors hover:border-emerald-500/40 hover:bg-zinc-900/80"
            >
              <div className={`rounded border ${colorClass} p-1.5`}>
                {icon}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-zinc-100">
                    {toolUse.tool}
                  </span>
                  <Badge variant="outline" className={`text-xs ${colorClass}`}>
                    {category}
                  </Badge>
                  <span className="ml-auto text-xs text-zinc-500">
                    {timeAgo < 60 ? `${timeAgo}s ago` : `${Math.floor(timeAgo / 60)}m ago`}
                  </span>
                </div>
                <div className="mt-1 truncate text-xs text-zinc-400">
                  {resultText}
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
    const categories: Record<string, number> = {};
    const toolCounts: Record<string, number> = {};
    
    tools.forEach(t => {
      const category = getToolCategory(t.tool);
      categories[category] = (categories[category] || 0) + 1;
      toolCounts[t.tool] = (toolCounts[t.tool] || 0) + 1;
    });
    
    const topTool = Object.entries(toolCounts)
      .sort((a, b) => b[1] - a[1])[0];
    
    return {
      total: tools.length,
      categories,
      topTool: topTool ? { name: topTool[0], count: topTool[1] } : null
    };
  }, [tools]);
  
  return (
    <div className="grid grid-cols-2 gap-2">
      <Card className="border-zinc-800/60 bg-zinc-900/70">
        <CardContent className="pt-4">
          <div className="text-2xl font-bold text-zinc-100">
            {stats.total}
          </div>
          <div className="text-xs text-zinc-500">
            Total Tools Used
          </div>
        </CardContent>
      </Card>

      <Card className="border-zinc-800/60 bg-zinc-900/70">
        <CardContent className="pt-4">
          <div className="text-2xl font-bold text-zinc-100">
            {Object.keys(stats.categories).length}
          </div>
          <div className="text-xs text-zinc-500">
            Categories Active
          </div>
        </CardContent>
      </Card>
      
      {stats.topTool && (
        <Card className="bg-slate-900/50 border-slate-700 col-span-2">
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-slate-200">
                  {stats.topTool.name}
                </div>
                <div className="text-xs text-slate-400">
                  Most Used Tool
                </div>
              </div>
              <Badge variant="outline" className="text-lg">
                {stats.topTool.count}x
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
