import type { MemoryEdge, MemoryNode } from "../types/memory";

export interface MemoryInsightEntry {
  id: string;
  label: string;
  strength: number;
  connections: number;
  kind: MemoryNode["kind"];
  tags: string[];
  milestone: boolean;
  source?: string;
  timestamp?: string;
}

export interface MemorySummary {
  totalNodes: number;
  totalEdges: number;
  conceptCount: number;
  stimulusCount: number;
  eventCount: number;
  selfCount: number;
  averageStrength: number;
  density: number;
}

export interface MemoryInsights {
  nodeById: Map<string, MemoryNode>;
  connectionIndex: Map<string, number>;
  milestones: MemoryInsightEntry[];
  dataEntries: MemoryInsightEntry[];
  summary: MemorySummary;
}

const asArray = <T,>(value: T[] | undefined): T[] => (value ? value.filter(Boolean) : []);

function countConnections(edges: MemoryEdge[]): Map<string, number> {
  const counts = new Map<string, number>();
  edges.forEach(edge => {
    counts.set(edge.from, (counts.get(edge.from) ?? 0) + 1);
    counts.set(edge.to, (counts.get(edge.to) ?? 0) + 1);
  });
  return counts;
}

function toInsightEntry(node: MemoryNode, connectionCount: number): MemoryInsightEntry {
  return {
    id: node.id,
    label: node.label,
    strength: node.strength,
    connections: connectionCount,
    kind: node.kind,
    tags: asArray(node.tags),
    milestone: Boolean(node.milestone || node.kind === "event"),
    source: node.source,
    timestamp: node.timestamp,
  };
}

export function computeMemoryInsights(nodes: MemoryNode[], edges: MemoryEdge[]): MemoryInsights {
  const nodeById = new Map(nodes.map(node => [node.id, node] as const));
  const connectionIndex = countConnections(edges);

  const baselineStrength = nodes.length
    ? nodes.reduce((sum, node) => sum + node.strength, 0) / nodes.length
    : 0;

  const summary: MemorySummary = {
    totalNodes: nodes.length,
    totalEdges: edges.length,
    conceptCount: nodes.filter(node => node.kind === "concept").length,
    stimulusCount: nodes.filter(node => node.kind === "stimulus").length,
    eventCount: nodes.filter(node => node.kind === "event").length,
    selfCount: nodes.filter(node => node.kind === "self").length,
    averageStrength: baselineStrength,
    density: nodes.length > 1 ? edges.length / (nodes.length * (nodes.length - 1)) : 0,
  };

  const entries = nodes.map(node => toInsightEntry(node, connectionIndex.get(node.id) ?? 0));

  const milestones = entries
    .filter(entry => entry.milestone)
    .sort((a, b) => {
      const scoreA = a.strength * 0.7 + a.connections * 0.3;
      const scoreB = b.strength * 0.7 + b.connections * 0.3;
      return scoreB - scoreA;
    });

  const dataEntries = entries
    .filter(entry => entry.kind === "stimulus")
    .sort((a, b) => {
      const scoreA = a.connections + a.strength;
      const scoreB = b.connections + b.strength;
      return scoreB - scoreA;
    });

  return { nodeById, connectionIndex, milestones, dataEntries, summary };
}

export interface MemoryNodeDetail {
  id: string;
  label: string;
  kind: MemoryNode["kind"];
  description?: string;
  tags: string[];
  source?: string;
  strength: number;
  connections: number;
  confidence?: number;
  timestamp?: string;
  related: Array<{ id: string; label: string; strength: number; direction: "inbound" | "outbound"; weight: number }>;
}

export function buildMemoryNodeDetail(
  node: MemoryNode,
  edges: MemoryEdge[],
  connectionIndex: Map<string, number>,
  nodeById: Map<string, MemoryNode>,
): MemoryNodeDetail {
  const related = edges
    .filter(edge => edge.from === node.id || edge.to === node.id)
    .map(edge => {
      const relatedId = edge.from === node.id ? edge.to : edge.from;
      const relatedNode = nodeById.get(relatedId);
      return {
        id: relatedId,
        label: relatedNode?.label ?? relatedId,
        strength: relatedNode?.strength ?? 0,
        direction: edge.from === node.id ? "outbound" : "inbound",
        weight: edge.weight,
      };
    })
    .sort((a, b) => b.weight - a.weight || b.strength - a.strength);

  return {
    id: node.id,
    label: node.label,
    kind: node.kind,
    description: node.description,
    tags: asArray(node.tags),
    source: node.source,
    strength: node.strength,
    connections: connectionIndex.get(node.id) ?? related.length,
    confidence: node.confidence,
    timestamp: node.timestamp,
    related,
  };
}
