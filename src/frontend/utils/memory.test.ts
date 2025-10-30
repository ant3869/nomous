import { describe, expect, it } from "vitest";
import type { MemoryEdge, MemoryNode } from "../types/memory";
import { buildMemoryNodeDetail, computeMemoryInsights } from "./memory";

describe("memory analytics", () => {
  const nodes: MemoryNode[] = [
    { id: "self", label: "Nomous", strength: 1, kind: "self", tags: ["agent"] },
    { id: "stimulus:1", label: "Camera Frame", strength: 0.6, kind: "stimulus", source: "vision", timestamp: "2024-10-01T12:00:00Z" },
    { id: "event:milestone", label: "First Autonomous Task", strength: 0.9, kind: "event", milestone: true, description: "Completed first unsupervised task." },
    { id: "concept:strategy", label: "Task Strategy", strength: 0.7, kind: "concept", tags: ["planning", "strategy"] },
  ];

  const edges: MemoryEdge[] = [
    { id: "e1", from: "self", to: "event:milestone", weight: 0.8 },
    { id: "e2", from: "stimulus:1", to: "concept:strategy", weight: 0.6 },
    { id: "e3", from: "event:milestone", to: "concept:strategy", weight: 0.9 },
  ];

  it("prioritises milestones by combined strength and connectivity", () => {
    const insights = computeMemoryInsights(nodes, edges);
    expect(insights.milestones.map(entry => entry.id)).toEqual([
      "event:milestone",
    ]);
    expect(insights.summary.totalNodes).toBe(4);
    expect(insights.summary.totalEdges).toBe(3);
    expect(insights.summary.density).toBeCloseTo(3 / 12);
  });

  it("derives data entry points from stimulus nodes", () => {
    const insights = computeMemoryInsights(nodes, edges);
    expect(insights.dataEntries).toHaveLength(1);
    expect(insights.dataEntries[0]).toMatchObject({ id: "stimulus:1", connections: 1, kind: "stimulus" });
  });

  it("builds node detail with related edges sorted by weight", () => {
    const insights = computeMemoryInsights(nodes, edges);
    const milestone = nodes[2];
    const detail = buildMemoryNodeDetail(milestone, edges, insights.connectionIndex, insights.nodeById);

    expect(detail.connections).toBe(2);
    expect(detail.related.map(item => item.id)).toEqual(["concept:strategy", "self"]);
    expect(detail.related[0].direction).toBe("outbound");
    expect(detail.tags).toEqual([]);
  });
});
