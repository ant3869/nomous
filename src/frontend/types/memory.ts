export type MemoryNodeKind = "stimulus" | "concept" | "event" | "self";

export interface MemoryNode {
  id: string;
  label: string;
  strength: number;
  kind: MemoryNodeKind;
  /** Optional descriptive text about the memory */
  description?: string;
  /** ISO timestamp of the last time this memory was updated */
  timestamp?: string;
  /** Optional tags used to group memories */
  tags?: string[];
  /** Marks the node as a major milestone regardless of its kind */
  milestone?: boolean;
  /** The origin or modality that produced this memory */
  source?: string;
  /** Confidence score reported by the runtime */
  confidence?: number;
}

export interface MemoryEdge {
  id: string;
  from: string;
  to: string;
  weight: number;
  /** Additional context about why the edge exists */
  context?: string;
  /** When the association was last strengthened */
  lastStrengthChange?: string;
}
