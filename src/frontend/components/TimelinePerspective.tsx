import React, { useMemo, useState } from "react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Slider } from "./ui/slider";
import { Layers, Share2, Clock4 } from "lucide-react";
import type { TimelineEvent } from "../types/memory";

type TimelineMode = "entity" | "event";

const EVENT_TYPE_COLORS: Record<string, { dot: string; glow: string }> = {
  discovery: { dot: "bg-cyan-400", glow: "bg-cyan-500/20" },
  reinforcement: { dot: "bg-emerald-400", glow: "bg-emerald-500/20" },
  update: { dot: "bg-amber-400", glow: "bg-amber-500/20" },
  forget: { dot: "bg-red-400", glow: "bg-red-500/20" },
};

interface TimelinePerspectiveProps {
  events: TimelineEvent[];
}

export function TimelinePerspective({ events }: TimelinePerspectiveProps) {
  const [mode, setMode] = useState<TimelineMode>("entity");
  const [windowSize, setWindowSize] = useState([30]);
  const [hovered, setHovered] = useState<TimelineEvent | null>(null);

  const sortedEvents = useMemo(() => {
    return [...events]
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .slice(-windowSize[0]);
  }, [events, windowSize]);

  const timeBounds = useMemo(() => {
    if (sortedEvents.length === 0) {
      return { min: 0, max: 1 };
    }
    const min = new Date(sortedEvents[0].timestamp).getTime();
    const max = new Date(sortedEvents[sortedEvents.length - 1].timestamp).getTime();
    return { min, max: Math.max(min + 1, max) };
  }, [sortedEvents]);

  const groupedEvents = useMemo(() => {
    const map = new Map<string, TimelineEvent[]>();
    sortedEvents.forEach(event => {
      const key = mode === "entity" ? event.entity_type || "unknown" : event.event_type;
      if (!map.has(key)) {
        map.set(key, []);
      }
      map.get(key)!.push(event);
    });
    return Array.from(map.entries()).sort((a, b) => b[1].length - a[1].length);
  }, [sortedEvents, mode]);

  const formatTimestamp = (ts?: string) => {
    if (!ts) return "unknown";
    try {
      return new Date(ts).toLocaleString();
    } catch {
      return ts;
    }
  };

  return (
    <Card className="border-zinc-800/60 bg-zinc-950/40">
      <CardContent className="space-y-3 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
            <Layers className="h-4 w-4 text-cyan-400" />
            <span>Perspective</span>
          </div>
          <Badge className="bg-zinc-900/60 text-zinc-400 border border-zinc-800 text-xs">
            {sortedEvents.length}
          </Badge>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            variant={mode === "entity" ? "primary" : "secondary"}
            className={mode === "entity" ? "bg-cyan-600/80 text-white" : "text-zinc-300"}
            onClick={() => setMode("entity")}
          >
            <Layers className="mr-2 h-3.5 w-3.5" />
            Entity lanes
          </Button>
          <Button
            type="button"
            variant={mode === "event" ? "primary" : "secondary"}
            className={mode === "event" ? "bg-emerald-600/80 text-white" : "text-zinc-300"}
            onClick={() => setMode("event")}
          >
            <Clock4 className="mr-2 h-3.5 w-3.5" />
            Event types
          </Button>
        </div>

        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px] text-zinc-400 uppercase tracking-[0.2em]">
            <span>Window</span>
            <span>{windowSize[0]} events</span>
          </div>
          <Slider
            defaultValue={windowSize}
            onValueChange={setWindowSize}
            min={10}
            max={80}
            step={5}
          />
        </div>

        {sortedEvents.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-800/60 bg-zinc-950/40 p-6 text-center text-sm text-zinc-500">
            Learning timeline has no events yet.
          </div>
        ) : (
          <div className="space-y-3">
            {groupedEvents.map(([key, bucket]) => (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-zinc-500">
                  <span>{key}</span>
                  <span className="text-zinc-400">{bucket.length} events</span>
                </div>
                <div className="relative h-12 overflow-hidden rounded-lg border border-zinc-800/60 bg-gradient-to-r from-zinc-950 via-zinc-900 to-zinc-950 px-2">
                  <div className="absolute left-2 right-2 top-1/2 h-px -translate-y-1/2 bg-zinc-800/60" />
                  {bucket.map(event => {
                    const ts = new Date(event.timestamp).getTime();
                    const position = ((ts - timeBounds.min) / (timeBounds.max - timeBounds.min)) * 100;
                    const colors = EVENT_TYPE_COLORS[event.event_type] || { dot: "bg-zinc-400", glow: "bg-zinc-500/10" };
                    const isActive = hovered?.id === event.id;
                    return (
                      <button
                        key={event.id}
                        type="button"
                        style={{ left: `${position}%` }}
                        onMouseEnter={() => setHovered(event)}
                        onFocus={() => setHovered(event)}
                        onMouseLeave={() => setHovered(prev => (prev?.id === event.id ? null : prev))}
                        onBlur={() => setHovered(prev => (prev?.id === event.id ? null : prev))}
                        className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full transition-all ${isActive ? "ring-2 ring-emerald-400/60" : "ring-0"}`}
                        aria-label={event.description}
                      >
                        <span
                          className={`block h-2 w-2 rounded-full ${colors.dot} shadow-[0_0_12px_rgba(16,185,129,0.4)]`}
                        />
                        <span
                          className={`absolute inset-0 -z-10 m-auto block h-4 w-4 rounded-full ${colors.glow}`}
                        />
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="rounded-lg border border-zinc-800/70 bg-black/40 p-3 text-sm text-zinc-300 min-h-[70px]">
          {hovered ? (
            <div className="space-y-1">
              <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Selected point</div>
              <div className="font-semibold text-zinc-100">{hovered.description}</div>
              <div className="text-[11px] text-zinc-500">
                {hovered.entity_name ? `${hovered.entity_name} • ` : ""}
                {hovered.event_type} • {formatTimestamp(hovered.timestamp)}
              </div>
            </div>
          ) : (
            <div className="text-xs text-zinc-500">Hover any point to inspect the precise description and timestamp.</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
