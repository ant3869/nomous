import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clock, Sparkles, RefreshCw, Users, MapPin, Package, Star, Calendar, Filter, TrendingUp } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

export interface TimelineEvent {
  id: string;
  entity_id?: string;
  entity_name?: string;
  entity_type?: "person" | "place" | "object" | "preference";
  event_type: "discovery" | "reinforcement" | "update" | "forget";
  description: string;
  metadata?: Record<string, unknown>;
  timestamp: string;
}

interface LearningTimelineProps {
  events: TimelineEvent[];
  selectedEntityId?: string | null;
  onSelectEntity?: (entityId: string) => void;
  limit?: number;
}

const EVENT_TYPE_CONFIG = {
  discovery: {
    icon: Sparkles,
    color: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
    label: "Discovery",
    bgGlow: "bg-cyan-500/5",
  },
  reinforcement: {
    icon: RefreshCw,
    color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
    label: "Reinforcement",
    bgGlow: "bg-emerald-500/5",
  },
  update: {
    icon: TrendingUp,
    color: "text-amber-400 bg-amber-500/10 border-amber-500/30",
    label: "Update",
    bgGlow: "bg-amber-500/5",
  },
  forget: {
    icon: Clock,
    color: "text-red-400 bg-red-500/10 border-red-500/30",
    label: "Forgotten",
    bgGlow: "bg-red-500/5",
  },
};

const ENTITY_TYPE_ICONS = {
  person: Users,
  place: MapPin,
  object: Package,
  preference: Star,
};

const ENTITY_TYPE_COLORS = {
  person: "text-emerald-300 bg-emerald-500/10 border-emerald-500/20",
  place: "text-cyan-300 bg-cyan-500/10 border-cyan-500/20",
  object: "text-amber-300 bg-amber-500/10 border-amber-500/20",
  preference: "text-purple-300 bg-purple-500/10 border-purple-500/20",
};

export function LearningTimeline({ events, selectedEntityId, onSelectEntity, limit = 50 }: LearningTimelineProps) {
  const [filterType, setFilterType] = useState<string>("all");
  const [filterEntity, setFilterEntity] = useState<string>("all");

  // Get unique entity types for filtering
  const entityTypes = useMemo(() => {
    const types = new Set<string>();
    events.forEach((e) => {
      if (e.entity_type) types.add(e.entity_type);
    });
    return Array.from(types).sort();
  }, [events]);

  // Filter events
  const filteredEvents = useMemo(() => {
    let filtered = events;

    if (filterType !== "all") {
      filtered = filtered.filter((e) => e.event_type === filterType);
    }

    if (filterEntity !== "all") {
      filtered = filtered.filter((e) => e.entity_type === filterEntity);
    }

    if (selectedEntityId) {
      filtered = filtered.filter((e) => e.entity_id === selectedEntityId);
    }

    return filtered.slice(0, limit);
  }, [events, filterType, filterEntity, selectedEntityId, limit]);

  // Group events by date
  const groupedEvents = useMemo(() => {
    const groups: Record<string, TimelineEvent[]> = {};
    filteredEvents.forEach((event) => {
      const date = new Date(event.timestamp).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });
      if (!groups[date]) groups[date] = [];
      groups[date].push(event);
    });
    return Object.entries(groups).sort(([a], [b]) => new Date(b).getTime() - new Date(a).getTime());
  }, [filteredEvents]);

  const formatTime = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "unknown";
    }
  };

  const getRelativeTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const seconds = Math.floor(diff / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);

      if (days > 1) return `${days} days ago`;
      if (days === 1) return "yesterday";
      if (hours > 0) return `${hours}h ago`;
      if (minutes > 0) return `${minutes}m ago`;
      return "just now";
    } catch {
      return "unknown";
    }
  };

  return (
    <Card className="border-zinc-800/60 bg-zinc-950/40">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-zinc-100">Timeline</h3>
          </div>
          <Badge className="bg-zinc-900/60 text-zinc-400 border border-zinc-800 text-xs">
            {filteredEvents.length}
          </Badge>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 mb-3">
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="w-[130px] h-8 text-[11px] bg-zinc-900/60 border-zinc-800">
              <SelectValue placeholder="Event type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Events</SelectItem>
              <SelectItem value="discovery">Discovery</SelectItem>
              <SelectItem value="reinforcement">Reinforcement</SelectItem>
              <SelectItem value="update">Update</SelectItem>
              <SelectItem value="forget">Forgotten</SelectItem>
            </SelectContent>
          </Select>

          <Select value={filterEntity} onValueChange={setFilterEntity}>
            <SelectTrigger className="w-[130px] h-8 text-[11px] bg-zinc-900/60 border-zinc-800">
              <SelectValue placeholder="Entity type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Entities</SelectItem>
              {entityTypes.map((type) => (
                <SelectItem key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {(filterType !== "all" || filterEntity !== "all") && (
            <Button
              variant="secondary"
              className="h-8 px-2 text-[11px] text-zinc-400 hover:text-zinc-100"
              onClick={() => {
                setFilterType("all");
                setFilterEntity("all");
              }}
            >
              Clear
            </Button>
          )}
        </div>

        {/* Timeline */}
        <div className="space-y-6 max-h-[500px] overflow-y-auto pr-2">
          <AnimatePresence mode="popLayout">
            {groupedEvents.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center py-8 text-zinc-500 text-sm"
              >
                No timeline events found
              </motion.div>
            ) : (
              groupedEvents.map(([date, dayEvents]) => (
                <motion.div
                  key={date}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <Calendar className="w-4 h-4 text-zinc-500" />
                    <h4 className="text-sm font-medium text-zinc-300">{date}</h4>
                    <div className="flex-1 h-px bg-zinc-800/60" />
                  </div>

                  <div className="space-y-3 pl-6 border-l-2 border-zinc-800/40">
                    {dayEvents.map((event, index) => {
                      const config = EVENT_TYPE_CONFIG[event.event_type];
                      const EventIcon = config.icon;
                      const EntityIcon = event.entity_type ? ENTITY_TYPE_ICONS[event.entity_type] : null;
                      const entityColor = event.entity_type ? ENTITY_TYPE_COLORS[event.entity_type] : "";

                      return (
                        <motion.div
                          key={event.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                          transition={{ delay: index * 0.05, duration: 0.2 }}
                          className={`relative rounded-lg border p-3 ${config.bgGlow} border-zinc-800/60 bg-zinc-900/40`}
                        >
                          {/* Timeline dot */}
                          <div className="absolute -left-[29px] top-4 w-2 h-2 rounded-full bg-zinc-700 ring-4 ring-zinc-950" />

                          <div className="flex items-start gap-3">
                            <div className={`p-2 rounded-lg ${config.color} border flex-shrink-0`}>
                              <EventIcon className="w-4 h-4" />
                            </div>

                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1 flex-wrap">
                                <Badge className={`text-[10px] px-1.5 py-0 ${config.color} border`}>
                                  {config.label}
                                </Badge>
                                <span className="text-[10px] text-zinc-500">{formatTime(event.timestamp)}</span>
                                <span className="text-[10px] text-zinc-600">•</span>
                                <span className="text-[10px] text-zinc-500">{getRelativeTime(event.timestamp)}</span>
                              </div>

                              <p className="text-sm text-zinc-200 mb-2">{event.description}</p>

                              {(event.entity_name || event.entity_type) && (
                                <div className="flex items-center gap-2">
                                  {EntityIcon && (
                                    <div className={`p-1 rounded ${entityColor} border`}>
                                      <EntityIcon className="w-3 h-3" />
                                    </div>
                                  )}
                                  {event.entity_name && (
                                    <Button
                                      variant="secondary"
                                      className="h-auto p-0 text-xs text-zinc-400 hover:text-zinc-100 bg-transparent hover:bg-transparent underline"
                                      onClick={() => {
                                        if (event.entity_id && onSelectEntity) {
                                          onSelectEntity(event.entity_id);
                                        }
                                      }}
                                    >
                                      {event.entity_name}
                                    </Button>
                                  )}
                                  {event.entity_type && (
                                    <Badge className={`text-[10px] px-1.5 py-0 ${entityColor} border`}>
                                      {event.entity_type}
                                    </Badge>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
}
