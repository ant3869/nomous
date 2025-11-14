import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Users, MapPin, Package, Search, X, Edit2, Trash2, Star } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";

export interface Entity {
  id: string;
  entity_type: "person" | "place" | "object" | "preference";
  name: string;
  description?: string;
  properties?: Record<string, unknown>;
  first_seen: string;
  last_seen: string;
  occurrence_count: number;
  importance: number;
  similarity?: number; // For semantic search results
}

interface EntityBrowserProps {
  entities: Entity[];
  selectedEntityId?: string | null;
  onSelect: (id: string) => void;
  onEdit?: (entity: Entity) => void;
  onDelete?: (id: string) => void;
  onSearch?: (query: string, entityType?: string) => void;
}

const ENTITY_TYPE_ICONS = {
  person: Users,
  place: MapPin,
  object: Package,
  preference: Star,
};

const ENTITY_TYPE_COLORS = {
  person: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  place: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
  object: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  preference: "text-purple-400 bg-purple-500/10 border-purple-500/30",
};

const ENTITY_TYPE_LABELS = {
  person: "People",
  place: "Places",
  object: "Objects",
  preference: "Preferences",
};

export function EntityBrowser({
  entities,
  selectedEntityId,
  onSelect,
  onEdit,
  onDelete,
  onSearch,
}: EntityBrowserProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"all" | "person" | "place" | "object" | "preference">("all");

  // Filter entities by tab and search query
  const filteredEntities = useMemo(() => {
    let filtered = entities;

    // Filter by type
    if (activeTab !== "all") {
      filtered = filtered.filter((e) => e.entity_type === activeTab);
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (e) =>
          e.name.toLowerCase().includes(query) ||
          e.description?.toLowerCase().includes(query) ||
          Object.values(e.properties || {}).some((v) =>
            String(v).toLowerCase().includes(query)
          )
      );
    }

    // Sort by importance and occurrence
    return filtered.sort((a, b) => {
      if (a.similarity !== undefined && b.similarity !== undefined) {
        return b.similarity - a.similarity;
      }
      const importanceDiff = b.importance - a.importance;
      if (Math.abs(importanceDiff) > 0.01) return importanceDiff;
      return b.occurrence_count - a.occurrence_count;
    });
  }, [entities, activeTab, searchQuery]);

  // Count entities by type
  const entityCounts = useMemo(() => {
    const counts: Record<Entity["entity_type"], number> = {
      person: 0,
      place: 0,
      object: 0,
      preference: 0,
    };
    entities.forEach((e) => {
      counts[e.entity_type]++;
    });
    return counts;
  }, [entities]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onSearch && searchQuery.trim()) {
      onSearch(searchQuery, activeTab === "all" ? undefined : activeTab);
    }
  };

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const seconds = Math.floor(diff / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);

      if (days > 0) return `${days}d ago`;
      if (hours > 0) return `${hours}h ago`;
      if (minutes > 0) return `${minutes}m ago`;
      return "just now";
    } catch {
      return "unknown";
    }
  };

  return (
    <Card className="border-zinc-800/60 bg-black/40">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-emerald-400" />
            <h3 className="font-semibold text-zinc-100">Entity Memory</h3>
          </div>
          <Badge className="bg-zinc-900/60 text-zinc-300 border border-zinc-700">
            {entities.length} {entities.length === 1 ? "entity" : "entities"}
          </Badge>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSearchSubmit} className="mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
            <Input
              type="text"
              placeholder="Search entities..."
              value={searchQuery}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
              className="pl-10 pr-10 bg-zinc-900/60 border-zinc-800 text-zinc-100 placeholder:text-zinc-500"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>

        {/* Entity Type Tabs */}
        <Tabs defaultValue={activeTab} value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)} className="w-full">
          <TabsList className="grid w-full grid-cols-5 bg-zinc-900/60 border border-zinc-800">
            <TabsTrigger value="all" className="text-xs">
              All ({entities.length})
            </TabsTrigger>
            <TabsTrigger value="person" className="text-xs">
              <Users className="w-3 h-3 mr-1" />
              {entityCounts.person}
            </TabsTrigger>
            <TabsTrigger value="place" className="text-xs">
              <MapPin className="w-3 h-3 mr-1" />
              {entityCounts.place}
            </TabsTrigger>
            <TabsTrigger value="object" className="text-xs">
              <Package className="w-3 h-3 mr-1" />
              {entityCounts.object}
            </TabsTrigger>
            <TabsTrigger value="preference" className="text-xs">
              <Star className="w-3 h-3 mr-1" />
              {entityCounts.preference}
            </TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab} className="mt-4">
            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
              <AnimatePresence mode="popLayout">
                {filteredEntities.length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center py-8 text-zinc-500 text-sm"
                  >
                    {searchQuery ? "No entities found" : `No ${activeTab === "all" ? "" : ENTITY_TYPE_LABELS[activeTab as keyof typeof ENTITY_TYPE_LABELS].toLowerCase()} entities yet`}
                  </motion.div>
                ) : (
                  filteredEntities.map((entity, index) => {
                    const Icon = ENTITY_TYPE_ICONS[entity.entity_type];
                    const isSelected = selectedEntityId === entity.id;
                    return (
                      <motion.div
                        key={entity.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ delay: index * 0.03, duration: 0.2 }}
                        onClick={() => onSelect(entity.id)}
                        className={`
                          rounded-lg border p-3 cursor-pointer transition-all
                          ${
                            isSelected
                              ? "border-emerald-500/60 bg-emerald-500/10 ring-1 ring-emerald-500/30"
                              : "border-zinc-800/60 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/60"
                          }
                        `}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-start gap-3 flex-1 min-w-0">
                            <div className={`p-2 rounded-lg ${ENTITY_TYPE_COLORS[entity.entity_type]} border flex-shrink-0`}>
                              <Icon className="w-4 h-4" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <h4 className="font-medium text-zinc-100 truncate">{entity.name}</h4>
                                <Badge className="text-[10px] px-1.5 py-0 bg-zinc-800/60 text-zinc-300">
                                  {entity.occurrence_count}x
                                </Badge>
                                {entity.similarity !== undefined && (
                                  <Badge className="text-[10px] px-1.5 py-0 bg-emerald-900/30 text-emerald-300 border border-emerald-500/30">
                                    {Math.round((entity.similarity || 0) * 100)}%
                                  </Badge>
                                )}
                              </div>
                              {entity.description && (
                                <p className="text-xs text-zinc-400 line-clamp-2 mb-2">{entity.description}</p>
                              )}
                              <div className="flex items-center gap-3 text-[10px] text-zinc-500">
                                <span>First seen {formatDate(entity.first_seen)}</span>
                                <span>•</span>
                                <span>Last seen {formatDate(entity.last_seen)}</span>
                                <span>•</span>
                                <span title="Importance">
                                  <Star className="w-3 h-3 inline mr-0.5" fill={entity.importance > 0.5 ? "currentColor" : "none"} />
                                  {Math.round(entity.importance * 100)}%
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-1 flex-shrink-0">
                            {onEdit && (
                              <Button
                                variant="secondary"
                                className="h-7 w-7 p-0 text-zinc-400 hover:text-zinc-100"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onEdit(entity);
                                }}
                              >
                                <Edit2 className="w-3 h-3" />
                              </Button>
                            )}
                            {onDelete && (
                              <Button
                                variant="secondary"
                                className="h-7 w-7 p-0 text-zinc-400 hover:text-red-400"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onDelete(entity.id);
                                }}
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    );
                  })
                )}
              </AnimatePresence>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
