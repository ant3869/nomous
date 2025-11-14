import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Sparkles, Sliders, Users, MapPin, Package, Star, Brain, X } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

export interface SemanticSearchResult {
  node_id: string;
  text: string;
  label: string;
  description?: string;
  kind?: string;
  timestamp?: string;
  importance?: number;
  similarity: number;
}

interface SemanticSearchProps {
  onSearch: (query: string, options: { limit: number; threshold: number; entityType?: string }) => void;
  results?: SemanticSearchResult[];
  isSearching?: boolean;
  onSelectResult?: (nodeId: string) => void;
  selectedResultId?: string | null;
}

const ENTITY_TYPE_ICONS = {
  person: Users,
  place: MapPin,
  object: Package,
  preference: Star,
  stimulus: Sparkles,
  concept: Brain,
  event: Sparkles,
  self: Star,
  behavior: Users,
};

const ENTITY_TYPE_COLORS = {
  person: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  place: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
  object: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  preference: "text-purple-400 bg-purple-500/10 border-purple-500/30",
  stimulus: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  concept: "text-purple-400 bg-purple-500/10 border-purple-500/30",
  event: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
  self: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  behavior: "text-teal-400 bg-teal-500/10 border-teal-500/30",
};

export function SemanticSearch({
  onSearch,
  results = [],
  isSearching = false,
  onSelectResult,
  selectedResultId,
}: SemanticSearchProps) {
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState([0.7]);
  const [limit, setLimit] = useState([10]);
  const [entityType, setEntityType] = useState<string>("all");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (query.trim()) {
        onSearch(query, {
          limit: limit[0],
          threshold: threshold[0],
          entityType: entityType === "all" ? undefined : entityType,
        });
      }
    },
    [query, limit, threshold, entityType, onSearch]
  );

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.9) return "text-emerald-400 bg-emerald-500/20 border-emerald-500/40";
    if (similarity >= 0.8) return "text-cyan-400 bg-cyan-500/20 border-cyan-500/40";
    if (similarity >= 0.7) return "text-amber-400 bg-amber-500/20 border-amber-500/40";
    return "text-zinc-400 bg-zinc-500/20 border-zinc-500/40";
  };

  const getSimilarityLabel = (similarity: number) => {
    if (similarity >= 0.9) return "Excellent";
    if (similarity >= 0.8) return "Good";
    if (similarity >= 0.7) return "Fair";
    return "Low";
  };

  const formatDate = (isoString?: string) => {
    if (!isoString) return null;
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
      return null;
    }
  };

  return (
    <Card className="border-zinc-800/60 bg-black/40">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            <h3 className="font-semibold text-zinc-100">Semantic Search</h3>
          </div>
          <Button
            variant="secondary"
            className="h-7 text-xs text-zinc-400 hover:text-zinc-100"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Sliders className="w-3 h-3 mr-1" />
            {showAdvanced ? "Hide" : "Show"} Options
          </Button>
        </div>

        {/* Search Form */}
        <form onSubmit={handleSearch} className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
            <Input
              type="text"
              placeholder="Search by meaning, not just keywords..."
              value={query}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
              className="pl-10 pr-10 bg-zinc-900/60 border-zinc-800 text-zinc-100 placeholder:text-zinc-500"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Advanced Options */}
          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="space-y-3 overflow-hidden"
              >
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <label className="text-xs text-zinc-400 flex items-center justify-between">
                      <span>Similarity Threshold</span>
                      <span className="font-mono text-emerald-400">{(threshold[0] * 100).toFixed(0)}%</span>
                    </label>
                    <Slider
                      defaultValue={threshold}
                      onValueChange={setThreshold}
                      min={0.5}
                      max={1.0}
                      step={0.05}
                    />
                    <p className="text-[10px] text-zinc-500">Higher = more similar results</p>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-zinc-400 flex items-center justify-between">
                      <span>Result Limit</span>
                      <span className="font-mono text-cyan-400">{limit[0]}</span>
                    </label>
                    <Slider defaultValue={limit} onValueChange={setLimit} min={5} max={50} step={5} />
                    <p className="text-[10px] text-zinc-500">Max results to return</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Entity Type Filter</label>
                  <Select value={entityType} onValueChange={setEntityType}>
                    <SelectTrigger className="w-full h-9 bg-zinc-900/60 border-zinc-800">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Types</SelectItem>
                      <SelectItem value="person">Person</SelectItem>
                      <SelectItem value="place">Place</SelectItem>
                      <SelectItem value="object">Object</SelectItem>
                      <SelectItem value="preference">Preference</SelectItem>
                      <SelectItem value="concept">Concept</SelectItem>
                      <SelectItem value="event">Event</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <Button type="submit" className="w-full" disabled={!query.trim() || isSearching}>
            {isSearching ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="mr-2"
                >
                  <Sparkles className="w-4 h-4" />
                </motion.div>
                Searching...
              </>
            ) : (
              <>
                <Search className="w-4 h-4 mr-2" />
                Search Semantically
              </>
            )}
          </Button>
        </form>

        {/* Results */}
        {results.length > 0 && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium text-zinc-300">Results</h4>
              <Badge className="bg-zinc-900/60 text-zinc-300 border border-zinc-700">
                {results.length} {results.length === 1 ? "match" : "matches"}
              </Badge>
            </div>

            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
              <AnimatePresence mode="popLayout">
                {results.map((result, index) => {
                  const Icon = result.kind ? ENTITY_TYPE_ICONS[result.kind as keyof typeof ENTITY_TYPE_ICONS] : Search;
                  const color = result.kind ? ENTITY_TYPE_COLORS[result.kind as keyof typeof ENTITY_TYPE_COLORS] : "text-zinc-400 bg-zinc-500/10 border-zinc-500/30";
                  const isSelected = selectedResultId === result.node_id;
                  const dateStr = formatDate(result.timestamp);

                  return (
                    <motion.div
                      key={result.node_id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ delay: index * 0.05, duration: 0.2 }}
                      onClick={() => onSelectResult?.(result.node_id)}
                      className={`
                        rounded-lg border p-3 cursor-pointer transition-all
                        ${
                          isSelected
                            ? "border-emerald-500/60 bg-emerald-500/10 ring-1 ring-emerald-500/30"
                            : "border-zinc-800/60 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/60"
                        }
                      `}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg ${color} border flex-shrink-0`}>
                          <Icon className="w-4 h-4" />
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <h5 className="font-medium text-zinc-100">{result.label}</h5>
                            <Badge className={`text-[10px] px-1.5 py-0 ${getSimilarityColor(result.similarity)} border`}>
                              {Math.round(result.similarity * 100)}%
                            </Badge>
                            <Badge className="text-[10px] px-1.5 py-0 bg-zinc-800/60 text-zinc-300">
                              {getSimilarityLabel(result.similarity)}
                            </Badge>
                          </div>

                          {result.description && (
                            <p className="text-xs text-zinc-400 line-clamp-2 mb-2">{result.description}</p>
                          )}

                          <div className="flex items-center gap-3 text-[10px] text-zinc-500 flex-wrap">
                            {result.kind && (
                              <Badge className={`text-[10px] px-1.5 py-0 ${color} border`}>
                                {result.kind}
                              </Badge>
                            )}
                            {dateStr && (
                              <>
                                <span>•</span>
                                <span>{dateStr}</span>
                              </>
                            )}
                            {result.importance !== undefined && (
                              <>
                                <span>•</span>
                                <span title="Importance">
                                  <Star className="w-3 h-3 inline mr-0.5" fill={result.importance > 0.5 ? "currentColor" : "none"} />
                                  {Math.round(result.importance * 100)}%
                                </span>
                              </>
                            )}
                          </div>

                          {result.text && result.text !== result.label && (
                            <div className="mt-2 p-2 rounded bg-zinc-900/60 border border-zinc-800/40">
                              <p className="text-xs text-zinc-400 italic line-clamp-3">{result.text}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
