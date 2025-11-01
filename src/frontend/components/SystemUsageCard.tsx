import React from "react";
import { Card, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Activity, Cpu, MemoryStick, ThermometerSun, Zap } from "lucide-react";
import type { SystemMetricsPayload } from "../types/system";

const formatBytes = (value: number): string => {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = size >= 10 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
};

const clampPercent = (value: number | undefined | null): number => {
  if (!Number.isFinite(value ?? NaN)) return 0;
  return Math.max(0, Math.min(100, Number(value)));
};

const getDeviceBadgeStyle = (deviceType: string): string => {
  return deviceType.toUpperCase() === "GPU"
    ? "bg-emerald-600/80"
    : "bg-amber-500/30 text-amber-100";
};

interface SystemUsageCardProps {
  metrics: SystemMetricsPayload | null;
}

export const SystemUsageCard: React.FC<SystemUsageCardProps> = ({ metrics }) => {
  const deviceType = metrics?.device.backend ?? "CPU";
  const deviceName = metrics?.device.name ?? "Unavailable";
  const reason = metrics?.device.reason ?? "Waiting for runtime telemetry…";
  const gpuInfo = metrics?.gpu ?? null;

  return (
    <Card className="bg-zinc-900/70 border-zinc-800/60 h-full">
      <CardContent className="p-4 space-y-3 text-zinc-200">
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold">System Resources</span>
            <span className="text-xs text-zinc-400">{reason}</span>
          </div>
          <Badge className={`px-3 py-1 text-xs ${getDeviceBadgeStyle(deviceType)}`}>
            {deviceType.toUpperCase()}
          </Badge>
        </div>

        <div className="text-xs text-zinc-400">{deviceName}</div>

        <div className="space-y-3 pt-1">
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="flex items-center gap-1 text-zinc-300"><Cpu className="h-3.5 w-3.5" /> CPU Usage</span>
              <span>{clampPercent(metrics?.cpu.percent).toFixed(0)}%</span>
            </div>
            <Progress value={clampPercent(metrics?.cpu.percent)} className="h-2" />
          </div>

          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="flex items-center gap-1 text-zinc-300"><MemoryStick className="h-3.5 w-3.5" /> RAM</span>
              <span>
                {formatBytes(metrics?.memory.used ?? 0)} / {formatBytes(metrics?.memory.total ?? 0)}
              </span>
            </div>
            <Progress value={clampPercent(metrics?.memory.percent)} className="h-2" />
          </div>

          {gpuInfo ? (
            <div className="space-y-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
              <div className="flex items-center justify-between text-xs text-emerald-200">
                <span className="flex items-center gap-1"><Zap className="h-3.5 w-3.5" /> GPU Usage</span>
                <span>{clampPercent(gpuInfo.percent).toFixed(0)}%</span>
              </div>
              <Progress value={clampPercent(gpuInfo.percent)} className="h-2 bg-emerald-900/50" />
              <div className="flex items-center justify-between text-xs text-emerald-200/90">
                <span className="flex items-center gap-1"><Activity className="h-3.5 w-3.5" /> VRAM</span>
                <span>{formatBytes(gpuInfo.memory_used)} / {formatBytes(gpuInfo.memory_total)}</span>
              </div>
              <Progress value={clampPercent(gpuInfo.memory_percent)} className="h-2 bg-emerald-900/50" />
              {typeof gpuInfo.temperature === "number" && (
                <div className="flex items-center gap-1 text-xs text-emerald-200/80">
                  <ThermometerSun className="h-3.5 w-3.5" /> {gpuInfo.temperature.toFixed(0)}°C
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-lg border border-zinc-700/40 bg-zinc-800/30 p-3 text-xs text-zinc-300">
              GPU metrics unavailable. The runtime is currently using CPU execution.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
