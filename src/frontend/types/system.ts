export interface ComputeDeviceInfo {
  backend: string;
  name: string;
  reason: string;
  cuda_version?: string | null;
  gpu_count?: number;
}

export interface CpuMetrics {
  percent: number;
  frequency?: number | null;
  cores?: number | null;
  load?: number[] | null;
}

export interface MemoryMetrics {
  total: number;
  used: number;
  available: number;
  percent: number;
}

export interface SwapMetrics {
  total: number;
  used: number;
  percent: number;
}

export interface GpuMetrics {
  name: string;
  percent: number;
  memory_total: number;
  memory_used: number;
  memory_percent: number;
  temperature?: number | null;
}

export interface SystemMetricsPayload {
  timestamp: number;
  device: ComputeDeviceInfo;
  cpu: CpuMetrics;
  memory: MemoryMetrics;
  swap?: SwapMetrics;
  gpu?: GpuMetrics | null;
}
