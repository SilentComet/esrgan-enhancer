/**
 * Frontend TypeScript Definitions
 * ===============================
 * 
 * Authored by: Frontend Engineering Team
 * Date: April 2026
 * Version: 1.0
 * 
 * Description:
 *     Exposes shared data contracts and interfaces for tasks, API settings,
 *     and UI state representation.
 */

export interface EnhancementTask {
  taskId: string;
  inputFile: File;
  inputUrl: string;
  outputUrl?: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  estimatedTime?: number;
  error?: string;
}

export interface ModelInfo {
  model: string;
  scale_factors: number[];
  max_file_size_mb: number;
  supported_formats: string[];
  device: string;
  cuda_available: boolean;
  precision: string;
  version: string;
}
