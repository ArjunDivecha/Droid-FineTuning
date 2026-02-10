import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface TrainingMetrics {
  current_step: number;
  total_steps: number;
  train_loss: number | null;  // Can be null before first measurement
  val_loss: number | null;    // Can be null before first measurement
  learning_rate: number;
  start_time: string;
  estimated_time_remaining: number | null;
  // Optional RL metrics (GSPO/GRPO)
  avg_reward?: number | null;
  success_rate?: number | null; // percentage (0-100) or fraction (0-1) depending on trainer
  kl?: number | null;
  entropy?: number | null;
}

export interface TrainingConfig {
  model_path: string;
  train_data_path: string;
  val_data_path: string;
  learning_rate: number;
  batch_size: number;
  max_seq_length: number;
  iterations: number;
  steps_per_report: number;
  steps_per_eval: number;
  save_every: number;
  early_stop: boolean;
  patience: number;
  adapter_name: string;
  // Full-Layer LoRA Configuration
  fine_tune_type?: string;
  lora_rank?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  lora_num_layers?: number;
  // Enhanced/GSPO-GRPO fields (optional)
  training_method?: string;
  group_size?: number;
  epsilon?: number;
  temperature?: number;
  max_completion_length?: number;
  importance_sampling_level?: string | null;
  grpo_loss_type?: string;
}

export type TrainingState = 'idle' | 'running' | 'paused' | 'completed' | 'error' | 'stopped';

interface TrainingSliceState {
  state: TrainingState;
  config: TrainingConfig | null;
  metrics: TrainingMetrics | null;
  logs: string[];
  error: string | null;
  isConnected: boolean;
}

const initialState: TrainingSliceState = {
  state: 'idle',
  config: null,
  metrics: null,
  logs: [],
  error: null,
  isConnected: false,
};

export const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    setTrainingState: (state, action: PayloadAction<TrainingState>) => {
      state.state = action.payload;
    },
    setTrainingConfig: (state, action: PayloadAction<TrainingConfig>) => {
      state.config = action.payload;
    },
    setTrainingMetrics: (state, action: PayloadAction<TrainingMetrics>) => {
      state.metrics = action.payload;
    },
    updateTrainingMetrics: (state, action: PayloadAction<Partial<TrainingMetrics>>) => {
      if (state.metrics) {
        state.metrics = { ...state.metrics, ...action.payload };
      }
    },
    addLogLine: (state, action: PayloadAction<string>) => {
      state.logs.push(action.payload);
      // Keep only last 1000 lines
      if (state.logs.length > 1000) {
        state.logs = state.logs.slice(-1000);
      }
    },
    clearLogs: (state) => {
      state.logs = [];
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    resetTraining: (state) => {
      // Complete reset to initial state
      state.state = 'idle';
      state.metrics = null;
      state.error = null;
      state.logs = [];
      state.config = null;
    },
    trainingStarted: (state) => {
      state.state = 'running';
      state.error = null;
      state.metrics = null; // Clear old metrics
      state.logs = []; // Clear logs
      // Also clear config to force new configuration on next training
    },
    trainingProgress: (state, action: PayloadAction<{ metrics: TrainingMetrics; log_line: string }>) => {
      // Handle both direct payload and nested data structures
      const data = action.payload;
      if (data && data.metrics) {
        const newMetrics = data.metrics;
        
        // If step regressed significantly, assume a fresh run and clear logs
        if (state.metrics && typeof newMetrics.current_step === 'number' && 
            newMetrics.current_step < state.metrics.current_step - 10) {
          state.logs = [];
        }
        
        // Merge metrics: preserve existing values if new ones are null/undefined
        // This ensures we don't lose val_loss when only train_loss is reported
        state.metrics = {
          ...state.metrics,
          ...newMetrics,
          // Explicitly handle nullable fields - don't overwrite valid values with null
          train_loss: newMetrics.train_loss !== undefined ? newMetrics.train_loss : (state.metrics?.train_loss ?? null),
          val_loss: newMetrics.val_loss !== undefined ? newMetrics.val_loss : (state.metrics?.val_loss ?? null),
          learning_rate: newMetrics.learning_rate !== undefined ? newMetrics.learning_rate : (state.metrics?.learning_rate ?? 0),
          estimated_time_remaining: newMetrics.estimated_time_remaining !== undefined 
            ? newMetrics.estimated_time_remaining 
            : (state.metrics?.estimated_time_remaining ?? null),
          // RL metrics
          avg_reward: newMetrics.avg_reward !== undefined ? newMetrics.avg_reward : (state.metrics?.avg_reward ?? null),
          success_rate: newMetrics.success_rate !== undefined ? newMetrics.success_rate : (state.metrics?.success_rate ?? null),
          kl: newMetrics.kl !== undefined ? newMetrics.kl : (state.metrics?.kl ?? null),
          entropy: newMetrics.entropy !== undefined ? newMetrics.entropy : (state.metrics?.entropy ?? null),
        };
        
        // Ensure state reflects running when progress arrives
        state.state = 'running';
      }
      // Only add non-empty log lines to avoid spam
      if (data && data.log_line && data.log_line.trim().length > 0) {
        state.logs.push(data.log_line);
        // Keep only last 1000 lines
        if (state.logs.length > 1000) {
          state.logs = state.logs.slice(-1000);
        }
      }
    },
    trainingCompleted: (state, action: PayloadAction<{ final_metrics: TrainingMetrics }>) => {
      state.state = 'completed';
      state.metrics = action.payload.final_metrics;
    },
    trainingStopped: (state) => {
      state.state = 'stopped';
    },
    trainingError: (state, action: PayloadAction<{ error: string }>) => {
      state.state = 'error';
      state.error = action.payload.error;
    },
  },
});

export const {
  setTrainingState,
  setTrainingConfig,
  setTrainingMetrics,
  updateTrainingMetrics,
  addLogLine,
  clearLogs,
  setError,
  setConnectionStatus,
  resetTraining,
  trainingStarted,
  trainingProgress,
  trainingCompleted,
  trainingStopped,
  trainingError,
} = trainingSlice.actions;
