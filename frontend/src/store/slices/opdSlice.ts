import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface OPDMetrics {
  step: number;
  total_steps: number;
  progress_pct: number;
  kl_loss: number | null;
  token_agreement_pct: number | null;
  student_entropy?: number | null;
  teacher_entropy?: number | null;
  started_at: string;
  completed_at?: string;
}

export interface OPDConfig {
  base_model_path: string;
  teacher_model_path: string;
  student_adapter_path: string;
  validation_prompts_path: string;
  num_steps: number;
  batch_size: number;
  temperature: number;
  kl_weight: number;
  learning_rate: number;
}

export type OPDState = 'idle' | 'running' | 'completed' | 'error' | 'stopped';

export interface OPDRun {
  run_id: string;
  status: OPDState;
  started_at: string;
  completed_at?: string;
  teacher_model: string;
  student_model: string;
  student_adapter: string;
  config: OPDConfig;
  final_metrics?: OPDMetrics;
}

interface OPDSliceState {
  state: OPDState;
  config: OPDConfig | null;
  metrics: OPDMetrics | null;
  currentRunId: string | null;
  runs: OPDRun[];
  metricsHistory: OPDMetrics[];
  error: string | null;
  estimatedDuration: number | null; // minutes
}

const initialState: OPDSliceState = {
  state: 'idle',
  config: null,
  metrics: null,
  currentRunId: null,
  runs: [],
  metricsHistory: [],
  error: null,
  estimatedDuration: null,
};

export const opdSlice = createSlice({
  name: 'opd',
  initialState,
  reducers: {
    setOPDState: (state, action: PayloadAction<OPDState>) => {
      state.state = action.payload;
    },
    setOPDConfig: (state, action: PayloadAction<OPDConfig>) => {
      state.config = action.payload;
    },
    setOPDMetrics: (state, action: PayloadAction<OPDMetrics>) => {
      state.metrics = action.payload;
    },
    updateOPDMetrics: (state, action: PayloadAction<Partial<OPDMetrics>>) => {
      if (state.metrics) {
        state.metrics = { ...state.metrics, ...action.payload };
      }
    },
    setCurrentRunId: (state, action: PayloadAction<string | null>) => {
      state.currentRunId = action.payload;
    },
    setRuns: (state, action: PayloadAction<OPDRun[]>) => {
      state.runs = action.payload;
    },
    addMetricsToHistory: (state, action: PayloadAction<OPDMetrics>) => {
      state.metricsHistory.push(action.payload);
      // Keep last 1000 metrics points
      if (state.metricsHistory.length > 1000) {
        state.metricsHistory = state.metricsHistory.slice(-1000);
      }
    },
    setMetricsHistory: (state, action: PayloadAction<OPDMetrics[]>) => {
      state.metricsHistory = action.payload;
    },
    clearMetricsHistory: (state) => {
      state.metricsHistory = [];
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setEstimatedDuration: (state, action: PayloadAction<number | null>) => {
      state.estimatedDuration = action.payload;
    },
    resetOPD: (state) => {
      state.state = 'idle';
      state.config = null;
      state.metrics = null;
      state.currentRunId = null;
      state.metricsHistory = [];
      state.error = null;
      state.estimatedDuration = null;
    },
    // WebSocket event handlers
    opdStarted: (state, action: PayloadAction<{ run_id: string; config: OPDConfig; estimated_duration_minutes: number }>) => {
      state.state = 'running';
      state.currentRunId = action.payload.run_id;
      state.config = action.payload.config;
      state.error = null;
      state.metricsHistory = [];
      state.estimatedDuration = action.payload.estimated_duration_minutes;
      state.metrics = {
        step: 0,
        total_steps: action.payload.config.num_steps,
        progress_pct: 0,
        kl_loss: null,
        token_agreement_pct: null,
        started_at: new Date().toISOString(),
      };
    },
    opdProgress: (state, action: PayloadAction<OPDMetrics>) => {
      state.state = 'running';
      state.metrics = action.payload;
      // Add to history if it's a new step
      if (!state.metricsHistory.length ||
          state.metricsHistory[state.metricsHistory.length - 1].step !== action.payload.step) {
        state.metricsHistory.push(action.payload);
        if (state.metricsHistory.length > 1000) {
          state.metricsHistory = state.metricsHistory.slice(-1000);
        }
      }
    },
    opdCompleted: (state, action: PayloadAction<{ run_id: string; final_metrics: OPDMetrics; message: string }>) => {
      state.state = 'completed';
      state.metrics = action.payload.final_metrics;
      if (action.payload.final_metrics.completed_at) {
        state.metrics.completed_at = action.payload.final_metrics.completed_at;
      }
    },
    opdStopped: (state, action: PayloadAction<{ final_step: number; checkpoint_path: string }>) => {
      state.state = 'stopped';
      if (state.metrics) {
        state.metrics.step = action.payload.final_step;
      }
    },
    opdError: (state, action: PayloadAction<{ error: string }>) => {
      state.state = 'error';
      state.error = action.payload.error;
    },
  },
});

export const {
  setOPDState,
  setOPDConfig,
  setOPDMetrics,
  updateOPDMetrics,
  setCurrentRunId,
  setRuns,
  addMetricsToHistory,
  setMetricsHistory,
  clearMetricsHistory,
  setError,
  setEstimatedDuration,
  resetOPD,
  opdStarted,
  opdProgress,
  opdCompleted,
  opdStopped,
  opdError,
} = opdSlice.actions;

export default opdSlice.reducer;
