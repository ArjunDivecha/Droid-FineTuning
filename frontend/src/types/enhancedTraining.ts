// frontend/src/types/enhancedTraining.ts
// Enhanced TypeScript types for GSPO and Dr. GRPO training methods

export enum TrainingMethod {
  SFT = 'sft',
  GSPO = 'gspo',
  DR_GRPO = 'dr_grpo',
  GRPO = 'grpo'
}

export interface TrainingMethodConfig {
  display_name: string;
  description: string;
  complexity: string;
  use_case: string;
  badge?: string;
  resource_intensity: 'low' | 'medium' | 'high' | 'very_high';
  estimated_speedup?: string;
  data_format: string;
  requires_reasoning_chains: boolean;
  requires_preferences: boolean;
  supports_batch: boolean;
  module_name: string;
  additional_params: string[];
}

export interface EnhancedTrainingConfig {
  // Existing base configuration
  model_path: string;
  train_data_path: string;
  val_data_path?: string;
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
  
  // Enhanced training method selection
  training_method: TrainingMethod;
  
  // Shared GRPO parameters
  group_size: number;
  epsilon: number;
  temperature: number;
  max_completion_length: number;
  importance_sampling_level?: string;
  grpo_loss_type?: string;
  epsilon_high?: number;
  reward_functions?: string;
  reward_weights?: string;
  reward_functions_file?: string;

  // GSPO specific parameters
  sparse_ratio?: number;
  efficiency_threshold?: number;
  sparse_optimization?: boolean;
  
  // Dr. GRPO specific parameters
  domain?: 'general' | 'medical' | 'scientific' | 'legal' | 'technical';
  expertise_level?: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  domain_adaptation_strength?: number;
  
  // GRPO specific parameters
  reasoning_steps?: number;
  multi_step_training?: boolean;
}

export interface ResourceEstimation {
  method: string;
  estimated_memory_gb: number;
  estimated_time_hours: number;
  resource_intensity: string;
  recommendations: string[];
}

export interface ValidationResult {
  valid: boolean;
  error?: string;
  format?: string;
  required_format?: string[];
  sample_format?: any;
  note?: string;
}

export interface TrainingMethodsResponse {
  success: boolean;
  methods: Record<string, TrainingMethodConfig>;
  error?: string;
}

export interface DataValidationRequest {
  method: string;
  data_path: string;
}

export interface DataValidationResponse {
  success: boolean;
  validation: ValidationResult;
  error?: string;
}

export interface ResourceEstimationRequest {
  method: string;
  model_path: string;
  dataset_size: number;
}

export interface ResourceEstimationResponse {
  success: boolean;
  estimation: ResourceEstimation;
  error?: string;
}

export interface StartTrainingResponse {
  success: boolean;
  message?: string;
  method?: string;
  config_path?: string;
  pid?: number;
  error?: string;
}

export interface SampleDataRequest {
  method: string;
  output_path: string;
  num_samples?: number;
}

export interface SampleDataResponse {
  success: boolean;
  message?: string;
  output_path?: string;
  method?: string;
  sample_count?: number;
  error?: string;
}

// Enhanced training state interface
export interface EnhancedTrainingState {
  // Base training state
  training_state: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  current_step: number;
  total_steps: number;
  train_loss: number;
  val_loss: number;
  learning_rate: number;
  start_time: string;
  estimated_time_remaining: string;
  
  // Enhanced state
  method: TrainingMethod;
  best_val_loss: number;
  best_model_step: number;
  
  // Method-specific metrics
  efficiency_score?: number; // For GSPO
  domain_adaptation_score?: number; // For Dr. GRPO
  reasoning_depth?: number; // For GRPO
}

// Form validation rules
export interface ValidationRules {
  required: string[];
  numeric: string[];
  fileExists: string[];
  range: Record<string, { min: number; max: number }>;
  enum: Record<string, string[]>;
}

export const getValidationRules = (method: TrainingMethod): ValidationRules => {
  const baseRules: ValidationRules = {
    required: ['model_path', 'train_data_path', 'adapter_name'],
    numeric: ['learning_rate', 'batch_size', 'max_seq_length', 'iterations'],
    fileExists: ['model_path', 'train_data_path'],
    range: {
      learning_rate: { min: 0.00001, max: 0.001 },
      batch_size: { min: 1, max: 8 },
      max_seq_length: { min: 512, max: 8192 },
      iterations: { min: 10, max: 1000 }
    },
    enum: {}
  };

  switch (method) {
    case TrainingMethod.GSPO:
      return {
        ...baseRules,
        range: {
          ...baseRules.range,
          sparse_ratio: { min: 0.1, max: 0.9 },
          efficiency_threshold: { min: 0.5, max: 1.0 }
        }
      };
    
    case TrainingMethod.DR_GRPO:
      return {
        ...baseRules,
        enum: {
          domain: ['general', 'medical', 'scientific', 'legal', 'technical'],
          expertise_level: ['beginner', 'intermediate', 'advanced', 'expert']
        },
        range: {
          ...baseRules.range,
          domain_adaptation_strength: { min: 0.1, max: 2.0 }
        }
      };
    
    case TrainingMethod.GRPO:
      return {
        ...baseRules,
        range: {
          ...baseRules.range,
          reasoning_steps: { min: 3, max: 15 }
        }
      };
    
    default:
      return baseRules;
  }
};
