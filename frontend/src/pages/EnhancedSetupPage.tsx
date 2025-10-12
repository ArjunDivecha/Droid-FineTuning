// frontend/src/pages/EnhancedSetupPageNew.tsx
// Enhanced setup page with GSPO and Dr. GRPO - matching your beautiful dark theme design

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Settings, Play, Database, Cpu, Zap, Brain, Target, Sparkles, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { RootState } from '../store/store';
import { addNotification } from '../store/slices/uiSlice';
import { setTrainingConfig } from '../store/slices/trainingSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';
const MAX_MODEL_FETCH_RETRIES = 5;

// Training method types
enum TrainingMethod {
  SFT = 'sft',
  GSPO = 'gspo', 
  DR_GRPO = 'dr_grpo',
  GRPO = 'grpo'
}

interface TrainingMethodConfig {
  name: string;
  display_name: string;
  description: string;
  complexity: string;
  use_case: string;
  badge?: string;
  resource_intensity: string;
  estimated_speedup?: string;
}

interface EnhancedTrainingConfig {
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
  adapter_name: string;
  training_method: TrainingMethod;
  // GRPO/GSPO/Dr.GRPO parameters (actual mlx-lm-lora flags)
  group_size?: number;
  epsilon?: number;
  temperature?: number;
  max_completion_length?: number;
  importance_sampling_level?: string;
  grpo_loss_type?: string;
  epsilon_high?: number;
  reward_functions?: string;
  reward_weights?: string;
  reward_functions_file?: string;
}

interface ResourceEstimation {
  estimated_memory_gb: number;
  estimated_time_hours: number;
  resource_intensity: string;
  recommendations: string[];
}

interface ValidationResult {
  valid: boolean;
  error?: string;
  samples_count?: number;
  format_issues?: string[];
}

const EnhancedSetupPage: React.FC = () => {
  const dispatch = useDispatch();
  
  // Available training methods
  const [availableMethods] = useState<Record<string, TrainingMethodConfig>>({
    [TrainingMethod.SFT]: {
      name: 'sft',
      display_name: 'Supervised Fine-Tuning',
      description: 'Standard instruction following fine-tuning with LoRA adapters',
      complexity: '⭐⭐',
      use_case: 'General instruction following and task adaptation',
      resource_intensity: 'medium',
      badge: 'Enhanced'
    },
    [TrainingMethod.GSPO]: {
      name: 'gspo',
      display_name: 'GSPO',
      description: 'GRPO with importance sampling for improved sample efficiency',
      complexity: '⭐⭐⭐⭐',
      use_case: 'Policy optimization with better convergence than standard GRPO',
      resource_intensity: 'high',
      estimated_speedup: 'Better sample efficiency',
      badge: '🆕 Efficient'
    },
    [TrainingMethod.DR_GRPO]: {
      name: 'dr_grpo',
      display_name: 'Dr. GRPO',
      description: 'Decoupled rewards GRPO for more stable training',
      complexity: '⭐⭐⭐⭐⭐',
      use_case: 'Stable policy optimization for complex tasks',
      resource_intensity: 'very high',
      badge: '🆕 Most Stable'
    },
    [TrainingMethod.GRPO]: {
      name: 'grpo',
      display_name: 'GRPO',
      description: 'Group Relative Policy Optimization for reasoning and instruction following',
      complexity: '⭐⭐⭐⭐',
      use_case: 'Improving reasoning quality and response quality',
      resource_intensity: 'high',
      badge: '🆕 RL'
    }
  });

  const [selectedMethod, setSelectedMethod] = useState<TrainingMethod>(TrainingMethod.SFT);
  const [resourceEstimation, setResourceEstimation] = useState<ResourceEstimation | null>(null);
  const [dataValidation, setDataValidation] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isEstimating, setIsEstimating] = useState(false);
  const retryTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<Array<{name: string, path: string}>>([]);
  
  // Form state matching actual mlx-lm-lora parameters
  const [formData, setFormData] = useState<Partial<EnhancedTrainingConfig>>({
    model_path: '',
    train_data_path: '',
    val_data_path: '',
    learning_rate: 1e-5,
    batch_size: 1,
    max_seq_length: 2048,
    iterations: 100,
    steps_per_report: 10,
    steps_per_eval: 25,
    save_every: 100,
    adapter_name: 'mlx_finetune',
    training_method: TrainingMethod.SFT,
    // GRPO parameters with defaults
    group_size: 4,
    epsilon: 0.0001,
    temperature: 0.8,
    max_completion_length: 512,
    importance_sampling_level: 'sequence',
    grpo_loss_type: 'grpo',
    epsilon_high: undefined,
    reward_functions: 'lexical_f1_reward,length_window_reward,keyword_coverage_reward',
    reward_weights: '[0.7,0.2,0.1]',
    reward_functions_file: 'rewards/style_rewards.py'
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const fetchModels = useCallback(async (attempt = 0) => {
    if (retryTimeout.current) {
      clearTimeout(retryTimeout.current);
      retryTimeout.current = null;
    }

    let scheduledRetry = false;

    if (attempt === 0) {
      setModelError(null);
    }

    setIsLoadingModels(true);

    try {
      await axios.get(`${BACKEND_URL}/health`, { timeout: 2000 });
      const response = await axios.get(`${BACKEND_URL}/models`, { timeout: 5000 });
      const models = (response.data.models || []).map((m: any) => ({
        name: m.name,
        path: m.path
      }));

      setAvailableModels(models);
      if (models.length > 0) {
        setFormData(prev => ({
          ...prev,
          model_path: prev.model_path || models[0].path
        }));
      }
    } catch (error) {
      console.error(`Failed to fetch models (attempt ${attempt + 1}):`, error);
      const nextAttempt = attempt + 1;
      if (nextAttempt <= MAX_MODEL_FETCH_RETRIES) {
        scheduledRetry = true;
        const delay = Math.min(4000, 750 * nextAttempt);
        retryTimeout.current = setTimeout(() => {
          fetchModels(nextAttempt);
        }, delay);
      } else {
        setModelError('Failed to load available models. Check backend connection.');
        setAvailableModels([]);
        dispatch(addNotification({
          type: 'error',
          title: 'Model Loading Error',
          message: 'Failed to load available models. Check backend connection.',
        }));
      }
    } finally {
      if (!scheduledRetry) {
        setIsLoadingModels(false);
      }
    }
  }, [dispatch]);

  useEffect(() => {
    fetchModels();
    return () => {
      if (retryTimeout.current) {
        clearTimeout(retryTimeout.current);
      }
    };
  }, [fetchModels]);

  const manualRetry = useCallback(() => {
    fetchModels();
  }, [fetchModels]);

  // Update form data when method changes
  useEffect(() => {
    setFormData(prev => ({ ...prev, training_method: selectedMethod }));
  }, [selectedMethod]);

  const getMethodIcon = (method: string) => {
    switch (method) {
      case 'sft': return <Settings className="w-5 h-5" />;
      case 'gspo': return <Zap className="w-5 h-5" />;
      case 'dr_grpo': return <Brain className="w-5 h-5" />;
      case 'grpo': return <Target className="w-5 h-5" />;
      default: return <Sparkles className="w-5 h-5" />;
    }
  };

  const getBadgeColor = (badge: string) => {
    if (badge.includes('Most Efficient')) return 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300';
    if (badge.includes('Domain Expert')) return 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300';
    if (badge.includes('New')) return 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300';
    return 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300';
  };

  const handleInputChange = (field: keyof EnhancedTrainingConfig, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleFileSelect = async (type: 'train' | 'val') => {
    try {
      if (window.electronAPI) {
        const result = await window.electronAPI.showOpenDialog({
          title: `Select ${type === 'train' ? 'Training' : 'Validation'} Data File`,
          filters: [
            { name: 'JSONL Files', extensions: ['jsonl'] },
            { name: 'JSON Files', extensions: ['json'] },
            { name: 'All Files', extensions: ['*'] }
          ],
          properties: ['openFile']
        });
        
        if (!result.canceled && result.filePaths.length > 0) {
          handleInputChange(type === 'train' ? 'train_data_path' : 'val_data_path', result.filePaths[0]);
        }
      } else {
        const path = prompt(`Enter path to ${type} data file (JSONL format):`);
        if (path) {
          handleInputChange(type === 'train' ? 'train_data_path' : 'val_data_path', path);
        }
      }
    } catch (error) {
      console.error('Error opening file dialog:', error);
    }
  };

  const validateTrainingData = async () => {
    if (!formData.train_data_path) {
      dispatch(addNotification({
        type: 'error',
        title: 'Validation Error',
        message: 'Please select a training data file first'
      }));
      return;
    }

    try {
      const response = await axios.post(`${BACKEND_URL}/api/training/validate-data`, {
        data_path: formData.train_data_path,
        method: selectedMethod
      });
      
      if (response.data.valid) {
        dispatch(addNotification({
          type: 'success',
          title: 'Data Validation Passed',
          message: `✓ Found ${response.data.num_samples} valid samples\n✓ Format: ${response.data.format}\n✓ Compatible with ${selectedMethod.toUpperCase()}`
        }));
      } else {
        dispatch(addNotification({
          type: 'error',
          title: 'Data Validation Failed',
          message: response.data.error || 'Invalid data format'
        }));
      }
    } catch (error: any) {
      dispatch(addNotification({
        type: 'error',
        title: 'Validation Failed',
        message: error.response?.data?.error || 'Failed to validate training data'
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      // Validate required fields
      if (!formData.model_path || !formData.train_data_path) {
        dispatch(addNotification({
          type: 'error',
          title: 'Validation Error',
          message: 'Please provide model path and training data path'
        }));
        return;
      }

      // Start training via enhanced backend API
      const response = await axios.post(`${BACKEND_URL}/api/training/start-enhanced`, formData);

      if (response.data.success) {
        dispatch(setTrainingConfig(formData as any));
        dispatch(addNotification({
          type: 'success',
          title: 'Training Started',
          message: response.data.message || `${selectedMethod.toUpperCase()} training has been initiated successfully`
        }));
      } else {
        throw new Error(response.data.error || 'Training failed to start');
      }
    } catch (error: any) {
      console.error('Training start error:', error);
      dispatch(addNotification({
        type: 'error',
        title: 'Training Failed',
        message: error.response?.data?.detail || error.message || 'Failed to start enhanced training'
      }));
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Enhanced Training Setup
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Configure advanced training methods including GSPO and Dr. GRPO
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Training Method Selection */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Training Method Selection
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Choose the training method that best fits your use case
            </p>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(availableMethods).map(([method, methodConfig]) => (
                <div 
                  key={method}
                  className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    selectedMethod === method 
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' 
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                  onClick={() => setSelectedMethod(method as TrainingMethod)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        selectedMethod === method 
                          ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/50 dark:text-primary-400'
                          : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                      }`}>
                        {getMethodIcon(method)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                          {methodConfig.display_name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {methodConfig.complexity}
                        </p>
                      </div>
                    </div>
                    {methodConfig.badge && (
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getBadgeColor(methodConfig.badge)}`}>
                        {methodConfig.badge}
                      </span>
                    )}
                  </div>
                  
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                    {methodConfig.description}
                  </p>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500 dark:text-gray-400">Resources:</span>
                      <span className={`px-2 py-1 rounded-full ${
                        methodConfig.resource_intensity === 'medium' ? 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300' :
                        methodConfig.resource_intensity === 'high' ? 'bg-error-100 text-error-700 dark:bg-error-900/30 dark:text-error-300' :
                        'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300'
                      }`}>
                        {methodConfig.resource_intensity}
                      </span>
                    </div>
                    
                    {methodConfig.estimated_speedup && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-500 dark:text-gray-400">Performance:</span>
                        <span className="text-success-600 dark:text-success-400 font-medium">
                          {methodConfig.estimated_speedup}
                        </span>
                      </div>
                    )}
                    
                    <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        <span className="font-medium">Best for:</span> {methodConfig.use_case}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Basic Configuration */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
              <Database className="w-5 h-5 mr-2" />
              Basic Configuration
            </h2>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Model Path
                </label>
                {modelError && (
                  <div className="text-error-600 bg-error-50 dark:bg-error-900/30 p-3 rounded-lg flex items-center justify-between mb-3">
                    <span>{modelError}</span>
                    <button
                      type="button"
                      onClick={manualRetry}
                      className="btn-secondary text-sm"
                    >
                      Retry
                    </button>
                  </div>
                )}
                {isLoadingModels ? (
                  <div className="flex items-center space-x-3 text-gray-600 dark:text-gray-300">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600" />
                    <span>Loading available models...</span>
                  </div>
                ) : availableModels.length > 0 ? (
                  <select
                    value={formData.model_path || ''}
                    onChange={(e) => handleInputChange('model_path', e.target.value)}
                    className="input-field"
                  >
                    <option value="">Select a model...</option>
                    {availableModels.map((model) => (
                      <option key={model.path} value={model.path}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={formData.model_path || ''}
                    onChange={(e) => handleInputChange('model_path', e.target.value)}
                    className="input-field"
                    placeholder="Path to your model directory"
                  />
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Adapter Name
                </label>
                <input
                  type="text"
                  value={formData.adapter_name || ''}
                  onChange={(e) => handleInputChange('adapter_name', e.target.value)}
                  className="input-field"
                  placeholder="Name for your adapter"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Training Data Path
                </label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={formData.train_data_path || ''}
                    onChange={(e) => handleInputChange('train_data_path', e.target.value)}
                    className="input-field flex-1"
                    placeholder="Path to training data (JSONL)"
                  />
                  <button
                    type="button"
                    onClick={() => handleFileSelect('train')}
                    className="btn-secondary"
                  >
                    <Upload className="w-4 h-4" />
                  </button>
                  <button
                    type="button"
                    onClick={validateTrainingData}
                    className="btn-primary text-sm px-3"
                    disabled={!formData.train_data_path}
                    title="Validate training data format and compatibility"
                  >
                    <CheckCircle className="w-4 h-4 mr-1 inline" />
                    Validate Data
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Validation Data Path (Optional)
                </label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={formData.val_data_path || ''}
                    onChange={(e) => handleInputChange('val_data_path', e.target.value)}
                    className="input-field flex-1"
                    placeholder="Path to validation data (JSONL)"
                  />
                  <button
                    type="button"
                    onClick={() => handleFileSelect('val')}
                    className="btn-secondary"
                  >
                    <Upload className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Training Parameters */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
              <Cpu className="w-5 h-5 mr-2" />
              Training Parameters
            </h2>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.00001"
                  value={formData.learning_rate || ''}
                  onChange={(e) => handleInputChange('learning_rate', parseFloat(e.target.value))}
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={formData.batch_size || ''}
                  onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Max Sequence Length
                </label>
                <input
                  type="number"
                  value={formData.max_seq_length || ''}
                  onChange={(e) => handleInputChange('max_seq_length', parseInt(e.target.value))}
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Iterations
                </label>
                <input
                  type="number"
                  value={formData.iterations || ''}
                  onChange={(e) => handleInputChange('iterations', parseInt(e.target.value))}
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Steps per Report
                </label>
                <input
                  type="number"
                  value={formData.steps_per_report || ''}
                  onChange={(e) => handleInputChange('steps_per_report', parseInt(e.target.value))}
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Steps per Eval
                </label>
                <input
                  type="number"
                  value={formData.steps_per_eval || ''}
                  onChange={(e) => handleInputChange('steps_per_eval', parseInt(e.target.value))}
                  className="input-field"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Method-specific Configuration */}
        {selectedMethod === TrainingMethod.GSPO && (
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
                <Zap className="w-5 h-5 mr-2" />
                GSPO Configuration
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                GRPO with importance sampling for improved efficiency
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Importance Sampling Level
                  </label>
                  <select
                    value={formData.importance_sampling_level || 'token'}
                    onChange={(e) => handleInputChange('importance_sampling_level', e.target.value)}
                    className="select-field"
                  >
                    <option value="token">Token-level (most fine-grained)</option>
                    <option value="sequence">Sequence-level</option>
                    <option value="">None (standard GRPO)</option>
                  </select>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Token-level focuses on individual tokens, sequence-level on full responses
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Group Size
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="16"
                    value={formData.group_size || 4}
                    onChange={(e) => handleInputChange('group_size', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Number of completions to generate per prompt (2-16, default: 4)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Epsilon
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.0001"
                    max="0.01"
                    value={formData.epsilon || 0.0001}
                    onChange={(e) => handleInputChange('epsilon', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Clipping parameter for numerical stability (default: 0.0001)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Temperature
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.6"
                    max="1.2"
                    value={formData.temperature || 0.8}
                    onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Sampling temperature (0.6-1.2, lower=deterministic, higher=creative)
                  </p>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Max Completion Length
                  </label>
                  <input
                    type="number"
                    min="128"
                    max="2048"
                    value={formData.max_completion_length || 512}
                    onChange={(e) => handleInputChange('max_completion_length', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Maximum tokens for generated completions (default: 512)
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedMethod === TrainingMethod.DR_GRPO && (
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                Dr. GRPO Configuration
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Decoupled rewards GRPO for more stable training
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Group Size
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="16"
                    value={formData.group_size || 4}
                    onChange={(e) => handleInputChange('group_size', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Number of completions to generate per prompt (2-16, default: 4)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Epsilon (Lower Bound)
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.0001"
                    max="0.01"
                    value={formData.epsilon || 0.0001}
                    onChange={(e) => handleInputChange('epsilon', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Lower clipping bound for stability (default: 0.0001)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Epsilon High (Upper Bound)
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.0001"
                    max="0.1"
                    value={formData.epsilon_high || ''}
                    onChange={(e) => handleInputChange('epsilon_high', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-field"
                    placeholder="Optional (for DAPO variant)"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Upper clipping bound (optional, for DAPO variant)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Temperature
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.6"
                    max="1.2"
                    value={formData.temperature || 0.8}
                    onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Sampling temperature (0.6-1.2, default: 0.8)
                  </p>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Max Completion Length
                  </label>
                  <input
                    type="number"
                    min="128"
                    max="2048"
                    value={formData.max_completion_length || 512}
                    onChange={(e) => handleInputChange('max_completion_length', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Maximum tokens for generated completions (default: 512)
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedMethod === TrainingMethod.GRPO && (
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
                <Target className="w-5 h-5 mr-2" />
                GRPO Configuration
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Group Relative Policy Optimization for reasoning tasks
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Group Size
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="16"
                    value={formData.group_size || 4}
                    onChange={(e) => handleInputChange('group_size', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Number of completions to generate per prompt (2-16, default: 4)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Epsilon
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.0001"
                    max="0.01"
                    value={formData.epsilon || 0.0001}
                    onChange={(e) => handleInputChange('epsilon', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Clipping parameter for stability (default: 0.0001)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Temperature
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.6"
                    max="1.2"
                    value={formData.temperature || 0.8}
                    onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Sampling temperature (0.6-1.2, default: 0.8)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Max Completion Length
                  </label>
                  <input
                    type="number"
                    min="128"
                    max="2048"
                    value={formData.max_completion_length || 512}
                    onChange={(e) => handleInputChange('max_completion_length', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Maximum tokens for generated completions (default: 512)
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedMethod !== TrainingMethod.SFT && (
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Reward Configuration
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Configure custom reward functions used during RL fine-tuning
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Reward Functions (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={formData.reward_functions || ''}
                    onChange={(e) => handleInputChange('reward_functions', e.target.value)}
                    className="input-field"
                    placeholder="lexical_f1_reward,length_window_reward,keyword_coverage_reward"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Provide reward function names defined in your reward file.
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Reward Weights (JSON array)
                  </label>
                  <input
                    type="text"
                    value={formData.reward_weights || ''}
                    onChange={(e) => handleInputChange('reward_weights', e.target.value)}
                    className="input-field"
                    placeholder="[0.7,0.2,0.1]"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Must sum to 1.0. Use JSON list syntax.
                  </p>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Reward Functions File
                  </label>
                  <input
                    type="text"
                    value={formData.reward_functions_file || ''}
                    onChange={(e) => handleInputChange('reward_functions_file', e.target.value)}
                    className="input-field"
                    placeholder="rewards/style_rewards.py"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Path (relative or absolute) to the Python module that defines your reward helpers.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            className="btn-primary flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>Start {availableMethods[selectedMethod]?.display_name} Training</span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default EnhancedSetupPage;
