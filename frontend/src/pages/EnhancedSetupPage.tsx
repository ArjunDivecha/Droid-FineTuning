// frontend/src/pages/EnhancedSetupPageNew.tsx
// Enhanced setup page with GSPO and Dr. GRPO - matching your beautiful dark theme design

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Settings, Play, Database, Cpu, Zap, Brain, Target, Sparkles, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { RootState } from '../store/store';
import { addNotification } from '../store/slices/uiSlice';
import { setTrainingConfig } from '../store/slices/trainingSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';

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
  early_stop: boolean;
  patience: number;
  adapter_name: string;
  training_method: TrainingMethod;
  sparse_ratio?: number;
  efficiency_threshold?: number;
  sparse_optimization?: boolean;
  domain?: string;
  expertise_level?: string;
  domain_adaptation_strength?: number;
  reasoning_steps?: number;
  multi_step_training?: boolean;
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
      description: 'Latest breakthrough in efficient reasoning model training with sparse optimization',
      complexity: '⭐⭐⭐⭐',
      use_case: 'Efficient reasoning tasks with resource constraints',
      resource_intensity: 'medium',
      estimated_speedup: '2x faster training',
      badge: '🆕 Most Efficient'
    },
    [TrainingMethod.DR_GRPO]: {
      name: 'dr_grpo',
      display_name: 'Dr. GRPO',
      description: 'Domain-specialized reasoning for expert knowledge applications',
      complexity: '⭐⭐⭐⭐⭐',
      use_case: 'Medical, scientific, and specialized domain reasoning',
      resource_intensity: 'high',
      badge: '🆕 Domain Expert'
    },
    [TrainingMethod.GRPO]: {
      name: 'grpo',
      display_name: 'GRPO',
      description: 'DeepSeek-R1 style multi-step reasoning capabilities',
      complexity: '⭐⭐⭐⭐',
      use_case: 'Complex multi-step reasoning and problem solving',
      resource_intensity: 'high',
      badge: '🆕 New'
    }
  });

  const [selectedMethod, setSelectedMethod] = useState<TrainingMethod>(TrainingMethod.SFT);
  const [resourceEstimation, setResourceEstimation] = useState<ResourceEstimation | null>(null);
  const [dataValidation, setDataValidation] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isEstimating, setIsEstimating] = useState(false);
  
  // Form state matching your existing design
  const [formData, setFormData] = useState<Partial<EnhancedTrainingConfig>>({
    model_path: '',
    train_data_path: '',
    val_data_path: '',
    learning_rate: 1e-5,
    batch_size: 1,
    max_seq_length: 32768,
    iterations: 7329,
    steps_per_report: 25,
    steps_per_eval: 200,
    save_every: 1000,
    early_stop: true,
    patience: 3,
    adapter_name: 'mlx_finetune',
    training_method: TrainingMethod.SFT,
    sparse_ratio: 0.7,
    efficiency_threshold: 0.85,
    sparse_optimization: true,
    domain: 'general',
    expertise_level: 'advanced',
    domain_adaptation_strength: 1.0,
    reasoning_steps: 8,
    multi_step_training: true
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

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

  const generateSampleData = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/training/generate-sample-data`, {
        method: selectedMethod,
        output_path: `/tmp/sample_${selectedMethod}_data.jsonl`,
        num_samples: 20
      });
      
      if (response.data.success) {
        handleInputChange('train_data_path', response.data.output_path);
        dispatch(addNotification({
          type: 'success',
          title: 'Sample Data Generated',
          message: `Sample data created: ${response.data.output_path}`
        }));
      }
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        title: 'Generation Failed',
        message: 'Failed to generate sample data'
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      dispatch(setTrainingConfig(formData as any));
      dispatch(addNotification({
        type: 'success',
        title: 'Training Started',
        message: `${selectedMethod.toUpperCase()} training has been initiated`
      }));
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        title: 'Training Failed',
        message: 'Failed to start enhanced training'
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
                <input
                  type="text"
                  value={formData.model_path || ''}
                  onChange={(e) => handleInputChange('model_path', e.target.value)}
                  className="input-field"
                  placeholder="Path to your model directory"
                />
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
                    onClick={generateSampleData}
                    className="btn-primary text-sm px-3"
                  >
                    Generate Sample
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

              <div className="md:col-span-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={formData.early_stop || false}
                    onChange={(e) => handleInputChange('early_stop', e.target.checked)}
                    className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Enable Early Stopping
                  </span>
                </label>
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
                Group Sparse Policy Optimization settings
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Sparse Ratio
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="0.9"
                    value={formData.sparse_ratio || ''}
                    onChange={(e) => handleInputChange('sparse_ratio', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Fraction of reasoning steps to optimize (0.1-0.9)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Efficiency Threshold
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    min="0.5"
                    max="1.0"
                    value={formData.efficiency_threshold || ''}
                    onChange={(e) => handleInputChange('efficiency_threshold', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Minimum efficiency score to maintain (0.5-1.0)
                  </p>
                </div>

                <div className="md:col-span-2">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.sparse_optimization || false}
                      onChange={(e) => handleInputChange('sparse_optimization', e.target.checked)}
                      className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable Sparse Optimization
                    </span>
                  </label>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-6">
                    Use sparse attention patterns for efficiency
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
                Domain-specialized reasoning settings
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Domain
                  </label>
                  <select
                    value={formData.domain || ''}
                    onChange={(e) => handleInputChange('domain', e.target.value)}
                    className="select-field"
                  >
                    <option value="general">General</option>
                    <option value="medical">Medical</option>
                    <option value="scientific">Scientific</option>
                    <option value="legal">Legal</option>
                    <option value="technical">Technical</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Expertise Level
                  </label>
                  <select
                    value={formData.expertise_level || ''}
                    onChange={(e) => handleInputChange('expertise_level', e.target.value)}
                    className="select-field"
                  >
                    <option value="beginner">Beginner</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                    <option value="expert">Expert</option>
                  </select>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Domain Adaptation Strength
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2.0"
                    value={formData.domain_adaptation_strength || ''}
                    onChange={(e) => handleInputChange('domain_adaptation_strength', parseFloat(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Strength of domain-specific adaptation (0.1-2.0)
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
                Multi-step reasoning settings
              </p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Reasoning Steps
                  </label>
                  <input
                    type="number"
                    min="3"
                    max="15"
                    value={formData.reasoning_steps || ''}
                    onChange={(e) => handleInputChange('reasoning_steps', parseInt(e.target.value))}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Number of reasoning steps to train (3-15)
                  </p>
                </div>

                <div className="flex items-center">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.multi_step_training || false}
                      onChange={(e) => handleInputChange('multi_step_training', e.target.checked)}
                      className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable Multi-Step Training
                    </span>
                  </label>
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