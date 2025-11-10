import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Settings, Play, Database, Cpu, Layers, Zap, Info } from 'lucide-react';
import { RootState } from '../store/store';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';
const STORAGE_KEY = 'nested_learning_last_config';

interface NestedLearningConfig {
  base_model_path: string;
  adapter_path: string;
  train_data_path: string;
  val_data_path?: string;

  // Nested Learning Parameters
  num_tiers: number;
  tier_update_frequencies: number[];  // e.g., [1, 2, 4] means tier0 every step, tier1 every 2 steps, tier2 every 4 steps
  tier_assignment_strategy: 'layer_depth' | 'parameter_importance' | 'manual';

  // Training Parameters
  learning_rate: number;
  batch_size: number;
  num_steps: number;
  max_seq_length: number;

  // LoRA Config
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;

  // Advanced
  warmup_steps: number;
  gradient_accumulation_steps: number;
  checkpoint_every: number;
  eval_every: number;

  output_path: string;
  experiment_name: string;
}

export const NestedLearningPage: React.FC = () => {
  const dispatch = useDispatch();

  // Default configuration
  const getDefaultConfig = (): NestedLearningConfig => ({
    base_model_path: '',
    adapter_path: '',
    train_data_path: '',
    val_data_path: '',

    // Default Nested Learning settings (3 tiers as per research)
    num_tiers: 3,
    tier_update_frequencies: [1, 2, 4],
    tier_assignment_strategy: 'layer_depth',

    // Training defaults
    learning_rate: 1e-5,
    batch_size: 4,
    num_steps: 1000,
    max_seq_length: 2048,

    // LoRA defaults
    lora_rank: 8,
    lora_alpha: 16,
    lora_dropout: 0.0,

    // Advanced defaults
    warmup_steps: 100,
    gradient_accumulation_steps: 2,
    checkpoint_every: 100,
    eval_every: 100,

    output_path: './nested_learning/checkpoints',
    experiment_name: 'nested_learning_experiment'
  });

  // Load saved config from localStorage or use defaults
  const loadSavedConfig = (): NestedLearningConfig => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        console.log('Loaded saved configuration:', parsed);
        dispatch(addNotification({
          type: 'info',
          title: 'Configuration Restored',
          message: 'Previous training configuration has been loaded.',
          autoHide: true,
        }));
        return parsed;
      }
    } catch (error) {
      console.error('Failed to load saved config:', error);
    }
    return getDefaultConfig();
  };

  const [formData, setFormData] = useState<NestedLearningConfig>(loadSavedConfig);

  const [isTraining, setIsTraining] = useState(false);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [availableAdapters, setAvailableAdapters] = useState<any[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);

  // Save configuration to localStorage
  const saveConfig = (config: NestedLearningConfig) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      console.log('Configuration saved to localStorage');
    } catch (error) {
      console.error('Failed to save config:', error);
    }
  };

  useEffect(() => {
    fetchAvailableModels();
    fetchAvailableAdapters();
  }, []);

  // Poll training status
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/nested-learning/status`);
        setTrainingStatus(response.data);

        // Update isTraining based on status
        if (response.data.status === 'running') {
          setIsTraining(true);
        } else if (response.data.status === 'completed' || response.data.status === 'error') {
          setIsTraining(false);
        }
      } catch (error) {
        console.error('Failed to fetch status:', error);
      }
    };

    // Poll every 2 seconds
    const interval = setInterval(pollStatus, 2000);
    pollStatus(); // Initial fetch

    return () => clearInterval(interval);
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/models`);
      setAvailableModels(response.data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchAvailableAdapters = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/adapters`);
      setAvailableAdapters(response.data.adapters || []);
    } catch (error) {
      console.error('Failed to fetch adapters:', error);
    }
  };

  const handleFileSelect = async (type: 'train' | 'val' | 'model' | 'adapter') => {
    try {
      if (window.electronAPI) {
        let filters: any[] = [];
        let title = '';
        let defaultPath = '';

        switch (type) {
          case 'train':
          case 'val':
            filters = [
              { name: 'JSONL Files', extensions: ['jsonl'] },
              { name: 'JSON Files', extensions: ['json'] },
              { name: 'All Files', extensions: ['*'] }
            ];
            title = `Select ${type === 'train' ? 'Training' : 'Validation'} Data File`;
            defaultPath = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen';
            break;
          case 'model':
            filters = [{ name: 'All Files', extensions: ['*'] }];
            title = 'Select Base Model Directory';
            defaultPath = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model';
            break;
          case 'adapter':
            filters = [
              { name: 'Adapter Files', extensions: ['safetensors', 'npz'] },
              { name: 'All Files', extensions: ['*'] }
            ];
            title = 'Select LoRA Adapter Directory';
            defaultPath = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters';
            break;
        }

        const result = await window.electronAPI.showOpenDialog({
          title,
          filters,
          defaultPath,
          properties: (type === 'model' || type === 'adapter') ? ['openDirectory'] : ['openFile']
        });

        if (!result.canceled && result.filePaths.length > 0) {
          const fieldMap: Record<string, keyof NestedLearningConfig> = {
            'train': 'train_data_path',
            'val': 'val_data_path',
            'model': 'base_model_path',
            'adapter': 'adapter_path'
          };

          setFormData(prev => ({
            ...prev,
            [fieldMap[type]]: result.filePaths[0]
          }));
        }
      } else {
        const path = prompt(`Enter path to ${type}:`);
        if (path) {
          const fieldMap: Record<string, keyof NestedLearningConfig> = {
            'train': 'train_data_path',
            'val': 'val_data_path',
            'model': 'base_model_path',
            'adapter': 'adapter_path'
          };

          setFormData(prev => ({
            ...prev,
            [fieldMap[type]]: path
          }));
        }
      }
    } catch (error) {
      console.error('Error opening file dialog:', error);
      dispatch(addNotification({
        type: 'error',
        title: 'File Dialog Error',
        message: 'Failed to open file dialog.',
      }));
    }
  };

  const handleTierFrequencyChange = (tierIndex: number, value: number) => {
    const newFrequencies = [...formData.tier_update_frequencies];
    newFrequencies[tierIndex] = value;
    setFormData(prev => ({
      ...prev,
      tier_update_frequencies: newFrequencies
    }));
  };

  const handleNumTiersChange = (numTiers: number) => {
    // Adjust tier frequencies array when number of tiers changes
    const newFrequencies = Array.from({ length: numTiers }, (_, i) =>
      Math.pow(2, i) // Default: 1, 2, 4, 8, ...
    );

    setFormData(prev => ({
      ...prev,
      num_tiers: numTiers,
      tier_update_frequencies: newFrequencies
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.base_model_path || !formData.train_data_path) {
      dispatch(addNotification({
        type: 'warning',
        title: 'Incomplete Setup',
        message: 'Please select base model and training data. Adapter is optional.',
      }));
      return;
    }

    setIsTraining(true);

    // Save configuration before starting training
    saveConfig(formData);

    try {
      const response = await axios.post(`${BACKEND_URL}/nested-learning/start`, formData);

      dispatch(addNotification({
        type: 'success',
        title: 'Nested Learning Started',
        message: 'Training has been initiated with multi-frequency parameter updates. Configuration saved.',
        autoHide: true,
      }));

      console.log('Training started:', response.data);
    } catch (error: any) {
      dispatch(addNotification({
        type: 'error',
        title: 'Training Start Failed',
        message: error.response?.data?.detail || 'Failed to start nested learning training.',
      }));
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <div className="flex items-center space-x-3 mb-2">
          <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
            <Layers className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Nested Learning
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Multi-frequency parameter updates for continual learning and catastrophic forgetting prevention
        </p>

        {/* Info Banner */}
        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <p className="font-medium mb-1">What is Nested Learning?</p>
              <p className="text-blue-700 dark:text-blue-300">
                Nested Learning organizes model parameters into tiers that update at different frequencies.
                Fast-changing parameters (Tier 0) update every step, while slower tiers update every 2, 4, or 8 steps.
                This creates a hierarchy of learning speeds that improves continual learning and prevents catastrophic forgetting.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Training Status Display */}
      {trainingStatus && trainingStatus.status !== 'idle' && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Training Status</h2>

          <div className="space-y-4">
            {/* Status Badge */}
            <div className="flex items-center space-x-3">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Status:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                trainingStatus.status === 'running' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                trainingStatus.status === 'completed' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400' :
                trainingStatus.status === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
              }`}>
                {trainingStatus.status.toUpperCase()}
              </span>
            </div>

            {/* Progress */}
            {trainingStatus.current_step !== undefined && trainingStatus.total_steps && (
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600 dark:text-gray-400">Progress</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {trainingStatus.current_step} / {trainingStatus.total_steps} steps
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(trainingStatus.current_step / trainingStatus.total_steps) * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* Experiment Name */}
            {trainingStatus.experiment_name && (
              <div className="flex items-center space-x-3">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Experiment:</span>
                <span className="font-mono text-sm text-gray-900 dark:text-gray-100">{trainingStatus.experiment_name}</span>
              </div>
            )}

            {/* Tier Stats */}
            {trainingStatus.tier_stats && (
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
                <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">Tier Update Counts</h3>
                <div className="grid grid-cols-3 gap-4">
                  {trainingStatus.tier_stats.tier_update_counts?.map((count: number, idx: number) => (
                    <div key={idx} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Tier {idx}</div>
                      <div className="text-lg font-bold text-purple-600 dark:text-purple-400">{count}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Every {trainingStatus.tier_stats.tier_update_frequencies[idx]} step{trainingStatus.tier_stats.tier_update_frequencies[idx] !== 1 ? 's' : ''}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error Message */}
            {trainingStatus.message && trainingStatus.status === 'error' && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <p className="text-sm text-red-900 dark:text-red-100 font-mono">{trainingStatus.message}</p>
              </div>
            )}
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Model & Adapter Selection */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-primary-600" />
              <h2 className="text-xl font-semibold">Model & Adapter Selection</h2>
            </div>
          </div>
          <div className="card-body space-y-6">
            {/* Base Model */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Base Model
                <span className="text-gray-500 dark:text-gray-400 font-normal ml-2">(e.g., Qwen2.5-7B)</span>
              </label>
              <div className="space-y-3">
                <select
                  className="select-field"
                  value={formData.base_model_path}
                  onChange={(e) => setFormData(prev => ({ ...prev, base_model_path: e.target.value }))}
                >
                  <option value="">Select a base model...</option>
                  {availableModels.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.name} ({model.model_type})
                    </option>
                  ))}
                </select>

                {formData.base_model_path && (
                  <div className="bg-gray-50 dark:bg-gray-800/50 p-3 rounded-lg">
                    <p className="text-xs text-gray-600 dark:text-gray-400 font-mono break-all">
                      {formData.base_model_path}
                    </p>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => handleFileSelect('model')}
                  className="btn-secondary flex items-center space-x-2 w-full"
                >
                  <Upload className="h-4 w-4" />
                  <span>Browse for Custom Model</span>
                </button>
              </div>
            </div>

            {/* Adapter */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Existing LoRA Adapter
                <span className="text-gray-500 dark:text-gray-400 font-normal ml-2">(Previously fine-tuned)</span>
              </label>
              <div className="space-y-3">
                <select
                  className="select-field"
                  value={formData.adapter_path}
                  onChange={(e) => setFormData(prev => ({ ...prev, adapter_path: e.target.value }))}
                >
                  <option value="">None (train from base model)</option>
                  {availableAdapters.map((adapter) => (
                    <option key={adapter.path} value={adapter.path}>
                      {adapter.name}
                    </option>
                  ))}
                </select>

                {formData.adapter_path && (
                  <div className="bg-gray-50 dark:bg-gray-800/50 p-3 rounded-lg">
                    <p className="text-xs text-gray-600 dark:text-gray-400 font-mono break-all">
                      {formData.adapter_path}
                    </p>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => handleFileSelect('adapter')}
                  className="btn-secondary flex items-center space-x-2 w-full"
                >
                  <Upload className="h-4 w-4" />
                  <span>Browse for Custom Adapter</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Dataset Configuration */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-primary-600" />
              <h2 className="text-xl font-semibold">Dataset Configuration</h2>
            </div>
          </div>
          <div className="card-body space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">Training Data (JSONL)</label>
              <div className="flex items-center space-x-3">
                <input
                  type="text"
                  className="input-field flex-1 font-mono text-sm"
                  value={formData.train_data_path}
                  onChange={(e) => setFormData(prev => ({ ...prev, train_data_path: e.target.value }))}
                  placeholder="/path/to/training_data.jsonl"
                />
                <button
                  type="button"
                  onClick={() => handleFileSelect('train')}
                  className="btn-secondary flex items-center space-x-2"
                >
                  <Upload className="h-4 w-4" />
                  <span>Browse</span>
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Validation Data (JSONL, Optional)</label>
              <div className="flex items-center space-x-3">
                <input
                  type="text"
                  className="input-field flex-1 font-mono text-sm"
                  value={formData.val_data_path || ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, val_data_path: e.target.value }))}
                  placeholder="/path/to/validation_data.jsonl"
                />
                <button
                  type="button"
                  onClick={() => handleFileSelect('val')}
                  className="btn-secondary flex items-center space-x-2"
                >
                  <Upload className="h-4 w-4" />
                  <span>Browse</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Nested Learning Configuration */}
        <div className="card border-2 border-purple-200 dark:border-purple-800">
          <div className="card-header bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20">
            <div className="flex items-center space-x-2">
              <Layers className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              <h2 className="text-xl font-semibold">Nested Learning Configuration</h2>
            </div>
          </div>
          <div className="card-body space-y-6">
            {/* Number of Tiers */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Number of Parameter Tiers
                <span className="text-gray-500 dark:text-gray-400 font-normal ml-2">(2-4 recommended)</span>
              </label>
              <select
                className="select-field"
                value={formData.num_tiers}
                onChange={(e) => handleNumTiersChange(parseInt(e.target.value))}
              >
                <option value={2}>2 Tiers (Fast + Slow)</option>
                <option value={3}>3 Tiers (Fast + Medium + Slow) - Recommended</option>
                <option value={4}>4 Tiers (Multi-level hierarchy)</option>
              </select>
            </div>

            {/* Tier Assignment Strategy */}
            <div>
              <label className="block text-sm font-medium mb-2">Tier Assignment Strategy</label>
              <div className="space-y-3">
                <label className="flex items-start space-x-3 p-3 rounded-lg border-2 cursor-pointer transition-all hover:bg-gray-50 dark:hover:bg-gray-800/50">
                  <input
                    type="radio"
                    name="tier_strategy"
                    value="layer_depth"
                    checked={formData.tier_assignment_strategy === 'layer_depth'}
                    onChange={(e) => setFormData(prev => ({ ...prev, tier_assignment_strategy: 'layer_depth' as any }))}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="font-medium">Layer Depth</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Shallow layers update frequently, deep layers update slowly (good for transfer learning)
                    </div>
                  </div>
                </label>

                <label className="flex items-start space-x-3 p-3 rounded-lg border-2 cursor-pointer transition-all hover:bg-gray-50 dark:hover:bg-gray-800/50">
                  <input
                    type="radio"
                    name="tier_strategy"
                    value="parameter_importance"
                    checked={formData.tier_assignment_strategy === 'parameter_importance'}
                    onChange={(e) => setFormData(prev => ({ ...prev, tier_assignment_strategy: 'parameter_importance' as any }))}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="font-medium">Parameter Importance</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Assign tiers based on gradient magnitude (adaptive to your task)
                    </div>
                  </div>
                </label>
              </div>
            </div>

            {/* Update Frequencies */}
            <div>
              <label className="block text-sm font-medium mb-3">
                Update Frequencies per Tier
                <span className="text-gray-500 dark:text-gray-400 font-normal ml-2">(steps between updates)</span>
              </label>
              <div className="space-y-3">
                {formData.tier_update_frequencies.map((freq, idx) => (
                  <div key={idx} className="flex items-center space-x-4">
                    <div className="w-24 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Tier {idx}
                      {idx === 0 && <span className="text-xs text-purple-600 dark:text-purple-400 ml-1">(Fastest)</span>}
                      {idx === formData.num_tiers - 1 && <span className="text-xs text-blue-600 dark:text-blue-400 ml-1">(Slowest)</span>}
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="16"
                      step="1"
                      value={freq}
                      onChange={(e) => handleTierFrequencyChange(idx, parseInt(e.target.value))}
                      className="flex-1"
                    />
                    <div className="w-32">
                      <input
                        type="number"
                        min="1"
                        max="16"
                        value={freq}
                        onChange={(e) => handleTierFrequencyChange(idx, parseInt(e.target.value))}
                        className="input-field text-center"
                      />
                    </div>
                    <div className="w-32 text-sm text-gray-600 dark:text-gray-400">
                      Every {freq} step{freq !== 1 ? 's' : ''}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div className="text-xs text-purple-900 dark:text-purple-100">
                  <strong>Example:</strong> With frequencies [1, 2, 4], Tier 0 updates every step,
                  Tier 1 every 2 steps, and Tier 2 every 4 steps. This creates a learning hierarchy.
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Training Parameters */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Settings className="h-5 w-5 text-primary-600" />
              <h2 className="text-xl font-semibold">Training Parameters</h2>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">Learning Rate</label>
                <input
                  type="number"
                  step="0.000001"
                  className="input-field"
                  value={formData.learning_rate}
                  onChange={(e) => setFormData(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Batch Size</label>
                <select
                  className="select-field"
                  value={formData.batch_size}
                  onChange={(e) => setFormData(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Training Steps</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.num_steps}
                  onChange={(e) => setFormData(prev => ({ ...prev, num_steps: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Max Sequence Length</label>
                <select
                  className="select-field"
                  value={formData.max_seq_length}
                  onChange={(e) => setFormData(prev => ({ ...prev, max_seq_length: parseInt(e.target.value) }))}
                >
                  <option value={512}>512</option>
                  <option value={1024}>1024</option>
                  <option value={2048}>2048</option>
                  <option value={4096}>4096</option>
                  <option value={8192}>8192</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">LoRA Rank</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.lora_rank}
                  onChange={(e) => setFormData(prev => ({ ...prev, lora_rank: parseInt(e.target.value) }))}
                  min="1"
                  max="256"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">LoRA Alpha</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.lora_alpha}
                  onChange={(e) => setFormData(prev => ({ ...prev, lora_alpha: parseInt(e.target.value) }))}
                  min="1"
                  max="512"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Warmup Steps</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.warmup_steps}
                  onChange={(e) => setFormData(prev => ({ ...prev, warmup_steps: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Gradient Accumulation Steps</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.gradient_accumulation_steps}
                  onChange={(e) => setFormData(prev => ({ ...prev, gradient_accumulation_steps: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Checkpoint Every N Steps</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.checkpoint_every}
                  onChange={(e) => setFormData(prev => ({ ...prev, checkpoint_every: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Evaluate Every N Steps</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.eval_every}
                  onChange={(e) => setFormData(prev => ({ ...prev, eval_every: parseInt(e.target.value) }))}
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium mb-2">Experiment Name</label>
                <input
                  type="text"
                  className="input-field"
                  value={formData.experiment_name}
                  onChange={(e) => setFormData(prev => ({ ...prev, experiment_name: e.target.value }))}
                  placeholder="my_nested_learning_experiment"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="flex justify-end space-x-4">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              // Reset form to defaults and clear saved config
              const defaultConfig = getDefaultConfig();
              setFormData(defaultConfig);
              localStorage.removeItem(STORAGE_KEY);
              dispatch(addNotification({
                type: 'info',
                title: 'Configuration Reset',
                message: 'Form has been reset to default values and saved configuration cleared.',
                autoHide: true,
              }));
            }}
          >
            Reset Form
          </button>
          <button
            type="submit"
            disabled={isTraining || !formData.base_model_path || !formData.train_data_path}
            className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isTraining ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                <span>Starting Training...</span>
              </>
            ) : (
              <>
                <Zap className="h-4 w-4" />
                <span>Start Nested Learning</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
