import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Settings, Play, Database, Cpu } from 'lucide-react';
import { RootState } from '../store/store';
import { setModels, setSelectedModel, setLoading, setError } from '../store/slices/modelsSlice';
import { setTrainingConfig, type TrainingConfig } from '../store/slices/trainingSlice';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';
const MAX_MODEL_FETCH_RETRIES = 5;

export const SetupPage: React.FC = () => {
  const dispatch = useDispatch();
  const { models, selectedModel, isLoading, error } = useSelector((state: RootState) => state.models);
  const { state: trainingState, config } = useSelector((state: RootState) => state.training);
  
  const retryTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [formData, setFormData] = useState<Partial<TrainingConfig>>({
    model_path: '',
    train_data_path: '',
    val_data_path: '',
    learning_rate: 1e-4,  // Updated: 10x increase for LoRA
    batch_size: 1,
    grad_accumulation_steps: 1,
    max_seq_length: 32768,
    iterations: 7329,
    steps_per_report: 25,
    steps_per_eval: 200,
    save_every: 1000,
    early_stop: true,
    patience: 3,
    adapter_name: 'mlx_finetune',
    // Full-Layer LoRA Configuration
    fine_tune_type: 'lora',
    lora_rank: 32,
    lora_alpha: 32,
    lora_dropout: 0.0,
    lora_num_layers: -1
  });

  const fetchModels = useCallback(async (attempt = 0) => {
    if (retryTimeout.current) {
      clearTimeout(retryTimeout.current);
      retryTimeout.current = null;
    }

    let scheduledRetry = false;

    if (attempt === 0) {
      dispatch(setError(null));
    }

    dispatch(setLoading(true));

    try {
      await axios.get(`${BACKEND_URL}/health`, { timeout: 2000 });
      const response = await axios.get(`${BACKEND_URL}/models`, { timeout: 5000 });

      const availableModels = response.data.models || [];
      dispatch(setModels(availableModels));

      if (availableModels.length > 0) {
        dispatch(setSelectedModel(availableModels[0]));
      }
    } catch (err) {
      console.error(`Model fetch attempt ${attempt + 1} failed:`, err);
      const nextAttempt = attempt + 1;
      if (nextAttempt <= MAX_MODEL_FETCH_RETRIES) {
        scheduledRetry = true;
        const delay = Math.min(4000, 750 * nextAttempt);
        retryTimeout.current = setTimeout(() => {
          fetchModels(nextAttempt);
        }, delay);
      } else {
        dispatch(setError('Failed to fetch models'));
        dispatch(addNotification({
          type: 'error',
          title: 'Model Loading Error',
          message: 'Failed to load available models. Check backend connection.',
        }));
      }
    } finally {
      if (!scheduledRetry) {
        dispatch(setLoading(false));
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

  useEffect(() => {
    if (selectedModel) {
      setFormData(prev => ({
        ...prev,
        model_path: selectedModel.path
      }));
    }
  }, [selectedModel]);

  const manualRetry = () => {
    fetchModels();
  };

  const handleModelChange = (modelPath: string) => {
    const model = models.find(m => m.path === modelPath);
    dispatch(setSelectedModel(model || null));
  };

  const handleFileSelect = async (type: 'train' | 'val') => {
    try {
      // Use Electron's file dialog if available
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
          setFormData(prev => ({
            ...prev,
            [type === 'train' ? 'train_data_path' : 'val_data_path']: result.filePaths[0]
          }));
        }
      } else {
        // Fallback for development without Electron
        const path = prompt(`Enter path to ${type} data file (JSONL format):`);
        if (path) {
          setFormData(prev => ({
            ...prev,
            [type === 'train' ? 'train_data_path' : 'val_data_path']: path
          }));
        }
      }
    } catch (error) {
      console.error('Error opening file dialog:', error);
      dispatch(addNotification({
        type: 'error',
        title: 'File Dialog Error',
        message: 'Failed to open file dialog. Please try again.',
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.model_path || !formData.train_data_path) {
      dispatch(addNotification({
        type: 'warning',
        title: 'Incomplete Setup',
        message: 'Please select a model and training data file.',
      }));
      return;
    }

    const trainingConfig: TrainingConfig = {
      model_path: formData.model_path!,
      train_data_path: formData.train_data_path!,
      val_data_path: formData.val_data_path || '',
      learning_rate: formData.learning_rate!,
      batch_size: formData.batch_size!,
      grad_accumulation_steps: formData.grad_accumulation_steps || 1,
      max_seq_length: formData.max_seq_length!,
      iterations: formData.iterations!,
      steps_per_report: formData.steps_per_report!,
      steps_per_eval: formData.steps_per_eval!,
      save_every: formData.save_every!,
      early_stop: formData.early_stop!,
      patience: formData.patience!,
      adapter_name: formData.adapter_name!,
      // Include LoRA configuration
      fine_tune_type: formData.fine_tune_type,
      lora_rank: formData.lora_rank,
      lora_alpha: formData.lora_alpha,
      lora_dropout: formData.lora_dropout,
      lora_num_layers: formData.lora_num_layers
    };

    dispatch(setTrainingConfig(trainingConfig));
    
    try {
      await axios.post(`${BACKEND_URL}/training/start`, trainingConfig);
      dispatch(addNotification({
        type: 'success',
        title: 'Training Started',
        message: 'Fine-tuning process has been initiated.',
        autoHide: true,
      }));
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        title: 'Training Start Failed',
        message: 'Failed to start training. Check configuration and try again.',
      }));
    }
  };

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Setup Fine-Tuning
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Configure your model, datasets, and training parameters
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Model Selection */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-primary-600" />
              <h2 className="text-xl font-semibold">Model Selection</h2>
            </div>
          </div>
          <div className="card-body space-y-4">
            {isLoading ? (
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600" />
                <span>Loading models...</span>
              </div>
            ) : error ? (
              <div className="text-error-600 bg-error-50 dark:bg-error-900/20 p-4 rounded-lg">
                {error}
                <button
                  type="button"
                  onClick={manualRetry}
                  className="ml-4 btn-secondary text-sm"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Available Models</label>
                  <select
                    className="select-field"
                    value={selectedModel?.path || ''}
                    onChange={(e) => handleModelChange(e.target.value)}
                  >
                    <option value="">Select a model...</option>
                    {models.map((model) => (
                      <option key={model.path} value={model.path}>
                        {model.name} ({model.model_type})
                      </option>
                    ))}
                  </select>
                </div>

                {selectedModel && (
                  <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg">
                    <h3 className="font-medium mb-2">{selectedModel.name}</h3>
                    <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <p><span className="font-medium">Type:</span> {selectedModel.model_type}</p>
                      <p><span className="font-medium">Vocab Size:</span> {selectedModel.vocab_size.toLocaleString()}</p>
                      <p><span className="font-medium">Path:</span> {selectedModel.path}</p>
                    </div>
                  </div>
                )}
              </div>
            )}
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
            {/* Training Data */}
            <div>
              <label className="block text-sm font-medium mb-2">Training Data (JSONL)</label>
              <div className="flex items-center space-x-3">
                <input
                  type="text"
                  className="input-field flex-1"
                  value={formData.train_data_path || ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, train_data_path: e.target.value }))}
                  placeholder="Path to training data file..."
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

            {/* Validation Data */}
            <div>
              <label className="block text-sm font-medium mb-2">Validation Data (JSONL, Optional)</label>
              <div className="flex items-center space-x-3">
                <input
                  type="text"
                  className="input-field flex-1"
                  value={formData.val_data_path || ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, val_data_path: e.target.value }))}
                  placeholder="Path to validation data file..."
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
                <label className="block text-sm font-medium mb-2">Gradient Accumulation Steps</label>
                <select
                  className="select-field"
                  value={formData.grad_accumulation_steps}
                  onChange={(e) => setFormData(prev => ({ ...prev, grad_accumulation_steps: parseInt(e.target.value) }))}
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Increases effective batch size without extra VRAM.
                </p>
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
                  <option value={16384}>16384</option>
                  <option value={32768}>32768</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Training Steps</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.iterations}
                  onChange={(e) => setFormData(prev => ({ ...prev, iterations: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Steps per Report</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.steps_per_report}
                  onChange={(e) => setFormData(prev => ({ ...prev, steps_per_report: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Steps per Evaluation</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.steps_per_eval}
                  onChange={(e) => setFormData(prev => ({ ...prev, steps_per_eval: parseInt(e.target.value) }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Adapter Name</label>
                <input
                  type="text"
                  className="input-field"
                  value={formData.adapter_name}
                  onChange={(e) => setFormData(prev => ({ ...prev, adapter_name: e.target.value }))}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Early Stop Patience</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.patience}
                  onChange={(e) => setFormData(prev => ({ ...prev, patience: parseInt(e.target.value) }))}
                />
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="early_stop"
                  checked={formData.early_stop}
                  onChange={(e) => setFormData(prev => ({ ...prev, early_stop: e.target.checked }))}
                  className="rounded border-gray-300"
                />
                <label htmlFor="early_stop" className="text-sm font-medium">Enable Early Stopping</label>
              </div>
            </div>
          </div>
        </div>

        {/* Full-Layer LoRA Configuration */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-primary-600" />
              <h2 className="text-xl font-semibold">Full-Layer LoRA Configuration</h2>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Applies LoRA adapters to all 7 weight matrices (Q, K, V, O + gate, up, down) across all transformer layers.
              Research shows this significantly outperforms attention-only training.
            </p>
          </div>
          <div className="card-body">
            {/* Info Banner */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
              <p className="text-sm text-blue-900 dark:text-blue-100">
                <strong>Full-Layer LoRA Training</strong> - Trains attention (Q, K, V, O) + MLP (gate, up, down) layers
                for comprehensive fine-tuning. Based on{' '}
                <a
                  href="https://thinkingmachines.ai/blog/lora/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-blue-700 dark:hover:text-blue-300"
                >
                  "LoRA Without Regret" research
                </a>
                .
              </p>
            </div>

            {/* LoRA Parameters Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* LoRA Rank */}
              <div>
                <label className="block text-sm font-medium mb-2">LoRA Rank</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  className="input-field"
                  value={formData.lora_rank}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    setFormData(prev => ({ ...prev, lora_rank: isNaN(val) || val < 1 ? 32 : val }));
                  }}
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Recommended: 32 for most datasets (1-256 range)
                </p>
              </div>

              {/* LoRA Alpha */}
              <div>
                <label className="block text-sm font-medium mb-2">LoRA Alpha (Scale)</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  step="1"
                  className="input-field"
                  value={formData.lora_alpha}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setFormData(prev => ({ ...prev, lora_alpha: isNaN(val) || val < 1 ? 32 : val }));
                  }}
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Typically equals rank (scaling factor)
                </p>
              </div>

              {/* LoRA Dropout */}
              <div>
                <label className="block text-sm font-medium mb-2">LoRA Dropout</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step="0.01"
                  className="input-field"
                  value={formData.lora_dropout}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    const clamped = isNaN(val) ? 0 : Math.min(1, Math.max(0, val));
                    setFormData(prev => ({ ...prev, lora_dropout: clamped }));
                  }}
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  0.0 recommended (0.0-0.1 for regularization)
                </p>
              </div>

              {/* Layer Coverage */}
              <div>
                <label className="block text-sm font-medium mb-2">Layer Coverage</label>
                <select
                  className="select-field"
                  value={formData.lora_num_layers}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    setFormData(prev => ({ ...prev, lora_num_layers: isNaN(val) ? -1 : val }));
                  }}
                >
                  <option value={-1}>All Layers (Recommended)</option>
                  <option value={24}>Top 24 Layers</option>
                  <option value={16}>Top 16 Layers</option>
                  <option value={8}>Top 8 Layers</option>
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  -1 applies LoRA to all transformer blocks
                </p>
              </div>
            </div>

            {/* Matrix Coverage Visualization */}
            <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
                Matrix Coverage
              </h4>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <p className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-2">Attention Layers (4):</p>
                  <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                    <li>✓ Query projection (q_proj)</li>
                    <li>✓ Key projection (k_proj)</li>
                    <li>✓ Value projection (v_proj)</li>
                    <li>✓ Output projection (o_proj)</li>
                  </ul>
                </div>
                <div>
                  <p className="text-xs font-medium text-green-600 dark:text-green-400 mb-2">MLP Layers (3):</p>
                  <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                    <li>✓ Gate projection (gate_proj)</li>
                    <li>✓ Up projection (up_proj)</li>
                    <li>✓ Down projection (down_proj)</li>
                  </ul>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  <strong>Total Coverage:</strong> 7 matrices × {formData.lora_num_layers === -1 ? 'all' : formData.lora_num_layers} transformer layers
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                  Expected trainable parameters: ~3.5-4% of total model parameters
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={trainingState === 'running' || !formData.model_path || !formData.train_data_path}
            className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="h-4 w-4" />
            <span>Start Training</span>
          </button>
        </div>
      </form>
    </div>
  );
};
