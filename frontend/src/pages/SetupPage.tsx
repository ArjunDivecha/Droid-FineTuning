import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Settings, Play, Database, Cpu } from 'lucide-react';
import { RootState } from '../store/store';
import { setModels, setSelectedModel, setLoading, setError } from '../store/slices/modelsSlice';
import { setTrainingConfig, type TrainingConfig } from '../store/slices/trainingSlice';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';
const STORAGE_KEY = 'setup_page_last_config';

export const SetupPage: React.FC = () => {
  const dispatch = useDispatch();
  const { models, selectedModel, isLoading, error } = useSelector((state: RootState) => state.models);
  const { state: trainingState, config } = useSelector((state: RootState) => state.training);
  
  const [fineTuneType, setFineTuneType] = useState<'lora' | 'full'>('lora');
  const [loraConfig, setLoraConfig] = useState({
    rank: 8,
    alpha: 16,
    dropout: 0.0,
    target_modules: 'all'
  });
  
  // Default configuration
  const getDefaultConfig = (): Partial<TrainingConfig> => ({
    model_path: '',
    train_data_path: '',
    val_data_path: '',
    learning_rate: 1e-5,
    batch_size: 1,
    max_seq_length: 1024,
    iterations: 7329,
    steps_per_report: 25,
    steps_per_eval: 200,
    save_every: 100,
    early_stop: true,
    patience: 3,
    adapter_name: 'mlx_finetune'
  });

  // Load saved config from localStorage or use defaults
  const loadSavedConfig = (): Partial<TrainingConfig> => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        console.log('Loaded saved configuration:', parsed);
        return parsed;
      }
    } catch (error) {
      console.error('Failed to load saved config:', error);
    }
    return getDefaultConfig();
  };
  
  const [formData, setFormData] = useState<Partial<TrainingConfig>>(loadSavedConfig);
  const [hasLoadedConfig, setHasLoadedConfig] = useState(false);

  // Show notification after component mounts (not during initialization)
  useEffect(() => {
    if (!hasLoadedConfig) {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          JSON.parse(saved); // Verify it's valid JSON
          dispatch(addNotification({
            type: 'info',
            title: 'Configuration Restored',
            message: 'Previous training configuration has been loaded.',
            autoHide: true,
          }));
        } catch (error) {
          // Invalid JSON, ignore
        }
      }
      setHasLoadedConfig(true);
    }
  }, [dispatch, hasLoadedConfig]);

  // Save configuration to localStorage
  const saveConfig = (config: Partial<TrainingConfig>) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      console.log('Configuration saved to localStorage');
    } catch (error) {
      console.error('Failed to save config:', error);
    }
  };

  // Save config whenever formData changes (debounced to avoid excessive saves)
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      saveConfig(formData);
    }, 500); // Debounce saves by 500ms
    
    return () => clearTimeout(timeoutId);
  }, [formData]);

  useEffect(() => {
    fetchModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      setFormData(prev => ({
        ...prev,
        model_path: selectedModel.path
      }));
    }
  }, [selectedModel]);

  const fetchModels = async () => {
    dispatch(setLoading(true));
    try {
      const response = await axios.get(`${BACKEND_URL}/models`);
      dispatch(setModels(response.data.models));
      if (response.data.models.length > 0) {
        dispatch(setSelectedModel(response.data.models[0]));
      }
    } catch (error) {
      dispatch(setError('Failed to fetch models'));
      dispatch(addNotification({
        type: 'error',
        title: 'Model Loading Error',
        message: 'Failed to load available models. Check backend connection.',
      }));
    }
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
      max_seq_length: formData.max_seq_length!,
      iterations: formData.iterations!,
      steps_per_report: formData.steps_per_report!,
      steps_per_eval: formData.steps_per_eval!,
      save_every: formData.save_every!,
      early_stop: formData.early_stop!,
      patience: formData.patience!,
      adapter_name: formData.adapter_name!
    };

    dispatch(setTrainingConfig(trainingConfig));
    
    // Save config before starting training
    saveConfig(formData);
    
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
                  onClick={fetchModels}
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

        {/* Fine-Tuning Method */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-primary-600" />
              <h2 className="text-xl font-semibold">Fine-Tuning Method</h2>
            </div>
          </div>
          <div className="card-body space-y-6">
            <div>
              <label className="block text-sm font-medium mb-3">Select Fine-Tuning Type</label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  type="button"
                  onClick={() => setFineTuneType('lora')}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    fineTuneType === 'lora'
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                  }`}
                >
                  <div className="font-semibold text-lg mb-1">LoRA (Recommended)</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Low-Rank Adaptation - Efficient fine-tuning with small adapter weights
                  </div>
                  <div className="mt-2 text-xs text-primary-600 dark:text-primary-400">
                    ✓ Fast • ✓ Low Memory • ✓ Portable
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => setFineTuneType('full')}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    fineTuneType === 'full'
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                  }`}
                >
                  <div className="font-semibold text-lg mb-1">Full Fine-Tuning</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Train all model weights - Maximum adaptation capability
                  </div>
                  <div className="mt-2 text-xs text-warning-600 dark:text-warning-400">
                    ⚠ Slower • ⚠ High Memory • ⚠ Large Output
                  </div>
                </button>
              </div>
            </div>

            {fineTuneType === 'lora' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    LoRA Rank <span className="text-gray-500 dark:text-gray-400 font-normal">(Higher = More capacity)</span>
                  </label>
                  <input
                    type="number"
                    className="input-field"
                    value={loraConfig.rank}
                    onChange={(e) => setLoraConfig(prev => ({ ...prev, rank: parseInt(e.target.value) }))}
                    min="1"
                    max="256"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    LoRA Alpha <span className="text-gray-500 dark:text-gray-400 font-normal">(Scaling factor)</span>
                  </label>
                  <input
                    type="number"
                    className="input-field"
                    value={loraConfig.alpha}
                    onChange={(e) => setLoraConfig(prev => ({ ...prev, alpha: parseInt(e.target.value) }))}
                    min="1"
                    max="512"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    LoRA Dropout <span className="text-gray-500 dark:text-gray-400 font-normal">(Regularization)</span>
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    className="input-field"
                    value={loraConfig.dropout}
                    onChange={(e) => setLoraConfig(prev => ({ ...prev, dropout: parseFloat(e.target.value) }))}
                    min="0"
                    max="0.5"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Target Modules</label>
                  <select
                    className="input-field"
                    value={loraConfig.target_modules}
                    onChange={(e) => setLoraConfig(prev => ({ ...prev, target_modules: e.target.value }))}
                  >
                    <option value="all">All Linear Layers</option>
                    <option value="attention">Attention Only</option>
                    <option value="mlp">MLP Only</option>
                  </select>
                </div>
              </div>
            )}
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
                <label className="block text-sm font-medium mb-2">Save Checkpoint Every (steps)</label>
                <input
                  type="number"
                  className="input-field"
                  value={formData.save_every}
                  onChange={(e) => setFormData(prev => ({ ...prev, save_every: parseInt(e.target.value) }))}
                  placeholder="25"
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

        {/* Submit */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!formData.model_path || !formData.train_data_path}
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
