import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Cloud, Upload, Settings, Play, Download, CheckCircle, Loader, Database, Cpu } from 'lucide-react';
import { RootState } from '../store/store';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';
const STORAGE_KEY = 'tinker_page_last_config';

interface TinkerConfig {
  base_model: string;
  train_data_path: string;
  val_data_path: string;
  adapter_name: string;
  learning_rate: number;
  batch_size: number;
  num_epochs: number;
  lora_rank: number;
  max_seq_length: number;
}

interface TinkerJob {
  job_id: string;
  adapter_name: string;
  status: 'training' | 'completed' | 'error';
  message: string;
  ready_for_download: boolean;
}

export const TinkerPage: React.FC = () => {
  const dispatch = useDispatch();
  const { models } = useSelector((state: RootState) => state.models);
  
  // Default configuration
  const getDefaultConfig = (): TinkerConfig => ({
    base_model: 'Qwen/Qwen3-4B-Instruct-2507',
    train_data_path: '',
    val_data_path: '',
    adapter_name: 'tinker_adapter',
    learning_rate: 1e-5,
    batch_size: 1,
    num_epochs: 3,
    lora_rank: 64,
    max_seq_length: 2048
  });

  // Load saved config
  const loadSavedConfig = (): TinkerConfig => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (error) {
      console.error('Failed to load saved config:', error);
    }
    return getDefaultConfig();
  };

  const [formData, setFormData] = useState<TinkerConfig>(loadSavedConfig);
  const [isTraining, setIsTraining] = useState(false);
  const [currentJob, setCurrentJob] = useState<TinkerJob | null>(null);
  const [statusCheckInterval, setStatusCheckInterval] = useState<NodeJS.Timeout | null>(null);
  const [tinkerModels, setTinkerModels] = useState<any[]>([]);

  // Save config to localStorage
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(formData));
      } catch (error) {
        console.error('Failed to save config:', error);
      }
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [formData]);

  // Load Tinker models on mount
  useEffect(() => {
    fetchTinkerModels();
  }, []);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
      }
    };
  }, [statusCheckInterval]);

  const fetchTinkerModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/tinker/models`);
      setTinkerModels(response.data.models || []);
    } catch (error) {
      console.error('Failed to fetch Tinker models:', error);
    }
  };

  const handleInputChange = (field: keyof TinkerConfig, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleFileSelect = async (type: 'train' | 'val') => {
    try {
      if (window.electronAPI) {
        const result = await window.electronAPI.showOpenDialog({
          title: `Select ${type === 'train' ? 'Training' : 'Validation'} Data File`,
          filters: [
            { name: 'JSONL Files', extensions: ['jsonl'] },
            { name: 'All Files', extensions: ['*'] }
          ],
          properties: ['openFile']
        });
        
        if (!result.canceled && result.filePaths.length > 0) {
          const field = type === 'train' ? 'train_data_path' : 'val_data_path';
          handleInputChange(field, result.filePaths[0]);
        }
      } else {
        dispatch(addNotification({
          type: 'warning',
          title: 'File Selection',
          message: 'Please enter the file path manually.',
        }));
      }
    } catch (error) {
      console.error('File selection error:', error);
    }
  };

  const startTraining = async () => {
    if (!formData.train_data_path) {
      dispatch(addNotification({
        type: 'error',
        title: 'Missing Data',
        message: 'Please select a training data file.',
      }));
      return;
    }

    if (!formData.adapter_name) {
      dispatch(addNotification({
        type: 'error',
        title: 'Missing Name',
        message: 'Please provide an adapter name.',
      }));
      return;
    }

    try {
      setIsTraining(true);
      
      dispatch(addNotification({
        type: 'info',
        title: 'Starting Tinker Training',
        message: 'Uploading data and starting cloud fine-tuning...',
      }));

      const response = await axios.post(`${BACKEND_URL}/api/tinker/start-training`, formData);
      
      if (response.data.success) {
        setCurrentJob({
          job_id: response.data.job_id,
          adapter_name: response.data.adapter_name,
          status: 'training',
          message: response.data.message,
          ready_for_download: false
        });

        dispatch(addNotification({
          type: 'success',
          title: 'Training Started',
          message: `Tinker job ${response.data.job_id} started successfully!`,
        }));

        // Start polling for status
        const interval = setInterval(() => checkTrainingStatus(response.data.job_id), 10000);
        setStatusCheckInterval(interval);
      } else {
        throw new Error(response.data.message || 'Failed to start training');
      }
    } catch (error: any) {
      console.error('Training start error:', error);
      setIsTraining(false);
      
      dispatch(addNotification({
        type: 'error',
        title: 'Training Failed',
        message: error.response?.data?.detail || error.message || 'Failed to start Tinker training',
      }));
    }
  };

  const checkTrainingStatus = async (jobId: string) => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/tinker/status/${jobId}`);
      const status = response.data;

      setCurrentJob(prev => prev ? { ...prev, ...status } : null);

      if (status.status === 'completed') {
        if (statusCheckInterval) {
          clearInterval(statusCheckInterval);
          setStatusCheckInterval(null);
        }

        dispatch(addNotification({
          type: 'success',
          title: 'Training Complete',
          message: 'Tinker training finished! Ready to download model.',
        }));
      } else if (status.status === 'error') {
        if (statusCheckInterval) {
          clearInterval(statusCheckInterval);
          setStatusCheckInterval(null);
        }
        setIsTraining(false);

        dispatch(addNotification({
          type: 'error',
          title: 'Training Error',
          message: status.message || 'Training failed',
        }));
      }
    } catch (error) {
      console.error('Status check error:', error);
    }
  };

  const downloadModel = async () => {
    if (!currentJob) return;

    try {
      dispatch(addNotification({
        type: 'info',
        title: 'Downloading Model',
        message: 'Downloading trained model from Tinker...',
      }));

      const response = await axios.post(`${BACKEND_URL}/api/tinker/download`, {
        job_id: currentJob.job_id,
        adapter_name: formData.adapter_name
      });

      if (response.data.success) {
        dispatch(addNotification({
          type: 'success',
          title: 'Download Complete',
          message: `Model saved to ${response.data.local_path}`,
        }));

        // Reset state
        setIsTraining(false);
        setCurrentJob(null);
        
        // Refresh models list
        fetchTinkerModels();
      } else {
        throw new Error(response.data.message || 'Download failed');
      }
    } catch (error: any) {
      console.error('Download error:', error);
      
      dispatch(addNotification({
        type: 'error',
        title: 'Download Failed',
        message: error.response?.data?.detail || error.message || 'Failed to download model',
      }));
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-3 mb-2">
          <Cloud className="w-8 h-8 text-blue-500" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Tinker Cloud Fine-Tuning
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-400">
          Train models using Tinker's distributed cloud infrastructure. Models trained here will appear in Compare and Fusion tabs.
        </p>
      </div>

      {/* Configuration Form */}
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Training Configuration
        </h2>

        <div className="space-y-4">
          {/* Base Model */}
          <div>
            <label className="block text-sm font-medium mb-2">
              <Cpu className="w-4 h-4 inline mr-2" />
              Base Model
            </label>
            <select
              value={formData.base_model}
              onChange={(e) => handleInputChange('base_model', e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
              disabled={isTraining}
            >
              <optgroup label="🦖 Large Models (70B+)">
                <option value="Qwen/Qwen3-235B-A22B-Instruct-2507">Qwen3-235B-A22B-Instruct (235B/22B active) - MoE ⭐</option>
                <option value="deepseek-ai/DeepSeek-V3.1">DeepSeek-V3.1 - MoE</option>
                <option value="meta-llama/Llama-3.3-70B-Instruct">Llama-3.3-70B-Instruct</option>
                <option value="meta-llama/Llama-3.1-70B">Llama-3.1-70B (Base)</option>
              </optgroup>
              <optgroup label="🦅 Medium Models (30B-32B)">
                <option value="Qwen/Qwen3-30B-A3B-Instruct-2507">Qwen3-30B-A3B-Instruct (30B/3B active) - MoE ⭐</option>
                <option value="Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B (30B/3B active) - MoE</option>
                <option value="Qwen/Qwen3-32B">Qwen3-32B</option>
                <option value="openai/gpt-oss-120b">GPT-OSS-120B - MoE Reasoning</option>
              </optgroup>
              <optgroup label="🦆 Small Models (8B)">
                <option value="Qwen/Qwen3-8B">Qwen3-8B</option>
                <option value="meta-llama/Llama-3.1-8B">Llama-3.1-8B (Base)</option>
                <option value="meta-llama/Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</option>
                <option value="openai/gpt-oss-20b">GPT-OSS-20B - MoE Reasoning</option>
              </optgroup>
              <optgroup label="🐣 Compact Models (1B-4B)">
                <option value="Qwen/Qwen3-4B-Instruct-2507">Qwen3-4B-Instruct-2507 ⭐</option>
                <option value="meta-llama/Llama-3.2-3B">Llama-3.2-3B (Base)</option>
                <option value="meta-llama/Llama-3.2-1B">Llama-3.2-1B (Base)</option>
              </optgroup>
            </select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              ⭐ = Recommended | MoE = Mixture of Experts (cost-effective)
            </p>
          </div>

          {/* Training Data */}
          <div>
            <label className="block text-sm font-medium mb-2">
              <Database className="w-4 h-4 inline mr-2" />
              Training Data (JSONL)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={formData.train_data_path}
                onChange={(e) => handleInputChange('train_data_path', e.target.value)}
                placeholder="/path/to/train.jsonl"
                className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              />
              <button
                onClick={() => handleFileSelect('train')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                disabled={isTraining}
              >
                <Upload className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Validation Data */}
          <div>
            <label className="block text-sm font-medium mb-2">
              <Database className="w-4 h-4 inline mr-2" />
              Validation Data (Optional)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={formData.val_data_path}
                onChange={(e) => handleInputChange('val_data_path', e.target.value)}
                placeholder="/path/to/val.jsonl"
                className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              />
              <button
                onClick={() => handleFileSelect('val')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                disabled={isTraining}
              >
                <Upload className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Adapter Name */}
          <div>
            <label className="block text-sm font-medium mb-2">Adapter Name</label>
            <input
              type="text"
              value={formData.adapter_name}
              onChange={(e) => handleInputChange('adapter_name', e.target.value)}
              placeholder="my_tinker_adapter"
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
              disabled={isTraining}
            />
          </div>

          {/* Hyperparameters Grid */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Learning Rate</label>
              <input
                type="number"
                step="0.000001"
                value={formData.learning_rate}
                onChange={(e) => handleInputChange('learning_rate', parseFloat(e.target.value))}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Batch Size</label>
              <input
                type="number"
                value={formData.batch_size}
                onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Number of Epochs</label>
              <input
                type="number"
                value={formData.num_epochs}
                onChange={(e) => handleInputChange('num_epochs', parseInt(e.target.value))}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">LoRA Rank</label>
              <select
                value={formData.lora_rank}
                onChange={(e) => handleInputChange('lora_rank', parseInt(e.target.value))}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              >
                <option value={8}>8</option>
                <option value={16}>16</option>
                <option value={32}>32</option>
                <option value={64}>64</option>
                <option value={128}>128</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Max Sequence Length</label>
              <input
                type="number"
                value={formData.max_seq_length}
                onChange={(e) => handleInputChange('max_seq_length', parseInt(e.target.value))}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                disabled={isTraining}
              />
            </div>
          </div>

          {/* Start Training Button */}
          <button
            onClick={startTraining}
            disabled={isTraining || !formData.train_data_path}
            className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
          >
            {isTraining ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                Training in Progress...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start Cloud Training
              </>
            )}
          </button>
        </div>
      </div>

      {/* Training Status */}
      {currentJob && (
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Training Status</h2>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Job ID:</span>
              <span className="text-sm text-gray-600 dark:text-gray-400">{currentJob.job_id}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status:</span>
              <span className={`text-sm font-medium ${
                currentJob.status === 'completed' ? 'text-green-600' :
                currentJob.status === 'error' ? 'text-red-600' :
                'text-blue-600'
              }`}>
                {currentJob.status === 'training' && <Loader className="w-4 h-4 inline animate-spin mr-1" />}
                {currentJob.status === 'completed' && <CheckCircle className="w-4 h-4 inline mr-1" />}
                {currentJob.status.toUpperCase()}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Message:</span>
              <span className="text-sm text-gray-600 dark:text-gray-400">{currentJob.message}</span>
            </div>

            {currentJob.ready_for_download && (
              <button
                onClick={downloadModel}
                className="w-full mt-4 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center justify-center gap-2 font-medium"
              >
                <Download className="w-5 h-5" />
                Download Trained Model
              </button>
            )}
          </div>
        </div>
      )}

      {/* Trained Models List */}
      {tinkerModels.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Tinker-Trained Models</h2>
          
          <div className="space-y-3">
            {tinkerModels.map((model, index) => (
              <div key={index} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{model.adapter_name}</span>
                  <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
                    Tinker
                  </span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <div>Base: {model.base_model}</div>
                  <div>Rank: {model.lora_rank} | Epochs: {model.num_epochs}</div>
                  <div>Completed: {new Date(model.completed_at).toLocaleString()}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
