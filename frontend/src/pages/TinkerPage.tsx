import React, { useEffect, useState, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Cloud, Upload, Settings, Play, Download, CheckCircle, Loader, Database, Cpu, RefreshCw } from 'lucide-react';
import { RootState } from '../store/store';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://127.0.0.1:8000';
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

interface CloudModel {
  checkpoint_id: string;
  created_at: string;
  tinker_path: string;
  base_model: string;
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
  const statusCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [tinkerModels, setTinkerModels] = useState<any[]>([]);
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [cloudModels, setCloudModels] = useState<CloudModel[]>([]);
  const [downloadingCloudId, setDownloadingCloudId] = useState<string | null>(null);

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

  // Load Tinker models and datasets on mount
  useEffect(() => {
    fetchTinkerModels();
    fetchAvailableDatasets();
    // Also fetch cloud models for initial base model
    fetchCloudModels(formData.base_model);
  }, []);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (statusCheckIntervalRef.current) {
        clearInterval(statusCheckIntervalRef.current);
      }
    };
  }, []);

  const fetchTinkerModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/tinker/models`);
      setTinkerModels(response.data.models || []);
    } catch (error) {
      console.error('Failed to fetch Tinker models:', error);
    }
  };

  const fetchAvailableDatasets = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/datasets`);
      if (response.data.datasets) {
        // Filter for .jsonl files only
        const jsonlFiles = response.data.datasets.filter((file: string) => file.endsWith('.jsonl'));
        setAvailableDatasets(jsonlFiles);
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    }
  };

  const fetchCloudModels = async (baseModel: string) => {
    if (!baseModel) return;
    try {
      const response = await axios.get(`${BACKEND_URL}/api/tinker/cloud-models/${encodeURIComponent(baseModel)}`);
      setCloudModels(response.data.models);
    } catch (error) {
      console.error('Failed to fetch cloud models:', error);
      // Don't notify on error to avoid spam if API key is missing etc.
    }
  };

  const handleInputChange = (field: keyof typeof formData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    if (field === 'base_model') {
      // Clear cloud models list when base model changes to avoid confusion
      setCloudModels([]);
    }
  };

  const handleFileSelect = async (type: 'train' | 'val') => {
    try {
      // In a real app, this would open a file picker.
      // For now, we'll scan for .jsonl files in common locations
      const response = await axios.get(`${BACKEND_URL}/api/datasets`);
      const datasets = response.data.datasets;

      if (datasets.length > 0) {
        // Simple heuristic: pick the first one or let user type
        // Ideally we'd show a modal. For now, just notify available datasets
        console.log('Available datasets:', datasets);
        if (type === 'train' && !formData.train_data_path) {
          setFormData(prev => ({ ...prev, train_data_path: datasets[0] }));
        }
      }
    } catch (error) {
      console.error('Failed to list datasets:', error);
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

        // Clear any existing interval
        if (statusCheckIntervalRef.current) {
          clearInterval(statusCheckIntervalRef.current);
        }

        // Start polling for status
        const interval = setInterval(() => checkTrainingStatus(response.data.job_id), 10000);
        statusCheckIntervalRef.current = interval;
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

      // Update job state
      setCurrentJob(prev => {
        // If we already marked it as completed, don't update again to avoid loops
        if (prev?.status === 'completed' && status.status === 'completed') {
          return prev;
        }
        return prev ? { ...prev, ...status } : null;
      });

      if (status.status === 'completed') {
        // Clear interval immediately
        if (statusCheckIntervalRef.current) {
          clearInterval(statusCheckIntervalRef.current);
          statusCheckIntervalRef.current = null;
        }

        setIsTraining(false);

        dispatch(addNotification({
          type: 'success',
          title: 'Training Complete',
          message: 'Tinker training finished! Ready to download model.',
        }));
      } else if (status.status === 'error') {
        if (statusCheckIntervalRef.current) {
          clearInterval(statusCheckIntervalRef.current);
          statusCheckIntervalRef.current = null;
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
        adapter_name: formData.adapter_name,
        base_model_id: formData.base_model
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

  const downloadCloudModel = async (model: CloudModel) => {
    try {
      setDownloadingCloudId(model.checkpoint_id);
      dispatch(addNotification({
        type: 'info',
        title: 'Downloading Cloud Model',
        message: `Downloading checkpoint ${model.checkpoint_id.substring(0, 8)}...`,
      }));

      // Generate a name if not provided (using checkpoint ID)
      const adapterName = `tinker_${model.base_model.split('/').pop()}_${model.checkpoint_id.substring(0, 8)}`;

      const response = await axios.post(`${BACKEND_URL}/api/tinker/download`, {
        job_id: '', // Not needed for cloud download
        adapter_name: adapterName,
        checkpoint_id: model.checkpoint_id,
        base_model_id: model.base_model
      });

      if (response.data.success) {
        dispatch(addNotification({
          type: 'success',
          title: 'Download Complete',
          message: `Model saved to ${response.data.local_path}`,
        }));

        // Refresh local models list
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
    } finally {
      setDownloadingCloudId(null);
    }
  };

  // Evaluation State
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<any>(null);
  const [selectedEvalModel, setSelectedEvalModel] = useState<string>("");

  const handleEvaluate = async () => {
    if (!selectedEvalModel) {
      dispatch(addNotification({
        type: 'error',
        title: 'Selection Required',
        message: 'Please select a model to evaluate.',
      }));
      return;
    }

    try {
      setIsEvaluating(true);
      setEvalResult(null);

      // Parse selection: "type:id:base_model:tinker_path"
      const [type, id, baseModel, tinkerPath] = selectedEvalModel.split('|');

      dispatch(addNotification({
        type: 'info',
        title: 'Starting Evaluation',
        message: 'Running LLM-as-a-judge evaluation on Tinker cloud...',
      }));

      const response = await axios.post(`${BACKEND_URL}/api/tinker/evaluate`, {
        checkpoint_id: id,
        tinker_path: tinkerPath && tinkerPath !== 'undefined' ? tinkerPath : undefined,
        base_model: baseModel
      });

      if (response.data.success) {
        setEvalResult(response.data);
        dispatch(addNotification({
          type: 'success',
          title: 'Evaluation Complete',
          message: 'Evaluation finished successfully.',
        }));
      } else {
        throw new Error(response.data.error || 'Evaluation failed');
      }
    } catch (error: any) {
      console.error('Evaluation error:', error);
      dispatch(addNotification({
        type: 'error',
        title: 'Evaluation Failed',
        message: error.response?.data?.detail || error.message || 'Failed to evaluate model',
      }));
    } finally {
      setIsEvaluating(false);
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
            <div className="flex items-center space-x-3">
              <input
                type="text"
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                value={formData.train_data_path}
                onChange={(e) => handleInputChange('train_data_path', e.target.value)}
                placeholder="Path to training data file..."
                disabled={isTraining}
              />
              <button
                type="button"
                onClick={() => handleFileSelect('train')}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center space-x-2"
                disabled={isTraining}
              >
                <Upload className="h-4 w-4" />
                <span>Browse</span>
              </button>
            </div>
          </div>

          {/* Validation Data */}
          <div>
            <label className="block text-sm font-medium mb-2">
              <Database className="w-4 h-4 inline mr-2" />
              Validation Data (Optional)
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="text"
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                value={formData.val_data_path}
                onChange={(e) => handleInputChange('val_data_path', e.target.value)}
                placeholder="Path to validation data file..."
                disabled={isTraining}
              />
              <button
                type="button"
                onClick={() => handleFileSelect('val')}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center space-x-2"
                disabled={isTraining}
              >
                <Upload className="h-4 w-4" />
                <span>Browse</span>
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

          {/* Training Status */}
          {currentJob && (
            <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="font-medium mb-2">Training Status</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Job ID:</span>
                  <span className="font-mono">{currentJob.job_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Status:</span>
                  <span className={`font-medium ${currentJob.status === 'completed' ? 'text-green-600' :
                    currentJob.status === 'error' ? 'text-red-600' :
                      'text-blue-600'
                    }`}>
                    {currentJob.status.toUpperCase()}
                  </span>
                </div>
                <div className="text-gray-500 mt-2">{currentJob.message}</div>

                {currentJob.ready_for_download && (
                  <button
                    onClick={downloadModel}
                    className="w-full mt-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Trained Model
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Model Evaluation */}
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5" />
            Model Evaluation
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Select Model to Evaluate</label>
              <select
                value={selectedEvalModel}
                onChange={(e) => setSelectedEvalModel(e.target.value)}
                className="w-full p-2 rounded border dark:bg-gray-800 dark:border-gray-700"
              >
                <option value="">-- Select a model --</option>
                {/* Local Models */}
                {tinkerModels.map((m, i) => (
                  <option key={`local-${i}`} value={`local|${m.checkpoint_id}|${m.base_model}|${m.tinker_path}`}>
                    Local: {m.adapter_name} ({m.base_model})
                  </option>
                ))}
                {/* Cloud Models */}
                {cloudModels.map((m, i) => (
                  <option key={`cloud-${i}`} value={`cloud|${m.checkpoint_id}|${m.base_model}|${m.tinker_path}`}>
                    Cloud: {m.checkpoint_id.substring(0, 8)}... ({m.base_model})
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleEvaluate}
              disabled={isEvaluating || !selectedEvalModel}
              className="w-full py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isEvaluating ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Run Evaluation
            </button>

            {/* Results Display */}
            {evalResult && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="font-medium mb-2">Evaluation Results</h3>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-2 mb-4">
                  {evalResult.metrics.map((m: any, i: number) => (
                    <div key={i} className="p-2 bg-white dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700">
                      <div className="text-xs text-gray-500">{m.metric}</div>
                      <div className="font-bold text-lg">{typeof m.value === 'number' ? m.value.toFixed(2) : m.value}</div>
                    </div>
                  ))}
                </div>

                {/* Samples */}
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {evalResult.samples.map((s: any, i: number) => (
                    <div key={i} className="text-xs p-2 border-b border-gray-200 dark:border-gray-700 last:border-0">
                      <div className="font-medium text-blue-600">Q: {s.input}</div>
                      <div className="text-gray-600 dark:text-gray-400">A: {s.output}</div>
                      <div className={`font-bold ${s.score === 'C' ? 'text-green-600' : 'text-red-600'}`}>
                        Grade: {s.score}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Trained Models List */}
        {tinkerModels.length > 0 && (
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
            <h2 className="text-xl font-semibold mb-4">Local Tinker Models</h2>

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

        {/* Cloud Models List */}
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm p-6 border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4 flex items-center justify-between">
            <span>Cloud Models (Tinker)</span>
            <button
              onClick={() => fetchCloudModels(formData.base_model)}
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              <RefreshCw className="w-4 h-4" /> Refresh
            </button>
          </h2>

          {cloudModels.length === 0 ? (
            <p className="text-gray-500 text-sm">No cloud models found for {formData.base_model}</p>
          ) : (
            <div className="space-y-3">
              {cloudModels.map((model, index) => (
                <div key={index} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 flex items-center justify-between">
                  <div>
                    <div className="font-medium mb-1">Checkpoint: {model.checkpoint_id.substring(0, 8)}...</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <div>Created: {model.created_at ? new Date(model.created_at).toLocaleString() : 'Unknown'}</div>
                      <div>Base: {model.base_model}</div>
                    </div>
                  </div>
                  <button
                    onClick={() => downloadCloudModel(model)}
                    disabled={downloadingCloudId === model.checkpoint_id}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-2 text-sm"
                  >
                    {downloadingCloudId === model.checkpoint_id ? (
                      <Loader className="w-4 h-4 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4" />
                    )}
                    Download
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>

  );
};
