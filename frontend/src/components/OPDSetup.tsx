import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { Play, Upload, HelpCircle } from 'lucide-react';
import { opdStarted, setError } from '../store/slices/opdSlice';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';

export const OPDSetup: React.FC = () => {
  const dispatch = useDispatch();
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Form state
  const [config, setConfig] = useState({
    base_model_path: '',
    teacher_model_path: '',
    student_adapter_path: '',
    validation_prompts_path: '',
    num_steps: 1000,
    batch_size: 4,
    temperature: 2.0,
    kl_weight: 0.8,
    learning_rate: 0.00001,
  });

  const handleInputChange = (field: string, value: string | number) => {
    setConfig(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleFileSelect = async (field: string) => {
    try {
      // Use Electron file dialog
      const result = await window.electronAPI.selectDirectory();
      if (result) {
        setConfig(prev => ({
          ...prev,
          [field]: result,
        }));
      }
    } catch (err) {
      console.error(`Error selecting ${field}:`, err);
      dispatch(addNotification({
        type: 'error',
        title: 'File Selection Error',
        message: `Failed to select ${field}`,
      }));
    }
  };

  const validateConfig = (): boolean => {
    if (!config.base_model_path) {
      dispatch(setError('Base model path is required'));
      return false;
    }
    if (!config.teacher_model_path) {
      dispatch(setError('Teacher model path is required'));
      return false;
    }
    if (!config.student_adapter_path) {
      dispatch(setError('Student adapter path is required'));
      return false;
    }
    if (!config.validation_prompts_path) {
      dispatch(setError('Validation prompts path is required'));
      return false;
    }
    if (config.num_steps < 1) {
      dispatch(setError('Number of steps must be at least 1'));
      return false;
    }
    return true;
  };

  const handleStartDistillation = async () => {
    if (!validateConfig()) return;

    setIsSubmitting(true);
    dispatch(setError(null));

    try {
      const response = await axios.post(`${BACKEND_URL}/opd/start`, config);

      if (response.data.status === 'success') {
        dispatch(opdStarted({
          run_id: response.data.run_id,
          config,
          estimated_duration_minutes: response.data.estimated_duration_minutes,
        }));

        dispatch(addNotification({
          type: 'success',
          title: 'Distillation Started',
          message: `Distillation training started (Run ID: ${response.data.run_id})`,
          autoHide: true,
        }));
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to start distillation';
      dispatch(setError(errorMsg));
      dispatch(addNotification({
        type: 'error',
        title: 'Start Failed',
        message: errorMsg,
      }));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Info Card */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <HelpCircle className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
          <div className="space-y-2">
            <p className="text-sm text-blue-900 dark:text-blue-100 font-medium">
              What is On-Policy Distillation?
            </p>
            <p className="text-sm text-blue-800 dark:text-blue-200">
              OPD transfers knowledge from a larger teacher model (e.g., Qwen 32B) to your fine-tuned student model (e.g., Qwen 7B),
              improving quality while maintaining fast inference. The teacher's outputs guide the student's learning through
              KL divergence loss.
            </p>
          </div>
        </div>
      </div>

      {/* Configuration Form */}
      <div className="card p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Configuration
        </h2>

        {/* Model Paths */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Model Paths</h3>

          {/* Base Model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Base Model Path (Student) *
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={config.base_model_path}
                onChange={(e) => handleInputChange('base_model_path', e.target.value)}
                className="input flex-1"
                placeholder="/path/to/qwen2.5-7b"
              />
              <button
                onClick={() => handleFileSelect('base_model_path')}
                className="btn-secondary flex items-center space-x-2"
              >
                <Upload className="h-4 w-4" />
                <span>Browse</span>
              </button>
            </div>
          </div>

          {/* Teacher Model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Teacher Model Path *
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={config.teacher_model_path}
                onChange={(e) => handleInputChange('teacher_model_path', e.target.value)}
                className="input flex-1"
                placeholder="/path/to/qwen2.5-32b-4bit"
              />
              <button
                onClick={() => handleFileSelect('teacher_model_path')}
                className="btn-secondary flex items-center space-x-2"
              >
                <Upload className="h-4 w-4" />
                <span>Browse</span>
              </button>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Larger model to transfer knowledge from (e.g., 32B quantized model)
            </p>
          </div>

          {/* Student Adapter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Student Adapter Path *
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={config.student_adapter_path}
                onChange={(e) => handleInputChange('student_adapter_path', e.target.value)}
                className="input flex-1"
                placeholder="/path/to/your-fine-tuned-adapter"
              />
              <button
                onClick={() => handleFileSelect('student_adapter_path')}
                className="btn-secondary flex items-center space-x-2"
              >
                <Upload className="h-4 w-4" />
                <span>Browse</span>
              </button>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Your fine-tuned LoRA adapter from previous training
            </p>
          </div>

          {/* Validation Prompts */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Validation Prompts File *
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={config.validation_prompts_path}
                onChange={(e) => handleInputChange('validation_prompts_path', e.target.value)}
                className="input flex-1"
                placeholder="/path/to/validation_prompts.jsonl"
              />
              <button
                onClick={() => handleFileSelect('validation_prompts_path')}
                className="btn-secondary flex items-center space-x-2"
              >
                <Upload className="h-4 w-4" />
                <span>Browse</span>
              </button>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              JSONL file with prompts for distillation (one per line: {`{"prompt": "..."}`})
            </p>
          </div>
        </div>

        {/* Training Parameters */}
        <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Training Parameters</h3>

          <div className="grid grid-cols-2 gap-4">
            {/* Number of Steps */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Number of Steps
              </label>
              <input
                type="number"
                value={config.num_steps}
                onChange={(e) => handleInputChange('num_steps', parseInt(e.target.value))}
                className="input w-full"
                min="1"
              />
            </div>

            {/* Batch Size */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={config.batch_size}
                onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}
                className="input w-full"
                min="1"
                max="8"
              />
            </div>
          </div>

          {/* Distillation Parameters */}
          <div className="grid grid-cols-3 gap-4">
            {/* Temperature */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Temperature
              </label>
              <input
                type="number"
                value={config.temperature}
                onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
                className="input w-full"
                step="0.1"
                min="0.1"
              />
            </div>

            {/* KL Weight */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                KL Weight
              </label>
              <input
                type="number"
                value={config.kl_weight}
                onChange={(e) => handleInputChange('kl_weight', parseFloat(e.target.value))}
                className="input w-full"
                step="0.1"
                min="0"
                max="1"
              />
            </div>

            {/* Learning Rate */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                value={config.learning_rate}
                onChange={(e) => handleInputChange('learning_rate', parseFloat(e.target.value))}
                className="input w-full"
                step="0.000001"
                min="0.000001"
              />
            </div>
          </div>
        </div>

        {/* Start Button */}
        <div className="flex justify-end pt-4">
          <button
            onClick={handleStartDistillation}
            disabled={isSubmitting}
            className="btn-primary flex items-center space-x-2"
          >
            {isSubmitting ? (
              <>
                <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Starting...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Start Distillation</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
