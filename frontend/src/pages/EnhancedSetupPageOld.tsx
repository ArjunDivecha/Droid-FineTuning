// frontend/src/pages/EnhancedSetupPage.tsx
// Enhanced setup page with GSPO and Dr. GRPO method selection - matching original design

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Settings, Play, Database, Cpu, Zap, Brain, Target, Sparkles, FileText, AlertCircle } from 'lucide-react';
import { RootState } from '../store/store';
import { addNotification } from '../store/slices/uiSlice';
import { setTrainingConfig } from '../store/slices/trainingSlice';
import axios from 'axios';
import { 
  TrainingMethod, 
  TrainingMethodConfig, 
  EnhancedTrainingConfig,
  ResourceEstimation,
  ValidationResult,
  getValidationRules
} from '../types/enhancedTraining';

const BACKEND_URL = 'http://localhost:8000';

interface EnhancedSetupPageProps {
  onStartTraining?: (config: EnhancedTrainingConfig) => void;
  isTraining?: boolean;
}

const EnhancedSetupPage: React.FC<EnhancedSetupPageProps> = ({ 
  onStartTraining, 
  isTraining = false 
}) => {
  const dispatch = useDispatch();
  // Available training methods (will be fetched from API)
  const [availableMethods, setAvailableMethods] = useState<Record<string, TrainingMethodConfig>>({});
  const [selectedMethod, setSelectedMethod] = useState<TrainingMethod>(TrainingMethod.SFT);
  
  // Resource estimation and validation
  const [resourceEstimation, setResourceEstimation] = useState<ResourceEstimation | null>(null);
  const [dataValidation, setDataValidation] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isEstimating, setIsEstimating] = useState(false);
  
  // Form state
  const [config, setConfig] = useState<EnhancedTrainingConfig>({
    // Base configuration
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
    early_stop: false,
    patience: 10,
    adapter_name: 'enhanced_adapter',
    
    // Enhanced configuration
    training_method: TrainingMethod.SFT,
    
    // GSPO parameters
    sparse_ratio: 0.7,
    efficiency_threshold: 0.85,
    sparse_optimization: true,
    
    // Dr. GRPO parameters  
    domain: 'general',
    expertise_level: 'advanced',
    domain_adaptation_strength: 1.0,
    
    // GRPO parameters
    reasoning_steps: 8,
    multi_step_training: true
  });
  
  // Form validation errors
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  // Load available training methods on component mount
  useEffect(() => {
    fetchAvailableMethods();
  }, []);
  
  // Update config when method changes
  useEffect(() => {
    setConfig(prev => ({ ...prev, training_method: selectedMethod }));
  }, [selectedMethod]);
  
  // Validate data when paths change
  useEffect(() => {
    if (config.train_data_path && selectedMethod) {
      validateData();
    }
  }, [config.train_data_path, selectedMethod]);
  
  // Estimate resources when key parameters change
  useEffect(() => {
    if (config.model_path && selectedMethod) {
      estimateResources();
    }
  }, [config.model_path, selectedMethod, config.iterations]);
  
  const fetchAvailableMethods = async () => {
    try {
      const response = await fetch('/api/training/methods');
      const data = await response.json();
      if (data.success) {
        setAvailableMethods(data.methods);
      }
    } catch (error) {
      console.error('Failed to fetch training methods:', error);
    }
  };
  
  const validateData = async () => {
    if (!config.train_data_path) return;
    
    setIsValidating(true);
    try {
      const response = await fetch('/api/training/validate-data', {\n        method: 'POST',\n        headers: { 'Content-Type': 'application/json' },\n        body: JSON.stringify({\n          method: selectedMethod,\n          data_path: config.train_data_path\n        })\n      });\n      \n      const data = await response.json();\n      if (data.success) {\n        setDataValidation(data.validation);\n      }\n    } catch (error) {\n      console.error('Data validation failed:', error);\n    } finally {\n      setIsValidating(false);\n    }\n  };\n  \n  const estimateResources = async () => {\n    if (!config.model_path) return;\n    \n    setIsEstimating(true);\n    try {\n      const response = await fetch('/api/training/estimate-resources', {\n        method: 'POST',\n        headers: { 'Content-Type': 'application/json' },\n        body: JSON.stringify({\n          method: selectedMethod,\n          model_path: config.model_path,\n          dataset_size: config.iterations // Approximation\n        })\n      });\n      \n      const data = await response.json();\n      if (data.success) {\n        setResourceEstimation(data.estimation);\n      }\n    } catch (error) {\n      console.error('Resource estimation failed:', error);\n    } finally {\n      setIsEstimating(false);\n    }\n  };\n  \n  const generateSampleData = async () => {\n    try {\n      const response = await fetch('/api/training/generate-sample-data', {\n        method: 'POST',\n        headers: { 'Content-Type': 'application/json' },\n        body: JSON.stringify({\n          method: selectedMethod,\n          output_path: `/tmp/sample_${selectedMethod}_data.jsonl`,\n          num_samples: 20\n        })\n      });\n      \n      const data = await response.json();\n      if (data.success) {\n        setConfig(prev => ({ ...prev, train_data_path: data.output_path }));\n        alert(`Sample data generated: ${data.output_path}`);\n      }\n    } catch (error) {\n      console.error('Sample data generation failed:', error);\n    }\n  };\n  \n  const validateForm = (): boolean => {\n    const newErrors: Record<string, string> = {};\n    const rules = getValidationRules(selectedMethod);\n    \n    // Check required fields\n    rules.required.forEach(field => {\n      if (!config[field as keyof EnhancedTrainingConfig]) {\n        newErrors[field] = `${field.replace('_', ' ')} is required`;\n      }\n    });\n    \n    // Check numeric ranges\n    Object.entries(rules.range).forEach(([field, range]) => {\n      const value = config[field as keyof EnhancedTrainingConfig] as number;\n      if (value < range.min || value > range.max) {\n        newErrors[field] = `${field} must be between ${range.min} and ${range.max}`;\n      }\n    });\n    \n    // Check data validation\n    if (dataValidation && !dataValidation.valid) {\n      newErrors.train_data_path = dataValidation.error || 'Invalid data format';\n    }\n    \n    setErrors(newErrors);\n    return Object.keys(newErrors).length === 0;\n  };\n  \n  const handleSubmit = async (e: React.FormEvent) => {\n    e.preventDefault();\n    \n    if (!validateForm()) {\n      return;\n    }\n    \n    // Start enhanced training\n    try {\n      const response = await fetch('/api/training/start-enhanced', {\n        method: 'POST',\n        headers: { 'Content-Type': 'application/json' },\n        body: JSON.stringify(config)\n      });\n      \n      const data = await response.json();\n      if (data.success) {\n        onStartTraining(config);\n      } else {\n        alert(`Training start failed: ${data.error}`);\n      }\n    } catch (error) {\n      console.error('Failed to start training:', error);\n      alert('Failed to start training');\n    }\n  };\n  \n  const handleInputChange = (field: keyof EnhancedTrainingConfig, value: any) => {\n    setConfig(prev => ({ ...prev, [field]: value }));\n    \n    // Clear error for this field\n    if (errors[field]) {\n      setErrors(prev => ({ ...prev, [field]: '' }));\n    }\n  };\n  \n  const renderMethodSelector = () => (\n    <div className=\"method-selector\">\n      <h3>Training Method Selection</h3>\n      <div className=\"method-grid\">\n        {Object.entries(availableMethods).map(([method, methodConfig]) => (\n          <div \n            key={method}\n            className={`method-card ${selectedMethod === method ? 'selected' : ''}`}\n            onClick={() => setSelectedMethod(method as TrainingMethod)}\n          >\n            <div className=\"method-header\">\n              <h4>{methodConfig.display_name}</h4>\n              {methodConfig.badge && (\n                <span className=\"method-badge\">{methodConfig.badge}</span>\n              )}\n            </div>\n            <p className=\"method-description\">{methodConfig.description}</p>\n            <div className=\"method-details\">\n              <div className=\"complexity\">\n                <span>Complexity: {methodConfig.complexity}</span>\n              </div>\n              <div className=\"resource-intensity\">\n                <span>Resources: {methodConfig.resource_intensity}</span>\n              </div>\n              {methodConfig.estimated_speedup && (\n                <div className=\"speedup\">\n                  <span>{methodConfig.estimated_speedup}</span>\n                </div>\n              )}\n            </div>\n            <p className=\"use-case\"><strong>Best for:</strong> {methodConfig.use_case}</p>\n          </div>\n        ))}\n      </div>\n    </div>\n  );\n  \n  const renderBasicConfiguration = () => (\n    <div className=\"config-section\">\n      <h3>Basic Configuration</h3>\n      <div className=\"config-grid\">\n        <div className=\"config-item\">\n          <label>Model Path</label>\n          <input \n            type=\"text\" \n            value={config.model_path}\n            onChange={(e) => handleInputChange('model_path', e.target.value)}\n            className={errors.model_path ? 'error' : ''}\n          />\n          {errors.model_path && <span className=\"error-text\">{errors.model_path}</span>}\n        </div>\n        \n        <div className=\"config-item\">\n          <label>Training Data Path</label>\n          <div className=\"input-with-button\">\n            <input \n              type=\"text\" \n              value={config.train_data_path}\n              onChange={(e) => handleInputChange('train_data_path', e.target.value)}\n              className={errors.train_data_path ? 'error' : ''}\n            />\n            <button \n              type=\"button\" \n              onClick={generateSampleData}\n              className=\"generate-sample-btn\"\n            >\n              Generate Sample\n            </button>\n          </div>\n          {isValidating && <span className=\"validating\">Validating...</span>}\n          {dataValidation && (\n            <div className={`validation-result ${dataValidation.valid ? 'valid' : 'invalid'}`}>\n              {dataValidation.valid ? '✓ Valid format' : `✗ ${dataValidation.error}`}\n            </div>\n          )}\n          {errors.train_data_path && <span className=\"error-text\">{errors.train_data_path}</span>}\n        </div>\n        \n        <div className=\"config-item\">\n          <label>Validation Data Path (Optional)</label>\n          <input \n            type=\"text\" \n            value={config.val_data_path}\n            onChange={(e) => handleInputChange('val_data_path', e.target.value)}\n          />\n        </div>\n        \n        <div className=\"config-item\">\n          <label>Adapter Name</label>\n          <input \n            type=\"text\" \n            value={config.adapter_name}\n            onChange={(e) => handleInputChange('adapter_name', e.target.value)}\n            className={errors.adapter_name ? 'error' : ''}\n          />\n          {errors.adapter_name && <span className=\"error-text\">{errors.adapter_name}</span>}\n        </div>\n      </div>\n    </div>\n  );\n  \n  const renderTrainingParameters = () => (\n    <div className=\"config-section\">\n      <h3>Training Parameters</h3>\n      <div className=\"config-grid\">\n        <div className=\"config-item\">\n          <label>Learning Rate</label>\n          <input \n            type=\"number\" \n            step=\"0.00001\"\n            value={config.learning_rate}\n            onChange={(e) => handleInputChange('learning_rate', parseFloat(e.target.value))}\n            className={errors.learning_rate ? 'error' : ''}\n          />\n          {errors.learning_rate && <span className=\"error-text\">{errors.learning_rate}</span>}\n        </div>\n        \n        <div className=\"config-item\">\n          <label>Batch Size</label>\n          <input \n            type=\"number\" \n            value={config.batch_size}\n            onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}\n            className={errors.batch_size ? 'error' : ''}\n          />\n          {errors.batch_size && <span className=\"error-text\">{errors.batch_size}</span>}\n        </div>\n        \n        <div className=\"config-item\">\n          <label>Max Sequence Length</label>\n          <input \n            type=\"number\" \n            value={config.max_seq_length}\n            onChange={(e) => handleInputChange('max_seq_length', parseInt(e.target.value))}\n            className={errors.max_seq_length ? 'error' : ''}\n          />\n          {errors.max_seq_length && <span className=\"error-text\">{errors.max_seq_length}</span>}\n        </div>\n        \n        <div className=\"config-item\">\n          <label>Iterations</label>\n          <input \n            type=\"number\" \n            value={config.iterations}\n            onChange={(e) => handleInputChange('iterations', parseInt(e.target.value))}\n            className={errors.iterations ? 'error' : ''}\n          />\n          {errors.iterations && <span className=\"error-text\">{errors.iterations}</span>}\n        </div>\n        \n        <div className=\"config-item\">\n          <label>\n            <input \n              type=\"checkbox\" \n              checked={config.early_stop}\n              onChange={(e) => handleInputChange('early_stop', e.target.checked)}\n            />\n            Enable Early Stopping\n          </label>\n        </div>\n        \n        {config.early_stop && (\n          <div className=\"config-item\">\n            <label>Patience (evaluations)</label>\n            <input \n              type=\"number\" \n              value={config.patience}\n              onChange={(e) => handleInputChange('patience', parseInt(e.target.value))}\n            />\n          </div>\n        )}\n      </div>\n    </div>\n  );\n  \n  const renderMethodSpecificConfig = () => {\n    if (selectedMethod === TrainingMethod.GSPO) {\n      return (\n        <div className=\"config-section gspo-config\">\n          <h3>GSPO Configuration</h3>\n          <div className=\"config-grid\">\n            <div className=\"config-item\">\n              <label>Sparse Ratio</label>\n              <input \n                type=\"number\" \n                step=\"0.1\"\n                min=\"0.1\"\n                max=\"0.9\"\n                value={config.sparse_ratio}\n                onChange={(e) => handleInputChange('sparse_ratio', parseFloat(e.target.value))}\n              />\n              <span className=\"help-text\">Fraction of reasoning steps to optimize (0.1-0.9)</span>\n            </div>\n            \n            <div className=\"config-item\">\n              <label>Efficiency Threshold</label>\n              <input \n                type=\"number\" \n                step=\"0.05\"\n                min=\"0.5\"\n                max=\"1.0\"\n                value={config.efficiency_threshold}\n                onChange={(e) => handleInputChange('efficiency_threshold', parseFloat(e.target.value))}\n              />\n              <span className=\"help-text\">Minimum efficiency score to maintain (0.5-1.0)</span>\n            </div>\n            \n            <div className=\"config-item\">\n              <label>\n                <input \n                  type=\"checkbox\" \n                  checked={config.sparse_optimization}\n                  onChange={(e) => handleInputChange('sparse_optimization', e.target.checked)}\n                />\n                Enable Sparse Optimization\n              </label>\n              <span className=\"help-text\">Use sparse attention patterns for efficiency</span>\n            </div>\n          </div>\n        </div>\n      );\n    }\n    \n    if (selectedMethod === TrainingMethod.DR_GRPO) {\n      return (\n        <div className=\"config-section dr-grpo-config\">\n          <h3>Dr. GRPO Configuration</h3>\n          <div className=\"config-grid\">\n            <div className=\"config-item\">\n              <label>Domain</label>\n              <select \n                value={config.domain}\n                onChange={(e) => handleInputChange('domain', e.target.value)}\n              >\n                <option value=\"general\">General</option>\n                <option value=\"medical\">Medical</option>\n                <option value=\"scientific\">Scientific</option>\n                <option value=\"legal\">Legal</option>\n                <option value=\"technical\">Technical</option>\n              </select>\n              <span className=\"help-text\">Specialized knowledge domain</span>\n            </div>\n            \n            <div className=\"config-item\">\n              <label>Expertise Level</label>\n              <select \n                value={config.expertise_level}\n                onChange={(e) => handleInputChange('expertise_level', e.target.value)}\n              >\n                <option value=\"beginner\">Beginner</option>\n                <option value=\"intermediate\">Intermediate</option>\n                <option value=\"advanced\">Advanced</option>\n                <option value=\"expert\">Expert</option>\n              </select>\n              <span className=\"help-text\">Target expertise level for reasoning</span>\n            </div>\n            \n            <div className=\"config-item\">\n              <label>Domain Adaptation Strength</label>\n              <input \n                type=\"number\" \n                step=\"0.1\"\n                min=\"0.1\"\n                max=\"2.0\"\n                value={config.domain_adaptation_strength}\n                onChange={(e) => handleInputChange('domain_adaptation_strength', parseFloat(e.target.value))}\n              />\n              <span className=\"help-text\">Strength of domain-specific adaptation (0.1-2.0)</span>\n            </div>\n          </div>\n        </div>\n      );\n    }\n    \n    if (selectedMethod === TrainingMethod.GRPO) {\n      return (\n        <div className=\"config-section grpo-config\">\n          <h3>GRPO Configuration</h3>\n          <div className=\"config-grid\">\n            <div className=\"config-item\">\n              <label>Reasoning Steps</label>\n              <input \n                type=\"number\" \n                min=\"3\"\n                max=\"15\"\n                value={config.reasoning_steps}\n                onChange={(e) => handleInputChange('reasoning_steps', parseInt(e.target.value))}\n              />\n              <span className=\"help-text\">Number of reasoning steps to train (3-15)</span>\n            </div>\n            \n            <div className=\"config-item\">\n              <label>\n                <input \n                  type=\"checkbox\" \n                  checked={config.multi_step_training}\n                  onChange={(e) => handleInputChange('multi_step_training', e.target.checked)}\n                />\n                Enable Multi-Step Training\n              </label>\n              <span className=\"help-text\">Train on intermediate reasoning steps</span>\n            </div>\n          </div>\n        </div>\n      );\n    }\n    \n    return null;\n  };\n  \n  const renderResourceEstimation = () => {\n    if (!resourceEstimation) return null;\n    \n    return (\n      <div className=\"resource-estimation\">\n        <h3>Resource Estimation</h3>\n        <div className=\"estimation-grid\">\n          <div className=\"estimation-item\">\n            <span className=\"label\">Memory:</span>\n            <span className=\"value\">{resourceEstimation.estimated_memory_gb} GB</span>\n          </div>\n          <div className=\"estimation-item\">\n            <span className=\"label\">Time:</span>\n            <span className=\"value\">{resourceEstimation.estimated_time_hours} hours</span>\n          </div>\n          <div className=\"estimation-item\">\n            <span className=\"label\">Intensity:</span>\n            <span className={`intensity ${resourceEstimation.resource_intensity}`}>\n              {resourceEstimation.resource_intensity}\n            </span>\n          </div>\n        </div>\n        \n        {resourceEstimation.recommendations.length > 0 && (\n          <div className=\"recommendations\">\n            <h4>Recommendations:</h4>\n            <ul>\n              {resourceEstimation.recommendations.map((rec, index) => (\n                <li key={index}>{rec}</li>\n              ))}\n            </ul>\n          </div>\n        )}\n      </div>\n    );\n  };\n  \n  return (\n    <div className=\"enhanced-setup-page\">\n      <div className=\"setup-header\">\n        <h1>Enhanced Training Setup</h1>\n        <p>Configure advanced training methods including GSPO and Dr. GRPO</p>\n      </div>\n      \n      <form onSubmit={handleSubmit} className=\"setup-form\">\n        {renderMethodSelector()}\n        {renderBasicConfiguration()}\n        {renderTrainingParameters()}\n        {renderMethodSpecificConfig()}\n        {renderResourceEstimation()}\n        \n        <div className=\"form-actions\">\n          <button \n            type=\"submit\" \n            disabled={isTraining || isValidating || isEstimating}\n            className=\"start-training-btn\"\n          >\n            {isTraining ? 'Training in Progress...' : `Start ${selectedMethod.toUpperCase()} Training`}\n          </button>\n        </div>\n      </form>\n    </div>\n  );\n};\n\nexport default EnhancedSetupPage;"