import React, { useState, useRef, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { 
  Send, 
  RefreshCw, 
  Copy, 
  Trash2,
  ArrowRight,
  Star,
  FolderOpen,
  BarChart3,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { RootState } from '../store/store';
import { LoadSessionModal } from '../components/LoadSessionModal';
import axios from 'axios';

interface Comparison {
  id: string;
  prompt: string;
  baseResponse: string;
  fineTunedResponse: string;
  rating?: 1 | 2 | 3 | 4 | 5;
  timestamp: Date;
}

interface Evaluation {
  id: string;
  adapter_name: string;
  is_base_model: boolean;
  overall_score: number;
  faithfulness: number;
  fact_recall: number;
  consistency: number;
  hallucination: number;
  num_questions: number;
  timestamp: Date;
  detailed_results?: any;
}

export const ComparePage: React.FC = () => {
  const { config } = useSelector((state: RootState) => state.training);
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [selectedComparison, setSelectedComparison] = useState<string | null>(null);
  const [isLoadSessionOpen, setIsLoadSessionOpen] = useState(false);
  const [loadedSession, setLoadedSession] = useState<any>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  // Evaluation state
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationProgress, setEvaluationProgress] = useState(0);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  // Comprehensive clipboard utilities with multiple fallbacks
  const clipboardUtils = {
    // Check if modern Clipboard API is available
    isClipboardAPIAvailable: () => {
      return (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        window.isSecureContext
      );
    },

    // Check if Electron API is available
    isElectronClipboardAvailable: () => {
      return (
        typeof window !== 'undefined' &&
        (window as any).electronAPI?.clipboard
      );
    },

    // Method 1: Modern Clipboard API
    async readFromClipboardAPI(): Promise<string | null> {
      try {
        if (this.isClipboardAPIAvailable()) {
          const text = await navigator.clipboard.readText();
          return text;
        }
      } catch (error) {
        console.warn('Clipboard API read failed:', error);
      }
      return null;
    },

    // Method 2: Electron Clipboard API
    readFromElectronAPI(): string | null {
      try {
        if (this.isElectronClipboardAvailable()) {
          const electronAPI = (window as any).electronAPI;
          return electronAPI.clipboard.readText();
        }
      } catch (error) {
        console.warn('Electron clipboard read failed:', error);
      }
      return null;
    },

    // Method 3: Traditional execCommand (deprecated but still works)
    async readFromExecCommand(): Promise<string | null> {
      try {
        // Create temporary textarea for paste operation
        const tempTextarea = document.createElement('textarea');
        tempTextarea.style.position = 'absolute';
        tempTextarea.style.left = '-9999px';
        tempTextarea.style.top = '-9999px';
        document.body.appendChild(tempTextarea);

        tempTextarea.focus();
        tempTextarea.select();

        // Execute paste command
        const success = document.execCommand('paste');
        const text = success ? tempTextarea.value : null;

        document.body.removeChild(tempTextarea);
        return text;
      } catch (error) {
        console.warn('execCommand paste failed:', error);
      }
      return null;
    },

    // Comprehensive read with all fallbacks
    async readFromClipboard(): Promise<string | null> {
      // Try modern API first
      let text = await this.readFromClipboardAPI();
      if (text) return text;

      // Try Electron API
      text = this.readFromElectronAPI();
      if (text) return text;

      // Try execCommand as last resort
      text = await this.readFromExecCommand();
      return text;
    }
  };

  // Enhanced paste handler with multiple fallback strategies
  const handleAdvancedPaste = async (e: ClipboardEvent) => {
    e.stopPropagation();
    e.preventDefault();

    const textarea = textareaRef.current;
    if (!textarea) return;

    let pastedText: string | null = null;

    try {
      // Strategy 1: Use event clipboardData (most reliable)
      if (e.clipboardData) {
        pastedText = e.clipboardData.getData('text/plain');
      }

      // Strategy 2: If no clipboardData, try comprehensive fallbacks
      if (!pastedText) {
        pastedText = await clipboardUtils.readFromClipboard();
      }

      if (pastedText) {
        // Get current cursor position
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;

        // Create new value
        const currentValue = textarea.value;
        const newValue = currentValue.substring(0, start) + pastedText + currentValue.substring(end);

        // Update textarea and React state
        textarea.value = newValue;
        setPrompt(newValue);

        // Set cursor position after pasted text
        const newCursorPos = start + pastedText.length;
        textarea.setSelectionRange(newCursorPos, newCursorPos);

        // Ensure focus
        textarea.focus();

        console.log('Paste successful:', pastedText.length, 'characters');
      } else {
        console.warn('No clipboard data available');
      }
    } catch (error) {
      console.error('Paste operation failed:', error);
    }
  };

  // Use DOM-level event handling for maximum compatibility
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    // Add paste event listener
    textarea.addEventListener('paste', handleAdvancedPaste);

    // Also handle keyboard shortcuts directly
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        // Let the paste event handler deal with it
        return;
      }
    };

    textarea.addEventListener('keydown', handleKeyDown);

    // Cleanup
    return () => {
      textarea.removeEventListener('paste', handleAdvancedPaste);
      textarea.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Simple change handler
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
  };

  // Real model inference function
  const generateResponses = async (inputPrompt: string) => {
    setIsGenerating(true);
    
    try {
      // Get the current configuration (could be from current training or loaded session)
      const currentConfig = loadedSession || config;
      
      if (!currentConfig) {
        throw new Error('No model configuration available. Please complete a training session first.');
      }
      
      // Generate responses from both models in parallel
      const [baseResponse, fineTunedResponse] = await Promise.all([
        axios.post('http://localhost:8000/model/test-base', {
          prompt: inputPrompt,
          max_tokens: 1024,
          temperature: 0.7
        }),
        axios.post('http://localhost:8000/model/test', {
          prompt: inputPrompt,
          max_tokens: 1024,
          temperature: 0.7
        })
      ]);

      const baseData = baseResponse.data;
      const fineTunedData = fineTunedResponse.data;
      
      const newComparison: Comparison = {
        id: Math.random().toString(36).substr(2, 9),
        prompt: inputPrompt,
        baseResponse: baseData.response,
        fineTunedResponse: fineTunedData.response,
        timestamp: new Date()
      };
      
      setComparisons(prev => [newComparison, ...prev]);
      setSelectedComparison(newComparison.id);
    } catch (error: any) {
      console.error('Failed to generate responses:', error);
      
      let baseErrorMessage = 'Error: Failed to generate base model response';
      let fineTunedErrorMessage = 'Error: Failed to generate fine-tuned response';
      
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        baseErrorMessage = `Error: ${detail}`;
        fineTunedErrorMessage = `Error: ${detail}`;
      } else if (error.message) {
        baseErrorMessage = `Error: ${error.message}`;
        fineTunedErrorMessage = `Error: ${error.message}`;
      }
      
      // Show error in the comparison
      const errorComparison: Comparison = {
        id: Math.random().toString(36).substr(2, 9),
        prompt: inputPrompt,
        baseResponse: baseErrorMessage,
        fineTunedResponse: fineTunedErrorMessage,
        timestamp: new Date()
      };
      
      setComparisons(prev => [errorComparison, ...prev]);
      setSelectedComparison(errorComparison.id);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isGenerating) {
      generateResponses(prompt.trim());
      setPrompt('');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const rateComparison = (comparisonId: string, rating: 1 | 2 | 3 | 4 | 5) => {
    setComparisons(prev => 
      prev.map(comp => 
        comp.id === comparisonId 
          ? { ...comp, rating }
          : comp
      )
    );
  };

  const deleteComparison = (comparisonId: string) => {
    setComparisons(prev => prev.filter(comp => comp.id !== comparisonId));
    if (selectedComparison === comparisonId) {
      setSelectedComparison(null);
    }
  };

  const handleSessionLoaded = (session: any) => {
    setLoadedSession(session);
    // You might want to update the Redux store here as well
    console.log('Loaded session:', session);
  };

  const handleStartEvaluation = async () => {
    if (!loadedSession) {
      alert('Please load a training session first');
      return;
    }
    
    setIsEvaluating(true);
    setEvaluationProgress(0);
    setEvaluationError(null);
    
    try {
      const adapterName = loadedSession.adapter_name || 'mlx_finetune';
      const newEvaluations: Evaluation[] = [];
      
      // Evaluate BOTH base model and adapter
      for (const isBase of [true, false]) {
        const response = await axios.post('http://localhost:8000/api/evaluation/start', {
          adapter_name: adapterName,
          training_data_path: null,
          num_questions: 20,
          evaluate_base_model: isBase
        });
        
        if (!response.data.success) {
          throw new Error(`Failed to start ${isBase ? 'base model' : 'adapter'} evaluation`);
        }
        
        // Poll for this evaluation to complete
        let hasAddedResult = false;
        await new Promise<void>((resolve, reject) => {
          const pollInterval = setInterval(async () => {
            try {
              const statusResponse = await axios.get('http://localhost:8000/api/evaluation/status');
              const status = statusResponse.data;
              
              // Update progress (0-50% for base, 50-100% for adapter)
              const progressOffset = isBase ? 0 : 50;
              setEvaluationProgress(progressOffset + (status.progress / 2));
              
              if (!status.running && !hasAddedResult) {
                clearInterval(pollInterval);
                hasAddedResult = true;
                
                if (status.error) {
                  reject(new Error(status.error));
                } else {
                  const resultResponse = await axios.get('http://localhost:8000/api/evaluation/result');
                  const result = resultResponse.data.result;
                  
                  newEvaluations.push({
                    id: `${Date.now()}-${isBase ? 'base' : 'adapter'}`,
                    adapter_name: result.adapter_name,
                    is_base_model: result.is_base_model || false,
                    overall_score: result.scores.overall,
                    faithfulness: result.scores.faithfulness,
                    fact_recall: result.scores.fact_recall,
                    consistency: result.scores.consistency,
                    hallucination: result.scores.hallucination,
                    num_questions: result.num_questions,
                    timestamp: new Date(),
                    detailed_results: result.detailed_results
                  });
                  
                  resolve();
                }
              }
            } catch (error) {
              clearInterval(pollInterval);
              reject(error);
            }
          }, 2000);
        });
      }
      
      // Add both evaluations at once
      setEvaluations(prev => [...newEvaluations, ...prev]);
      setIsEvaluating(false);
      setEvaluationProgress(100);
      
    } catch (error: any) {
      console.error('Failed to complete evaluation:', error);
      setEvaluationError(error.message || 'Failed to complete evaluation');
      setIsEvaluating(false);
    }
  };

  const selectedComp = comparisons.find(comp => comp.id === selectedComparison);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Model Comparison
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Compare responses between your base model and fine-tuned model
        </p>
      </div>

      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <div className="card-body">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Base Model</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {config?.model_path.split('/').pop() || 'No model selected'}
            </p>
            <div className="mt-2">
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
                Original Model
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">Fine-tuned Model</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {loadedSession ? 
                    `${loadedSession.model_name} + ${loadedSession.adapter_name}` : 
                    (config?.adapter_name ? `${config.model_path.split('/').pop()} + ${config.adapter_name}` : 'No trained model available')
                  }
                </p>
              </div>
              <button
                onClick={() => setIsLoadSessionOpen(true)}
                className="btn-secondary flex items-center space-x-2 text-xs"
                title="Load a different trained model"
              >
                <FolderOpen className="h-3 w-3" />
                <span>Load Trained</span>
              </button>
            </div>
            <div className="mt-2 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                  (loadedSession || config?.adapter_name) ? 
                    'bg-success-100 dark:bg-success-900/30 text-success-800 dark:text-success-200' :
                    'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'
                }`}>
                  {loadedSession ? 'Loaded Session' : 
                   config?.adapter_name ? 'Model Available' : 'No Model'}
                </span>
                {(loadedSession || config?.adapter_name) && (
                  <span className="text-xs text-success-600 dark:text-success-400 font-medium">
                    ✓ Ready for inference
                  </span>
                )}
              </div>
              {loadedSession && (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {new Date(loadedSession.timestamp).toLocaleDateString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Prompt Input */}
      <div className="card">
        <div className="card-body">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Enter your prompt to compare model responses
              </label>
              <textarea
                ref={textareaRef}
                value={prompt}
                onChange={handleChange}
                placeholder="Enter a prompt to test both models... (Paste works with Ctrl+V/Cmd+V or right-click)"
                rows={4}
                className="input-field resize-none"
                disabled={isGenerating}
                autoComplete="off"
                spellCheck="true"
                tabIndex={0}
              />
            </div>
            
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={!prompt.trim() || isGenerating}
                className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    <span>Generate Comparison</span>
                  </>
                )}
              </button>
              
              <button
                type="button"
                onClick={handleStartEvaluation}
                disabled={!loadedSession || isEvaluating}
                className="btn-secondary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Evaluate both base model and adapter for comparison"
              >
                {isEvaluating ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Evaluating Both...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-4 w-4" />
                    <span>Evaluate Base vs Adapter</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Comparison Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Comparison History */}
        <div className="lg:col-span-1">
          <div className="card h-fit max-h-96 overflow-hidden">
            <div className="card-header">
              <h3 className="text-lg font-semibold">Comparison History</h3>
            </div>
            <div className="overflow-y-auto max-h-80">
              {evaluations.length === 0 && comparisons.length === 0 ? (
                <div className="p-6 text-center text-gray-500 dark:text-gray-400">
                  <ArrowRight className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No comparisons yet</p>
                  <p className="text-sm">Generate your first comparison above</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Evaluations Section */}
                  {evaluations.length > 0 && (() => {
                    // Group evaluations by adapter name, find pairs within 5 minutes
                    const pairs: Array<{base: Evaluation, adapter: Evaluation}> = [];
                    const baseEvals = evaluations.filter(e => e.is_base_model);
                    const adapterEvals = evaluations.filter(e => !e.is_base_model);
                    
                    for (const base of baseEvals) {
                      const matchingAdapter = adapterEvals.find(a => 
                        a.adapter_name === base.adapter_name &&
                        Math.abs(a.timestamp.getTime() - base.timestamp.getTime()) < 5 * 60 * 1000
                      );
                      if (matchingAdapter) {
                        pairs.push({ base, adapter: matchingAdapter });
                      }
                    }
                    
                    return (
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 px-3 py-2 bg-gray-50 dark:bg-gray-800">
                          Evaluation Results
                        </h4>
                        <div className="p-3 space-y-4">
                          {pairs.map((pair, idx) => {
                            const baseEval = pair.base;
                            const adapterEval = pair.adapter;
                            
                            const improvement = adapterEval.overall_score - baseEval.overall_score;
                            
                            return (
                              <div key={idx} className="card">
                                <div className="card-header flex items-center justify-between">
                                  <div>
                                    <h5 className="font-semibold text-gray-900 dark:text-gray-100">
                                      {adapterEval.adapter_name}
                                    </h5>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                      {adapterEval.num_questions} questions • {adapterEval.timestamp.toLocaleDateString()}
                                    </p>
                                  </div>
                                  <div className={`text-right ${improvement > 0 ? 'text-success-600 dark:text-success-400' : 'text-error-600 dark:text-error-400'}`}>
                                    <div className="text-2xl font-bold">
                                      {improvement > 0 ? '+' : ''}{improvement.toFixed(1)}
                                    </div>
                                    <div className="text-xs">Improvement</div>
                                  </div>
                                </div>
                                <div className="card-body p-0">
                                  <table className="w-full text-sm">
                                    <thead className="bg-gray-50 dark:bg-gray-800">
                                      <tr>
                                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Metric</th>
                                        <th className="px-4 py-2 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Base Model</th>
                                        <th className="px-4 py-2 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Fine-tuned</th>
                                        <th className="px-4 py-2 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Δ</th>
                                      </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                      {[
                                        { label: 'Overall Score', base: baseEval.overall_score, adapter: adapterEval.overall_score },
                                        { label: 'Faithfulness', base: baseEval.faithfulness, adapter: adapterEval.faithfulness },
                                        { label: 'Fact Recall', base: baseEval.fact_recall, adapter: adapterEval.fact_recall },
                                        { label: 'Consistency', base: baseEval.consistency, adapter: adapterEval.consistency },
                                      ].map((row, i) => {
                                        const delta = row.adapter - row.base;
                                        return (
                                          <tr key={i} className={i === 0 ? 'bg-primary-50 dark:bg-primary-900/10' : ''}>
                                            <td className={`px-4 py-3 ${i === 0 ? 'font-semibold' : ''} text-gray-900 dark:text-gray-100`}>
                                              {row.label}
                                            </td>
                                            <td className="px-4 py-3 text-center text-gray-600 dark:text-gray-400">
                                              {row.base}/100
                                            </td>
                                            <td className="px-4 py-3 text-center font-medium text-gray-900 dark:text-gray-100">
                                              {row.adapter}/100
                                            </td>
                                            <td className={`px-4 py-3 text-center font-semibold ${
                                              delta > 0 ? 'text-success-600 dark:text-success-400' : 
                                              delta < 0 ? 'text-error-600 dark:text-error-400' : 
                                              'text-gray-500 dark:text-gray-400'
                                            }`}>
                                              {delta > 0 ? '+' : ''}{delta.toFixed(1)}
                                            </td>
                                          </tr>
                                        );
                                      })}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })()}
                  
                  {/* Comparisons Section */}
                  {comparisons.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 px-3 py-2 bg-gray-50 dark:bg-gray-800">
                        Response Comparisons
                      </h4>
                      <div className="space-y-1">
                        {comparisons.map((comparison) => (
                          <div
                            key={comparison.id}
                            onClick={() => setSelectedComparison(comparison.id)}
                            className={`p-3 cursor-pointer border-l-4 transition-colors ${
                              selectedComparison === comparison.id
                                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                                : 'border-transparent hover:bg-gray-50 dark:hover:bg-gray-800'
                            }`}
                          >
                            <div className="text-sm font-medium text-gray-900 dark:text-gray-100 line-clamp-2">
                              {comparison.prompt}
                            </div>
                            <div className="flex items-center justify-between mt-2">
                              <div className="text-xs text-gray-500 dark:text-gray-400">
                                {comparison.timestamp.toLocaleDateString()}
                              </div>
                              {comparison.rating && (
                                <div className="flex items-center space-x-1">
                                  {Array.from({ length: 5 }, (_, i) => (
                                    <Star
                                      key={i}
                                      className={`h-3 w-3 ${
                                        i < comparison.rating!
                                          ? 'text-yellow-400 fill-current'
                                          : 'text-gray-300'
                                      }`}
                                    />
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Comparison View */}
        <div className="lg:col-span-2">
          {selectedComp ? (
            <div className="space-y-6">
              {/* Prompt */}
              <div className="card">
                <div className="card-header">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Prompt</h3>
                    <button
                      onClick={() => deleteComparison(selectedComp.id)}
                      className="text-error-600 hover:text-error-700 p-1"
                      title="Delete comparison"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <div className="card-body">
                  <p className="text-gray-900 dark:text-gray-100">{selectedComp.prompt}</p>
                </div>
              </div>

              {/* Responses */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {/* Base Model Response */}
                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">Base Model</h4>
                      <button
                        onClick={() => copyToClipboard(selectedComp.baseResponse)}
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 p-1"
                        title="Copy response"
                      >
                        <Copy className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <div className="card-body">
                    <div className="prose dark:prose-invert max-w-none">
                      <p className="whitespace-pre-wrap text-sm">
                        {selectedComp.baseResponse}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Fine-tuned Model Response */}
                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">Fine-tuned Model</h4>
                      <button
                        onClick={() => copyToClipboard(selectedComp.fineTunedResponse)}
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 p-1"
                        title="Copy response"
                      >
                        <Copy className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <div className="card-body">
                    <div className="prose dark:prose-invert max-w-none">
                      <p className="whitespace-pre-wrap text-sm">
                        {selectedComp.fineTunedResponse}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Rating */}
              <div className="card">
                <div className="card-header">
                  <h4 className="font-semibold">Rate Fine-tuned Response</h4>
                </div>
                <div className="card-body">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Quality:</span>
                    <div className="flex items-center space-x-1">
                      {Array.from({ length: 5 }, (_, i) => (
                        <button
                          key={i}
                          onClick={() => rateComparison(selectedComp.id, (i + 1) as 1 | 2 | 3 | 4 | 5)}
                          className="p-1 hover:scale-110 transition-transform"
                        >
                          <Star
                            className={`h-6 w-6 ${
                              selectedComp.rating && i < selectedComp.rating
                                ? 'text-yellow-400 fill-current'
                                : 'text-gray-300 hover:text-yellow-300'
                            }`}
                          />
                        </button>
                      ))}
                    </div>
                    {selectedComp.rating && (
                      <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">
                        ({selectedComp.rating}/5)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96">
              <div className="text-center text-gray-500 dark:text-gray-400">
                <ArrowRight className="h-12 w-12 mx-auto mb-4" />
                <p>Select a comparison from the history to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Load Session Modal */}
      <LoadSessionModal
        isOpen={isLoadSessionOpen}
        onClose={() => setIsLoadSessionOpen(false)}
        onSessionLoaded={handleSessionLoaded}
      />
      
      {/* Evaluation Progress Modal */}
      {isEvaluating && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Evaluating Base Model vs Adapter...
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <span>Progress</span>
                  <span>{Math.round(evaluationProgress)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${evaluationProgress}%` }}
                  />
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Evaluating both base model and fine-tuned adapter using Cerebras AI...
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-500">
                This should take 2-3 minutes for 40 questions total.
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Evaluation Error Modal */}
      {evaluationError && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center space-x-3 mb-4">
              <AlertCircle className="w-6 h-6 text-error-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Evaluation Failed
              </h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              {evaluationError}
            </p>
            <button
              onClick={() => setEvaluationError(null)}
              className="btn-primary w-full"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};