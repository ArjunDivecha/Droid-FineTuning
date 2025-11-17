import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Info,
  Zap,
  GitMerge,
  BarChart3,
} from 'lucide-react';
import { EvaluationCard } from '../components/EvaluationCard';
import axios from 'axios';

interface AdapterInfo {
  name: string;
  base_model: string;
  training_date: string | null;
  iterations: number | null;
  dataset: string | null;
  path: string;
}

interface AdaptersByBaseModel {
  base_model: string;
  adapters: AdapterInfo[];
}

interface EvaluationResult {
  adapter_name: string;
  is_base_model: boolean;
  overall_score: number;
  faithfulness: number;
  fact_recall: number;
  consistency: number;
  hallucination: number;
}

interface TierEvaluationResult {
  id: string;
  adapter_name: string;
  is_base_model: boolean;
  tier0?: {
    quality_score: number;
    grade: string;
    time_seconds: number;
  } | null;
  tier1: {
    quality_score: number;
    grade: string;
    perplexity: number;
    avg_loss: number;
    time_seconds: number;
  };
  total_time_seconds: number;
}

interface FusionResult {
  base_model_result: EvaluationResult;
  individual_results: EvaluationResult[];
  fused_result: EvaluationResult;
  fusion_info: {
    adapter_names: string[];
    method: string;
    weights: number[];
    output_name: string;
    timestamp: string;
  };
}

const FusionPage: React.FC = () => {
  const [adapterGroups, setAdapterGroups] = useState<AdaptersByBaseModel[]>([]);
  const [selectedAdapters, setSelectedAdapters] = useState<string[]>([]);
  const [selectedBaseModel, setSelectedBaseModel] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFusing, setIsFusing] = useState(false);
  const [fusionProgress, setFusionProgress] = useState(0);
  const [fusionStatus, setFusionStatus] = useState('');
  const [fusionResult, setFusionResult] = useState<FusionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Tier 0+1 Evaluation state
  const [tierEvaluations, setTierEvaluations] = useState<Map<string, TierEvaluationResult>>(new Map());
  const [isEvaluatingTiers, setIsEvaluatingTiers] = useState(false);
  const [evaluatingAdapter, setEvaluatingAdapter] = useState<string | null>(null);

  // Load available adapters on mount
  useEffect(() => {
    loadAdapters();
  }, []);

  // Poll fusion status when fusing
  useEffect(() => {
    if (isFusing) {
      const interval = setInterval(async () => {
        try {
          const response = await axios.get('http://localhost:8000/api/fusion/status');
          const status = response.data;
          
          setFusionProgress(status.progress);
          setFusionStatus(status.current_step);
          
          if (status.status === 'completed') {
            setIsFusing(false);
            // Fetch result
            const resultResponse = await axios.get('http://localhost:8000/api/fusion/result');
            setFusionResult(resultResponse.data);
          } else if (status.status === 'error') {
            setIsFusing(false);
            setError(status.error);
          }
        } catch (err) {
          console.error('Error polling fusion status:', err);
        }
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isFusing]);

  const loadAdapters = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<AdaptersByBaseModel[]>('http://localhost:8000/api/fusion/list-adapters');
      setAdapterGroups(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load adapters');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAdapterToggle = (adapterName: string, baseModel: string) => {
    if (selectedAdapters.includes(adapterName)) {
      // Deselect
      const newSelected = selectedAdapters.filter(a => a !== adapterName);
      setSelectedAdapters(newSelected);
      if (newSelected.length === 0) {
        setSelectedBaseModel(null);
      }
    } else {
      // Select
      if (selectedBaseModel === null) {
        // First selection - set base model
        setSelectedBaseModel(baseModel);
        setSelectedAdapters([adapterName]);
      } else if (selectedBaseModel === baseModel && selectedAdapters.length < 5) {
        // Add to selection (max 5)
        setSelectedAdapters([...selectedAdapters, adapterName]);
      }
    }
  };

  const handleStartFusion = async () => {
    if (selectedAdapters.length < 2) {
      setError('Please select at least 2 adapters');
      return;
    }

    setIsFusing(true);
    setError(null);
    setFusionResult(null);
    setFusionProgress(0);

    try {
      const method = selectedAdapters.length === 2 ? 'slerp' : 'weighted';
      await axios.post('http://localhost:8000/api/fusion/fuse', {
        adapter_names: selectedAdapters,
        method: method,
        weights: null, // Use default equal weights
        output_name: null // Auto-generate name
      });
    } catch (err: any) {
      setIsFusing(false);
      setError(err.response?.data?.detail || 'Failed to start fusion');
    }
  };

  const handleReset = () => {
    setSelectedAdapters([]);
    setSelectedBaseModel(null);
    setFusionResult(null);
    setError(null);
    setFusionProgress(0);
    setFusionStatus('');
    setTierEvaluations(new Map());
  };

  const handleEvaluateAdapter = async (adapterName: string) => {
    setIsEvaluatingTiers(true);
    setEvaluatingAdapter(adapterName);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/api/evaluate/adapter', {
        adapter_name: adapterName,
        max_samples: 20
      });
      
      if (response.data.success) {
        const result = response.data.result;
        setTierEvaluations(prev => {
          const newMap = new Map(prev);
          newMap.set(adapterName, {
            id: `tier-${Date.now()}-${adapterName}`,
            adapter_name: adapterName,
            is_base_model: false,
            tier0: result.tier0,
            tier1: result.tier1,
            total_time_seconds: result.total_time_seconds
          });
          return newMap;
        });
      }
    } catch (err: any) {
      setError(`Failed to evaluate ${adapterName}: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsEvaluatingTiers(false);
      setEvaluatingAdapter(null);
    }
  };

  const handleEvaluateSelected = async () => {
    if (selectedAdapters.length === 0) {
      setError('Please select adapters to evaluate');
      return;
    }
    
    setIsEvaluatingTiers(true);
    setError(null);
    
    // Evaluate base model first (Tier 1 only)
    try {
      setEvaluatingAdapter('base_model');
      const baseResponse = await axios.post('http://localhost:8000/api/evaluate/base-model', {
        max_samples: 20
      });
      
      if (baseResponse.data.success) {
        const result = baseResponse.data.result;
        setTierEvaluations(prev => {
          const newMap = new Map(prev);
          newMap.set('base_model', {
            id: `tier-${Date.now()}-base`,
            adapter_name: 'base_model',
            is_base_model: true,
            tier0: null,
            tier1: result.tier1,
            total_time_seconds: result.total_time_seconds
          });
          return newMap;
        });
      }
    } catch (err: any) {
      console.error('Base model evaluation error:', err);
      setError(`Base model evaluation failed: ${err.response?.data?.detail || err.message}`);
    }
    
    // Evaluate each selected adapter (Tier 0 + Tier 1)
    for (const adapterName of selectedAdapters) {
      try {
        setEvaluatingAdapter(adapterName);
        await handleEvaluateAdapter(adapterName);
      } catch (err) {
        console.error(`Error evaluating ${adapterName}:`, err);
      }
    }
    
    setIsEvaluatingTiers(false);
    setEvaluatingAdapter(null);
  };

  const getDeltaColor = (delta: number) => {
    if (delta > 0) return 'text-success-600 dark:text-success-400';
    if (delta < 0) return 'text-error-600 dark:text-error-400';
    return 'text-gray-500 dark:text-gray-400';
  };

  const getDeltaIcon = (delta: number) => {
    if (delta > 0) return <TrendingUp className="w-4 h-4" />;
    if (delta < 0) return <TrendingDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-success-600 dark:text-success-400';
    if (score >= 60) return 'text-warning-600 dark:text-warning-400';
    return 'text-error-600 dark:text-error-400';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return 'bg-success-100 dark:bg-success-900/20';
    if (score >= 60) return 'bg-warning-100 dark:bg-warning-900/20';
    return 'bg-error-100 dark:bg-error-900/20';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-3">
          <Sparkles className="w-8 h-8 text-primary-500" />
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Adapter Fusion
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Blend multiple adapters using SLERP interpolation to create hybrid models with combined capabilities
        </p>
      </div>

      {/* Info Banner */}
      <div className="card bg-primary-50 dark:bg-primary-900/20 border-primary-200 dark:border-primary-800">
        <div className="card-body">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-primary-600 dark:text-primary-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-primary-900 dark:text-primary-100">
              <p className="font-semibold mb-1">How Adapter Fusion Works:</p>
              <ul className="list-disc list-inside space-y-1 text-primary-800 dark:text-primary-200">
                <li>Select 2-5 adapters trained on the <strong>same base model</strong></li>
                <li>Uses <strong>SLERP</strong> (Spherical Linear Interpolation) for smooth blending</li>
                <li>Evaluates each adapter individually, then the fused result</li>
                <li>Compare all variants side-by-side to find the best combination</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Adapter Selection */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Select Adapters to Fuse</h2>
            <button
              onClick={loadAdapters}
              disabled={isLoading || isFusing}
              className="btn-secondary flex items-center space-x-2 text-sm"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
        <div className="card-body">
          {isLoading ? (
            <div className="text-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-3 text-primary-500" />
              <p className="text-gray-600 dark:text-gray-400">Loading adapters...</p>
            </div>
          ) : adapterGroups.length === 0 ? (
            <div className="text-center py-12">
              <AlertCircle className="w-12 h-12 mx-auto mb-3 text-gray-400" />
              <p className="text-gray-600 dark:text-gray-400">No adapters found</p>
              <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                Train some models first to use fusion
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {adapterGroups.map((group) => {
                const isGroupDisabled = selectedBaseModel !== null && selectedBaseModel !== group.base_model;
                const isGroupActive = selectedBaseModel === group.base_model;

                return (
                  <div
                    key={group.base_model}
                    className={`border-2 rounded-lg p-4 transition-all ${
                      isGroupActive
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/10'
                        : isGroupDisabled
                        ? 'border-gray-200 dark:border-gray-700 opacity-50'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                          {group.base_model}
                        </h3>
                        {isGroupActive && (
                          <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-primary-500 text-white">
                            Active
                          </span>
                        )}
                        {isGroupDisabled && (
                          <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-gray-400 text-white">
                            Disabled
                          </span>
                        )}
                      </div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {group.adapters.length} adapter{group.adapters.length !== 1 ? 's' : ''}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {group.adapters.map((adapter) => {
                        const isSelected = selectedAdapters.includes(adapter.name);
                        const isDisabled = isGroupDisabled || (selectedAdapters.length >= 5 && !isSelected);

                        return (
                          <label
                            key={adapter.name}
                            className={`flex items-start space-x-3 p-3 rounded-lg border-2 cursor-pointer transition-all ${
                              isSelected
                                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                                : isDisabled
                                ? 'border-gray-200 dark:border-gray-700 opacity-50 cursor-not-allowed'
                                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => handleAdapterToggle(adapter.name, group.base_model)}
                              disabled={isDisabled || isFusing}
                              className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                            />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between">
                              <div className="font-medium text-gray-900 dark:text-gray-100 truncate">
                                {adapter.name}
                                </div>
                                {tierEvaluations.has(adapter.name) && (
                                  <div className="flex items-center space-x-1 ml-2">
                                    <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                                      tierEvaluations.get(adapter.name)!.tier1.quality_score >= 80 ? 'bg-success-100 dark:bg-success-900/30 text-success-700 dark:text-success-300' :
                                      tierEvaluations.get(adapter.name)!.tier1.quality_score >= 60 ? 'bg-warning-100 dark:bg-warning-900/30 text-warning-700 dark:text-warning-300' :
                                      'bg-error-100 dark:bg-error-900/30 text-error-700 dark:text-error-300'
                                    }`}>
                                      T1: {tierEvaluations.get(adapter.name)!.tier1.quality_score.toFixed(0)}
                                    </span>
                                    {tierEvaluations.get(adapter.name)!.tier0 && (
                                      <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                                        T0: {tierEvaluations.get(adapter.name)!.tier0!.quality_score.toFixed(0)}
                                      </span>
                                    )}
                                  </div>
                                )}
                              </div>
                              <div className="text-xs text-gray-500 dark:text-gray-400 space-y-0.5 mt-1">
                                {adapter.iterations && (
                                  <div>{adapter.iterations} iterations</div>
                                )}
                                {adapter.training_date && (
                                  <div>{new Date(adapter.training_date).toLocaleDateString()}</div>
                                )}
                                {adapter.dataset && (
                                  <div className="truncate" title={adapter.dataset}>
                                    {adapter.dataset}
                                  </div>
                                )}
                              </div>
                              {!tierEvaluations.has(adapter.name) && !isSelected && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleEvaluateAdapter(adapter.name);
                                  }}
                                  disabled={isEvaluatingTiers || evaluatingAdapter === adapter.name}
                                  className="mt-2 text-xs btn-secondary py-1 px-2"
                                  title="Quick evaluate this adapter"
                                >
                                  {evaluatingAdapter === adapter.name ? (
                                    <span className="flex items-center space-x-1">
                                      <RefreshCw className="w-3 h-3 animate-spin" />
                                      <span>Evaluating...</span>
                                    </span>
                                  ) : (
                                    'Evaluate'
                                  )}
                                </button>
                              )}
                            </div>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Selection Summary */}
          {selectedAdapters.length > 0 && (
            <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-gray-900 dark:text-gray-100">
                    {selectedAdapters.length} adapter{selectedAdapters.length !== 1 ? 's' : ''} selected
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Base Model: {selectedBaseModel}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    Method: {selectedAdapters.length === 2 ? 'SLERP (Spherical Interpolation)' : 'Weighted Average'}
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleReset}
                    disabled={isFusing || isEvaluatingTiers}
                    className="btn-secondary text-sm"
                  >
                    Reset
                  </button>
                  <button
                    onClick={handleEvaluateSelected}
                    disabled={selectedAdapters.length === 0 || isEvaluatingTiers || isFusing}
                    className="btn-secondary flex items-center space-x-2"
                    title="Evaluate base model and selected adapters using Tier 0+1 (Fast & Deterministic)"
                  >
                    {isEvaluatingTiers ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Evaluating...</span>
                      </>
                    ) : (
                      <>
                        <BarChart3 className="w-4 h-4" />
                        <span>Evaluate Base + Selected</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleStartFusion}
                    disabled={selectedAdapters.length < 2 || isFusing || isEvaluatingTiers}
                    className="btn-primary flex items-center space-x-2"
                  >
                    <GitMerge className="w-4 h-4" />
                    <span>Start Fusion & Evaluation</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tier Evaluation Results */}
      {tierEvaluations.size > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="text-xl font-semibold">Tier 0+1 Evaluation Results</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Separate Tier 0 (Mathematical) and Tier 1 (Perplexity) scores for base model and adapters
            </p>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from(tierEvaluations.values()).map((evaluation) => (
                <EvaluationCard key={evaluation.id} result={evaluation} compact={true} />
              ))}
            </div>
            
            {/* Comparison Summary */}
            {(() => {
              const baseModelEval = tierEvaluations.get('base_model');
              const adapterEvals = selectedAdapters
                .map(name => tierEvaluations.get(name))
                .filter(Boolean);
              
              if (adapterEvals.length >= 1 || baseModelEval) {
                const allEvals: TierEvaluationResult[] = [
                  ...(baseModelEval ? [baseModelEval] : []),
                  ...adapterEvals.filter((e): e is TierEvaluationResult => e !== undefined)
                ].sort((a, b) => b.tier1.quality_score - a.tier1.quality_score);
                
                return (
                  <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                      Comparison (by Tier 1 Score)
                    </h3>
                    <div className="space-y-2">
                      {allEvals.map((evaluation, idx) => {
                        const isBaseModel = evaluation.is_base_model;
                        const isBest = idx === 0;
                        const improvement = baseModelEval && !isBaseModel 
                          ? evaluation.tier1.quality_score - baseModelEval.tier1.quality_score
                          : null;
                        
                        return (
                          <div key={evaluation.id} className={`flex items-center justify-between p-2 rounded ${
                            isBaseModel ? 'bg-blue-50 dark:bg-blue-900/20' : 'bg-white dark:bg-gray-700'
                          }`}>
                            <div className="flex items-center space-x-2">
                              {isBest && <span className="text-success-600 dark:text-success-400 font-bold">🏆</span>}
                              <span className={`font-medium ${isBaseModel ? 'text-blue-700 dark:text-blue-300' : ''}`}>
                                {isBaseModel ? 'Base Model' : evaluation.adapter_name}
                              </span>
                              {isBaseModel && <span className="text-xs px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">BASE</span>}
                            </div>
                            <div className="flex items-center space-x-3">
                              <span className="text-sm text-gray-600 dark:text-gray-400">
                                T1: <span className="font-semibold">{evaluation.tier1.quality_score.toFixed(1)}</span>
                              </span>
                              {evaluation.tier0 && (
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  T0: <span className="font-semibold">{evaluation.tier0.quality_score.toFixed(1)}</span>
                                </span>
                              )}
                              {improvement !== null && (
                                <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                                  improvement > 0 
                                    ? 'bg-success-100 dark:bg-success-900/30 text-success-700 dark:text-success-300'
                                    : improvement < 0
                                    ? 'bg-error-100 dark:bg-error-900/30 text-error-700 dark:text-error-300'
                                    : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                                }`}>
                                  {improvement > 0 ? '+' : ''}{improvement.toFixed(1)} vs base
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              }
              return null;
            })()}
          </div>
        </div>
      )}

      {/* Fusion Progress */}
      {isFusing && (
        <div className="card">
          <div className="card-body">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <RefreshCw className="w-5 h-5 animate-spin text-primary-500" />
                  <span className="font-semibold text-gray-900 dark:text-gray-100">
                    {fusionStatus || 'Processing...'}
                  </span>
                </div>
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {fusionProgress}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-primary-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${fusionProgress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="card border-error-200 dark:border-error-800 bg-error-50 dark:bg-error-900/20">
          <div className="card-body">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-error-600 dark:text-error-400 flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold text-error-900 dark:text-error-100">Error</div>
                <div className="text-sm text-error-800 dark:text-error-200 mt-1">{error}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Table */}
      {fusionResult && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <CheckCircle2 className="w-6 h-6 text-success-500" />
                <h2 className="text-xl font-semibold">Fusion Results</h2>
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {new Date(fusionResult.fusion_info.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
          <div className="card-body">
            {/* Fusion Info */}
            <div className="mb-6 p-4 bg-gradient-to-r from-primary-50 to-purple-50 dark:from-primary-900/20 dark:to-purple-900/20 rounded-lg border border-primary-200 dark:border-primary-800">
              <div className="flex items-start space-x-3">
                <Zap className="w-5 h-5 text-primary-600 dark:text-primary-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="font-semibold text-primary-900 dark:text-primary-100 mb-2">
                    Fused Adapter: {fusionResult.fusion_info.output_name}
                  </div>
                  <div className="text-sm text-primary-800 dark:text-primary-200 space-y-1">
                    <div>Method: <span className="font-medium">{fusionResult.fusion_info.method.toUpperCase()}</span></div>
                    <div>Source Adapters: <span className="font-medium">{fusionResult.fusion_info.adapter_names.join(', ')}</span></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-gray-300 dark:border-gray-600">
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">
                      Metric
                    </th>
                    <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800">
                      Base Model
                    </th>
                    {fusionResult.individual_results.map((result) => (
                      <th
                        key={result.adapter_name}
                        className="px-4 py-3 text-center text-sm font-semibold text-gray-700 dark:text-gray-300"
                      >
                        <div className="truncate max-w-[120px]" title={result.adapter_name}>
                          {result.adapter_name}
                        </div>
                      </th>
                    ))}
                    <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700 dark:text-gray-300 bg-gradient-to-r from-primary-100 to-purple-100 dark:from-primary-900/30 dark:to-purple-900/30">
                      <div className="flex items-center justify-center space-x-1">
                        <Sparkles className="w-4 h-4" />
                        <span>Fused</span>
                      </div>
                    </th>
                    <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700 dark:text-gray-300">
                      Best Δ
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {[
                    { label: 'Overall Score', key: 'overall_score' as const },
                    { label: 'Faithfulness', key: 'faithfulness' as const },
                    { label: 'Fact Recall', key: 'fact_recall' as const },
                    { label: 'Consistency', key: 'consistency' as const },
                  ].map((metric, idx) => {
                    const baseScore = fusionResult.base_model_result[metric.key];
                    const individualScores = fusionResult.individual_results.map(r => r[metric.key]);
                    const fusedScore = fusionResult.fused_result[metric.key];
                    const maxIndividualScore = Math.max(...individualScores);
                    const fusedDelta = fusedScore - baseScore;
                    const bestDelta = Math.max(fusedDelta, ...individualScores.map(s => s - baseScore));

                    return (
                      <tr
                        key={metric.key}
                        className={idx === 0 ? 'bg-gray-50 dark:bg-gray-800/50' : ''}
                      >
                        <td className={`px-4 py-3 ${idx === 0 ? 'font-semibold' : 'font-medium'} text-gray-900 dark:text-gray-100`}>
                          {metric.label}
                        </td>
                        <td className="px-4 py-3 text-center bg-gray-50 dark:bg-gray-800/50">
                          <span className="text-gray-600 dark:text-gray-400">
                            {baseScore}/100
                          </span>
                        </td>
                        {fusionResult.individual_results.map((result) => {
                          const score = result[metric.key];
                          const delta = score - baseScore;
                          return (
                            <td key={result.adapter_name} className="px-4 py-3 text-center">
                              <div className="flex flex-col items-center space-y-1">
                                <span className={`font-medium ${getScoreColor(score)}`}>
                                  {score}/100
                                </span>
                                <div className={`flex items-center space-x-1 text-xs ${getDeltaColor(delta)}`}>
                                  {getDeltaIcon(delta)}
                                  <span>{delta > 0 ? '+' : ''}{delta.toFixed(1)}</span>
                                </div>
                              </div>
                            </td>
                          );
                        })}
                        <td className="px-4 py-3 text-center bg-gradient-to-r from-primary-50 to-purple-50 dark:from-primary-900/20 dark:to-purple-900/20">
                          <div className="flex flex-col items-center space-y-1">
                            <span className={`font-bold text-lg ${getScoreColor(fusedScore)}`}>
                              {fusedScore}/100
                            </span>
                            <div className={`flex items-center space-x-1 text-sm font-semibold ${getDeltaColor(fusedDelta)}`}>
                              {getDeltaIcon(fusedDelta)}
                              <span>{fusedDelta > 0 ? '+' : ''}{fusedDelta.toFixed(1)}</span>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <div className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-sm font-semibold ${
                            bestDelta > 0 ? 'bg-success-100 dark:bg-success-900/30 text-success-700 dark:text-success-300' :
                            bestDelta < 0 ? 'bg-error-100 dark:bg-error-900/30 text-error-700 dark:text-error-300' :
                            'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                          }`}>
                            {getDeltaIcon(bestDelta)}
                            <span>{bestDelta > 0 ? '+' : ''}{bestDelta.toFixed(1)}</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Summary Stats */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Base Model Average</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {((fusionResult.base_model_result.overall_score +
                     fusionResult.base_model_result.faithfulness +
                     fusionResult.base_model_result.fact_recall +
                     fusionResult.base_model_result.consistency) / 4).toFixed(1)}
                </div>
              </div>
              <div className="p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
                <div className="text-sm text-primary-600 dark:text-primary-400 mb-1">Best Individual</div>
                <div className="text-2xl font-bold text-primary-900 dark:text-primary-100">
                  {Math.max(...fusionResult.individual_results.map(r =>
                    (r.overall_score + r.faithfulness + r.fact_recall + r.consistency) / 4
                  )).toFixed(1)}
                </div>
              </div>
              <div className="p-4 bg-gradient-to-r from-primary-100 to-purple-100 dark:from-primary-900/30 dark:to-purple-900/30 rounded-lg">
                <div className="text-sm text-primary-700 dark:text-primary-300 mb-1 font-semibold">Fused Model Average</div>
                <div className="text-2xl font-bold text-primary-900 dark:text-primary-100">
                  {((fusionResult.fused_result.overall_score +
                     fusionResult.fused_result.faithfulness +
                     fusionResult.fused_result.fact_recall +
                     fusionResult.fused_result.consistency) / 4).toFixed(1)}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FusionPage;
