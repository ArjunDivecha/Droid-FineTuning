import React from 'react';
import { CheckCircle, AlertCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface EvaluationResult {
  adapter_name?: string;
  model_name?: string;
  is_base_model?: boolean;
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
  base_model_comparison?: {
    base_perplexity: number;
    adapter_perplexity: number;
    perplexity_reduction_pct: number;
    quality_improvement: number;
  };
}

interface EvaluationCardProps {
  result: EvaluationResult;
  compact?: boolean;
}

export const EvaluationCard: React.FC<EvaluationCardProps> = ({ result, compact = false }) => {
  const name = result.adapter_name || result.model_name || 'Unknown';
  const isBaseModel = result.is_base_model || false;
  const tier0Score = result.tier0?.quality_score;
  const tier1Score = result.tier1.quality_score;
  
  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return 'text-green-600 dark:text-green-400';
      case 'B': return 'text-blue-600 dark:text-blue-400';
      case 'C': return 'text-yellow-600 dark:text-yellow-400';
      case 'D': return 'text-orange-600 dark:text-orange-400';
      case 'F': return 'text-red-600 dark:text-red-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getScoreBarWidth = (score: number) => {
    return Math.max(0, Math.min(100, score));
  };

  if (compact) {
    return (
      <div className="card">
        <div className="card-body p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">
              {isBaseModel ? 'Base Model' : name}
            </h3>
            <span className={`text-lg font-bold ${getGradeColor(result.tier1.grade)}`}>
              {result.tier1.grade}
            </span>
          </div>
          
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600 dark:text-gray-400">Tier 1 (Perplexity)</span>
                <span className="font-medium">{tier1Score.toFixed(1)}/100</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-primary-500 h-2 rounded-full transition-all"
                  style={{ width: `${getScoreBarWidth(tier1Score)}%` }}
                />
              </div>
            </div>
            
            {tier0Score !== undefined && tier0Score !== null && (
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600 dark:text-gray-400">Tier 0 (Mathematical)</span>
                  <span className="font-medium">{tier0Score.toFixed(1)}/100</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all"
                    style={{ width: `${getScoreBarWidth(tier0Score)}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold">
          {isBaseModel ? 'Base Model Evaluation' : `Adapter: ${name}`}
        </h3>
      </div>
      <div className="card-body">
        {/* Tier 1 Score */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                Tier 1 Score (Perplexity Analysis)
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <span className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {tier1Score.toFixed(1)}
              </span>
              <span className={`text-2xl font-bold ${getGradeColor(result.tier1.grade)}`}>
                {result.tier1.grade}
              </span>
            </div>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2">
            <div
              className="bg-primary-500 h-3 rounded-full transition-all"
              style={{ width: `${getScoreBarWidth(tier1Score)}%` }}
            />
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Perplexity: {result.tier1.perplexity.toFixed(2)} | 
            Avg Loss: {result.tier1.avg_loss.toFixed(4)} | 
            Time: {result.tier1.time_seconds.toFixed(1)}s
          </div>
        </div>

        {/* Tier 0 Score (if available) */}
        {tier0Score !== undefined && tier0Score !== null ? (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-gray-900 dark:text-gray-100">
                  Tier 0 Score (Mathematical Analysis)
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {tier0Score.toFixed(1)}
                </span>
                <span className={`text-2xl font-bold ${getGradeColor(result.tier0!.grade)}`}>
                  {result.tier0!.grade}
                </span>
              </div>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2">
              <div
                className="bg-purple-500 h-3 rounded-full transition-all"
                style={{ width: `${getScoreBarWidth(tier0Score)}%` }}
              />
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Time: {result.tier0!.time_seconds.toFixed(1)}s
            </div>
          </div>
        ) : (
          <div className="mb-6 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Tier 0 Score: N/A (Base models cannot be evaluated with Tier 0)
            </div>
          </div>
        )}

        {/* Base Model Comparison */}
        {result.base_model_comparison && (
          <div className="mt-4 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              <span className="font-semibold text-primary-900 dark:text-primary-100">
                vs Base Model
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Perplexity Reduction:</span>
                <span className="font-medium text-green-600 dark:text-green-400">
                  {result.base_model_comparison.perplexity_reduction_pct.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Quality Improvement:</span>
                <span className="font-medium text-green-600 dark:text-green-400">
                  +{result.base_model_comparison.quality_improvement.toFixed(1)} points
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Total Evaluation Time: {result.total_time_seconds.toFixed(1)}s
          </div>
        </div>
      </div>
    </div>
  );
};











