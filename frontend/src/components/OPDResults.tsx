import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { CheckCircle, XCircle, StopCircle, RefreshCw, Download } from 'lucide-react';
import { resetOPD } from '../store/slices/opdSlice';

export const OPDResults: React.FC = () => {
  const dispatch = useDispatch();
  const { state, metrics, currentRunId, error } = useSelector((state: RootState) => state.opd);

  const handleNewDistillation = () => {
    dispatch(resetOPD());
  };

  const getStatusIcon = () => {
    switch (state) {
      case 'completed':
        return <CheckCircle className="h-12 w-12 text-success-600 dark:text-success-400" />;
      case 'error':
        return <XCircle className="h-12 w-12 text-error-600 dark:text-error-400" />;
      case 'stopped':
        return <StopCircle className="h-12 w-12 text-gray-600 dark:text-gray-400" />;
      default:
        return null;
    }
  };

  const getStatusMessage = () => {
    switch (state) {
      case 'completed':
        return 'Distillation completed successfully!';
      case 'error':
        return 'Distillation failed with an error.';
      case 'stopped':
        return 'Distillation was stopped early.';
      default:
        return 'Distillation finished.';
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <div className="card p-8">
        <div className="flex flex-col items-center text-center space-y-4">
          {getStatusIcon()}

          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
              {getStatusMessage()}
            </h2>
            {error && (
              <p className="text-sm text-error-600 dark:text-error-400">
                {error}
              </p>
            )}
          </div>

          {/* Metrics Summary */}
          {metrics && (
            <div className="grid grid-cols-2 gap-6 w-full max-w-md mt-6">
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Final KL Loss</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {metrics.kl_loss !== null ? metrics.kl_loss.toFixed(4) : '--'}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Token Agreement</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {metrics.token_agreement_pct !== null ? `${metrics.token_agreement_pct.toFixed(1)}%` : '--'}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Steps Completed</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {metrics.step} / {metrics.total_steps}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Progress</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {metrics.progress_pct.toFixed(1)}%
                </p>
              </div>
            </div>
          )}

          {/* Run Info */}
          {currentRunId && (
            <div className="w-full max-w-md mt-6 pt-6 border-t border-gray-200 dark:border-gray-700 text-sm text-gray-600 dark:text-gray-400">
              <p>Run ID: <span className="font-mono text-gray-900 dark:text-gray-100">{currentRunId}</span></p>
              {metrics?.started_at && (
                <p className="mt-1">Started: <span className="text-gray-900 dark:text-gray-100">{new Date(metrics.started_at).toLocaleString()}</span></p>
              )}
              {metrics?.completed_at && (
                <p className="mt-1">Completed: <span className="text-gray-900 dark:text-gray-100">{new Date(metrics.completed_at).toLocaleString()}</span></p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex space-x-3 mt-6">
            <button
              onClick={handleNewDistillation}
              className="btn-primary flex items-center space-x-2"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Start New Distillation</span>
            </button>
          </div>
        </div>
      </div>

      {/* Next Steps Card */}
      {state === 'completed' && (
        <div className="card p-6 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800">
          <h3 className="text-lg font-semibold text-success-900 dark:text-success-100 mb-3">
            Next Steps
          </h3>
          <ul className="space-y-2 text-sm text-success-800 dark:text-success-200">
            <li className="flex items-start space-x-2">
              <span className="text-success-600 dark:text-success-400">•</span>
              <span>Your distilled model adapter has been saved to the output directory</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-success-600 dark:text-success-400">•</span>
              <span>Test the distilled model on the Compare page to verify quality improvements</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-success-600 dark:text-success-400">•</span>
              <span>Compare inference speed: distilled 7B model vs original 32B teacher</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-success-600 dark:text-success-400">•</span>
              <span>Review metrics history to understand the knowledge transfer process</span>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
};
