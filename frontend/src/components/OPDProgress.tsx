import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { Square, TrendingDown, Percent, Clock } from 'lucide-react';
import { opdStopped, setError } from '../store/slices/opdSlice';
import { addNotification } from '../store/slices/uiSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';

export const OPDProgress: React.FC = () => {
  const dispatch = useDispatch();
  const { metrics, currentRunId, estimatedDuration } = useSelector((state: RootState) => state.opd);

  const handleStopDistillation = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/opd/stop`);
      dispatch(opdStopped(response.data));
      dispatch(addNotification({
        type: 'info',
        title: 'Distillation Stopped',
        message: 'Distillation training has been stopped',
        autoHide: true,
      }));
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to stop distillation';
      dispatch(setError(errorMsg));
      dispatch(addNotification({
        type: 'error',
        title: 'Stop Failed',
        message: errorMsg,
      }));
    }
  };

  const progress = metrics?.progress_pct || 0;

  return (
    <div className="space-y-6">
      {/* Progress Card */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Training Progress
          </h2>
          <button
            onClick={handleStopDistillation}
            className="btn-secondary flex items-center space-x-2 text-error-600 dark:text-error-400 hover:bg-error-50 dark:hover:bg-error-900/20"
          >
            <Square className="h-4 w-4" />
            <span>Stop</span>
          </button>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>Step {metrics?.step || 0} of {metrics?.total_steps || 0}</span>
            <span>{progress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className="bg-primary-600 h-full transition-all duration-300 rounded-full"
              style={{ width: `${Math.min(100, progress)}%` }}
            />
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-4">
          {/* KL Loss */}
          <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingDown className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                KL Loss
              </span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {metrics?.kl_loss !== null && metrics?.kl_loss !== undefined
                ? metrics.kl_loss.toFixed(4)
                : '--'}
            </div>
          </div>

          {/* Token Agreement */}
          <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Percent className="h-4 w-4 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Token Agreement
              </span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {metrics?.token_agreement_pct !== null && metrics?.token_agreement_pct !== undefined
                ? `${metrics.token_agreement_pct.toFixed(1)}%`
                : '--'}
            </div>
          </div>

          {/* Estimated Time */}
          <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Clock className="h-4 w-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Est. Duration
              </span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {estimatedDuration ? `${estimatedDuration}m` : '--'}
            </div>
          </div>
        </div>

        {/* Run Info */}
        <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            <p>Run ID: <span className="font-mono text-gray-900 dark:text-gray-100">{currentRunId}</span></p>
            <p className="mt-1">Started: <span className="text-gray-900 dark:text-gray-100">{metrics?.started_at ? new Date(metrics.started_at).toLocaleString() : '--'}</span></p>
          </div>
        </div>
      </div>

      {/* Info Card */}
      <div className="card p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>What's happening:</strong> The student model is learning from the teacher's token-level predictions.
          KL loss should decrease over time, and token agreement should increase, indicating better alignment with the teacher.
        </p>
      </div>
    </div>
  );
};
