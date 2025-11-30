import React, { useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { 
  Play, 
  Pause, 
  Square, 
  Activity, 
  Clock, 
  TrendingDown, 
  Cpu,
  Eye,
  EyeOff
} from 'lucide-react';
import { RootState } from '../store/store';
import { setShowLogs, addNotification } from '../store/slices/uiSlice';
import { resetTraining, clearLogs } from '../store/slices/trainingSlice';
import { TrainingChart } from '../components/TrainingChart';
import { LogViewer } from '../components/LogViewer';
import TrainingErrorDisplay from '../components/TrainingErrorDisplay';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';

export const TrainingPage: React.FC = () => {
  const dispatch = useDispatch();
  const { state: trainingState, metrics, config, logs } = useSelector((state: RootState) => state.training);
  const { showLogs } = useSelector((state: RootState) => state.ui);

  // DO NOT clear data on mount - preserve logs and graphs when navigating between tabs
  // Data is only cleared when a NEW training session starts (handled in useWebSocket hook)

  const handleStopTraining = async () => {
    try {
      await axios.post(`${BACKEND_URL}/training/stop`);
      dispatch(addNotification({
        type: 'info',
        title: 'Training Stopped',
        message: 'Training process is being terminated...',
        autoHide: true,
      }));
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        title: 'Stop Failed',
        message: 'Failed to stop training process.',
      }));
    }
  };

  const formatTime = (seconds: number | null) => {
    if (!seconds) return '--:--:--';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getProgress = () => {
    if (!metrics) return 0;
    return (metrics.current_step / metrics.total_steps) * 100;
  };

  const toggleLogs = () => {
    dispatch(setShowLogs(!showLogs));
  };

  // Only show "No Training Session" if idle AND no metrics/logs
  if (trainingState === 'idle' && !metrics && logs.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
            <Play className="h-8 w-8 text-gray-400" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            No Training Session
          </h2>
          <p className="text-gray-500 dark:text-gray-400">
            Configure your model and start training from the Setup page.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Training Monitor
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Real-time training progress and metrics
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={toggleLogs}
            className="btn-secondary flex items-center space-x-2"
          >
            {showLogs ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            <span>{showLogs ? 'Hide' : 'Show'} Logs</span>
          </button>

          {trainingState === 'running' && (
            <button
              onClick={handleStopTraining}
              className="btn-danger flex items-center space-x-2"
            >
              <Square className="h-4 w-4" />
              <span>Stop Training</span>
            </button>
          )}
        </div>
      </div>

      {/* Training Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Status Card */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Status</p>
                <p className={`text-lg font-semibold capitalize ${
                  trainingState === 'running' 
                    ? 'text-success-600' 
                    : trainingState === 'completed'
                    ? 'text-success-600'
                    : trainingState === 'error'
                    ? 'text-error-600'
                    : 'text-gray-600'
                }`}>
                  {trainingState}
                </p>
              </div>
              <div className={`p-3 rounded-full ${
                trainingState === 'running' 
                  ? 'bg-success-100 dark:bg-success-900/30' 
                  : trainingState === 'completed'
                  ? 'bg-success-100 dark:bg-success-900/30'
                  : trainingState === 'error'
                  ? 'bg-error-100 dark:bg-error-900/30'
                  : 'bg-gray-100 dark:bg-gray-800'
              }`}>
                <Activity className={`h-6 w-6 ${
                  trainingState === 'running' 
                    ? 'text-success-600' 
                    : trainingState === 'completed'
                    ? 'text-success-600'
                    : trainingState === 'error'
                    ? 'text-error-600'
                    : 'text-gray-400'
                }`} />
              </div>
            </div>
          </div>
        </div>

        {/* Progress Card */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Progress</p>
              <Cpu className="h-5 w-5 text-primary-600" />
            </div>
            {metrics ? (
              <div className="space-y-2">
                <p className="text-lg font-semibold">
                  {getProgress().toFixed(1)}%
                </p>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${getProgress()}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {metrics.current_step.toLocaleString()} / {metrics.total_steps.toLocaleString()} steps
                </p>
              </div>
            ) : (
              <p className="text-lg font-semibold text-gray-400">--</p>
            )}
          </div>
        </div>

        {/* Time Remaining Card */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Time Remaining</p>
              <Clock className="h-5 w-5 text-primary-600" />
            </div>
            <p className="text-lg font-semibold">
              {formatTime(metrics?.estimated_time_remaining || null)}
            </p>
          </div>
        </div>

        {/* Current Loss Card */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Train Loss</p>
              <TrendingDown className="h-5 w-5 text-primary-600" />
            </div>
            <p className="text-lg font-semibold">
              {metrics?.train_loss != null ? metrics.train_loss.toFixed(4) : '--'}
            </p>
            {metrics?.val_loss != null && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Val: {metrics.val_loss.toFixed(4)}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* RL Metrics (GSPO/GRPO) */}
      {(metrics?.avg_reward != null || metrics?.success_rate != null || metrics?.kl != null || metrics?.entropy != null) && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Avg Reward */}
          <div className="card">
            <div className="card-body">
              <p className="text-sm text-gray-500 dark:text-gray-400">Avg Reward</p>
              <p className="text-lg font-semibold">{metrics?.avg_reward != null ? Number(metrics.avg_reward).toFixed(4) : '--'}</p>
            </div>
          </div>
          {/* Success Rate */}
          <div className="card">
            <div className="card-body">
              <p className="text-sm text-gray-500 dark:text-gray-400">Success Rate</p>
              <p className="text-lg font-semibold">
                {metrics?.success_rate != null
                  ? `${Number(metrics.success_rate > 1 ? metrics.success_rate : metrics.success_rate * 100).toFixed(1)}%`
                  : '--'}
              </p>
            </div>
          </div>
          {/* KL */}
          <div className="card">
            <div className="card-body">
              <p className="text-sm text-gray-500 dark:text-gray-400">KL</p>
              <p className="text-lg font-semibold">{metrics?.kl != null ? Number(metrics.kl).toFixed(4) : '--'}</p>
            </div>
          </div>
          {/* Entropy */}
          <div className="card">
            <div className="card-body">
              <p className="text-sm text-gray-500 dark:text-gray-400">Entropy</p>
              <p className="text-lg font-semibold">{metrics?.entropy != null ? Number(metrics.entropy).toFixed(4) : '--'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Training Configuration */}
      {config && (
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Training Configuration</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
              {config.training_method && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Method:</span>
                  <p className="font-medium">{config.training_method.toUpperCase()}</p>
                </div>
              )}
              <div>
                <span className="text-gray-500 dark:text-gray-400">Learning Rate:</span>
                <p className="font-medium">{config.learning_rate}</p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Batch Size:</span>
                <p className="font-medium">{config.batch_size}</p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Max Seq Length:</span>
                <p className="font-medium">{config.max_seq_length}</p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Adapter Name:</span>
                <p className="font-medium">{config.adapter_name}</p>
              </div>
              {/* RL Params */}
              {config.group_size != null && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Group Size:</span>
                  <p className="font-medium">{config.group_size}</p>
                </div>
              )}
              {config.temperature != null && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Temperature:</span>
                  <p className="font-medium">{config.temperature}</p>
                </div>
              )}
              {config.epsilon != null && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Epsilon:</span>
                  <p className="font-medium">{config.epsilon}</p>
                </div>
              )}
              {config.max_completion_length != null && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Max Completion Len:</span>
                  <p className="font-medium">{config.max_completion_length}</p>
                </div>
              )}
              {config.importance_sampling_level && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Importance Sampling:</span>
                  <p className="font-medium">{config.importance_sampling_level}</p>
                </div>
              )}
              {config.grpo_loss_type && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">GRPO Loss Type:</span>
                  <p className="font-medium">{config.grpo_loss_type}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Training Chart and Logs */}
      <div className={`${showLogs ? 'grid grid-cols-1 lg:grid-cols-2' : 'grid grid-cols-1'} gap-6`}>
        {/* Training Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Training Metrics</h3>
          </div>
          <div className="card-body">
            <TrainingChart />
          </div>
        </div>

        {/* Training Logs */}
        {showLogs && (
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-semibold">Training Logs</h3>
            </div>
            <div className="card-body p-0">
              <LogViewer logs={logs} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
