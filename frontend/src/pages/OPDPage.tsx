import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { Beaker, Loader2 } from 'lucide-react';
import { OPDSetup } from '../components/OPDSetup';
import { OPDProgress } from '../components/OPDProgress';
import { OPDResults } from '../components/OPDResults';
import { setRuns } from '../store/slices/opdSlice';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:8000';

export const OPDPage: React.FC = () => {
  const dispatch = useDispatch();
  const { state: opdState, metrics, error } = useSelector((state: RootState) => state.opd);

  // Load runs on mount
  useEffect(() => {
    const loadRuns = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/opd/runs`);
        if (response.data.runs) {
          dispatch(setRuns(response.data.runs));
        }
      } catch (err) {
        console.error('Failed to load OPD runs:', err);
      }
    };

    loadRuns();
  }, [dispatch]);

  // Poll status when running
  useEffect(() => {
    if (opdState !== 'running') return;

    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/opd/status`);
        // Status updates handled by WebSocket, this is just a fallback
      } catch (err) {
        console.error('Failed to poll OPD status:', err);
      }
    }, 3000);

    return () => clearInterval(pollInterval);
  }, [opdState]);

  // Render view based on state
  const renderView = () => {
    switch (opdState) {
      case 'idle':
        return <OPDSetup />;

      case 'running':
        return <OPDProgress />;

      case 'completed':
      case 'error':
      case 'stopped':
        return <OPDResults />;

      default:
        return <OPDSetup />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <Beaker className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                On-Policy Distillation
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Knowledge distillation from larger teacher models
              </p>
            </div>
          </div>
        </div>

        {/* Status Badge */}
        <div className="flex items-center space-x-2">
          {opdState === 'running' && (
            <div className="flex items-center space-x-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <Loader2 className="h-4 w-4 text-blue-600 dark:text-blue-400 animate-spin" />
              <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                Distillation Running
              </span>
            </div>
          )}
          {opdState === 'completed' && (
            <div className="px-4 py-2 bg-success-100 dark:bg-success-900/30 rounded-lg">
              <span className="text-sm font-medium text-success-900 dark:text-success-100">
                Completed
              </span>
            </div>
          )}
          {opdState === 'error' && (
            <div className="px-4 py-2 bg-error-100 dark:bg-error-900/30 rounded-lg">
              <span className="text-sm font-medium text-error-900 dark:text-error-100">
                Error
              </span>
            </div>
          )}
          {opdState === 'stopped' && (
            <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Stopped
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-lg p-4">
          <p className="text-sm text-error-900 dark:text-error-100">
            <strong>Error:</strong> {error}
          </p>
        </div>
      )}

      {/* Main Content */}
      {renderView()}
    </div>
  );
};
