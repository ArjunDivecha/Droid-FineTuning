import React from 'react';
import { AlertCircle, X } from 'lucide-react';

interface TrainingErrorDisplayProps {
  error: string | null;
  onDismiss?: () => void;
}

const TrainingErrorDisplay: React.FC<TrainingErrorDisplayProps> = ({ error, onDismiss }) => {
  if (!error) return null;

  return (
    <div className="card border-error-200 dark:border-error-800 bg-error-50 dark:bg-error-900/20">
      <div className="card-body">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-6 h-6 text-error-600 dark:text-error-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-error-900 dark:text-error-100 text-lg">
                Training Error
              </h3>
              {onDismiss && (
                <button
                  onClick={onDismiss}
                  className="text-error-600 hover:text-error-800 dark:hover:text-error-300"
                  aria-label="Dismiss error"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
            <div className="mt-2 text-sm text-error-800 dark:text-error-200">
              <pre className="whitespace-pre-wrap font-mono text-xs bg-error-100 dark:bg-error-900/40 p-3 rounded mt-2 overflow-x-auto">
                {error}
              </pre>
            </div>
            <p className="mt-3 text-xs text-error-700 dark:text-error-300">
              Check the training debug log for more details or try adjusting your configuration.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingErrorDisplay;
