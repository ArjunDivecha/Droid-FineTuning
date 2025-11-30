import React from 'react';
import { X, Download, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';

interface ExportFormat {
  id: string;
  name: string;
  description: string;
  extension: string;
  size: string;
  requires?: string;
}

interface ExportModalProps {
  isOpen: boolean;
  adapterName: string | null;
  formats: ExportFormat[];
  selectedFormat: string;
  setSelectedFormat: (format: string) => void;
  importToOllama: boolean;
  setImportToOllama: (value: boolean) => void;
  ollamaModelName: string;
  setOllamaModelName: (name: string) => void;
  copyToLmStudio: boolean;
  setCopyToLmStudio: (value: boolean) => void;
  onExport: () => void;
  onClose: () => void;
  exportStatus: string;
  exportProgress: number;
  exportStep: string;
  exportError: string | null;
}

const ExportModelModal: React.FC<ExportModalProps> = ({
  isOpen,
  adapterName,
  formats,
  selectedFormat,
  setSelectedFormat,
  importToOllama,
  setImportToOllama,
  ollamaModelName,
  setOllamaModelName,
  copyToLmStudio,
  setCopyToLmStudio,
  onExport,
  onClose,
  exportStatus,
  exportProgress,
  exportStep,
  exportError,
}) => {
  if (!isOpen) return null;

  const isExporting = exportStatus === 'starting' || exportStatus === 'running';
  const isCompleted = exportStatus === 'completed';
  const isError = exportStatus === 'error';

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <Download className="w-6 h-6 text-primary-500" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Export Model
            </h2>
          </div>
          {!isExporting && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X className="w-6 h-6" />
            </button>
          )}
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">
          {/* Model Info */}
          <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400">Exporting Model:</div>
            <div className="text-lg font-semibold text-gray-900 dark:text-gray-100 mt-1">
              {adapterName}
            </div>
          </div>

          {!isExporting && !isCompleted && !isError && (
            <>
              {/* Format Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Export Format
                </label>
                <div className="space-y-2">
                  {formats.map((format) => (
                    <label
                      key={format.id}
                      className={`flex items-start p-4 rounded-lg border-2 cursor-pointer transition-all ${
                        selectedFormat === format.id
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <input
                        type="radio"
                        name="export-format"
                        value={format.id}
                        checked={selectedFormat === format.id}
                        onChange={(e) => setSelectedFormat(e.target.value)}
                        className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300"
                      />
                      <div className="ml-3 flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900 dark:text-gray-100">
                            {format.name}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {format.size}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {format.description}
                        </p>
                        {format.requires && (
                          <p className="text-xs text-warning-600 dark:text-warning-400 mt-1">
                            Requires: {format.requires}
                          </p>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* llama.cpp Notice */}
              {selectedFormat.startsWith('gguf_') && (
                <div className="p-4 bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="w-5 h-5 text-warning-600 dark:text-warning-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-semibold text-warning-900 dark:text-warning-100">
                        Requires llama.cpp
                      </div>
                      <p className="text-sm text-warning-800 dark:text-warning-200 mt-1">
                        GGUF conversion requires llama.cpp tools. Install with:
                      </p>
                      <code className="block text-xs bg-warning-100 dark:bg-warning-900/40 p-2 rounded mt-2 text-warning-900 dark:text-warning-100">
                        brew install llama.cpp
                      </code>
                      <p className="text-xs text-warning-700 dark:text-warning-300 mt-2">
                        Or use "Merged Model (MLX)" format which doesn't require llama.cpp
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Ollama Import Option */}
              {selectedFormat.startsWith('gguf_') && (
                <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg space-y-4">
                  <label className="flex items-start space-x-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={importToOllama}
                      onChange={(e) => setImportToOllama(e.target.checked)}
                      className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        Import to Ollama
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        Automatically import the model into Ollama after export
                      </p>
                    </div>
                  </label>

                  {importToOllama && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Ollama Model Name
                      </label>
                      <input
                        type="text"
                        value={ollamaModelName}
                        onChange={(e) => setOllamaModelName(e.target.value)}
                        placeholder="my-custom-model"
                        className="input w-full"
                      />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Use lowercase with hyphens (e.g., my-custom-model)
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* LM Studio Option */}
              <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <label className="flex items-start space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={copyToLmStudio}
                    onChange={(e) => setCopyToLmStudio(e.target.checked)}
                    className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 dark:text-gray-100">
                      Copy to LM Studio
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Automatically copy the GGUF file to LM Studio's models directory
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Location: ~/.cache/lm-studio/models/custom-exports/
                    </p>
                  </div>
                </label>
              </div>
            </>
          )}

          {/* Export Progress */}
          {isExporting && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <RefreshCw className="w-5 h-5 animate-spin text-primary-500" />
                  <span className="font-semibold text-gray-900 dark:text-gray-100">
                    {exportStep || 'Exporting...'}
                  </span>
                </div>
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {exportProgress}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-primary-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${exportProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                This may take several minutes depending on the model size...
              </p>
            </div>
          )}

          {/* Success */}
          {isCompleted && (
            <div className="p-4 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-lg">
              <div className="flex items-start space-x-3">
                <CheckCircle2 className="w-5 h-5 text-success-600 dark:text-success-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-success-900 dark:text-success-100">
                    Export Completed Successfully!
                  </div>
                  <p className="text-sm text-success-800 dark:text-success-200 mt-1">
                    Your model has been exported and is ready to use.
                  </p>
                  {importToOllama && (
                    <p className="text-sm text-success-800 dark:text-success-200 mt-2">
                      Model imported to Ollama as: <span className="font-mono font-semibold">{ollamaModelName}</span>
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {isError && exportError && (
            <div className="p-4 bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-lg">
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-error-600 dark:text-error-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-error-900 dark:text-error-100">
                    Export Failed
                  </div>
                  <p className="text-sm text-error-800 dark:text-error-200 mt-1">
                    {exportError}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
          {!isExporting && (
            <>
              <button
                onClick={onClose}
                className="btn-secondary"
              >
                {isCompleted || isError ? 'Close' : 'Cancel'}
              </button>
              {!isCompleted && !isError && (
                <button
                  onClick={onExport}
                  disabled={isExporting}
                  className={`btn-primary flex items-center space-x-2 ${isExporting ? 'opacity-75 cursor-not-allowed' : ''}`}
                >
                  {isExporting ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Exporting...</span>
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4" />
                      <span>Start Export</span>
                    </>
                  )}
                </button>
              )}
            </>
          )}
          {isExporting && (
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Please wait...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExportModelModal;
