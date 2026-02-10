import { useCallback, useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store/store';
import {
  setConnectionStatus,
  setTrainingConfig,
  trainingStarted,
  trainingProgress,
  trainingCompleted,
  trainingStopped,
  trainingError,
  addLogLine,
  clearLogs,
} from '../store/slices/trainingSlice';
import {
  setTrainingState,
  setTrainingMetrics,
} from '../store/slices/trainingSlice';
import {
  opdProgress,
  opdCompleted,
  opdStopped,
  opdError,
} from '../store/slices/opdSlice';
import { addNotification } from '../store/slices/uiSlice';

const WEBSOCKET_URL = 'ws://127.0.0.1:8000/ws';

export const useWebSocket = () => {
  const dispatch = useDispatch();
  const { isConnected, metrics: currentMetrics } = useSelector((state: RootState) => state.training);
  const socketRef = useRef<WebSocket | null>(null);
  const lastLoggedStep = useRef<number>(-1);
  const lastRunStartTime = useRef<string | null>(null);
  const hasInitialized = useRef<boolean>(false);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    
    const socket = new WebSocket(WEBSOCKET_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      dispatch(setConnectionStatus(true));
      dispatch(addNotification({
        type: 'success',
        title: 'Connected',
        message: 'WebSocket connection established'
      }));
    };

    socket.onclose = () => {
      dispatch(setConnectionStatus(false));
      dispatch(addNotification({
        type: 'warning',
        title: 'Disconnected',
        message: 'WebSocket connection lost'
      }));
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'training_state': {
            const wsState = data.data?.state;
            const wsMetrics = data.data?.metrics;
            if (wsState === 'running') {
              // Detect new training by comparing with Redux state metrics
              const currentStartTime = currentMetrics?.start_time;
              const newStartTime = wsMetrics?.start_time;
              const startTimeChanged = newStartTime && newStartTime !== currentStartTime && newStartTime !== lastRunStartTime.current;
              const stepRegressed = currentMetrics && wsMetrics?.current_step && wsMetrics.current_step < currentMetrics.current_step;
              
              if (startTimeChanged || stepRegressed || !hasInitialized.current) {
                // New training session detected - clear everything
                dispatch(trainingStarted());
                lastLoggedStep.current = -1;
                if (newStartTime) {
                  lastRunStartTime.current = newStartTime as string;
                }
              } else {
                // Same training session - just update state
                dispatch(setTrainingState('running'));
              }
              hasInitialized.current = true;
              
              if (wsMetrics) {
                dispatch(trainingProgress({ metrics: wsMetrics, log_line: '' }));
              }
            } else if (wsState === 'completed') {
              if (wsMetrics) {
                dispatch(trainingCompleted({ final_metrics: wsMetrics }));
              } else {
                dispatch(setTrainingState('completed'));
              }
            } else if (wsState === 'error') {
              if (wsMetrics) {
                dispatch(setTrainingMetrics(wsMetrics));
              }
              dispatch(trainingError({ error: 'Training failed' }));
            } else {
              // idle or unknown: reflect state and surface metrics without forcing 'running'
              dispatch(setTrainingState('idle'));
              if (wsMetrics) {
                dispatch(setTrainingMetrics(wsMetrics));
              }
            }
            break;
          }
          case 'training_started':
            dispatch(clearLogs()); // Clear previous logs when new training starts
            lastLoggedStep.current = -1; // Reset step tracking
            // Run boundary will be confirmed via polling using metrics.start_time
            dispatch(trainingStarted());
            break;
          case 'training_progress':
            // Accept real-time progress over WebSocket; polling remains as fallback
            // Ensure metrics have all required fields
            if (data.data?.metrics) {
              const metrics = data.data.metrics;
              const safeMetrics = {
                current_step: metrics.current_step ?? 0,
                total_steps: metrics.total_steps ?? 0,
                train_loss: metrics.train_loss ?? null,
                val_loss: metrics.val_loss ?? null,
                learning_rate: metrics.learning_rate ?? 0,
                start_time: metrics.start_time ?? '',
                estimated_time_remaining: metrics.estimated_time_remaining ?? null,
                avg_reward: metrics.avg_reward ?? null,
                success_rate: metrics.success_rate ?? null,
                kl: metrics.kl ?? null,
                entropy: metrics.entropy ?? null,
                ...metrics
              };
              dispatch(trainingProgress({
                metrics: safeMetrics,
                log_line: data.data.log_line ?? ''
              }));
            }
            break;
          case 'training_completed':
            dispatch(trainingCompleted(data.data));
            break;
          case 'training_stopped':
            dispatch(trainingStopped());
            break;
          case 'training_error':
            dispatch(trainingError(data.data));
            break;
          // OPD events
          case 'opd_progress':
            dispatch(opdProgress(data.data));
            break;
          case 'opd_completed':
            dispatch(opdCompleted(data.data));
            dispatch(addNotification({
              type: 'success',
              title: 'Distillation Complete',
              message: data.data.message || 'Distillation training completed successfully',
              autoHide: true,
            }));
            break;
          case 'opd_stopped':
            dispatch(opdStopped(data.data));
            dispatch(addNotification({
              type: 'info',
              title: 'Distillation Stopped',
              message: 'Distillation training was stopped',
              autoHide: true,
            }));
            break;
          case 'opd_error':
            dispatch(opdError(data.data));
            dispatch(addNotification({
              type: 'error',
              title: 'Distillation Error',
              message: data.data.error || 'An error occurred during distillation',
            }));
            break;
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      dispatch(addNotification({
        type: 'error',
        title: 'Connection Error',
        message: 'WebSocket connection failed'
      }));
    };

  }, [dispatch]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
      dispatch(setConnectionStatus(false));
    }
  }, [dispatch]);

  const send = useCallback((event: string, data: any) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ type: event, payload: data });
      socketRef.current.send(message);
    } else {
    }
  }, []);

  // Removed fetchLogs - using polling for real-time updates only

  // Auto-connect on mount and start polling for training status
  useEffect(() => {
    connect();
    
    // Initial status fetch - DO NOT clear logs here, preserve existing data
    const fetchInitialData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/training/status');
        if (response.ok) {
          const status = await response.json();
          
          // Load the training config and state
          if (status.config) {
            dispatch(setTrainingConfig(status.config));
          }
          
          // Check if this is a NEW training session (start_time changed)
          const startTimeChanged = status.metrics?.start_time && 
            status.metrics.start_time !== lastRunStartTime.current;
          
          if (startTimeChanged) {
            // NEW training session - clear logs and reset step tracking
            dispatch(clearLogs());
            lastLoggedStep.current = -1;
            lastRunStartTime.current = status.metrics.start_time as string;
          } else if (status.metrics?.start_time) {
            // Same training session - just update tracking
            lastRunStartTime.current = status.metrics.start_time as string;
          }
          
          if (status.state === 'completed') {
            dispatch(trainingCompleted({ final_metrics: status.metrics }));
          } else if (status.state === 'error') {
            // Set metrics even for error state so UI can display data
            if (status.metrics) {
              dispatch(trainingProgress({
                metrics: status.metrics,
                log_line: ''
              }));
            }
            dispatch(trainingError({ error: 'Training failed' }));
          } else if (status.state === 'running') {
            // Update state without clearing logs (logs only cleared for NEW training above)
            if (startTimeChanged) {
              dispatch(trainingStarted());
            } else {
              dispatch(setTrainingState('running'));
            }
            if (status.metrics) {
              dispatch(trainingProgress({
                metrics: status.metrics,
                log_line: ''
              }));
            }
          } else if (status.state === 'idle' && status.metrics) {
            // Handle idle state with existing metrics (from previous run)
            if (status.config) {
              dispatch(setTrainingConfig(status.config));
            }
            dispatch(trainingProgress({
              metrics: status.metrics,
              log_line: ''
            }));
          }
        }
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };
    
    fetchInitialData();
    
    // Poll training status every 2 seconds as fallback
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/training/status');
        if (response.ok) {
          const status = await response.json();
          
          // Update training state based on API status 
          if (status.state === 'running') {
            const metrics = status.metrics || {};
            const currentStep: number = metrics.current_step ?? 0;
            const runStartTime: string | null = metrics.start_time ?? null;

            // Detect a new run by comparing with Redux state metrics (more reliable)
            const currentStartTime = currentMetrics?.start_time;
            const startTimeChanged = runStartTime && runStartTime !== currentStartTime && runStartTime !== lastRunStartTime.current;
            const stepRegressed = lastLoggedStep.current !== -1 && currentStep < lastLoggedStep.current;

            if (startTimeChanged || stepRegressed) {
              dispatch(clearLogs());
              lastLoggedStep.current = -1;
              if (runStartTime) {
                lastRunStartTime.current = runStartTime;
              }
              dispatch(trainingStarted());
            }

            // Only log when step changes - show iter, train loss, val loss
            let logLine = '';
            if (currentStep !== lastLoggedStep.current) {
              const trainLoss = metrics.train_loss != null ? Number(metrics.train_loss).toFixed(4) : 'N/A';
              const valLoss = metrics.val_loss != null ? Number(metrics.val_loss).toFixed(4) : 'N/A';
              logLine = `Iter ${currentStep}: Train loss ${trainLoss}, Val loss ${valLoss}`;
              lastLoggedStep.current = currentStep;
            }

            // Ensure metrics has all required fields with proper defaults
            const safeMetrics = {
              current_step: metrics.current_step ?? 0,
              total_steps: metrics.total_steps ?? 0,
              train_loss: metrics.train_loss ?? null,
              val_loss: metrics.val_loss ?? null,
              learning_rate: metrics.learning_rate ?? 0,
              start_time: metrics.start_time ?? '',
              estimated_time_remaining: metrics.estimated_time_remaining ?? null,
              avg_reward: metrics.avg_reward ?? null,
              success_rate: metrics.success_rate ?? null,
              kl: metrics.kl ?? null,
              entropy: metrics.entropy ?? null,
              ...metrics  // Allow actual values to override defaults
            };

            dispatch(trainingProgress({
              metrics: safeMetrics,
              log_line: logLine
            }));
          } else if (status.state === 'completed') {
            // Ensure we set the training config if we don't have it
            if (status.config) {
              dispatch(setTrainingConfig(status.config));
            }
            dispatch(trainingCompleted({ final_metrics: status.metrics }));
          } else if (status.state === 'error') {
            // Set config and metrics even for error state so UI can display data
            if (status.config) {
              dispatch(setTrainingConfig(status.config));
            }
            if (status.metrics) {
              dispatch(trainingProgress({
                metrics: status.metrics,
                log_line: ''
              }));
            }
            dispatch(trainingError({ error: 'Training failed' }));
          } else if (status.state === 'idle' && status.metrics) {
            // Handle idle state with existing metrics (from previous completed/error run)
            if (status.config) {
              dispatch(setTrainingConfig(status.config));
            }
            // Show the metrics from the previous run
            dispatch(trainingProgress({
              metrics: status.metrics,
              log_line: ''
            }));
          }
        }
      } catch (error) {
        // Silent - don't spam console with polling errors
      }
    }, 2000);
    
    // Cleanup on unmount
    return () => {
      disconnect();
      clearInterval(pollInterval);
    };
  }, [connect, disconnect, dispatch]);

  return {
    connect,
    disconnect,
    send,
    isConnected,
  };
};