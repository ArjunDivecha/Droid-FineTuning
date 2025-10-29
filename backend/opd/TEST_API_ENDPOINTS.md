# OPD API Endpoints Testing Guide

This document provides curl commands to test the OPD FastAPI endpoints.

## Prerequisites

1. Start the FastAPI server:
```bash
cd /home/user/Droid-FineTuning
python3 backend/main.py
```

The server will be available at `http://localhost:8000`

## Endpoint Tests

### 1. Health Check

Test that the server is running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-28T14:30:22.123456"
}
```

### 2. GET /opd/status

Check current OPD status:

```bash
curl http://localhost:8000/opd/status
```

Expected response (when idle):
```json
{
  "state": "idle",
  "run_id": null,
  "metrics": {},
  "config": null
}
```

### 3. GET /opd/runs

List all previous OPD runs:

```bash
curl http://localhost:8000/opd/runs
```

Expected response:
```json
{
  "runs": []
}
```

### 4. POST /opd/start

Start a new distillation run:

**IMPORTANT**: Update the paths below to match your system!

```bash
curl -X POST http://localhost:8000/opd/start \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_path": "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct",
    "teacher_model_path": "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-32B-MLX-4bit",
    "student_adapter_path": "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/my_adapter",
    "validation_prompts_path": "./OnPolicyDistill/test_prompts.jsonl",
    "num_steps": 10,
    "batch_size": 2,
    "temperature": 2.0,
    "kl_weight": 0.8,
    "learning_rate": 0.00001
  }'
```

Expected response:
```json
{
  "status": "success",
  "run_id": "distill_20250128_143022",
  "message": "Distillation training started",
  "estimated_duration_minutes": 5,
  "memory_required_gb": 48
}
```

### 5. Monitor Progress

After starting, monitor the progress:

```bash
# Check status
curl http://localhost:8000/opd/status

# Get detailed metrics
curl http://localhost:8000/opd/metrics
```

Expected response during training:
```json
{
  "state": "running",
  "run_id": "distill_20250128_143022",
  "metrics": {
    "step": 5,
    "total_steps": 10,
    "progress_pct": 50.0,
    "kl_loss": 0.234,
    "token_agreement_pct": 78.5,
    "started_at": "2025-01-28T14:30:22"
  }
}
```

### 6. POST /opd/stop

Stop a running distillation:

```bash
curl -X POST http://localhost:8000/opd/stop
```

Expected response:
```json
{
  "status": "stopped",
  "final_step": 5,
  "checkpoint_path": "./OnPolicyDistill/checkpoints/distill_20250128_143022/checkpoint_step_5.safetensors",
  "message": "Distillation stopped by user"
}
```

### 7. GET /opd/metrics?run_id={run_id}

Get detailed metrics for a specific run:

```bash
curl "http://localhost:8000/opd/metrics?run_id=distill_20250128_143022"
```

Expected response:
```json
{
  "run_id": "distill_20250128_143022",
  "total_steps": 10,
  "metrics_history": [
    {
      "step": 0,
      "kl_loss": 1.234,
      "token_agreement_pct": 45.2,
      "student_entropy": 4.12,
      "teacher_entropy": 3.98,
      "timestamp": "2025-01-28T14:30:22"
    },
    {
      "step": 1,
      "kl_loss": 1.102,
      "token_agreement_pct": 52.3,
      "timestamp": "2025-01-28T14:31:00"
    }
  ]
}
```

## WebSocket Testing

The OPD training broadcasts real-time progress via WebSocket at `ws://localhost:8000/ws`.

### Using websocat (install with: `brew install websocat`)

```bash
websocat ws://localhost:8000/ws
```

You'll receive real-time updates:
```json
{
  "type": "opd_progress",
  "data": {
    "step": 5,
    "total_steps": 10,
    "progress_pct": 50.0,
    "kl_loss": 0.234,
    "token_agreement_pct": 78.5
  }
}
```

When training completes:
```json
{
  "type": "opd_completed",
  "data": {
    "run_id": "distill_20250128_143022",
    "final_metrics": { ... },
    "message": "Distillation completed successfully"
  }
}
```

## Error Testing

### 1. Start when already running

```bash
# Start first distillation
curl -X POST http://localhost:8000/opd/start -H "Content-Type: application/json" -d '{ ... }'

# Try to start another (should fail)
curl -X POST http://localhost:8000/opd/start -H "Content-Type: application/json" -d '{ ... }'
```

Expected error:
```json
{
  "detail": "Distillation is already running"
}
```

### 2. Missing required fields

```bash
curl -X POST http://localhost:8000/opd/start \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 10}'
```

Expected error:
```json
{
  "detail": "Missing required field: teacher_model_path"
}
```

## Integration Test Script

Save this as `test_opd_api.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

echo "1. Testing health check..."
curl -s $BASE_URL/health | jq

echo -e "\n2. Testing OPD status (should be idle)..."
curl -s $BASE_URL/opd/status | jq

echo -e "\n3. Testing runs list (should be empty)..."
curl -s $BASE_URL/opd/runs | jq

echo -e "\n4. Starting distillation..."
RUN_ID=$(curl -s -X POST $BASE_URL/opd/start \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_path": "/path/to/qwen-7b",
    "teacher_model_path": "/path/to/qwen-32b",
    "student_adapter_path": "/path/to/adapter",
    "validation_prompts_path": "./OnPolicyDistill/test_prompts.jsonl",
    "num_steps": 10,
    "batch_size": 2
  }' | jq -r '.run_id')

echo "Run ID: $RUN_ID"

echo -e "\n5. Monitoring for 10 seconds..."
sleep 10

echo -e "\n6. Checking status..."
curl -s $BASE_URL/opd/status | jq

echo -e "\n7. Getting metrics..."
curl -s "$BASE_URL/opd/metrics?run_id=$RUN_ID" | jq

echo -e "\n8. Stopping distillation..."
curl -s -X POST $BASE_URL/opd/stop | jq

echo -e "\nTest complete!"
```

Make it executable:
```bash
chmod +x test_opd_api.sh
./test_opd_api.sh
```

## Troubleshooting

### Server won't start

- Check if port 8000 is already in use: `lsof -i :8000`
- Check backend/main.py for syntax errors
- Check Python dependencies are installed: `pip install -r backend/requirements.txt`

### OPD won't start

- Verify all model paths exist
- Check you have enough RAM (need ~48GB for 32B teacher + 7B student)
- Check logs at: `./OnPolicyDistill/runs/{run_id}.log`

### Metrics not updating

- Check the distillation process is actually running: `ps aux | grep run_distillation`
- Check the log file for errors: `tail -f ./OnPolicyDistill/runs/{run_id}.log`
- Verify metrics file exists: `ls ./OnPolicyDistill/metrics/`

### WebSocket not connecting

- Ensure WebSocket endpoint is `/ws` not `/opd/ws`
- Check CORS settings in main.py if connecting from browser
- Use browser DevTools Network tab to debug WebSocket connection
