# OPD Implementation - Testing Checklist

## ✅ Implementation Status

### Backend (100% Complete)
- ✅ **Phase 0**: Setup & Testing Infrastructure
- ✅ **Phase 1**: Core Components (TeacherModel, StudentModel, Loss, DataLoader)
- ✅ **Phase 2**: Training Loop, Utils & CLI
- ✅ **Phase 3**: FastAPI Endpoints & WebSocket Integration

### Frontend (100% Complete)
- ✅ **Phase 4**: React/Redux GUI Components
  - ✅ Redux opdSlice.ts
  - ✅ OPDPage.tsx
  - ✅ OPDSetup.tsx
  - ✅ OPDProgress.tsx
  - ✅ OPDResults.tsx
  - ✅ Navigation integration
  - ✅ WebSocket handlers

---

## 📋 Pre-Testing Verification

### Files Present
```bash
# Backend OPD Module
backend/opd/
├── teacher_model.py          ✅
├── student_model.py          ✅
├── distillation_loss.py      ✅
├── data_loader.py            ✅
├── distillation_trainer.py   ✅
├── utils.py                  ✅
├── run_distillation.py       ✅
├── config.py                 ✅
└── TEST_API_ENDPOINTS.md     ✅

# Frontend Components
frontend/src/
├── store/slices/opdSlice.ts  ✅
├── pages/OPDPage.tsx         ✅
├── components/
│   ├── OPDSetup.tsx          ✅
│   ├── OPDProgress.tsx       ✅
│   └── OPDResults.tsx        ✅
└── hooks/useWebSocket.ts     ✅ (updated)

# Backend API Integration
backend/main.py               ✅ (OPDManager + 5 endpoints)
```

### Dependencies
```bash
# Check backend dependencies
cd backend
pip3 list | grep -E "mlx|psutil|fastapi|uvicorn"

# Check frontend dependencies
cd frontend
npm list | grep -E "react|redux|lucide"
```

---

## 🧪 Testing Plan

### Test 1: Backend CLI (Standalone)
**Purpose**: Verify core distillation works without GUI

```bash
cd /Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning

python3 backend/opd/run_distillation.py \
  --teacher-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-32B-MLX-4bit" \
  --student-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct" \
  --adapter-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/7b" \
  --prompts-path ./OnPolicyDistill/test_prompts.jsonl \
  --output-path ./OnPolicyDistill/checkpoints/cli_test \
  --steps 10 \
  --batch-size 2
```

**Expected Results**:
- ✅ Models load successfully
- ✅ Training runs for 10 steps
- ✅ Checkpoints saved to `./OnPolicyDistill/checkpoints/cli_test/`
- ✅ Metrics logged to `./OnPolicyDistill/metrics/`
- ✅ No crashes or errors

---

### Test 2: Backend API (FastAPI Endpoints)
**Purpose**: Verify API endpoints work correctly

#### Step 1: Start Backend Server
```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Step 2: Test Health Check
```bash
curl http://localhost:8000/health
```
**Expected**: `{"status": "healthy", "timestamp": "..."}`

#### Step 3: Test OPD Status (Idle)
```bash
curl http://localhost:8000/opd/status
```
**Expected**: `{"state": "idle", "run_id": null, ...}`

#### Step 4: Test OPD Runs List
```bash
curl http://localhost:8000/opd/runs
```
**Expected**: `{"runs": []}`

#### Step 5: Start Distillation via API
```bash
curl -X POST http://localhost:8000/opd/start \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_path": "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct",
    "teacher_model_path": "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-32B-MLX-4bit",
    "student_adapter_path": "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/7b",
    "validation_prompts_path": "./OnPolicyDistill/test_prompts.jsonl",
    "num_steps": 20,
    "batch_size": 2,
    "temperature": 2.0,
    "kl_weight": 0.8,
    "learning_rate": 0.00001
  }'
```
**Expected**: `{"status": "success", "run_id": "distill_...", ...}`

#### Step 6: Monitor Status
```bash
# Check status while running
curl http://localhost:8000/opd/status

# Get metrics
curl http://localhost:8000/opd/metrics
```

#### Step 7: Stop Distillation (Optional)
```bash
curl -X POST http://localhost:8000/opd/stop
```

---

### Test 3: Frontend GUI (Full Integration)
**Purpose**: Verify complete user experience

#### Step 1: Build Frontend
```bash
cd frontend
npm run build
```

#### Step 2: Start Electron App
```bash
cd ..
npm start
```

#### Step 3: Navigate to Distillation Page
- Click "Distillation" in sidebar (beaker icon)
- Verify page loads with setup form

#### Step 4: Configure Distillation
- **Teacher Model**: Browse and select Qwen 32B path
- **Student Model**: Browse and select Qwen 7B path
- **Student Adapter**: Browse and select 7B adapter
- **Validation Prompts**: Browse and select test_prompts.jsonl
- **Steps**: 20
- **Batch Size**: 2
- **Temperature**: 2.0
- **KL Weight**: 0.8
- **Learning Rate**: 0.00001

#### Step 5: Start Training
- Click "Start Distillation"
- Verify page switches to Progress view
- Check for:
  - ✅ Progress bar updating
  - ✅ KL Loss displayed
  - ✅ Token Agreement displayed
  - ✅ Duration timer
  - ✅ Purple indicator in sidebar

#### Step 6: Monitor Real-time Updates
- Watch metrics update every 2 seconds
- Verify WebSocket connection is working
- Check console for any errors

#### Step 7: View Results
- Wait for completion (or stop early)
- Verify Results view shows:
  - ✅ Final status (completed/stopped)
  - ✅ Final metrics
  - ✅ Run metadata
  - ✅ "Start New Distillation" button

---

## 🔍 What to Check

### Backend Logs
```bash
# Check for errors in backend
tail -f backend/logs/backend.log

# Check OPD run logs
tail -f OnPolicyDistill/runs/*.log
```

### Frontend Console
- Open DevTools (Cmd+Option+I)
- Check Console tab for errors
- Check Network tab for API calls
- Check WebSocket connection status

### Memory Usage
```bash
# Monitor memory during training
top -pid $(pgrep -f "python.*run_distillation")
```

---

## ✅ Success Criteria

### Backend CLI
- [ ] Models load without errors
- [ ] Training completes all steps
- [ ] Checkpoints saved correctly
- [ ] Metrics logged to JSONL
- [ ] Memory stays within bounds (<60GB)

### Backend API
- [ ] All 5 endpoints respond correctly
- [ ] WebSocket broadcasts events
- [ ] Process management works (start/stop)
- [ ] Run metadata persisted
- [ ] Error handling works

### Frontend GUI
- [ ] Page loads without errors
- [ ] Form validation works
- [ ] File browser integration works
- [ ] Real-time updates display
- [ ] Status transitions work (idle→running→completed)
- [ ] Sidebar indicator updates
- [ ] Results view shows correct data

### Integration
- [ ] Frontend can start distillation via API
- [ ] WebSocket updates reach frontend
- [ ] Stop button works
- [ ] Multiple runs can be started sequentially
- [ ] Previous runs list populates

---

## 🐛 Known Issues to Watch For

### Backend
- ⚠️ First run may be slow (model loading + compilation)
- ⚠️ Teacher caching needs warm-up (first batch slower)
- ⚠️ Memory spikes during model loading

### Frontend
- ⚠️ File paths need to be absolute (no relative paths)
- ⚠️ Electron file browser may need permissions
- ⚠️ WebSocket reconnection on backend restart

### General
- ⚠️ Port 8000 must be available
- ⚠️ Sufficient disk space for checkpoints
- ⚠️ MLX requires Apple Silicon Mac

---

## 📝 Testing Notes

### Test Environment
- **OS**: macOS (Apple Silicon)
- **RAM**: 128 GB
- **Branch**: `claude/opd-011CUa2H4hPGQQ2BL84vcTm6`
- **Models**: Qwen 32B (teacher), Qwen 7B (student)

### Test Data
- **Prompts**: `./OnPolicyDistill/test_prompts.jsonl` (20 prompts)
- **Output**: `./OnPolicyDistill/checkpoints/`
- **Logs**: `./OnPolicyDistill/runs/`

---

## 🚀 Ready to Test!

All components are implemented and committed. The system is ready for end-to-end testing.

**Recommended Testing Order**:
1. ✅ Backend CLI (fastest, validates core)
2. ✅ Backend API (validates endpoints)
3. ✅ Frontend GUI (validates full UX)

Start with Test 1 (CLI) to verify the core functionality works, then proceed to Test 2 (API) and Test 3 (GUI).
