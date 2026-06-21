"""
=============================================================================
SCRIPT NAME: main.py
=============================================================================

INPUT FILES:
- <root>/models/<model>/config.json         : MLX model configs (for /models).
- <root>/sessions/*.json                     : saved training sessions.
- <config.train_data_path> (user-provided)   : JSONL training data.
- <root>/data/*.jsonl                        : generated sample datasets.

OUTPUT FILES:
- <root>/adapters/<name>/adapters.safetensors : trained LoRA weights.
- <root>/runs/<name>_<ts>/config.yaml         : reproducible run config.
- <root>/sessions/<id>.json                   : saved session files.
- <root>/data/sample_<ts>.jsonl               : generated sample data.
- <root>/outputs/logs/backend.log             : backend log.

VERSION: 1.0
LAST UPDATED: 2026-06-20
AUTHOR: Arjun

DESCRIPTION:
FastAPI backend for Droid-FineTuning v2. Replaces the original hand-rolled
backend with a thin service that wraps the `mlx-lm-lora` training engine.

This app implements the EXACT HTTP + WebSocket contract the existing React
frontend expects (verified against frontend/src/**). The SFT training path is
fully functional; Compare-tab evaluation (Tier 0+1 perplexity/weight analysis
and LLM-as-judge via DeepSeek) is implemented in app/evaluator.py. OPD /
nested-learning / fusion endpoints still return HTTP 501 with a clear message
and will be wired to mlx-lm-lora's online/teacher flows in a later phase.

The frontend contract (all hardcoded to localhost:8000):
  GET  /training/status                      -> {state, metrics, config}
  POST /training/start                        (body = TrainingConfig)
  POST /training/stop
  GET  /models                                -> {models:[{name,path,model_type,vocab_size}]}
  GET  /adapters                              -> {adapters:[{name,path}]}
  GET  /sessions                              -> {sessions:[...]}
  POST /sessions/{id}/load                    -> session object
  DELETE /sessions/{id}
  POST /api/training/generate-sample-data     (body {num_samples}) -> {success, output_path}
  POST /model/test                            (body {prompt,max_tokens,temperature})
  POST /api/evaluate/base-model               (Tier 1 perplexity) -> {success, result}
  POST /api/evaluate/adapter                  (Tier 0+1) -> {success, result}
  POST /api/evaluation/start | /status | /result  (LLM judge via DeepSeek)
  WS   /ws                                    (streams training_progress/completed/error)
  OPD/nested/fusion/*                         -> 501 (not implemented in v2 yet)

DEPENDENCIES:
- fastapi, uvicorn, websockets, pydantic, pyyaml, psutil
- mlx-lm-lora, mlx-lm, mlx (for /model/test generation)

USAGE:
    cd backend
    ../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Electron (src/main.ts) spawns this same command on app launch.
=============================================================================
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from . import paths
from .training_runner import TrainingConfig, TrainingRunner
from . import evaluator as evaluator_mod

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(paths.LOGS_DIR / "backend.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("droid.backend")

# --------------------------------------------------------------------------- #
# App + singletons
# --------------------------------------------------------------------------- #
app = FastAPI(title="Droid-FineTuning v2 API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One active training runner for the whole app (frontend assumes single run).
runner = TrainingRunner()


# --------------------------------------------------------------------------- #
# Pydantic models (mirror frontend TrainingConfig + request bodies)
# --------------------------------------------------------------------------- #
class TrainingConfigModel(BaseModel):
    """Mirrors frontend/src/store/slices/trainingSlice.ts TrainingConfig."""

    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    train_data_path: str
    val_data_path: str = ""
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_length: int = 1024
    iterations: int = 1000
    steps_per_report: int = 10
    steps_per_eval: int = 200
    save_every: int = 100
    early_stop: bool = True
    patience: int = 3
    adapter_name: str = "mlx_finetune"
    # Optional: path to a saved LoRA .safetensors checkpoint to resume from.
    resume_from: str = ""


class GenerateSampleDataRequest(BaseModel):
    num_samples: int = 20


class ModelTestRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    # Optional, used by ModelTestModal when testing a fine-tuned model.
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None


class EvaluateAdapterRequest(BaseModel):
    """Body for POST /api/evaluate/adapter (Tier 0 + Tier 1)."""

    adapter_name: str
    max_samples: int = 20


class EvaluateBaseModelRequest(BaseModel):
    """Body for POST /api/evaluate/base-model (Tier 1 only)."""

    max_samples: int = 20


class EvaluationStartRequest(BaseModel):
    """Body for POST /api/evaluation/start (LLM-as-judge)."""

    adapter_name: Optional[str] = None
    training_data_path: Optional[str] = None
    num_questions: int = 20
    evaluate_base_model: bool = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_model_info(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Read MLX model metadata from a model directory's config.json.

    Returns {name, path, model_type, vocab_size} or None if the directory is
    NOT a valid MLX-format model. We require BOTH config.json AND at least one
    *.safetensors file — this filters out GGUF models (inference-only, can't
    be fine-tuned by mlx-lm-lora) and incomplete/auxiliary directories.
    """
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        return None
    # Must have at least one safetensors weights file (MLX format). GGUF
    # models only contain .gguf files and are skipped.
    has_safetensors = any(model_dir.glob("*.safetensors"))
    if not has_safetensors:
        return None
    try:
        with open(config_file) as f:
            cfg = json.load(f)
    except Exception:
        return None
    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "model_type": cfg.get("model_type", "unknown"),
        "vocab_size": int(cfg.get("vocab_size", 0) or 0),
    }


def _scan_models_dir(root: Path) -> List[Dict[str, Any]]:
    """Recursively scan a models root for MLX-format model directories.

    Handles both flat layouts (root/<model>/) and nested publisher/model
    layouts (root/<publisher>/<model>/) like LM Studio's. Returns up to two
    levels deep; a model dir is identified by having config.json + safetensors.
    """
    found: List[Dict[str, Any]] = []
    if not root.is_dir():
        return found
    # Check direct children (flat layout) and one level of nesting.
    candidates: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidates.append(child)
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir():
                    candidates.append(grandchild)
    for cand in candidates:
        info = _read_model_info(cand)
        if info is not None and info["path"] not in {m["path"] for m in found}:
            found.append(info)
    return found


def _not_implemented(detail: str) -> HTTPException:
    return HTTPException(status_code=501, detail=detail)


def _latest_run_config() -> tuple[Optional[str], Optional[str]]:
    """Read the most recent run's config.yaml from disk.

    Returns (model_path, adapter_path) from the newest run directory under
    paths.RUNS_DIR, or (None, None) if none can be parsed. This is the
    durable, restart-surviving source of truth for "what was last trained",
    complementing the in-memory runner.config which is lost on restart.
    """
    try:
        import yaml  # part of mlx-lm-lora's deps
    except Exception:  # noqa: BLE001
        return None, None
    if not paths.RUNS_DIR.is_dir():
        return None, None
    candidates = sorted(
        (p for p in paths.RUNS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        cfg = run_dir / "config.yaml"
        if not cfg.is_file():
            continue
        try:
            with open(cfg) as f:
                data = yaml.safe_load(f) or {}
        except Exception:  # noqa: BLE001
            continue
        model = data.get("model") or data.get("model_path")
        adapter = data.get("adapter_path") or data.get("adapter_name")
        if not model:
            continue
        # adapter_name (str) -> resolve to adapters/<name>
        if adapter and not Path(str(adapter)).is_dir():
            adapter = str(paths.ADAPTERS_DIR / str(adapter))
        return model, (str(adapter) if adapter else None)
    return None, None


def _resolve_model_and_adapter(
    req_model_path: Optional[str],
    req_adapter_path: Optional[str],
    include_adapter: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve which model + adapter to use for a /model/test* call.

    Priority:
      1. Explicit values on the request (if provided).
      2. The current/last training run's config (runner.config) — works
         during training and after a run completes in the same process.
      3. The most recent run's config.yaml on disk — survives backend
         restarts (runner.config is in-memory and lost on relaunch).

    `include_adapter` controls whether the adapter is attached. The Compare
    page's "base model" call passes include_adapter=False (base only); the
    "fine-tuned" call passes include_adapter=True.

    Returns (model_path, adapter_path_or_None). Raises HTTPException(400)
    if no model can be resolved.
    """
    model_path = req_model_path
    adapter_path = req_adapter_path if include_adapter else None

    # Priority 2: in-memory last-run config.
    if model_path is None and runner.config is not None:
        model_path = runner.config.model_path
    if include_adapter and adapter_path is None and runner.config is not None:
        adapter_name = runner.config.adapter_name
        if adapter_name:
            adapter_path = str(paths.ADAPTERS_DIR / adapter_name)

    # Priority 3: disk-based fallback (survives restarts).
    if not model_path:
        disk_model, disk_adapter = _latest_run_config()
        if not model_path and disk_model:
            model_path = disk_model
        if include_adapter and not adapter_path and disk_adapter:
            adapter_path = disk_adapter

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="No model specified and no training run config available. "
            "Either pass model_path or start/load a training run first.",
        )
    # If an adapter was resolved, only attach it if the dir actually exists
    # (a run may be in progress and not have saved weights yet).
    if adapter_path and not Path(adapter_path).is_dir():
        adapter_path = None
    return model_path, adapter_path


# --------------------------------------------------------------------------- #
# WebSocket connection manager
# --------------------------------------------------------------------------- #
class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts training updates."""

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a message to every connected client. Drops dead clients."""
        dead: List[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# --------------------------------------------------------------------------- #
# LLM-as-judge evaluation manager (async start / poll status / fetch result)
# --------------------------------------------------------------------------- #
class EvaluationManager:
    """Runs LLM-judge evaluations on a background thread.

    The frontend's "Evaluate (LLM Judge)" flow is asynchronous:
      1. POST /api/evaluation/start  -> kicks off a run, returns immediately
      2. GET  /api/evaluation/status -> {running, progress, error}
      3. GET  /api/evaluation/result -> {success, result} once finished
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.running: bool = False
        self.progress: float = 0.0
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self._stop_flag = False

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "progress": self.progress,
                "error": self.error,
            }

    def get_result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self.result is None else dict(self.result)

    def start(
        self,
        adapter_name: Optional[str],
        num_questions: int,
        evaluate_base_model: bool,
    ) -> bool:
        """Start a new evaluation. Returns False if one is already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.progress = 0.0
            self.error = None
            self.result = None
            self._stop_flag = False

        self._thread = threading.Thread(
            target=self._run,
            args=(adapter_name, num_questions, evaluate_base_model),
            daemon=True,
        )
        self._thread.start()
        return True

    def _set(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _run(
        self,
        adapter_name: Optional[str],
        num_questions: int,
        evaluate_base_model: bool,
    ) -> None:
        try:

            def on_progress(done: int, total: int) -> None:
                self._set(progress=(done / total) * 100.0 if total else 0.0)

            res = evaluator_mod.llm_judge_evaluate(
                adapter_name=adapter_name,
                num_questions=num_questions,
                evaluate_base_model=evaluate_base_model,
                on_progress=on_progress,
            )
            self._set(result=res, progress=100.0)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM-judge evaluation failed")
            self._set(error=str(exc))
        finally:
            self._set(running=False)


evaluation_manager = EvaluationManager()


async def _drain_runner_updates() -> None:
    """Forward queued MetricsUpdates from the training thread to WS clients.

    Runs as a background task for the lifetime of one training run. Exits once
    the run is no longer active AND the queue has been fully drained (so the
    terminal completed/error/stopped frame is always delivered).
    """
    while True:
        update = runner.next_update()
        if update is None:
            # No queued update. If the training thread is still alive, keep
            # polling; otherwise the run is fully done and drained, so exit.
            if runner.is_running:
                await asyncio.sleep(0.1)
                continue
            return

        if update.state:
            # Terminal / state-transition message.
            if update.state == "completed":
                msg = {
                    "type": "training_completed",
                    "data": {"final_metrics": update.metrics},
                }
            elif update.state == "error":
                msg = {
                    "type": "training_error",
                    "data": {"error": runner.error or "Training failed"},
                }
            elif update.state == "stopped":
                msg = {"type": "training_stopped", "data": {}}
            elif update.state == "running":
                msg = {
                    "type": "training_started",
                    "data": {"metrics": update.metrics},
                }
            else:
                msg = {"type": "training_state", "data": {"state": update.state}}
            await manager.broadcast(msg)

        # Always send a progress frame for the metrics + log line.
        if update.metrics:
            await manager.broadcast(
                {
                    "type": "training_progress",
                    "data": {
                        "metrics": update.metrics,
                        "log_line": update.log_line,
                    },
                }
            )

        if update.state in {"completed", "error", "stopped"}:
            return


# --------------------------------------------------------------------------- #
# Routes: training
# --------------------------------------------------------------------------- #
@app.get("/training/status")
async def training_status() -> Dict[str, Any]:
    """Return current training state, metrics, and config.

    This is the Electron health-check endpoint (src/main.ts polls it on boot)
    and the frontend's 2-second polling target (useWebSocket.ts).
    """
    return runner.snapshot()


@app.post("/training/start")
async def training_start(config: TrainingConfigModel) -> Dict[str, Any]:
    """Start an SFT training run via mlx-lm-lora.

    Returns 409 if a run is already in progress.
    """
    if runner.is_running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    if not config.model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    if not config.train_data_path:
        raise HTTPException(status_code=400, detail="train_data_path is required")

    tc = TrainingConfig(**config.model_dump())
    try:
        runner.start(tc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to start: {exc}") from exc

    # Kick off the background drain task that forwards updates to WS clients.
    asyncio.create_task(_drain_runner_updates())

    return {"status": "started", "adapter_name": config.adapter_name}


@app.post("/training/stop")
async def training_stop() -> Dict[str, Any]:
    """Request the active run to stop."""
    runner.stop()
    return {"status": "stopped"}


@app.get("/training/latest-checkpoint")
async def latest_checkpoint() -> Dict[str, Any]:
    """Return the most recently modified LoRA checkpoint across all adapters.

    Disk-based (survives backend restarts, unlike the in-memory runner.config).
    The Setup page uses this to auto-default the "Resume from Checkpoint"
    field + adapter name to the last-trained model, so the user doesn't have
    to know where checkpoints live.

    Preference per adapter dir: adapters.safetensors (the latest saved) wins
    over the numbered 0001NNN_adapters.safetensors checkpoints. Across
    adapters, the newest mtime wins.

    Returns {adapter_name, adapter_path, checkpoint_path, checkpoint_mtime}
    or {adapter_name: null, checkpoint_path: null} if none exist.
    """
    best: Optional[Dict[str, Any]] = None
    if paths.ADAPTERS_DIR.is_dir():
        for adapter_dir in paths.ADAPTERS_DIR.iterdir():
            if not adapter_dir.is_dir():
                continue
            # Prefer adapters.safetensors; fall back to the newest numbered ckpt.
            candidates = sorted(
                adapter_dir.glob("*.safetensors"),
                key=lambda p: (
                    p.name != "adapters.safetensors",  # False (0) sorts first
                    -p.stat().st_mtime,  # then newest mtime
                ),
            )
            if not candidates:
                continue
            ckpt = candidates[0]
            mtime = ckpt.stat().st_mtime
            if best is None or mtime > best["checkpoint_mtime"]:
                best = {
                    "adapter_name": adapter_dir.name,
                    "adapter_path": str(adapter_dir),
                    "checkpoint_path": str(ckpt),
                    "checkpoint_mtime": mtime,
                }
    if best is None:
        return {"adapter_name": None, "checkpoint_path": None}
    return best


# --------------------------------------------------------------------------- #
# Routes: models + adapters
# --------------------------------------------------------------------------- #
@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List MLX base models available for fine-tuning.

    Scans two locations:
      1. <root>/models (the project's models dir, or DROID_MODELS_DIR)
      2. ~/.lmstudio/models (LM Studio's library, in publisher/model layout)

    Only MLX-format models (config.json + *.safetensors) are returned — GGUF
    models are skipped because mlx-lm-lora can't fine-tune them. Returns
    {models: []} if no trainable models are present.
    """
    models: List[Dict[str, Any]] = []
    seen_paths: set = set()
    for info in _scan_models_dir(paths.MODELS_DIR) + _scan_models_dir(
        paths.LM_STUDIO_MODELS_DIR
    ):
        if info["path"] not in seen_paths:
            seen_paths.add(info["path"])
            models.append(info)
    return {"models": models}


@app.get("/adapters")
async def list_adapters() -> Dict[str, Any]:
    """List trained LoRA adapter directories."""
    adapters: List[Dict[str, Any]] = []
    if paths.ADAPTERS_DIR.is_dir():
        for child in sorted(paths.ADAPTERS_DIR.iterdir()):
            if child.is_dir():
                adapters.append({"name": child.name, "path": str(child)})
    return {"adapters": adapters}


# --------------------------------------------------------------------------- #
# Routes: sessions
# --------------------------------------------------------------------------- #
@app.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List saved training session JSON files."""
    sessions: List[Dict[str, Any]] = []
    if paths.SESSIONS_DIR.is_dir():
        for child in sorted(paths.SESSIONS_DIR.iterdir(), reverse=True):
            if child.is_file() and child.suffix == ".json":
                try:
                    with open(child) as f:
                        data = json.load(f)
                    sessions.append({"id": child.stem, **data})
                except Exception:
                    continue
    return {"sessions": sessions}


@app.post("/sessions/{session_id}/load")
async def load_session(session_id: str) -> Dict[str, Any]:
    """Load a saved session by id (filename stem)."""
    path = paths.SESSIONS_DIR / f"{session_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    with open(path) as f:
        return json.load(f)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a saved session file."""
    path = paths.SESSIONS_DIR / f"{session_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    path.unlink()
    return {"status": "deleted", "id": session_id}


# --------------------------------------------------------------------------- #
# Routes: sample data generation + model test
# --------------------------------------------------------------------------- #
@app.post("/api/training/generate-sample-data")
async def generate_sample_data(req: GenerateSampleDataRequest) -> Dict[str, Any]:
    """Write a small canned SFT JSONL so the Setup page can bootstrap a run
    without the user having data ready yet.

    Each line is a {"text": "<prompt> <completion>"} object, which is the
    simplest mlx-lm-lora SFT format.
    """
    n = max(1, min(int(req.num_samples), 1000))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = paths.DATA_DIR / f"sample_{ts}.jsonl"
    samples: List[str] = []
    for i in range(n):
        text = (
            f"User: What is {i + 1} plus {i + 1}?\n"
            f"Assistant: {i + 1} plus {i + 1} is {2 * (i + 1)}."
        )
        samples.append(json.dumps({"text": text}))
    out_path.write_text("\n".join(samples) + "\n")
    logger.info("Wrote %d sample SFT rows to %s", n, out_path)
    return {"success": True, "output_path": str(out_path), "num_samples": n}


@app.post("/model/test")
async def model_test(req: ModelTestRequest) -> Dict[str, Any]:
    """Generate a completion from the fine-tuned (adapter-applied) model.

    Defaults to the current/last training run's model + adapter when the
    request doesn't specify them, so the Compare page works without changes.
    """
    model_path, adapter_path = _resolve_model_and_adapter(
        req.model_path, req.adapter_path, include_adapter=True
    )
    return await _run_generation(req, model_path, adapter_path)


@app.post("/model/test-base")
async def model_test_base(req: ModelTestRequest) -> Dict[str, Any]:
    """Generate a completion from the BASE model only (no adapter).

    Used by the Compare page's left-hand panel. Same defaulting as
    /model/test but the adapter is never attached.
    """
    model_path, adapter_path = _resolve_model_and_adapter(
        req.model_path, req.adapter_path, include_adapter=False
    )
    return await _run_generation(req, model_path, adapter_path)


async def _run_generation(
    req: ModelTestRequest,
    model_path: str,
    adapter_path: Optional[str],
) -> Dict[str, Any]:
    """Shared generation backend for /model/test and /model/test-base."""
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"mlx-lm unavailable: {exc}") from exc

    try:
        # mlx_lm.load() takes the model path as the first positional arg
        # (path_or_hf_repo); it does NOT accept a `model=` keyword. The adapter
        # is attached via the `adapter_path` kwarg.
        model, tokenizer = load(model_path, adapter_path=adapter_path)

        # mlx_lm.generate() forwards **kwargs to generate_step, which expects a
        # `sampler` callable rather than a `temp` kwarg. Build one explicitly.
        sampler = make_sampler(temp=float(req.temperature or 0.0))
        response = generate(
            model,
            tokenizer,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            sampler=sampler,
        )
        return {
            "response": response,
            "prompt": req.prompt,
            "model_path": model_path,
            "adapter_path": adapter_path,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a WS connection and stream training updates.

    The frontend (useWebSocket.ts) connects to ws://127.0.0.1:8000/ws and
    handles inbound message types: training_state, training_started,
    training_progress, training_completed, training_stopped, training_error,
    opd_*.
    """
    await manager.connect(ws)
    # Send an initial state frame so the client knows current status.
    snap = runner.snapshot()
    await ws.send_json(
        {
            "type": "training_state",
            "data": {
                "state": snap["state"],
                "metrics": snap["metrics"],
            },
        }
    )
    try:
        while True:
            # The frontend uses send() for outbound control messages; we read
            # them but currently only log (no commands defined in v2).
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# --------------------------------------------------------------------------- #
# Routes: not-yet-implemented (501) — wired in later phases
# --------------------------------------------------------------------------- #
_NOT_IMPL_MSG = "Not implemented in v2 backend yet (planned for a later phase)."


@app.post("/opd/start")
async def opd_start() -> None:
    raise _not_implemented(_NOT_IMPL_MSG)


@app.post("/opd/stop")
async def opd_stop() -> None:
    raise _not_implemented(_NOT_IMPL_MSG)


@app.get("/opd/runs")
async def opd_runs() -> Dict[str, Any]:
    return {"runs": []}


@app.get("/opd/status")
async def opd_status() -> Dict[str, Any]:
    return {"state": "idle"}


@app.get("/nested-learning/status")
async def nested_status() -> Dict[str, Any]:
    return {"state": "idle"}


@app.post("/nested-learning/start")
async def nested_start() -> None:
    raise _not_implemented(_NOT_IMPL_MSG)


@app.get("/api/fusion/status")
async def fusion_status() -> Dict[str, Any]:
    return {"status": "idle"}


@app.get("/api/fusion/result")
async def fusion_result() -> None:
    raise _not_implemented(_NOT_IMPL_MSG)


@app.get("/api/fusion/list-adapters")
async def fusion_list_adapters() -> Any:
    # Reuse the adapters list so the Fusion page at least renders adapters.
    return await list_adapters()


@app.post("/api/fusion/fuse")
async def fusion_fuse() -> None:
    raise _not_implemented(_NOT_IMPL_MSG)


@app.post("/api/evaluate/adapter")
async def evaluate_adapter(req: EvaluateAdapterRequest) -> Dict[str, Any]:
    """Evaluate a LoRA adapter with Tier 0 (mathematical) + Tier 1 (perplexity)."""
    try:
        result = await asyncio.to_thread(
            evaluator_mod.evaluate_adapter,
            req.adapter_name,
            req.max_samples,
        )
        return {"success": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Adapter evaluation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/evaluate/base-model")
async def evaluate_base_model(req: EvaluateBaseModelRequest) -> Dict[str, Any]:
    """Evaluate the base model with Tier 1 (perplexity) only."""
    try:
        result = await asyncio.to_thread(
            evaluator_mod.evaluate_base_model,
            req.max_samples,
        )
        return {"success": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Base model evaluation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Routes: LLM-as-judge evaluation (async start/status/result)
# --------------------------------------------------------------------------- #
@app.post("/api/evaluation/start")
async def start_evaluation(req: EvaluationStartRequest) -> Dict[str, Any]:
    """Start an LLM-as-judge (DeepSeek) evaluation. Runs on a background thread."""
    if not req.adapter_name and not req.evaluate_base_model:
        raise HTTPException(
            status_code=400,
            detail="adapter_name is required (or set evaluate_base_model=true).",
        )
    started = evaluation_manager.start(
        adapter_name=req.adapter_name,
        num_questions=req.num_questions,
        evaluate_base_model=req.evaluate_base_model,
    )
    if not started:
        raise HTTPException(status_code=409, detail="An evaluation is already running.")
    return {
        "success": True,
        "message": "Evaluation started",
        "adapter_name": req.adapter_name or "base_model",
    }


@app.get("/api/evaluation/status")
async def get_evaluation_status() -> Dict[str, Any]:
    """Poll the running LLM-judge evaluation's progress."""
    return evaluation_manager.get_status()


@app.get("/api/evaluation/result")
async def get_evaluation_result() -> Dict[str, Any]:
    """Fetch the completed LLM-judge evaluation result."""
    result = evaluation_manager.get_result()
    if result is None:
        raise HTTPException(status_code=404, detail="No evaluation result available")
    return {"success": True, "result": result}


@app.on_event("startup")
async def _startup() -> None:
    logger.info(
        "Droid-FineTuning v2 backend up. models=%s adapters=%s",
        paths.MODELS_DIR,
        paths.ADAPTERS_DIR,
    )
