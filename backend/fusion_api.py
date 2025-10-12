"""
=============================================================================
SCRIPT NAME: fusion_api.py
=============================================================================

INPUT FILES:
- Adapter directories from local_qwen/artifacts/lora_adapters
- Session JSON files for base model metadata

OUTPUT FILES:
- Fused adapter safetensors files
- Evaluation results JSON

VERSION: 1.0
LAST UPDATED: 2025-09-29
AUTHOR: Droid-FineTuning

DESCRIPTION:
FastAPI endpoints for adapter fusion functionality. Allows users to:
- List available adapters grouped by base model
- Validate adapter compatibility
- Fuse 2-5 adapters using SLERP or weighted averaging
- Evaluate all adapters + fused adapter vs base model

DEPENDENCIES:
- fastapi
- safetensors
- numpy
- torch (optional, for SLERP)

=============================================================================
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add adapter_fusion to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'adapter_fusion'))
from fusion_adapters import AdapterFusion

# Add evaluation path
sys.path.append('/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune')

# Import model exporter
from model_export import ModelExporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fusion", tags=["fusion"])

# Global state for fusion operations
fusion_state = {
    "is_running": False,
    "progress": 0,
    "status": "idle",
    "error": None,
    "current_step": "",
    "result": None
}

# Global state for export operations
export_state = {
    "is_running": False,
    "progress": 0,
    "status": "idle",
    "error": None,
    "current_step": "",
    "result": None
}

# Pydantic models
class AdapterInfo(BaseModel):
    name: str
    base_model: str
    training_date: Optional[str] = None
    iterations: Optional[int] = None
    dataset: Optional[str] = None
    path: str

class AdaptersByBaseModel(BaseModel):
    base_model: str
    adapters: List[AdapterInfo]

class FusionRequest(BaseModel):
    adapter_names: List[str]
    method: str = "slerp"  # "slerp" or "weighted"
    weights: Optional[List[float]] = None
    output_name: Optional[str] = None
    evaluation_dataset: Optional[str] = None  # Optional: specific dataset for all evaluations

class EvaluationResult(BaseModel):
    adapter_name: str
    is_base_model: bool
    overall_score: float
    faithfulness: float
    fact_recall: float
    consistency: float
    hallucination: float

class FusionEvaluationResult(BaseModel):
    base_model_result: EvaluationResult
    individual_results: List[EvaluationResult]
    fused_result: EvaluationResult
    fusion_info: Dict[str, Any]

def get_session_info(adapter_name: str) -> Dict[str, Any]:
    """Get session information for an adapter to determine base model."""
    sessions_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
    
    # Try to find matching session file
    if os.path.exists(sessions_dir):
        for session_file in os.listdir(sessions_dir):
            if session_file.endswith('.json'):
                session_path = os.path.join(sessions_dir, session_file)
                try:
                    with open(session_path, 'r') as f:
                        session_data = json.load(f)
                        # Check both direct adapter_name and config.adapter_name
                        config_adapter = session_data.get('config', {}).get('adapter_name')
                        direct_adapter = session_data.get('adapter_name')
                        
                        if config_adapter == adapter_name or direct_adapter == adapter_name:
                            return session_data
                except Exception as e:
                    logger.warning(f"Error reading session {session_file}: {e}")
    
    return {}

@router.get("/list-adapters", response_model=List[AdaptersByBaseModel])
async def list_adapters():
    """List all available adapters grouped by base model."""
    try:
        fusion = AdapterFusion()
        adapter_names = fusion.list_available_adapters()
        
        logger.info(f"Found {len(adapter_names)} adapters")
        
        # Group adapters by base model
        adapters_by_model: Dict[str, List[AdapterInfo]] = {}
        
        for adapter_name in adapter_names:
            logger.info(f"Processing adapter: {adapter_name}")
            session_info = get_session_info(adapter_name)
            
            # Extract base model name from model_path or model_name
            base_model = 'Unknown Model'
            
            # Try to get from config.model_path first
            config = session_info.get('config', {})
            model_path = config.get('model_path') or session_info.get('model_path')
            
            if model_path:
                # Extract model name from path (e.g., "Qwen2.5-0.5B-Instruct" from full path)
                base_model = model_path.rstrip('/').split('/')[-1]
            elif session_info.get('model_name'):
                base_model = session_info.get('model_name')
                if '/' in base_model:
                    base_model = base_model.split('/')[-1]
            else:
                # Fallback: Try to read adapter_config.json
                adapter_dir = os.path.join(fusion.base_adapter_dir, adapter_name)
                adapter_config_path = os.path.join(adapter_dir, 'adapter_config.json')
                
                if os.path.exists(adapter_config_path):
                    try:
                        with open(adapter_config_path, 'r') as f:
                            adapter_config = json.load(f)
                            # Try multiple possible keys for base model path
                            base_model_path = (
                                adapter_config.get('base_model_name_or_path') or
                                adapter_config.get('model') or
                                adapter_config.get('base_model')
                            )
                            if base_model_path:
                                base_model = base_model_path.rstrip('/').split('/')[-1]
                    except Exception as e:
                        logger.warning(f"Could not read adapter_config.json for {adapter_name}: {e}")
                
                # Skip fusion_report.txt lookup to avoid recursion/hanging
                # Just use "Fused Model" for fused adapters
                if base_model == 'Unknown Model':
                    fusion_report_path = os.path.join(adapter_dir, 'fusion_report.txt')
                    if os.path.exists(fusion_report_path):
                        base_model = 'Fused Model'
            
            # Get training metadata
            training_date = session_info.get('timestamp')
            iterations = config.get('iterations') or session_info.get('num_iters')
            train_data_path = config.get('train_data_path') or session_info.get('train_data_path')
            dataset = train_data_path.split('/')[-1] if train_data_path else None
            
            # Create adapter info
            adapter_info = AdapterInfo(
                name=adapter_name,
                base_model=base_model,
                training_date=training_date,
                iterations=iterations,
                dataset=dataset,
                path=os.path.join(fusion.base_adapter_dir, adapter_name)
            )
            
            if base_model not in adapters_by_model:
                adapters_by_model[base_model] = []
            adapters_by_model[base_model].append(adapter_info)
        
        # Convert to response format
        result = [
            AdaptersByBaseModel(base_model=model, adapters=adapters)
            for model, adapters in sorted(adapters_by_model.items())
        ]
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing adapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_adapters(adapter_names: List[str]):
    """Validate that selected adapters are compatible for fusion."""
    try:
        if len(adapter_names) < 2:
            return {"compatible": False, "error": "At least 2 adapters required"}
        
        if len(adapter_names) > 5:
            return {"compatible": False, "error": "Maximum 5 adapters allowed"}
        
        fusion = AdapterFusion()
        
        # Load all adapters
        loaded_adapters = []
        for adapter_name in adapter_names:
            try:
                weights = fusion.load_adapter_weights(adapter_name, use_best=True)
                loaded_adapters.append(weights)
            except Exception as e:
                return {"compatible": False, "error": f"Failed to load {adapter_name}: {str(e)}"}
        
        # Validate compatibility
        is_compatible = fusion.validate_adapter_compatibility(loaded_adapters)
        
        if is_compatible:
            # Calculate layer statistics
            all_keys = set()
            for adapter in loaded_adapters:
                all_keys.update(adapter.keys())
            
            common_keys = set(loaded_adapters[0].keys())
            for adapter in loaded_adapters[1:]:
                common_keys &= set(adapter.keys())
            
            return {
                "compatible": True,
                "num_adapters": len(adapter_names),
                "total_layers": len(all_keys),
                "common_layers": len(common_keys),
                "message": f"Adapters are compatible. {len(all_keys)} total layers, {len(common_keys)} common layers. Missing layers will use base model weights."
            }
        else:
            # Try to get more specific error from logs
            error_msg = "Adapters are not compatible. "
            # Check if it's a rank mismatch by examining the adapters
            try:
                ranks = []
                for adapter in loaded_adapters:
                    for key in adapter.keys():
                        if 'lora_b' in key:
                            ranks.append(adapter[key].shape[0])
                            break
                if len(set(ranks)) > 1:
                    error_msg += f"LoRA rank mismatch detected: {set(ranks)}. All adapters must have the same rank."
                else:
                    error_msg += "Incompatible dimensions or layer structure."
            except:
                error_msg += "Incompatible dimensions or keys."
            
            return {
                "compatible": False,
                "error": error_msg
            }
            
    except Exception as e:
        logger.error(f"Error validating adapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fuse")
async def fuse_adapters(request: FusionRequest, background_tasks: BackgroundTasks):
    """Fuse selected adapters and evaluate all variants."""
    global fusion_state
    
    if fusion_state["is_running"]:
        raise HTTPException(status_code=400, detail="Fusion already in progress")
    
    # Validate request
    if len(request.adapter_names) < 2 or len(request.adapter_names) > 5:
        raise HTTPException(status_code=400, detail="Must select 2-5 adapters")
    
    # Start fusion in background
    background_tasks.add_task(run_fusion_and_evaluation, request)
    
    fusion_state["is_running"] = True
    fusion_state["progress"] = 0
    fusion_state["status"] = "starting"
    fusion_state["error"] = None
    fusion_state["current_step"] = "Initializing fusion..."
    
    return {"success": True, "message": "Fusion started"}

async def run_fusion_and_evaluation(request: FusionRequest):
    """Background task to fuse adapters and run evaluations."""
    global fusion_state
    
    try:
        fusion = AdapterFusion()
        
        # Step 1: Load adapters (10% progress)
        fusion_state["current_step"] = "Loading adapters..."
        fusion_state["progress"] = 10
        
        loaded_adapters = []
        for adapter_name in request.adapter_names:
            weights = fusion.load_adapter_weights(adapter_name, use_best=True)
            loaded_adapters.append(weights)
        
        # Step 2: Validate compatibility (20% progress)
        fusion_state["current_step"] = "Validating compatibility..."
        fusion_state["progress"] = 20
        
        is_compatible = fusion.validate_adapter_compatibility(loaded_adapters)
        if not is_compatible:
            raise Exception("Adapters are not compatible")
        
        # Calculate all_keys for fusion
        all_keys = set()
        for adapter in loaded_adapters:
            all_keys.update(adapter.keys())
        
        # Step 3: Perform fusion (30% progress)
        fusion_state["current_step"] = f"Fusing adapters using {request.method}..."
        fusion_state["progress"] = 30
        
        # Set default weights if not provided
        if request.weights is None:
            request.weights = [1.0 / len(loaded_adapters)] * len(loaded_adapters)
        
        # Perform fusion
        if request.method == "slerp" and len(loaded_adapters) == 2:
            t = request.weights[1]
            fused_weights = fusion.slerp_fusion(loaded_adapters[0], loaded_adapters[1], t)
        else:
            fused_weights = fusion.weighted_average_fusion(loaded_adapters, request.weights, all_keys=all_keys)
        
        # Step 4: Save fused adapter (40% progress)
        fusion_state["current_step"] = "Saving fused adapter..."
        fusion_state["progress"] = 40
        
        output_name = request.output_name or f"fused_{'_'.join(request.adapter_names[:2])}"
        output_dir = os.path.join(fusion.base_adapter_dir, output_name)
        fusion.save_fused_adapter(fused_weights, output_dir, output_name, source_adapter_names=request.adapter_names)
        fusion.generate_fusion_report(request.adapter_names, request.weights, request.method, output_dir)
        
        # Step 5: Run evaluations (40-100% progress)
        fusion_state["current_step"] = "Running evaluations..."
        
        evaluation_results = []
        total_evals = len(request.adapter_names) + 2  # individual + base + fused
        
        # Evaluate base model
        fusion_state["current_step"] = "Evaluating base model..."
        fusion_state["progress"] = 45
        base_result = await run_evaluation(request.adapter_names[0], is_base=True, evaluation_dataset=request.evaluation_dataset)
        
        # Evaluate each individual adapter
        for i, adapter_name in enumerate(request.adapter_names):
            progress = 45 + (i + 1) * (40 / total_evals)
            fusion_state["current_step"] = f"Evaluating {adapter_name}..."
            fusion_state["progress"] = int(progress)
            
            result = await run_evaluation(adapter_name, is_base=False, evaluation_dataset=request.evaluation_dataset)
            evaluation_results.append(result)
        
        # Evaluate fused adapter
        fusion_state["current_step"] = "Evaluating fused adapter..."
        fusion_state["progress"] = 90
        fused_result = await run_evaluation(output_name, is_base=False, evaluation_dataset=request.evaluation_dataset)
        
        # Compile results
        fusion_state["result"] = {
            "base_model_result": base_result,
            "individual_results": evaluation_results,
            "fused_result": fused_result,
            "fusion_info": {
                "adapter_names": request.adapter_names,
                "method": request.method,
                "weights": request.weights,
                "output_name": output_name,
                "output_path": output_dir,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        fusion_state["status"] = "completed"
        fusion_state["progress"] = 100
        fusion_state["current_step"] = "Fusion and evaluation completed!"
        
    except Exception as e:
        logger.error(f"Fusion failed: {e}")
        fusion_state["status"] = "error"
        fusion_state["error"] = str(e)
        fusion_state["current_step"] = f"Error: {str(e)}"
    finally:
        fusion_state["is_running"] = False

async def run_evaluation(adapter_name: str, is_base: bool, evaluation_dataset: Optional[str] = None) -> Dict[str, Any]:
    """Run evaluation for a single adapter using the existing evaluation API."""
    import aiohttp
    import asyncio
    
    try:
        # If evaluation_dataset is just a filename, try to resolve it to full path
        resolved_dataset = None
        if evaluation_dataset:
            if os.path.isabs(evaluation_dataset) and os.path.exists(evaluation_dataset):
                # Already a full path
                resolved_dataset = evaluation_dataset
            else:
                # Try to find the file in common locations
                session_info = get_session_info(adapter_name)
                config = session_info.get('config', {})
                train_data_path = config.get('train_data_path') or session_info.get('train_data_path')
                
                if train_data_path:
                    # Get directory and try to find the dataset file
                    data_dir = os.path.dirname(train_data_path)
                    potential_path = os.path.join(data_dir, evaluation_dataset)
                    if os.path.exists(potential_path):
                        resolved_dataset = potential_path
                    else:
                        logger.warning(f"Could not resolve dataset '{evaluation_dataset}' to full path, using as-is")
                        resolved_dataset = evaluation_dataset
        
        # Start evaluation via existing API
        async with aiohttp.ClientSession() as session:
            # Start the evaluation
            async with session.post(
                'http://localhost:8000/api/evaluation/start',
                json={
                    "adapter_name": adapter_name,
                    "training_data_path": resolved_dataset,  # Use resolved dataset or None for default
                    "num_questions": 20,
                    "evaluate_base_model": is_base
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to start evaluation: {await response.text()}")
            
            # Poll for completion
            while True:
                await asyncio.sleep(2)
                async with session.get('http://localhost:8000/api/evaluation/status') as status_response:
                    status = await status_response.json()
                    
                    if not status['running']:
                        if status['error']:
                            raise Exception(status['error'])
                        break
            
            # Get result
            async with session.get('http://localhost:8000/api/evaluation/result') as result_response:
                if result_response.status != 200:
                    raise Exception("Failed to get evaluation result")
                
                result_data = await result_response.json()
                result = result_data['result']
                
                return {
                    "adapter_name": adapter_name,
                    "is_base_model": is_base,
                    "overall_score": result['scores']['overall'],
                    "faithfulness": result['scores']['faithfulness'],
                    "fact_recall": result['scores']['fact_recall'],
                    "consistency": result['scores']['consistency'],
                    "hallucination": result['scores'].get('hallucination', 0)
                }
    
    except Exception as e:
        logger.error(f"Evaluation failed for {adapter_name}: {e}")
        logger.error(f"Full error details:", exc_info=True)
        # Re-raise the exception instead of returning mock data
        raise Exception(f"Evaluation failed for {adapter_name}: {str(e)}")

@router.get("/status")
async def get_fusion_status():
    """Get current fusion operation status."""
    return fusion_state

@router.get("/result")
async def get_fusion_result():
    """Get the result of the last fusion operation."""
    if fusion_state["result"] is None:
        raise HTTPException(status_code=404, detail="No fusion result available")
    
    return fusion_state["result"]

@router.post("/reset")
async def reset_fusion():
    """Reset fusion state."""
    global fusion_state
    fusion_state = {
        "is_running": False,
        "progress": 0,
        "status": "idle",
        "error": None,
        "current_step": "",
        "result": None
    }
    return {"success": True}

# ============================================================================
# MODEL EXPORT ENDPOINTS
# ============================================================================

class ExportRequest(BaseModel):
    adapter_name: str
    format: str  # "merged", "gguf_q4", "gguf_q5", "gguf_q8", "gguf_f16"
    output_name: Optional[str] = None
    use_best: bool = True
    import_to_ollama: bool = False
    ollama_model_name: Optional[str] = None
    copy_to_lm_studio: bool = False

class ExportFormat(BaseModel):
    id: str
    name: str
    description: str
    extension: str
    size: str
    requires: Optional[str] = None

@router.get("/export-formats", response_model=List[ExportFormat])
async def get_export_formats():
    """Get available export formats."""
    exporter = ModelExporter()
    formats = exporter.get_export_formats()
    return formats

@router.post("/export-model")
async def export_model(request: ExportRequest, background_tasks: BackgroundTasks):
    """Export an adapter (merged with base model) in various formats."""
    global export_state
    
    if export_state["is_running"]:
        raise HTTPException(status_code=400, detail="Export already in progress")
    
    # Start export in background
    background_tasks.add_task(run_model_export, request)
    
    export_state["is_running"] = True
    export_state["progress"] = 0
    export_state["status"] = "starting"
    export_state["error"] = None
    export_state["current_step"] = "Initializing export..."
    export_state["result"] = None
    
    return {"success": True, "message": "Export started"}

async def run_model_export(request: ExportRequest):
    """Background task to export model."""
    global export_state
    
    def progress_callback(progress: int, message: str):
        export_state["progress"] = progress
        export_state["current_step"] = message
    
    try:
        exporter = ModelExporter()
        
        # Step 1: Merge adapter with base model
        export_state["current_step"] = "Merging adapter with base model..."
        export_state["progress"] = 10
        
        merged_dir = exporter.merge_adapter_with_base(
            adapter_name=request.adapter_name,
            use_best=request.use_best,
            progress_callback=progress_callback
        )
        
        result = {
            "adapter_name": request.adapter_name,
            "format": request.format,
            "merged_dir": merged_dir,
            "files": []
        }
        
        # Step 2: Convert to requested format
        if request.format.startswith("gguf_"):
            quantization_map = {
                "gguf_q4": "Q4_K_M",
                "gguf_q5": "Q5_K_M",
                "gguf_q8": "Q8_0",
                "gguf_f16": "F16"
            }
            
            quantization = quantization_map.get(request.format, "Q4_K_M")
            
            export_state["current_step"] = f"Converting to GGUF ({quantization})..."
            export_state["progress"] = 50
            
            gguf_path = exporter.convert_to_gguf(
                model_path=merged_dir,
                quantization=quantization,
                progress_callback=progress_callback
            )
            
            result["gguf_path"] = gguf_path
            result["files"].append(gguf_path)
            
            # Step 3: Generate Ollama Modelfile
            export_state["current_step"] = "Generating Ollama Modelfile..."
            export_state["progress"] = 90
            
            ollama_name = request.ollama_model_name or f"{request.adapter_name.lower().replace('_', '-')}"
            
            modelfile_path = exporter.generate_ollama_modelfile(
                model_path=gguf_path,
                model_name=ollama_name
            )
            
            result["modelfile_path"] = modelfile_path
            result["ollama_model_name"] = ollama_name
            
            # Step 4: Copy to LM Studio if requested
            if request.copy_to_lm_studio:
                export_state["current_step"] = "Copying to LM Studio..."
                export_state["progress"] = 93
                
                lm_studio_path = exporter.copy_to_lm_studio(gguf_path, request.adapter_name)
                result["lm_studio_path"] = lm_studio_path
            
            # Step 5: Import to Ollama if requested
            if request.import_to_ollama:
                export_state["current_step"] = "Importing to Ollama..."
                export_state["progress"] = 96
                
                import_result = exporter.import_to_ollama(modelfile_path, ollama_name)
                result["ollama_import"] = import_result
        
        else:
            # Just merged format
            result["files"].append(os.path.join(merged_dir, "model.safetensors"))
        
        export_state["result"] = result
        export_state["status"] = "completed"
        export_state["progress"] = 100
        export_state["current_step"] = "Export completed successfully!"
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        export_state["status"] = "error"
        export_state["error"] = str(e)
        export_state["current_step"] = f"Error: {str(e)}"
    finally:
        export_state["is_running"] = False

@router.get("/export-status")
async def get_export_status():
    """Get current export operation status."""
    return export_state

@router.get("/export-result")
async def get_export_result():
    """Get the result of the last export operation."""
    if export_state["result"] is None:
        raise HTTPException(status_code=404, detail="No export result available")
    
    return export_state["result"]

@router.post("/export-reset")
async def reset_export():
    """Reset export state."""
    global export_state
    export_state = {
        "is_running": False,
        "progress": 0,
        "status": "idle",
        "error": None,
        "current_step": "",
        "result": None
    }
    return {"success": True}
