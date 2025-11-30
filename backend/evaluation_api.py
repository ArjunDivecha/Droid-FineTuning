"""
Backend API endpoints for adapter evaluation.

This module provides REST API endpoints for evaluating the performance of trained adapters
against their training data. It measures faithfulness, fact recall, consistency, and 
hallucination rates using an external evaluation model (Cerebras).

The evaluation process:
1. Loads training data used for the adapter
2. Generates responses from the adapter for test questions
3. Evaluates those responses using Cerebras for quality metrics
4. Provides detailed scoring and analysis

Routes:
- POST /api/evaluation/start - Start evaluation of an adapter
- GET /api/evaluation/status - Get current evaluation progress and status
- GET /api/evaluation/result - Get evaluation results after completion
"""

import sys
import os
sys.path.append('/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/adapter_fusion')

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

# Import the evaluator
from evaluate_adapters import AdapterEvaluator

# Thread pool for running blocking evaluation
executor = ThreadPoolExecutor(max_workers=1)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for evaluation progress
evaluation_status = {
    "running": False,
    "progress": 0,
    "current_question": 0,
    "total_questions": 0,
    "adapter_name": "",
    "result": None,
    "error": None
}

class ComparisonRequest(BaseModel):
    """Request model for comparing base model vs adapter."""
    adapter_name: str
    training_data_path: Optional[str] = None
    num_questions: int = 20

class EvaluationRequest(BaseModel):
    """
    Request model for starting an adapter evaluation.
    
    Attributes:
        adapter_name (str): Name of the adapter to evaluate
        training_data_path (Optional[str]): Path to training data file. If not provided,
                                          the system will try to find it from session files
                                          or adapter configuration.
        num_questions (int): Number of questions to evaluate (default: 20)
        evaluate_base_model (bool): Whether to evaluate the base model instead of the adapter (default: False)
    """
    adapter_name: str
    training_data_path: Optional[str] = None
    num_questions: int = 20
    evaluate_base_model: bool = False

@router.post("/api/evaluation/start")
async def start_evaluation(request: EvaluationRequest):
    """
    Start adapter evaluation asynchronously.
    
    Args:
        request (EvaluationRequest): Evaluation parameters
        
    Returns:
        dict: Success message with adapter name
        
    Raises:
        HTTPException: If evaluation is already running (400) or 
                      if there's an error starting evaluation (500)
                      
    Example:
        POST /api/evaluation/start
        {
            "adapter_name": "my_adapter",
            "num_questions": 20,
            "evaluate_base_model": false
        }
        
        Response:
        {
            "success": true,
            "message": "Evaluation started",
            "adapter_name": "my_adapter"
        }
    """
    global evaluation_status
    
    if evaluation_status["running"]:
        raise HTTPException(status_code=400, detail="Evaluation already running")
    
    try:
        # Reset status
        evaluation_status["running"] = True
        evaluation_status["progress"] = 0
        evaluation_status["current_question"] = 0
        evaluation_status["total_questions"] = request.num_questions
        evaluation_status["adapter_name"] = request.adapter_name
        evaluation_status["result"] = None
        evaluation_status["error"] = None
        
        # Start evaluation in background
        asyncio.create_task(run_evaluation(
            request.adapter_name,
            request.training_data_path,
            request.num_questions,
            request.evaluate_base_model
        ))
        
        return {
            "success": True,
            "message": "Evaluation started",
            "adapter_name": request.adapter_name
        }
    
    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}")
        evaluation_status["running"] = False
        evaluation_status["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/evaluation/status")
async def get_evaluation_status():
    """
    Get current evaluation status and progress.
    
    Returns:
        dict: Current evaluation metrics
        
    Example:
        GET /api/evaluation/status
        
        Response:
        {
            "running": true,
            "progress": 45,
            "current_question": 9,
            "total_questions": 20,
            "adapter_name": "my_adapter",
            "error": null
        }
    """
    return {
        "running": evaluation_status["running"],
        "progress": evaluation_status["progress"],
        "current_question": evaluation_status["current_question"],
        "total_questions": evaluation_status["total_questions"],
        "adapter_name": evaluation_status["adapter_name"],
        "error": evaluation_status["error"]
    }

@router.get("/api/evaluation/result")
async def get_evaluation_result():
    """
    Get evaluation result after completion.
    
    Returns:
        dict: Evaluation results with scores and detailed analysis
        
    Raises:
        HTTPException: If no evaluation result is available (404)
        
    Example:
        GET /api/evaluation/result
        
        Response:
        {
            "success": true,
            "result": {
                "adapter_name": "my_adapter",
                "is_base_model": false,
                "adapter_config": {...},
                "training_data_path": "/path/to/train.jsonl",
                "num_questions": 20,
                "evaluation_date": "2025-01-29T10:30:00",
                "scores": {
                    "overall": 85.5,
                    "faithfulness": 90.0,
                    "fact_recall": 80.0,
                    "consistency": 85.0,
                    "hallucination": 95.0
                },
                "detailed_results": [...]
            }
        }
    """
    if evaluation_status["result"] is None:
        raise HTTPException(status_code=404, detail="No evaluation result available")
    
    return {
        "success": True,
        "result": evaluation_status["result"]
    }

@router.post("/api/evaluation/compare")
async def start_comparison(request: ComparisonRequest):
    """Start comparison evaluation (base model vs adapter on same questions)."""
    global evaluation_status
    
    if evaluation_status["running"]:
        raise HTTPException(status_code=400, detail="Evaluation already running")
    
    try:
        # Reset status
        evaluation_status["running"] = True
        evaluation_status["progress"] = 0
        evaluation_status["current_question"] = 0
        evaluation_status["total_questions"] = request.num_questions
        evaluation_status["adapter_name"] = request.adapter_name
        evaluation_status["result"] = None
        evaluation_status["error"] = None
        
        # Start comparison in background
        asyncio.create_task(run_comparison(
            request.adapter_name,
            request.training_data_path,
            request.num_questions
        ))
        
        return {
            "success": True,
            "message": "Comparison started",
            "adapter_name": request.adapter_name
        }
    
    except Exception as e:
        logger.error(f"Failed to start comparison: {e}")
        evaluation_status["running"] = False
        evaluation_status["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

async def run_comparison(adapter_name: str, training_data_path: Optional[str], num_questions: int):
    """Run comparison in background thread."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, run_comparison_sync, adapter_name, training_data_path, num_questions)

def run_comparison_sync(adapter_name: str, training_data_path: Optional[str], num_questions: int):
    """Synchronous comparison function."""
    global evaluation_status
    
    try:
        evaluator = AdapterEvaluator()
        
        # Get training data path if not provided
        if not training_data_path:
            try:
                session_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
                import glob
                import json
                
                session_files = glob.glob(os.path.join(session_dir, "session_*.json"))
                for session_file in session_files:
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                            if session_data.get('adapter_name') == adapter_name:
                                training_data_path = session_data.get('config', {}).get('train_data_path', '')
                                if training_data_path:
                                    logger.info(f"Found original training data from session: {training_data_path}")
                                    break
                    except:
                        continue
                
                if not training_data_path:
                    config = evaluator.load_adapter_config(adapter_name)
                    training_data_path = config.get('data', '')
                    
                    if training_data_path and os.path.isdir(training_data_path):
                        jsonl_files = glob.glob(os.path.join(training_data_path, "*.jsonl"))
                        if jsonl_files:
                            training_data_path = jsonl_files[0]
            except Exception as e:
                logger.error(f"Failed to load training data path: {e}")
                raise ValueError(f"Could not load training data path: {e}")
        
        if not training_data_path or not os.path.exists(training_data_path):
            raise ValueError(f"Training data path not found: {training_data_path}")
        
        logger.info(f"Using training data: {training_data_path}")
        
        # Progress callback
        def update_progress(current, total, progress):
            evaluation_status["current_question"] = current
            evaluation_status["total_questions"] = total
            evaluation_status["progress"] = progress
            logger.info(f"Progress update: {current}/{total} ({progress}%)")
        
        # Run comparison
        logger.info(f"Starting comparison: {adapter_name}")
        result = evaluator.compare_base_vs_adapter(adapter_name, training_data_path, num_questions, progress_callback=update_progress)
        
        # Save comparison report
        output_dir = "./evaluation_results"
        json_file, txt_file = evaluator.save_comparison_report(result, output_dir)
        
        # Add file paths to result
        result['report_files'] = {
            'json': os.path.abspath(json_file),
            'text': os.path.abspath(txt_file)
        }
        
        # Update status
        evaluation_status["running"] = False
        evaluation_status["progress"] = 100
        evaluation_status["result"] = result
        
        logger.info(f"Comparison complete. Base: {result['base_model_scores']['overall']}/100, Adapter: {result['adapter_scores']['overall']}/100")
        logger.info(f"Reports saved: {json_file}, {txt_file}")
    
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        evaluation_status["running"] = False
        evaluation_status["error"] = str(e)

def run_evaluation_sync(adapter_name: str, training_data_path: Optional[str], num_questions: int, evaluate_base_model: bool = False):
    """
    Synchronous evaluation function to run in thread pool.
    
    This function performs the actual evaluation work in a separate thread to avoid 
    blocking the main event loop.
    
    Args:
        adapter_name (str): Name of the adapter to evaluate
        training_data_path (Optional[str]): Path to training data file
        num_questions (int): Number of questions to evaluate
        evaluate_base_model (bool): Whether to evaluate base model instead of adapter
    """
    global evaluation_status
    
    try:
        evaluator = AdapterEvaluator()
        
        # Get training data path if not provided
        if not training_data_path:
            try:
                # First try to get original training data from session
                session_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
                import glob
                import json
                
                # Find session file for this adapter
                session_files = glob.glob(os.path.join(session_dir, "session_*.json"))
                for session_file in session_files:
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                            if session_data.get('adapter_name') == adapter_name:
                                training_data_path = session_data.get('config', {}).get('train_data_path', '')
                                if training_data_path:
                                    logger.info(f"Found original training data from session: {training_data_path}")
                                    break
                    except:
                        continue
                
                # Fallback to adapter config if session not found
                if not training_data_path:
                    config = evaluator.load_adapter_config(adapter_name)
                    training_data_path = config.get('data', '')
                    
                    # If it's a directory, look for JSONL files
                    if training_data_path and os.path.isdir(training_data_path):
                        jsonl_files = glob.glob(os.path.join(training_data_path, "*.jsonl"))
                        if jsonl_files:
                            training_data_path = jsonl_files[0]
                            logger.info(f"Using train.jsonl from adapter config (original dataset unknown)")
            except Exception as e:
                logger.error(f"Failed to load training data path: {e}")
                raise ValueError(f"Could not load training data path: {e}")
        
        if not training_data_path or not os.path.exists(training_data_path):
            raise ValueError(f"Training data path not found: {training_data_path}")
        
        logger.info(f"Using training data: {training_data_path}")
        
        # Progress callback
        def update_progress(current, total, progress):
            """
            Update evaluation progress status.
            
            Args:
                current (int): Current question number
                total (int): Total number of questions
                progress (int): Progress percentage
            """
            evaluation_status["current_question"] = current
            evaluation_status["total_questions"] = total
            evaluation_status["progress"] = progress
            logger.info(f"Progress update: {current}/{total} ({progress}%)")
        
        # Run evaluation
        eval_type = "base model" if evaluate_base_model else "adapter"
        logger.info(f"Starting evaluation: {adapter_name} ({eval_type})")
        result = evaluator.evaluate_adapter(adapter_name, training_data_path, num_questions, progress_callback=update_progress, use_base_model=evaluate_base_model)
        
        # Save report to disk
        output_dir = "./evaluation_results"
        json_file, summary_file = evaluator.save_report(result, output_dir)
        logger.info(f"Report saved: {json_file}")
        
        # Update status
        evaluation_status["running"] = False
        evaluation_status["progress"] = 100
        evaluation_status["result"] = result
        
        logger.info(f"Evaluation complete: {result['scores']['overall']}/100")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        evaluation_status["running"] = False
        evaluation_status["error"] = str(e)

async def run_evaluation(adapter_name: str, training_data_path: Optional[str], num_questions: int, evaluate_base_model: bool = False):
    """
    Run evaluation in background thread.
    
    Args:
        adapter_name (str): Name of the adapter to evaluate
        training_data_path (Optional[str]): Path to training data file
        num_questions (int): Number of questions to evaluate
        evaluate_base_model (bool): Whether to evaluate base model instead of adapter
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, run_evaluation_sync, adapter_name, training_data_path, num_questions, evaluate_base_model)

def setup_evaluation_routes(app):
    """
    Setup evaluation routes on FastAPI app.
    
    Args:
        app (FastAPI): FastAPI application instance
    """
    app.include_router(router)
    logger.info("Evaluation API routes registered")
