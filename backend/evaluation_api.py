"""
Backend API endpoints for adapter evaluation.
"""

import sys
import os
sys.path.append('/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/adapter_fusion')

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging

# Import the evaluator
from evaluate_adapters import AdapterEvaluator

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

class EvaluationRequest(BaseModel):
    adapter_name: str
    training_data_path: Optional[str] = None
    num_questions: int = 20

@router.post("/api/evaluation/start")
async def start_evaluation(request: EvaluationRequest):
    """Start adapter evaluation."""
    global evaluation_status
    
    if evaluation_status["running"]:
        raise HTTPException(status_code=400, detail="Evaluation already running")
    
    try:
        # Reset status
        evaluation_status = {
            "running": True,
            "progress": 0,
            "current_question": 0,
            "total_questions": request.num_questions,
            "adapter_name": request.adapter_name,
            "result": None,
            "error": None
        }
        
        # Start evaluation in background
        asyncio.create_task(run_evaluation(
            request.adapter_name,
            request.training_data_path,
            request.num_questions
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
    """Get current evaluation status."""
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
    """Get evaluation result."""
    if evaluation_status["result"] is None:
        raise HTTPException(status_code=404, detail="No evaluation result available")
    
    return {
        "success": True,
        "result": evaluation_status["result"]
    }

async def run_evaluation(adapter_name: str, training_data_path: Optional[str], num_questions: int):
    """Run evaluation in background."""
    global evaluation_status
    
    try:
        evaluator = AdapterEvaluator()
        
        # Get training data path if not provided
        if not training_data_path:
            config = evaluator.load_adapter_config(adapter_name)
            training_data_path = config.get('data', '')
            
            # If it's a directory, look for JSONL files
            if os.path.isdir(training_data_path):
                import glob
                jsonl_files = glob.glob(os.path.join(training_data_path, "*.jsonl"))
                if jsonl_files:
                    training_data_path = jsonl_files[0]
        
        if not training_data_path or not os.path.exists(training_data_path):
            raise ValueError("Training data path not found")
        
        # Run evaluation
        logger.info(f"Starting evaluation: {adapter_name}")
        result = evaluator.evaluate_adapter(adapter_name, training_data_path, num_questions)
        
        # Update status
        evaluation_status["running"] = False
        evaluation_status["progress"] = 100
        evaluation_status["result"] = result
        
        logger.info(f"Evaluation complete: {result['scores']['overall']}/100")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        evaluation_status["running"] = False
        evaluation_status["error"] = str(e)

def setup_evaluation_routes(app):
    """Setup evaluation routes on FastAPI app."""
    app.include_router(router)
    logger.info("Evaluation API routes registered")
