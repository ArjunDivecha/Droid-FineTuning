"""
Utility functions for On-Policy Distillation
"""

import os
import sys
import psutil
import logging
from pathlib import Path
from typing import Dict, Optional
import json

logger = logging.getLogger(__name__)


def get_memory_usage_gb() -> Dict[str, float]:
    """
    Get current system memory usage in GB.

    Returns:
        Dictionary with 'used', 'available', 'percent', 'total'
    """
    memory = psutil.virtual_memory()

    return {
        'used': memory.used / (1024 ** 3),
        'available': memory.available / (1024 ** 3),
        'percent': memory.percent,
        'total': memory.total / (1024 ** 3)
    }


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "./OnPolicyDistill/logs"
):
    """
    Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file name
        log_dir: Directory for log files
    """
    # Create log directory
    if log_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / log_file
    else:
        log_file_path = None

    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file_path:
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=handlers
    )

    logger.info(f"Logging configured (level={logging.getLevelName(level)})")
    if log_file_path:
        logger.info(f"Log file: {log_file_path}")


def validate_paths(config: 'OPDConfig') -> bool:
    """
    Validate all paths in configuration exist.

    Args:
        config: OPD configuration

    Returns:
        True if all paths valid, False otherwise
    """
    paths_to_check = {
        'base_model_path': config.base_model_path,
        'teacher_model_path': config.teacher_model_path,
        'student_adapter_path': config.student_adapter_path,
        'validation_prompts_path': config.validation_prompts_path
    }

    all_valid = True

    for name, path in paths_to_check.items():
        if not Path(path).exists():
            logger.error(f"{name} does not exist: {path}")
            all_valid = False
        else:
            logger.info(f"✓ {name}: {path}")

    return all_valid


def estimate_memory_requirements(config: 'OPDConfig') -> Dict[str, float]:
    """
    Estimate memory requirements for distillation.

    Returns:
        Dictionary with estimated memory usage in GB
    """
    # Rough estimates (in GB)
    estimates = {}

    # Teacher model (depends on quantization)
    # Assume 4-bit quantized 32B model
    estimates['teacher_model'] = 16.0

    # Student model (7B FP16)
    estimates['student_model'] = 14.0

    # Gradients (LoRA only, much smaller)
    estimates['gradients'] = 2.0

    # Activations (depends on batch size)
    estimates['activations'] = config.batch_size * 1.0

    # Cache (if enabled)
    if config.use_cache:
        estimates['cache'] = config.cache_size_mb / 1024

    # Total
    estimates['total_estimated'] = sum(estimates.values())

    # Current system
    memory_info = get_memory_usage_gb()
    estimates['system_available'] = memory_info['available']
    estimates['system_total'] = memory_info['total']

    # Check if sufficient
    estimates['sufficient'] = estimates['system_available'] > estimates['total_estimated']

    return estimates


def print_memory_report(estimates: Dict[str, float]):
    """Print formatted memory requirements report"""
    logger.info("="*60)
    logger.info("Memory Requirements Estimate")
    logger.info("="*60)
    logger.info(f"  Teacher model:      {estimates.get('teacher_model', 0):.1f} GB")
    logger.info(f"  Student model:      {estimates.get('student_model', 0):.1f} GB")
    logger.info(f"  Gradients:          {estimates.get('gradients', 0):.1f} GB")
    logger.info(f"  Activations:        {estimates.get('activations', 0):.1f} GB")
    logger.info(f"  Cache:              {estimates.get('cache', 0):.1f} GB")
    logger.info(f"  -" * 30)
    logger.info(f"  Total estimated:    {estimates.get('total_estimated', 0):.1f} GB")
    logger.info(f"")
    logger.info(f"  System total:       {estimates.get('system_total', 0):.1f} GB")
    logger.info(f"  System available:   {estimates.get('system_available', 0):.1f} GB")
    logger.info(f"")

    if estimates.get('sufficient', False):
        logger.info(f"  ✓ Sufficient memory available")
    else:
        logger.warning(f"  ⚠ May not have sufficient memory!")

    logger.info("="*60)


def save_config(config: 'OPDConfig', output_path: str):
    """
    Save configuration to file.

    Args:
        config: OPD configuration
        output_path: Where to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config.save(str(output_path))
    logger.info(f"Configuration saved to {output_path}")


def load_config(config_path: str) -> 'OPDConfig':
    """
    Load configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        OPD configuration
    """
    from .config import OPDConfig

    config = OPDConfig.load(config_path)
    logger.info(f"Configuration loaded from {config_path}")

    return config


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_alignment_score(
    student_tokens: list,
    teacher_tokens: list
) -> float:
    """
    Compute simple token alignment score.

    Args:
        student_tokens: Student generated tokens
        teacher_tokens: Teacher generated tokens

    Returns:
        Alignment score (0.0 to 1.0)
    """
    if not student_tokens or not teacher_tokens:
        return 0.0

    # Simple exact match ratio
    matches = sum(1 for s, t in zip(student_tokens, teacher_tokens) if s == t)
    max_len = max(len(student_tokens), len(teacher_tokens))

    return matches / max_len if max_len > 0 else 0.0


def get_git_revision() -> Optional[str]:
    """Get current git revision hash"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None


def create_run_manifest(config: 'OPDConfig', output_path: str):
    """
    Create a manifest file with all run information.

    Args:
        config: OPD configuration
        output_path: Where to save manifest
    """
    from datetime import datetime

    manifest = {
        'run_id': config.run_id,
        'timestamp': datetime.now().isoformat(),
        'git_revision': get_git_revision(),
        'config': config.to_dict(),
        'system': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Run manifest saved to {output_path}")
