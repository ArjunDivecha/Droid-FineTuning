"""
Backend Factory for MLX Fine-Tuning GUI
Supports switching between Metal (local) and CUDA (cloud) backends
"""

import os
import mlx.core as mx
import logging

logger = logging.getLogger(__name__)

class BackendFactory:
    """Factory class for managing MLX backends"""
    
    @staticmethod
    def get_backend():
        """Get the appropriate MLX backend based on environment variable"""
        backend_type = os.getenv("MLX_BACKEND", "metal").lower()
        
        if backend_type == "cuda":
            try:
                mx.set_default_device(mx.cuda)
                logger.info("MLX backend set to CUDA")
                return mx
            except Exception as e:
                logger.error(f"Failed to set CUDA backend: {e}")
                # Fallback to Metal if CUDA fails
                mx.set_default_device(mx.metal)
                logger.info("Falling back to Metal backend")
                return mx
        else:
            # Default to Metal for local development
            mx.set_default_device(mx.metal)
            logger.info("MLX backend set to Metal")
            return mx
    
    @staticmethod
    def is_cuda_available():
        """Check if CUDA backend is available"""
        try:
            # Try to set CUDA backend temporarily
            current_device = mx.default_device()
            mx.set_default_device(mx.cuda)
            mx.set_default_device(current_device)  # Restore original
            return True
        except:
            return False
    
    @staticmethod
    def get_current_backend():
        """Get string representation of current backend"""
        device = mx.default_device()
        if device.type == mx.DeviceType.cuda:
            return "cuda"
        elif device.type == mx.DeviceType.metal:
            return "metal"
        else:
            return "cpu"

# Global instance
backend_factory = BackendFactory()
