"""
=============================================================================
SCRIPT NAME: model_export.py
=============================================================================

DESCRIPTION:
Handles exporting and merging LoRA adapters with base models for use in
Ollama, LM Studio, and other inference platforms.

Supports:
- Merging LoRA adapters with base models using mlx_lm.fuse
- Converting to GGUF format for Ollama/LM Studio
- Generating Ollama Modelfiles
- Quantization options

VERSION: 1.0
LAST UPDATED: 2025-09-30
=============================================================================
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelExporter:
    """Handles model export and conversion for various inference platforms."""
    
    def __init__(self, 
                 adapter_base_dir: str = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters",
                 export_base_dir: str = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/exported_models"):
        self.adapter_base_dir = adapter_base_dir
        self.export_base_dir = export_base_dir
        self.lm_studio_dir = os.path.expanduser("~/.cache/lm-studio/models")
        os.makedirs(export_base_dir, exist_ok=True)
    
    def get_adapter_base_model(self, adapter_name: str) -> Optional[str]:
        """Get the base model path for an adapter by reading its session file or fusion report."""
        adapter_dir = os.path.join(self.adapter_base_dir, adapter_name)
        
        # Check if this is a fused adapter
        fusion_report_path = os.path.join(adapter_dir, "fusion_report.txt")
        if os.path.exists(fusion_report_path):
            logger.info(f"Detected fused adapter, reading fusion report")
            try:
                with open(fusion_report_path, 'r') as f:
                    content = f.read()
                    # Look for source adapters in the report
                    if 'Source Adapters:' in content:
                        lines = content.split('\n')
                        for line in lines:
                            # Find first source adapter (e.g., "1. adapter_name (weight: 0.5000)")
                            if line.strip().startswith('1.'):
                                # Extract adapter name
                                source_adapter = line.split('(')[0].strip().split('.', 1)[1].strip()
                                logger.info(f"Found source adapter: {source_adapter}")
                                # Recursively get base model of source adapter
                                return self.get_adapter_base_model(source_adapter)
            except Exception as e:
                logger.warning(f"Error reading fusion report: {e}")
        
        # Try to read from session file
        sessions_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
        
        if os.path.exists(sessions_dir):
            for session_file in os.listdir(sessions_dir):
                if session_file.endswith('.json'):
                    session_path = os.path.join(sessions_dir, session_file)
                    try:
                        with open(session_path, 'r') as f:
                            session_data = json.load(f)
                            config = session_data.get('config', {})
                            config_adapter = config.get('adapter_name')
                            direct_adapter = session_data.get('adapter_name')
                            
                            if config_adapter == adapter_name or direct_adapter == adapter_name:
                                model_path = config.get('model_path') or session_data.get('model_path')
                                return model_path
                    except Exception as e:
                        logger.warning(f"Error reading session {session_file}: {e}")
        
        # Fallback: try reading adapter_config.json in the adapter directory
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model_path = (
                        adapter_config.get('base_model_name_or_path') or
                        adapter_config.get('model') or
                        adapter_config.get('base_model')
                    )
                    if base_model_path:
                        logger.info(f"Found base model in adapter_config.json: {base_model_path}")
                        return base_model_path
            except Exception as e:
                logger.warning(f"Error reading adapter_config.json: {e}")
        
        return None
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """Get information about an adapter including base model."""
        adapter_dir = os.path.join(self.adapter_base_dir, adapter_name)
        
        if not os.path.exists(adapter_dir):
            raise FileNotFoundError(f"Adapter not found: {adapter_name}")
        
        base_model = self.get_adapter_base_model(adapter_name)
        
        # Check for adapter files
        has_best = os.path.exists(os.path.join(adapter_dir, "best_adapters.safetensors"))
        has_latest = os.path.exists(os.path.join(adapter_dir, "adapters.safetensors"))
        
        return {
            "adapter_name": adapter_name,
            "adapter_path": adapter_dir,
            "base_model": base_model,
            "has_best": has_best,
            "has_latest": has_latest,
            "is_fused": os.path.exists(os.path.join(adapter_dir, "fusion_report.txt"))
        }
    
    def merge_adapter_with_base(self, 
                                adapter_name: str, 
                                output_dir: Optional[str] = None,
                                use_best: bool = True,
                                progress_callback: Optional[callable] = None) -> str:
        """
        Merge LoRA adapter with base model to create a standalone model.
        Uses mlx_lm.fuse functionality.
        """
        logger.info(f"Starting merge for adapter: {adapter_name}")
        
        # Get adapter info
        adapter_info = self.get_adapter_info(adapter_name)
        adapter_path = adapter_info["adapter_path"]
        base_model = adapter_info["base_model"]
        
        if not base_model:
            raise ValueError(f"Cannot determine base model for adapter: {adapter_name}")
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.export_base_dir, f"{adapter_name}_merged_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which adapter file to use
        adapter_file = "best_adapters.safetensors" if use_best and adapter_info["has_best"] else "adapters.safetensors"
        
        logger.info(f"Merging adapter: {adapter_path}/{adapter_file}")
        logger.info(f"With base model: {base_model}")
        logger.info(f"Output directory: {output_dir}")
        
        if progress_callback:
            progress_callback(10, "Starting merge process...")
        
        # Build fuse command using mlx_lm fuse subcommand (new style)
        cmd = [
            "python3", "-m", "mlx_lm", "fuse",
            "--model", base_model,
            "--adapter-path", adapter_path,
            "--save-path", output_dir,
            "--de-quantize"  # De-quantize if base model is quantized
        ]
        
        try:
            if progress_callback:
                progress_callback(20, "Running merge command...")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Merge output: {result.stdout}")
            
            if progress_callback:
                progress_callback(90, "Merge completed, saving metadata...")
            
            # Save merge metadata
            metadata = {
                "adapter_name": adapter_name,
                "base_model": base_model,
                "adapter_file": adapter_file,
                "merged_at": datetime.now().isoformat(),
                "output_dir": output_dir
            }
            
            metadata_path = os.path.join(output_dir, "merge_info.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if progress_callback:
                progress_callback(100, "Merge completed successfully!")
            
            logger.info(f"Successfully merged adapter to: {output_dir}")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Merge failed: {e.stderr}")
            raise Exception(f"Failed to merge adapter: {e.stderr}")
    
    def convert_to_gguf(self,
                       model_path: str,
                       output_path: Optional[str] = None,
                       quantization: str = "Q4_K_M",
                       progress_callback: Optional[callable] = None) -> str:
        """
        Convert merged model to GGUF format for Ollama/LM Studio.
        
        Quantization options:
        - Q4_K_M: 4-bit medium (recommended, good balance)
        - Q5_K_M: 5-bit medium (better quality, larger size)
        - Q8_0: 8-bit (best quality, largest size)
        - F16: 16-bit float (unquantized, very large)
        """
        logger.info(f"Converting model to GGUF: {model_path}")
        
        if progress_callback:
            progress_callback(10, "Preparing GGUF conversion...")
        
        # Create output path
        if output_path is None:
            model_name = os.path.basename(model_path.rstrip('/'))
            output_path = os.path.join(self.export_base_dir, f"{model_name}_{quantization}.gguf")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # First convert to GGUF FP16
        if progress_callback:
            progress_callback(20, "Converting to GGUF FP16...")
        
        temp_fp16_path = output_path.replace(".gguf", "_fp16.gguf")
        
        # Try using llama.cpp convert script if available
        convert_script = self._find_llama_cpp_convert_script()
        
        if convert_script:
            logger.info(f"Using llama.cpp convert script: {convert_script}")
            cmd = [
                "python3", convert_script,
                model_path,
                "--outfile", temp_fp16_path,
                "--outtype", "f16"
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"FP16 conversion failed: {e.stderr}")
                raise Exception(f"Failed to convert to GGUF FP16: {e.stderr}")
        else:
            # Fallback: Try using direct Python conversion
            logger.warning("llama.cpp convert script not found, using alternative method")
            self._convert_to_gguf_alternative(model_path, temp_fp16_path)
        
        # Quantize if needed
        if quantization != "F16":
            if progress_callback:
                progress_callback(60, f"Quantizing to {quantization}...")
            
            quantize_cmd = self._find_llama_cpp_quantize()
            
            if quantize_cmd:
                cmd = [quantize_cmd, temp_fp16_path, output_path, quantization]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    # Remove temp FP16 file
                    if os.path.exists(temp_fp16_path):
                        os.remove(temp_fp16_path)
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Quantization failed: {e.stderr}")
                    raise Exception(f"Failed to quantize: {e.stderr}")
            else:
                logger.warning("llama.cpp quantize not found, keeping FP16 version")
                output_path = temp_fp16_path
        else:
            output_path = temp_fp16_path
        
        if progress_callback:
            progress_callback(100, "GGUF conversion completed!")
        
        logger.info(f"Successfully converted to GGUF: {output_path}")
        return output_path
    
    def _find_llama_cpp_convert_script(self) -> Optional[str]:
        """Find llama.cpp convert script."""
        possible_paths = [
            "/opt/homebrew/bin/convert_hf_to_gguf.py",
            "/usr/local/bin/convert_hf_to_gguf.py",
            "/opt/homebrew/opt/llama.cpp/bin/convert_hf_to_gguf.py",
            os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
            os.path.expanduser("~/llama.cpp/convert.py"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try which command
        try:
            result = subprocess.run(["which", "convert_hf_to_gguf.py"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def _find_llama_cpp_quantize(self) -> Optional[str]:
        """Find llama.cpp quantize binary."""
        possible_paths = [
            "/opt/homebrew/bin/llama-quantize",
            "/usr/local/bin/llama-quantize",
            "/opt/homebrew/bin/quantize",
            "/usr/local/bin/quantize",
            os.path.expanduser("~/llama.cpp/llama-quantize"),
            os.path.expanduser("~/llama.cpp/quantize"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try which command
        try:
            result = subprocess.run(["which", "llama-quantize"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        try:
            result = subprocess.run(["which", "quantize"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def _convert_to_gguf_alternative(self, model_path: str, output_path: str):
        """Alternative GGUF conversion method using Python."""
        logger.info("Using alternative GGUF conversion method")
        
        # This is a placeholder - you would need to implement actual conversion
        # For now, we'll just create a note that conversion failed
        raise NotImplementedError(
            "llama.cpp tools not found. Please install llama.cpp to use GGUF conversion.\n"
            "Instructions: brew install llama.cpp OR git clone https://github.com/ggerganov/llama.cpp"
        )
    
    def generate_ollama_modelfile(self,
                                  model_path: str,
                                  model_name: str,
                                  output_path: Optional[str] = None,
                                  template: Optional[str] = None,
                                  system_prompt: Optional[str] = None) -> str:
        """Generate Ollama Modelfile for importing the model."""
        
        if output_path is None:
            output_path = os.path.join(os.path.dirname(model_path), "Modelfile")
        
        # Default template for Qwen models
        if template is None:
            template = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        # Generate Modelfile
        modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"{template}\"\"\"

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
"""
        
        with open(output_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Generated Ollama Modelfile: {output_path}")
        
        # Generate import instructions
        instructions_path = output_path.replace("Modelfile", "ollama_import_instructions.txt")
        with open(instructions_path, 'w') as f:
            f.write(f"""How to import this model into Ollama:

1. Open Terminal

2. Run the following command:
   cd {os.path.dirname(output_path)}
   ollama create {model_name} -f Modelfile

3. Test the model:
   ollama run {model_name} "Hello!"

4. The model is now available in Ollama!

Model file location: {model_path}
""")
        
        logger.info(f"Generated import instructions: {instructions_path}")
        
        return output_path
    
    def import_to_ollama(self, 
                        modelfile_path: str,
                        model_name: str) -> Dict[str, Any]:
        """Import model into Ollama using the Modelfile."""
        
        try:
            # Change to directory containing Modelfile
            model_dir = os.path.dirname(modelfile_path)
            
            cmd = [
                "ollama", "create", model_name,
                "-f", modelfile_path
            ]
            
            logger.info(f"Importing to Ollama: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=model_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "success": True,
                "model_name": model_name,
                "message": "Model successfully imported to Ollama",
                "output": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama import failed: {e.stderr}")
            return {
                "success": False,
                "error": e.stderr,
                "message": "Failed to import to Ollama"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Ollama not found",
                "message": "Ollama is not installed. Please install from https://ollama.ai"
            }
    
    def get_export_formats(self) -> List[Dict[str, str]]:
        """Get available export formats."""
        return [
            {
                "id": "merged",
                "name": "Merged Model (MLX)",
                "description": "Full model weights merged with adapter, compatible with MLX",
                "extension": "safetensors",
                "size": "Large (~1-10GB)"
            },
            {
                "id": "gguf_q4",
                "name": "GGUF Q4_K_M (Recommended)",
                "description": "4-bit quantized, best balance of size/quality",
                "extension": "gguf",
                "size": "Medium (~500MB-2GB)",
                "requires": "llama.cpp"
            },
            {
                "id": "gguf_q5",
                "name": "GGUF Q5_K_M",
                "description": "5-bit quantized, better quality",
                "extension": "gguf",
                "size": "Medium-Large (~700MB-3GB)",
                "requires": "llama.cpp"
            },
            {
                "id": "gguf_q8",
                "name": "GGUF Q8_0",
                "description": "8-bit quantized, best quality",
                "extension": "gguf",
                "size": "Large (~1GB-5GB)",
                "requires": "llama.cpp"
            },
            {
                "id": "gguf_f16",
                "name": "GGUF F16 (Unquantized)",
                "description": "16-bit float, maximum quality",
                "extension": "gguf",
                "size": "Very Large (~2GB-10GB)",
                "requires": "llama.cpp"
            }
        ]
    
    def copy_to_lm_studio(self, gguf_path: str, model_name: str) -> str:
        """Copy GGUF file to LM Studio models directory with proper structure."""
        if not os.path.exists(self.lm_studio_dir):
            raise FileNotFoundError(f"LM Studio models directory not found: {self.lm_studio_dir}")
        
        # Create proper LM Studio directory structure: publisher/model-name/
        publisher = "custom-exports"
        model_dir = os.path.join(self.lm_studio_dir, publisher, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy GGUF file into the model directory
        dest_path = os.path.join(model_dir, f"{model_name}.gguf")
        
        logger.info(f"Copying GGUF to LM Studio: {dest_path}")
        
        import shutil
        shutil.copy2(gguf_path, dest_path)
        
        logger.info(f"Model copied to LM Studio: {dest_path}")
        logger.info(f"Model will appear in LM Studio as: {publisher}/{model_name}")
        
        return dest_path
