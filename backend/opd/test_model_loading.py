"""
Test script for model loading and memory profiling

This script tests:
1. Loading Qwen 32B teacher model
2. Loading Qwen 7B student model
3. Loading both simultaneously
4. Memory usage measurement
5. Basic generate_with_logprobs implementation

Usage:
    python backend/opd/test_model_loading.py --teacher-path /path/to/qwen32b --student-path /path/to/qwen7b
"""

import argparse
import sys
import time
import gc
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
except ImportError:
    print("ERROR: MLX not found. Please install mlx and mlx-lm:")
    print("  pip install mlx mlx-lm")
    sys.exit(1)


def get_memory_usage_gb():
    """Get current memory usage in GB (MLX memory)"""
    try:
        # MLX memory stats if available
        memory_info = mx.metal.get_active_memory()
        return memory_info / (1024 ** 3)  # Convert to GB
    except:
        # Fallback: estimate from peak memory
        return mx.metal.get_peak_memory() / (1024 ** 3)


def print_memory_stats(label: str):
    """Print memory statistics"""
    try:
        active_mem = mx.metal.get_active_memory() / (1024 ** 3)
        peak_mem = mx.metal.get_peak_memory() / (1024 ** 3)
        cache_mem = mx.metal.get_cache_memory() / (1024 ** 3)

        print(f"\n{'='*60}")
        print(f"Memory Stats: {label}")
        print(f"{'='*60}")
        print(f"  Active Memory:  {active_mem:.2f} GB")
        print(f"  Peak Memory:    {peak_mem:.2f} GB")
        print(f"  Cache Memory:   {cache_mem:.2f} GB")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Unable to get memory stats: {e}")


def test_teacher_loading(teacher_path: str):
    """Test loading teacher model (Qwen 32B)"""
    print("\n" + "="*60)
    print("TEST 1: Loading Teacher Model (Qwen 32B)")
    print("="*60)

    if not Path(teacher_path).exists():
        print(f"ERROR: Teacher model not found at {teacher_path}")
        print("Please specify the correct path with --teacher-path")
        return None

    print(f"Loading teacher from: {teacher_path}")
    start_time = time.time()

    try:
        model, tokenizer = load(teacher_path)
        load_time = time.time() - start_time

        print(f"✓ Teacher loaded successfully in {load_time:.2f}s")
        print(f"  Model type: {type(model)}")
        print(f"  Tokenizer vocab size: {len(tokenizer)}")

        print_memory_stats("After Teacher Load")

        return model, tokenizer

    except Exception as e:
        print(f"✗ Failed to load teacher: {e}")
        return None


def test_student_loading(student_path: str, adapter_path: str = None):
    """Test loading student model (Qwen 7B) with optional LoRA adapter"""
    print("\n" + "="*60)
    print("TEST 2: Loading Student Model (Qwen 7B)")
    print("="*60)

    if not Path(student_path).exists():
        print(f"ERROR: Student model not found at {student_path}")
        print("Please specify the correct path with --student-path")
        return None

    print(f"Loading student from: {student_path}")
    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")

    start_time = time.time()

    try:
        if adapter_path and Path(adapter_path).exists():
            model, tokenizer = load(student_path, adapter_path=adapter_path)
            print(f"✓ Student loaded with adapter in {time.time() - start_time:.2f}s")
        else:
            model, tokenizer = load(student_path)
            print(f"✓ Student loaded (no adapter) in {time.time() - start_time:.2f}s")

        print(f"  Model type: {type(model)}")
        print(f"  Tokenizer vocab size: {len(tokenizer)}")

        print_memory_stats("After Student Load")

        return model, tokenizer

    except Exception as e:
        print(f"✗ Failed to load student: {e}")
        return None


def test_simultaneous_loading(teacher_path: str, student_path: str, adapter_path: str = None):
    """Test loading both models simultaneously"""
    print("\n" + "="*60)
    print("TEST 3: Loading Both Models Simultaneously")
    print("="*60)

    # Reset memory
    gc.collect()
    mx.metal.clear_cache()

    print("Loading teacher model...")
    teacher_result = test_teacher_loading(teacher_path)
    if teacher_result is None:
        return False

    teacher_model, teacher_tokenizer = teacher_result

    print("\nNow loading student model (teacher still in memory)...")
    student_result = test_student_loading(student_path, adapter_path)
    if student_result is None:
        return False

    student_model, student_tokenizer = student_result

    print_memory_stats("After Both Models Loaded")

    print("\n✓ Both models loaded successfully!")
    print("  This confirms you have enough memory for distillation.")

    return True


def generate_with_logprobs(model, tokenizer, prompt: str, max_tokens: int = 20):
    """
    Generate text and extract token-level logprobs.

    This is the manual generation loop approach (Option C).

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        dict with 'tokens', 'token_ids', 'text', 'logprobs'
    """
    print("\n" + "="*60)
    print("TEST 4: generate_with_logprobs() Implementation")
    print("="*60)

    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    tokens = input_ids.copy()

    generated_tokens = []
    generated_token_ids = []
    token_logprobs = []

    # Get EOS token
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

    print("\nGenerating tokens:")
    start_time = time.time()

    try:
        for step in range(max_tokens):
            # Forward pass
            # Convert tokens to MLX array
            input_tensor = mx.array([tokens])

            # Get logits from model
            outputs = model(input_tensor)

            # Get logits for last position (next token prediction)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[0, -1, :]  # (vocab_size,)
            else:
                logits = outputs[0, -1, :]  # Assume outputs are logits directly

            # Compute log probabilities
            logprobs = mx.log_softmax(logits, axis=-1)

            # Sample next token (greedy decoding for determinism)
            next_token_id = int(mx.argmax(logprobs).item())

            # Get logprob of selected token
            token_logprob = float(logprobs[next_token_id].item())

            # Decode token
            next_token = tokenizer.decode([next_token_id])

            # Store
            generated_token_ids.append(next_token_id)
            generated_tokens.append(next_token)
            token_logprobs.append(token_logprob)
            tokens.append(next_token_id)

            print(f"  Step {step+1}: '{next_token}' (id={next_token_id}, logprob={token_logprob:.4f})")

            # Check for EOS
            if eos_token_id and next_token_id == eos_token_id:
                print(f"  [EOS token reached at step {step+1}]")
                break

        generation_time = time.time() - start_time
        generated_text = ''.join(generated_tokens)

        print(f"\n✓ Generation complete in {generation_time:.2f}s")
        print(f"  Generated text: {generated_text}")
        print(f"  Tokens generated: {len(generated_tokens)}")
        print(f"  Average logprob: {sum(token_logprobs)/len(token_logprobs):.4f}")

        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens': generated_tokens,
            'token_ids': generated_token_ids,
            'logprobs': token_logprobs,
            'generation_time': generation_time
        }

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_generate_with_logprobs(model, tokenizer):
    """Test the generate_with_logprobs function"""
    test_prompt = "The capital of France is"

    result = generate_with_logprobs(model, tokenizer, test_prompt, max_tokens=10)

    if result:
        print("\n✓ generate_with_logprobs() works correctly!")
        print("  This is the core function for extracting teacher logprobs.")
    else:
        print("\n✗ generate_with_logprobs() failed")

    return result is not None


def main():
    parser = argparse.ArgumentParser(description="Test OPD model loading and memory usage")
    parser.add_argument(
        "--teacher-path",
        type=str,
        required=True,
        help="Path to teacher model (Qwen 32B)"
    )
    parser.add_argument(
        "--student-path",
        type=str,
        required=True,
        help="Path to student model (Qwen 7B)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to student's LoRA adapter (optional)"
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="Test generate_with_logprobs function"
    )

    args = parser.parse_args()

    print("="*60)
    print("OPD Model Loading & Memory Profiling Test")
    print("="*60)
    print(f"Teacher: {args.teacher_path}")
    print(f"Student: {args.student_path}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print("="*60)

    # Initial memory
    print_memory_stats("Initial State")

    # Test 1: Teacher loading
    teacher_result = test_teacher_loading(args.teacher_path)
    if teacher_result is None:
        print("\n✗ Teacher loading failed. Exiting.")
        return 1

    # Test 2: Student loading (separate)
    gc.collect()
    mx.metal.clear_cache()
    print("\n[Clearing memory for separate student test...]")

    student_result = test_student_loading(args.student_path, args.adapter_path)
    if student_result is None:
        print("\n✗ Student loading failed. Exiting.")
        return 1

    # Test 3: Simultaneous loading
    gc.collect()
    mx.metal.clear_cache()
    print("\n[Clearing memory for simultaneous test...]")

    success = test_simultaneous_loading(args.teacher_path, args.student_path, args.adapter_path)
    if not success:
        print("\n✗ Simultaneous loading failed. Exiting.")
        return 1

    # Test 4: generate_with_logprobs (if requested)
    if args.test_generation:
        print("\n[Testing generate_with_logprobs function...]")
        # Use teacher model for test
        teacher_model, teacher_tokenizer = teacher_result
        success = test_generate_with_logprobs(teacher_model, teacher_tokenizer)
        if not success:
            print("\n✗ generate_with_logprobs test failed")
            return 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ All tests passed successfully!")
    print("\nYou are ready to proceed with OPD implementation.")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
