#!/bin/bash
# Test script for OPD distillation training
# This runs a quick test with your Qwen models

# Your model paths (from Phase 0 testing)
TEACHER="/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-32B-MLX-4bit"
STUDENT="/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"

# Create sample validation prompts if not exists
PROMPTS_FILE="./OnPolicyDistill/test_prompts.jsonl"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Creating sample prompts file..."
    python3 -c "
from backend.opd.data_loader import create_sample_prompts_file
create_sample_prompts_file('$PROMPTS_FILE', num_prompts=50)
"
fi

# For testing, we need an adapter path
# If you don't have one yet, we'll create a dummy one
ADAPTER_PATH="./OnPolicyDistill/test_adapter"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "⚠ Warning: No adapter found at $ADAPTER_PATH"
    echo "For a real test, you need a fine-tuned LoRA adapter from SFT training."
    echo "For now, we'll use the base model path as a placeholder."
    ADAPTER_PATH="$STUDENT"
fi

# Output path
OUTPUT_PATH="./OnPolicyDistill/checkpoints/test_run"

echo "================================"
echo "OPD Distillation Test"
echo "================================"
echo "Teacher: Qwen 32B (4-bit)"
echo "Student: Qwen 7B"
echo "Steps: 10 (quick test)"
echo "================================"
echo ""

# Run distillation with minimal steps for testing
python3 backend/opd/run_distillation.py \
    --teacher-path "$TEACHER" \
    --student-path "$STUDENT" \
    --adapter-path "$ADAPTER_PATH" \
    --prompts-path "$PROMPTS_FILE" \
    --output-path "$OUTPUT_PATH" \
    --steps 10 \
    --batch-size 2 \
    --temperature 2.0 \
    --max-prompts 20 \
    --max-tokens 50 \
    --checkpoint-every 5 \
    --eval-every 5 \
    --log-level DEBUG

echo ""
echo "================================"
echo "Test complete!"
echo "Check output at: $OUTPUT_PATH"
echo "================================"
