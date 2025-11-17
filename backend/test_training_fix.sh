#!/bin/bash
# Quick test to verify training works with the fix

echo "=========================================="
echo "Testing Nested Learning Training with Fix"
echo "=========================================="

# Configuration
BASE_MODEL="/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
ADAPTER="/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/big/checkpoints/best"
TRAIN_DATA="/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/chunked_stories/train.jsonl"
OUTPUT_PATH="outputs/test_fix"

echo ""
echo "Running 5 training steps to verify fix..."
echo ""

python3 backend/run_nested_learning_cli.py \
  --base-model "$BASE_MODEL" \
  --adapter-path "$ADAPTER" \
  --train-data "$TRAIN_DATA" \
  --output-path "$OUTPUT_PATH" \
  --experiment-name "fix_verification" \
  --num-steps 5 \
  --eval-every 5 \
  --learning-rate 1e-4 \
  --batch-size 1 \
  --num-tiers 2 \
  --tier-frequencies "1,2"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Check the logs above for 'matches_update=True'"
echo "=========================================="
