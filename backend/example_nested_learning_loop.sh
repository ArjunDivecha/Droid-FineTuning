#!/bin/bash
# Example script showing how to loop over nested learning parameters
# This demonstrates parameter sweeps for testing and experimentation

# Activate virtual environment
source /Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/activate

# Set your paths (MODIFY THESE!)
BASE_MODEL="/path/to/your/model"
TRAIN_DATA="/path/to/your/train.jsonl"
VAL_DATA="/path/to/your/val.jsonl"  # Optional
ADAPTER=""  # Optional adapter path

# Output directory
OUTPUT_DIR="./nested_learning_experiments"

# ============================================================================
# Example 1: Test different learning rates and batch sizes
# ============================================================================
echo "Example 1: Learning rate and batch size sweep"
for lr in 1e-5 5e-5 1e-4; do
    for bs in 1 2; do
        echo "Testing lr=$lr, batch_size=$bs"
        python run_nested_learning_cli.py \
            --base-model-path "$BASE_MODEL" \
            --train-data-path "$TRAIN_DATA" \
            --learning-rate $lr \
            --batch-size $bs \
            --num-steps 100 \
            --max-seq-length 128 \
            --experiment-name "lr_${lr}_bs_${bs}" \
            --output-path "$OUTPUT_DIR/lr_bs_sweep"
    done
done

# ============================================================================
# Example 2: Test different tier configurations
# ============================================================================
echo "Example 2: Tier configuration sweep"
# 2 tiers: [1, 2]
python run_nested_learning_cli.py \
    --base-model-path "$BASE_MODEL" \
    --train-data-path "$TRAIN_DATA" \
    --num-tiers 2 \
    --tier-update-frequencies 1 2 \
    --num-steps 100 \
    --experiment-name "2tiers_1_2" \
    --output-path "$OUTPUT_DIR/tier_sweep"

# 3 tiers: [1, 2, 4]
python run_nested_learning_cli.py \
    --base-model-path "$BASE_MODEL" \
    --train-data-path "$TRAIN_DATA" \
    --num-tiers 3 \
    --tier-update-frequencies 1 2 4 \
    --num-steps 100 \
    --experiment-name "3tiers_1_2_4" \
    --output-path "$OUTPUT_DIR/tier_sweep"

# 4 tiers: [1, 2, 4, 8]
python run_nested_learning_cli.py \
    --base-model-path "$BASE_MODEL" \
    --train-data-path "$TRAIN_DATA" \
    --num-tiers 4 \
    --tier-update-frequencies 1 2 4 8 \
    --num-steps 100 \
    --experiment-name "4tiers_1_2_4_8" \
    --output-path "$OUTPUT_DIR/tier_sweep"

# ============================================================================
# Example 3: Test different LoRA ranks
# ============================================================================
echo "Example 3: LoRA rank sweep"
for rank in 4 8 16 32; do
    alpha=$((rank * 2))  # Common practice: alpha = 2 * rank
    echo "Testing LoRA rank=$rank, alpha=$alpha"
    python run_nested_learning_cli.py \
        --base-model-path "$BASE_MODEL" \
        --train-data-path "$TRAIN_DATA" \
        --lora-rank $rank \
        --lora-alpha $alpha \
        --num-steps 100 \
        --experiment-name "lora_r${rank}_a${alpha}" \
        --output-path "$OUTPUT_DIR/lora_sweep"
done

# ============================================================================
# Example 4: Test different sequence lengths (memory comparison)
# ============================================================================
echo "Example 4: Sequence length sweep (memory testing)"
for seq_len in 64 128 256; do
    echo "Testing max_seq_length=$seq_len"
    python run_nested_learning_cli.py \
        --base-model-path "$BASE_MODEL" \
        --train-data-path "$TRAIN_DATA" \
        --max-seq-length $seq_len \
        --batch-size 1 \
        --num-steps 50 \
        --experiment-name "seqlen_${seq_len}" \
        --output-path "$OUTPUT_DIR/seqlen_sweep"
done

# ============================================================================
# Example 5: Test tier assignment strategies
# ============================================================================
echo "Example 5: Tier assignment strategy comparison"
for strategy in layer_depth parameter_importance; do
    echo "Testing strategy=$strategy"
    python run_nested_learning_cli.py \
        --base-model-path "$BASE_MODEL" \
        --train-data-path "$TRAIN_DATA" \
        --tier-assignment-strategy $strategy \
        --num-steps 100 \
        --experiment-name "strategy_${strategy}" \
        --output-path "$OUTPUT_DIR/strategy_sweep"
done

# ============================================================================
# Example 6: Test early stopping parameters
# ============================================================================
echo "Example 6: Early stopping patience sweep"
for patience in 3 5 10; do
    echo "Testing patience=$patience"
    python run_nested_learning_cli.py \
        --base-model-path "$BASE_MODEL" \
        --train-data-path "$TRAIN_DATA" \
        --val-data-path "$VAL_DATA" \
        --early-stop \
        --patience $patience \
        --min-delta 0.0001 \
        --num-steps 500 \
        --eval-every 50 \
        --experiment-name "patience_${patience}" \
        --output-path "$OUTPUT_DIR/earlystop_sweep"
done

# ============================================================================
# Example 7: Validate configurations before running (dry run)
# ============================================================================
echo "Example 7: Validate multiple configs without training"
for lr in 1e-5 1e-4 1e-3; do
    for bs in 1 2 4; do
        echo "Validating lr=$lr, batch_size=$bs"
        python run_nested_learning_cli.py \
            --base-model-path "$BASE_MODEL" \
            --train-data-path "$TRAIN_DATA" \
            --learning-rate $lr \
            --batch-size $bs \
            --validate-only \
            --experiment-name "validate_lr${lr}_bs${bs}"
    done
done

# ============================================================================
# Example 8: Comprehensive sweep with nested loops
# ============================================================================
echo "Example 8: Comprehensive parameter sweep"
for lr in 1e-5 5e-5; do
    for rank in 8 16; do
        for num_tiers in 2 3; do
            # Generate tier frequencies based on num_tiers
            if [ $num_tiers -eq 2 ]; then
                freqs="1 2"
            else
                freqs="1 2 4"
            fi

            echo "Testing lr=$lr, rank=$rank, tiers=$num_tiers, freq=$freqs"
            python run_nested_learning_cli.py \
                --base-model-path "$BASE_MODEL" \
                --train-data-path "$TRAIN_DATA" \
                --learning-rate $lr \
                --lora-rank $rank \
                --lora-alpha $((rank * 2)) \
                --num-tiers $num_tiers \
                --tier-update-frequencies $freqs \
                --num-steps 200 \
                --experiment-name "lr${lr}_r${rank}_t${num_tiers}" \
                --output-path "$OUTPUT_DIR/comprehensive_sweep"
        done
    done
done

echo "All experiments complete! Results in: $OUTPUT_DIR"
