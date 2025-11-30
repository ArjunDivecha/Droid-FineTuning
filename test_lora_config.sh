#!/bin/bash

echo "🔍 Verifying LoRA Configuration Fix"
echo "===================================="
echo ""

# Check if training config exists
if [ -f "/tmp/gui_training_config.yaml" ]; then
    echo "✅ Training config found"
    echo ""
    echo "📄 LoRA Parameters in config:"
    echo "----------------------------"
    grep -A 10 "lora_parameters:" /tmp/gui_training_config.yaml
    echo ""
else
    echo "⚠️  No training config found at /tmp/gui_training_config.yaml"
    echo "   Start a training run first"
    echo ""
fi

# Check if LoRA config exists
if [ -f "/tmp/lora_config.yaml" ]; then
    echo "✅ LoRA config found"
    echo ""
    echo "📄 LoRA Config Contents:"
    echo "------------------------"
    cat /tmp/lora_config.yaml
    echo ""
else
    echo "⚠️  No LoRA config found at /tmp/lora_config.yaml"
    echo "   This gets created when training starts"
    echo ""
fi

# Check training logs for the confirmation message
echo "🔍 Checking training logs for LoRA parameters..."
echo "------------------------------------------------"

LOG_FILE="/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/logs/gui_training.log"

if [ -f "$LOG_FILE" ]; then
    if grep -q "Using GUI-provided LoRA parameters" "$LOG_FILE"; then
        echo "✅ FOUND: Training script is using GUI parameters!"
        echo ""
        grep -A 4 "Using GUI-provided LoRA parameters" "$LOG_FILE" | tail -5
        echo ""
    elif grep -q "Using fallback LoRA parameters" "$LOG_FILE"; then
        echo "❌ WARNING: Training script is using FALLBACK parameters!"
        echo "   The lora_parameters dict is not being passed correctly"
        echo ""
    else
        echo "⚠️  No LoRA parameter confirmation found in logs yet"
        echo "   Training may not have started"
        echo ""
    fi
    
    # Check for trainable parameters percentage
    echo "🔍 Checking trainable parameters..."
    echo "-----------------------------------"
    TRAINABLE=$(grep "Trainable parameters" "$LOG_FILE" | head -1)
    if [ -n "$TRAINABLE" ]; then
        echo "$TRAINABLE"
        
        # Extract percentage
        PERCENT=$(echo "$TRAINABLE" | grep -oE '[0-9]+\.[0-9]+%' | head -1)
        if [ -n "$PERCENT" ]; then
            PERCENT_NUM=$(echo "$PERCENT" | sed 's/%//')
            if (( $(echo "$PERCENT_NUM > 3.0" | bc -l) )); then
                echo "✅ GOOD: $PERCENT trainable (full-layer LoRA working!)"
            else
                echo "❌ BAD: $PERCENT trainable (should be ~3.5-4%)"
            fi
        fi
        echo ""
    else
        echo "⚠️  Trainable parameters not logged yet"
        echo ""
    fi
else
    echo "⚠️  Training log not found: $LOG_FILE"
    echo ""
fi

echo "===================================="
echo "📝 Summary:"
echo ""
echo "To verify the fix is working, you should see:"
echo "  1. ✅ 'Using GUI-provided LoRA parameters' in logs"
echo "  2. ✅ 'Keys: 7 matrices' in logs"
echo "  3. ✅ ~3.5-4% trainable parameters"
echo ""
echo "If you see 'Using fallback LoRA parameters', the fix"
echo "is not working and the dict is not being passed."
echo ""
