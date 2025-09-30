#!/bin/bash
# Quick script to check training errors

LOG_FILE="adapters/training_debug.log"

if [ -f "$LOG_FILE" ]; then
    echo "=== Training Debug Log ==="
    cat "$LOG_FILE"
    echo ""
else
    echo "No training debug log found at: $LOG_FILE"
    echo ""
    echo "Checking for any training processes..."
    ps aux | grep -E "mlx_lm" | grep -v grep
fi