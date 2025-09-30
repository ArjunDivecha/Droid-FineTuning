#!/bin/bash

# Script to add adapter_config.json to existing fused adapters
# This is needed for mlx_lm.fuse to work with fused adapters

ADAPTERS_DIR="/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"

echo "Fixing fused adapters by adding adapter_config.json..."

# Find all fused adapters
find "$ADAPTERS_DIR" -name "fusion_report.txt" -type f | while read report; do
  dir=$(dirname "$report")
  adapter_name=$(basename "$dir")
  
  # Skip if adapter_config.json already exists
  if [ -f "$dir/adapter_config.json" ]; then
    echo "✓ $adapter_name already has adapter_config.json"
    continue
  fi
  
  echo "Processing: $adapter_name"
  
  # Read fusion report to find source adapters
  source_adapter=$(grep -A 1 "Source Adapters:" "$report" | grep "1\." | sed 's/.*1\. \([^ ]*\).*/\1/')
  
  if [ -n "$source_adapter" ]; then
    source_config="$ADAPTERS_DIR/$source_adapter/adapter_config.json"
    
    if [ -f "$source_config" ]; then
      cp "$source_config" "$dir/adapter_config.json"
      echo "  ✓ Copied adapter_config.json from $source_adapter"
    else
      echo "  ✗ Source adapter $source_adapter doesn't have adapter_config.json"
    fi
  else
    echo "  ✗ Could not find source adapter in fusion report"
  fi
done

echo ""
echo "Done! All fused adapters should now have adapter_config.json"
