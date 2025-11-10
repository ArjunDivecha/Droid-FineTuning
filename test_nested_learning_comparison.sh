#!/bin/bash
# Test script for Nested Learning Adapters Comparison
# This script demonstrates that nested learning adapters work without AttributeError

echo "=============================================="
echo "NESTED LEARNING ADAPTERS COMPARISON TEST"
echo "=============================================="
echo ""
echo "Testing prompt: 'What are emerging markets?'"
echo ""

# Test Base Model
echo "-------------------------------------------"
echo "1. Testing Base Model (Qwen2.5-0.5B-Instruct)"
echo "-------------------------------------------"
BASE_RESPONSE=$(curl -s -X POST http://localhost:8000/models/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are emerging markets?", "model_name": "Qwen2.5-0.5B-Instruct", "max_tokens": 100}')

echo "Base Model Response:"
echo "$BASE_RESPONSE" | python3 -m json.tool
echo ""

# Test Nested Learning Adapter b1
echo "-------------------------------------------"
echo "2. Testing Nested Learning Adapter: b1 (nested)"
echo "-------------------------------------------"
B1_RESPONSE=$(curl -s -X POST http://localhost:8000/models/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are emerging markets?", "model_name": "Qwen2.5-0.5B-Instruct", "adapter_name": "b1 (nested)", "max_tokens": 100}')

echo "b1 (nested) Response:"
echo "$B1_RESPONSE" | python3 -m json.tool
echo ""

# Test Nested Learning Adapter b2
echo "-------------------------------------------"
echo "3. Testing Nested Learning Adapter: b2 (nested)"
echo "-------------------------------------------"
B2_RESPONSE=$(curl -s -X POST http://localhost:8000/models/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are emerging markets?", "model_name": "Qwen2.5-0.5B-Instruct", "adapter_name": "b2 (nested)", "max_tokens": 100}')

echo "b2 (nested) Response:"
echo "$B2_RESPONSE" | python3 -m json.tool
echo ""

# Test Nested Learning Adapter b3
echo "-------------------------------------------"
echo "4. Testing Nested Learning Adapter: b3 (nested)"
echo "-------------------------------------------"
B3_RESPONSE=$(curl -s -X POST http://localhost:8000/models/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are emerging markets?", "model_name": "Qwen2.5-0.5B-Instruct", "adapter_name": "b3 (nested)", "max_tokens": 100}')

echo "b3 (nested) Response:"
echo "$B3_RESPONSE" | python3 -m json.tool
echo ""

# Summary
echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo ""

# Check if all responses were successful
BASE_SUCCESS=$(echo "$BASE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))")
B1_SUCCESS=$(echo "$B1_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))")
B2_SUCCESS=$(echo "$B2_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))")
B3_SUCCESS=$(echo "$B3_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))")

echo "Base Model:           $BASE_SUCCESS"
echo "b1 (nested) Adapter:  $B1_SUCCESS"
echo "b2 (nested) Adapter:  $B2_SUCCESS"
echo "b3 (nested) Adapter:  $B3_SUCCESS"
echo ""

if [ "$BASE_SUCCESS" = "True" ] && [ "$B1_SUCCESS" = "True" ] && [ "$B2_SUCCESS" = "True" ] && [ "$B3_SUCCESS" = "True" ]; then
    echo "✓ ALL TESTS PASSED - No AttributeError detected!"
    echo "✓ Nested learning adapters are working correctly"
else
    echo "✗ Some tests failed - Please check the responses above"
fi

echo ""
echo "=============================================="
echo "END OF TEST"
echo "=============================================="
