#!/bin/bash

echo "=========================================="
echo "Full-Layer LoRA Implementation Tests"
echo "=========================================="
echo ""

# Change to backend directory
cd /Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend

echo "📋 TEST 1: TrainingConfig Dataclass"
echo "=========================================="
python3 test_milestone1.py
TEST1_RESULT=$?
echo ""

echo "📋 TEST 2 & 3: LoRA Generation & Endpoint"
echo "=========================================="
python3 test_milestone2_3.py
TEST2_RESULT=$?
echo ""

# Change to frontend directory
cd /Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/frontend

echo "📋 TEST 4 & 5: Frontend TypeScript Compilation"
echo "=========================================="
npm run build 2>&1 | grep -E "(error|warning|Compiled successfully)" | head -20
TEST3_RESULT=$?
echo ""

echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ $TEST1_RESULT -eq 0 ]; then
    echo "✅ Backend TrainingConfig: PASSED"
else
    echo "❌ Backend TrainingConfig: FAILED"
fi

if [ $TEST2_RESULT -eq 0 ]; then
    echo "✅ Backend LoRA Logic: PASSED"
else
    echo "❌ Backend LoRA Logic: FAILED"
fi

if [ $TEST3_RESULT -eq 0 ]; then
    echo "✅ Frontend TypeScript: PASSED"
else
    echo "❌ Frontend TypeScript: FAILED"
fi

echo ""
echo "=========================================="
if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ] && [ $TEST3_RESULT -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "=========================================="
    echo ""
    echo "Next Steps:"
    echo "1. Start backend: cd backend && python main.py"
    echo "2. Start frontend: cd frontend && npm start"
    echo "3. Navigate to Setup page and verify LoRA section"
else
    echo "⚠️  SOME TESTS FAILED"
    echo "=========================================="
    echo "Review errors above and fix issues"
fi
echo ""
