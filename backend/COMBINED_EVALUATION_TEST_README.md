# Combined Evaluation Test & Scoring System

## Overview

This document describes the test suite and enhanced scoring system for the Combined Tier 0+1 Evaluation System.

## Files Created

### 1. `test_combined_evaluation.py`
**Purpose**: Comprehensive test script for combined evaluation system

**Features**:
- Tests multiple adapters automatically
- Displays enhanced score cards with visual breakdown
- Creates comparison tables
- Saves results to JSON and text files
- Includes scoring system explanation

**Usage**:
```bash
cd backend
python test_combined_evaluation.py
```

**Output Files**:
- `test_combined_evaluation_results.json`: Full evaluation results
- `test_combined_evaluation_summary.txt`: Human-readable summary

### 2. Enhanced `combined_evaluator.py`
**New Features**:
- `CombinedEvaluator.print_score_card()`: Standard scoring display format
- Visual score breakdown with bar charts
- Clear Tier 0 (40%) and Tier 1 (60%) contribution display
- Enhanced comparison mode

## Scoring System

### Formula
```
Combined Score = (Tier 0 Score × 0.4) + (Tier 1 Score × 0.6)
```

### Tier Breakdown

**Tier 0 (Mathematical Analysis) - 40% Weight**:
- Spectral properties (singular values, effective rank)
- Training dynamics (loss curves, overfitting)
- Weight statistics (norms, sparsity)
- Time: <5 seconds
- Score Range: 0-100

**Tier 1 (Perplexity Analysis) - 60% Weight**:
- Perplexity on validation data
- Actual language modeling performance
- Works for both base models and adapters
- Time: 10-20 seconds
- Score Range: 0-100

### Grade Scale
- **A**: 90-100 (Excellent)
- **B**: 80-89 (Good)
- **C**: 70-79 (Fair)
- **D**: 60-69 (Poor)
- **F**: 0-59 (Failing)

## Score Card Display Format

The standard score card shows:

1. **Overall Score**: Large, prominent display
2. **Score Breakdown**: Box showing Tier 0 and Tier 1 contributions
3. **Visual Bar Chart**: ASCII bars showing score levels
4. **Tier 0 Details**: Spectral norm, effective rank, concentration, etc.
5. **Tier 1 Details**: Perplexity, average loss
6. **Base Model Comparison**: If included
7. **Warnings**: Any issues detected

### Example Output

```
================================================================================
📊 EVALUATION SCORE CARD: 4b-nested
================================================================================

🏆 OVERALL SCORE: 85.9/100
   Grade: B
   ⏱️  Total Time: 12.4s

📈 SCORE BREAKDOWN:
   ┌─────────────────────────────────────────────────────────┐
   │ Tier 0 (Mathematical):  65.0/100 × 40% =  26.0 points │
   │ Tier 1 (Perplexity):    99.9/100 × 60% =  59.9 points │
   ├─────────────────────────────────────────────────────────┤
   │ Combined Score:                    85.9/100      │
   └─────────────────────────────────────────────────────────┘

📊 VISUAL SCORE BREAKDOWN:
   Tier 0 (40%)        │████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░│  65.0/100 (weighted:  26.0)
   Tier 1 (60%)        │████████████████████████████████████████████████████│  99.9/100 (weighted:  59.9)
   COMBINED            │████████████████████████████████████████░░░░░░░░░░░░│  85.9/100
```

## Testing Workflow

1. **Run Test Script**:
   ```bash
   python test_combined_evaluation.py
   ```

2. **Review Score Cards**: Each adapter gets a detailed score card

3. **Check Comparison Table**: See all adapters ranked by score

4. **Review Output Files**: 
   - JSON for programmatic access
   - TXT for human reading

## Integration

The scoring system is now integrated into:

1. **`combined_evaluator.py`**: CLI tool uses enhanced display
2. **`test_combined_evaluation.py`**: Test script uses same display
3. **Future GUI**: Can use `CombinedEvaluator.print_score_card()` method

## Next Steps

1. Run the test to verify everything works
2. Review score cards to understand the scoring
3. Use `CombinedEvaluator.print_score_card()` in GUI integration
4. Customize scoring weights if needed (currently 40/60)

## Notes

- The scoring system emphasizes actual performance (Tier 1) while considering mathematical properties (Tier 0)
- All scores are deterministic and reproducible
- The visual breakdown makes it easy to see which tier contributes more to the final score
- Base model comparison helps understand adapter improvement

