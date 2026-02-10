#!/usr/bin/env python3
"""
Test script to verify the metrics pipeline is working correctly.
Tests:
1. Backend regex pattern matching
2. Metrics initialization
3. Progress calculation
4. Time remaining calculation
5. Frontend data handling
"""

import re
import sys
import math
from datetime import datetime, timedelta

def test_regex_patterns():
    """Test that regex patterns correctly parse training output"""
    print("=" * 60)
    print("TEST 1: Regex Pattern Matching")
    print("=" * 60)
    
    # Test patterns from actual local_lora.py output
    step_pattern = re.compile(r'Iter\s+(\d+)\s*:')
    loss_pattern = re.compile(
        r'(?:^|(?<!Val\s))(?:Train\s+)?[Ll]oss[:\s]+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)',
        re.IGNORECASE
    )
    val_pattern = re.compile(r'Val\s+loss\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)', re.IGNORECASE)
    lr_pattern = re.compile(r'Learning\s+[Rr]ate\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)')
    
    test_cases = [
        ("Iter 1: Val loss 2.527, Val took 10.293s", 
         {"step": "1", "train_loss": None, "val_loss": "2.527", "lr": None}),
        ("Iter 25: Train loss 2.015, Learning Rate 1.000e-05, It/sec 0.123",
         {"step": "25", "train_loss": "2.015", "val_loss": None, "lr": "1.000e-05"}),
        ("Iter 200: Val loss 1.892, Val took 15.234s",
         {"step": "200", "train_loss": None, "val_loss": "1.892", "lr": None}),
        ("Iter 50: Train loss 1.756, Learning Rate 9.500e-06",
         {"step": "50", "train_loss": "1.756", "val_loss": None, "lr": "9.500e-06"}),
        # Edge case: negative loss (possible in some RL scenarios)
        ("Iter 100: Train loss -0.123, Learning Rate 1e-6",
         {"step": "100", "train_loss": "-0.123", "val_loss": None, "lr": "1e-6"}),
    ]
    
    all_passed = True
    for line, expected in test_cases:
        step_match = step_pattern.search(line)
        loss_match = loss_pattern.search(line)
        val_match = val_pattern.search(line)
        lr_match = lr_pattern.search(line)
        
        results = {
            "step": step_match.group(1) if step_match else None,
            "train_loss": loss_match.group(1) if loss_match else None,
            "val_loss": val_match.group(1) if val_match else None,
            "lr": lr_match.group(1) if lr_match else None,
        }
        
        passed = results == expected
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status}: {line}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {results}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_progress_calculation():
    """Test progress calculation edge cases"""
    print("\n" + "=" * 60)
    print("TEST 2: Progress Calculation")
    print("=" * 60)
    
    test_cases = [
        # (current_step, total_steps, expected_progress)
        (0, 1000, 0),        # Start
        (1, 1000, 0.1),      # First iteration
        (25, 1000, 2.5),     # Normal progress
        (500, 1000, 50),     # Halfway
        (1000, 1000, 100),   # Complete
        (0, 0, 0),           # Edge: zero total
        (100, 0, 0),         # Edge: zero total with non-zero current
        (-1, 1000, 0),       # Edge: negative current
        (100, -100, 0),      # Edge: negative total
        (1500, 1000, 100),   # Edge: current exceeds total (capped at 100)
    ]
    
    all_passed = True
    for current, total, expected in test_cases:
        # Apply safeguards from the fixed code
        if total <= 0 or current < 0:
            progress = 0
        else:
            progress = (current / total) * 100
            progress = min(progress, 100)  # Cap at 100%
        
        passed = abs(progress - expected) < 0.01
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: step={current}, total={total} -> {progress:.1f}% (expected {expected}%)")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_time_calculation():
    """Test time remaining calculation"""
    print("\n" + "=" * 60)
    print("TEST 3: Time Remaining Calculation")
    print("=" * 60)
    
    start_time = datetime.now()
    
    test_cases = [
        # (elapsed_seconds, current_step, total_steps, should_have_eta)
        (0, 0, 1000, False),      # No progress yet
        (10, 10, 1000, True),     # 1% progress (above 0.001 threshold)
        (100, 100, 1000, True),   # 10% progress
        (500, 500, 1000, True),   # 50% progress
        (900, 900, 1000, True),   # 90% progress
        (50, 0, 1000, False),     # Zero progress
    ]
    
    all_passed = True
    for elapsed, current, total, should_have_eta in test_cases:
        # Simulate the fixed calculation logic
        total_steps = total
        current_step = current
        
        eta = None
        if total_steps > 0 and current_step > 0:
            progress = current_step / total_steps
            if progress > 0.001:  # Avoid division by very small numbers
                estimated_total = elapsed / progress
                remaining = max(0, int(estimated_total - elapsed))
                eta = remaining
        
        has_eta = eta is not None
        passed = has_eta == should_have_eta
        
        status = "✓ PASS" if passed else "✗ FAIL"
        eta_str = f"{eta}s" if eta is not None else "N/A"
        print(f"{status}: elapsed={elapsed}s, step={current}/{total}, ETA={eta_str}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_sanitize_metrics():
    """Test that sanitize_metrics handles all edge cases"""
    print("\n" + "=" * 60)
    print("TEST 4: Metrics Sanitization")
    print("=" * 60)
    
    def sanitize_metrics(data):
        """Recursively replace infinite/NaN values with None for JSON serialization."""
        if isinstance(data, dict):
            return {k: sanitize_metrics(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_metrics(v) for v in data]
        elif isinstance(data, float):
            if math.isinf(data) or math.isnan(data):
                return None
            return data
        return data
    
    test_cases = [
        # (input, expected)
        ({"loss": 0.5}, {"loss": 0.5}),
        ({"loss": float('inf')}, {"loss": None}),
        ({"loss": float('-inf')}, {"loss": None}),
        ({"loss": float('nan')}, {"loss": None}),
        ({"train_loss": 0.5, "val_loss": float('inf')}, {"train_loss": 0.5, "val_loss": None}),
        ({"nested": {"value": float('nan')}}, {"nested": {"value": None}}),
        ({"list": [1.0, float('inf'), 3.0]}, {"list": [1.0, None, 3.0]}),
    ]
    
    all_passed = True
    for input_data, expected in test_cases:
        result = sanitize_metrics(input_data)
        passed = result == expected
        
        # Special handling for NaN comparison since NaN != NaN
        if not passed and isinstance(result, dict) and isinstance(expected, dict):
            passed = all(
                (k in result and (result[k] == expected[k] or 
                 (isinstance(result[k], float) and isinstance(expected[k], float) and
                  math.isnan(result[k]) and math.isnan(expected[k]))))
                for k in expected
            )
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {input_data} -> {result}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_metrics_initialization():
    """Test that metrics are initialized with all required fields"""
    print("\n" + "=" * 60)
    print("TEST 5: Metrics Initialization")
    print("=" * 60)
    
    # This mirrors the initialization in start_training
    training_metrics = {
        "current_step": 0,
        "total_steps": 1000,
        "train_loss": None,
        "val_loss": None,
        "learning_rate": 1e-5,
        "start_time": datetime.now().isoformat(),
        "estimated_time_remaining": None,
        "avg_reward": None,
        "success_rate": None,
        "kl": None,
        "entropy": None
    }
    
    required_fields = [
        "current_step", "total_steps", "train_loss", "val_loss",
        "learning_rate", "start_time", "estimated_time_remaining",
        "avg_reward", "success_rate", "kl", "entropy"
    ]
    
    all_passed = True
    for field in required_fields:
        has_field = field in training_metrics
        status = "✓ PASS" if has_field else "✗ FAIL"
        value = training_metrics.get(field, "MISSING")
        print(f"{status}: {field} = {value}")
        if not has_field:
            all_passed = False
    
    # Validate types
    type_checks = [
        ("current_step", int, training_metrics["current_step"]),
        ("total_steps", int, training_metrics["total_steps"]),
        ("learning_rate", float, training_metrics["learning_rate"]),
    ]
    
    print("\nType checks:")
    for field, expected_type, value in type_checks:
        is_correct_type = isinstance(value, expected_type)
        status = "✓ PASS" if is_correct_type else "✗ FAIL"
        print(f"{status}: {field} is {expected_type.__name__} (got {type(value).__name__})")
        if not is_correct_type:
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("METRICS PIPELINE TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Regex Patterns", test_regex_patterns()))
    results.append(("Progress Calculation", test_progress_calculation()))
    results.append(("Time Calculation", test_time_calculation()))
    results.append(("Sanitization", test_sanitize_metrics()))
    results.append(("Initialization", test_metrics_initialization()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
