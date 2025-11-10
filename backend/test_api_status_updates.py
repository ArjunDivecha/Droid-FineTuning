#!/usr/bin/env python3
"""
Test script to verify nested learning API status updates work correctly.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_status_updates():
    """Test that status updates correctly during training."""

    print("=" * 70)
    print("TESTING NESTED LEARNING API STATUS UPDATES")
    print("=" * 70)

    # 1. Check initial status
    print("\n1. Checking initial status...")
    response = requests.get(f"{BASE_URL}/nested-learning/status")
    status = response.json()
    print(f"   Initial status: {status['status']}")
    assert status['status'] in ['idle', 'completed'], f"Unexpected initial status: {status['status']}"

    # 2. Start training with minimal configuration
    print("\n2. Starting training...")
    config = {
        "base_model_path": "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct",
        "adapter_path": "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/7b",
        "train_data_path": "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/test_data_nested_learning.jsonl",
        "num_tiers": 3,
        "tier_update_frequencies": [1, 2, 4],
        "tier_assignment_strategy": "layer_depth",
        "learning_rate": 1e-5,
        "batch_size": 2,
        "num_steps": 20,
        "checkpoint_every": 10,
        "eval_every": 10,
        "experiment_name": "api_status_test",
        "output_path": "./test_api_status_output"
    }

    response = requests.post(f"{BASE_URL}/nested-learning/start", json=config)
    result = response.json()
    print(f"   Start response: {result}")

    if not result.get('success'):
        print(f"   ✗ Failed to start training: {result.get('message')}")
        return False

    # 3. Poll status for updates
    print("\n3. Polling status updates...")
    print("   (This should show progress from 0 to 20 steps)")

    last_step = -1
    max_polls = 60  # Max 60 seconds
    poll_count = 0

    while poll_count < max_polls:
        time.sleep(1)
        poll_count += 1

        response = requests.get(f"{BASE_URL}/nested-learning/status")
        status = response.json()

        current_step = status.get('current_step', 0)
        total_steps = status.get('total_steps', 0)
        current_status = status['status']

        # Print update if step changed
        if current_step != last_step:
            tier_stats = status.get('tier_stats', {})
            tier_info = ""
            if tier_stats:
                tier_params = tier_stats.get('tier_parameters', {})
                if tier_params:
                    updates = [tier_params.get(f'tier_{i}', {}).get('update_count', 0)
                              for i in range(3)]
                    tier_info = f" | Tier updates: {updates}"

            print(f"   Step {current_step}/{total_steps} | Status: {current_status}{tier_info}")
            last_step = current_step

        # Check if training completed
        if current_status == 'completed':
            print("\n   ✓ Training completed!")
            break
        elif current_status == 'error':
            print(f"\n   ✗ Training failed: {status.get('message')}")
            return False

    if poll_count >= max_polls:
        print(f"\n   ✗ Timeout: Status did not update to 'completed' after {max_polls} seconds")
        return False

    # 4. Verify final status
    print("\n4. Verifying final status...")
    response = requests.get(f"{BASE_URL}/nested-learning/status")
    final_status = response.json()

    print(f"   Final step: {final_status.get('current_step')}/{final_status.get('total_steps')}")
    print(f"   Status: {final_status['status']}")

    if final_status.get('tier_stats'):
        tier_params = final_status['tier_stats'].get('tier_parameters', {})
        print("\n   Tier Statistics:")
        for i in range(3):
            tier = tier_params.get(f'tier_{i}', {})
            print(f"     Tier {i}: {tier.get('update_count', 0)} updates (freq: {tier.get('frequency', 0)})")

    # 5. Verify tier update counts are correct
    print("\n5. Verifying tier update counts...")
    tier_params = final_status['tier_stats']['tier_parameters']

    tier_0_updates = tier_params['tier_0']['update_count']
    tier_1_updates = tier_params['tier_1']['update_count']
    tier_2_updates = tier_params['tier_2']['update_count']

    expected_0 = 20  # Every step
    expected_1 = 10  # Every 2 steps
    expected_2 = 5   # Every 4 steps

    checks = [
        (tier_0_updates == expected_0, f"Tier 0: {tier_0_updates} == {expected_0}"),
        (tier_1_updates == expected_1, f"Tier 1: {tier_1_updates} == {expected_1}"),
        (tier_2_updates == expected_2, f"Tier 2: {tier_2_updates} == {expected_2}"),
    ]

    all_passed = True
    for passed, check in checks:
        symbol = "✓" if passed else "✗"
        print(f"   {symbol} {check}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Status updates working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Check tier update counts")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    try:
        success = test_status_updates()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
