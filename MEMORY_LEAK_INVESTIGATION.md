# MLX Memory Leak Investigation Summary

**Date:** November 30, 2025  
**Status:** Critical / Unresolved  
**Impact:** Prevents fine-tuning of even small models (0.5B params) on high-end hardware (128GB RAM).

## 1. The Problem
When attempting to fine-tune `Qwen2.5-0.5B-Instruct` using MLX, the Python process consumes excessive memory, rapidly growing from ~7GB to over **100GB**, causing system instability or process termination.

*   **Hardware:** MacBook Pro M-series, 128GB RAM.
*   **Software:** macOS 25.1.0, Python 3.12.
*   **Symptoms:** 
    *   Rapid memory spike during initialization/early training.
    *   `wired_memory` (OS metric) sometimes appears stable (~13GB), but **Process RSS** (Activity Monitor) confirms >100GB usage.
    *   System freezes or kills the process.

## 2. Timeline of Investigation & Fixes

### Phase 1: Configuration & Version Checks
*   **Hypothesis:** Incorrect batch size or sequence length.
*   **Action:** Reduced batch size to 1, sequence length to various values.
*   **Result:** **Failed.** Leak persists even with minimal load.
*   **Action:** Downgraded `mlx-lm-lora` from 0.8.5 to 0.8.1.
*   **Result:** **Failed.**

### Phase 2: MLX Version Downgrades
*   **Hypothesis:** Regression in recent MLX versions (0.30.0).
*   **Action:** Downgraded `mlx` to 0.29.3.
*   **Result:** **Failed.** Same 82GB+ spike.
*   **Action:** Major downgrade to `mlx` 0.17.3 / `mlx-lm` 0.18.2 (known stable versions from older reports).
*   **Result:** **Failed.** While `wired_memory` metrics looked lower (26GB), the actual process memory still leaked to unmanageable levels.

### Phase 3: Cache Management Attempts
*   **Hypothesis:** MLX computation graph or cache not clearing.
*   **Action:** Implemented aggressive garbage collection, `mx.metal.clear_cache()`, and `mx.set_cache_limit()`.
*   **Result:** **Failed.** No impact on the memory growth.

### Phase 4: Branch Isolation
*   **Hypothesis:** Specific code in the `tinker` branch causing issues.
*   **Action:** Switched to `main` branch to test clean state.
*   **Result:** **Failed.** Both branches exhibit identical memory explosion, confirming the issue is likely in the core library interactions or the specific model configuration, not the application logic.

## 3. Current Technical State
*   **Framework:** `mlx==0.30.0`, `mlx-lm==0.28.3`.
*   **Codebase:** `tinker` branch (merged with `full-lora`).
*   **Critical Finding:** The memory usage is **not** reflected correctly in standard `wired_memory` checks via Python scripts. Activity Monitor is the only reliable source, showing the true scale of the leak (100GB+).

## 4. Next Steps
1.  **Isolate Reproduction:** Create a minimal script (0 dependencies outside MLX) to reproduce the leak.
2.  **Profile Object Graph:** Use `objgraph` or `tracemalloc` to see if Python objects are retaining references to MLX arrays.
3.  **Submit Issue:** If minimal reproduction succeeds, open a formal issue on the MLX GitHub repository with the reproduction script.

---
*This document summarizes the troubleshooting steps taken as of Nov 30, 2025.*
