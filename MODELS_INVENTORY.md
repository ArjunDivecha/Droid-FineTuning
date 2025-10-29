# Models Inventory for Droid-FineTuning Stock System

**Last Updated:** October 26, 2025  
**Purpose:** This document lists all AI models available for fine-tuning in your multi-model stock market system. Think of it as a library card catalog—each model is a "brain" you can train on stock data (e.g., news sentiment, price predictions). Load them via the `config/models.json` file in your Python scripts.

## Quick Start
- **Total Models:** 1 (expand by adding to `config/models.json`).
- **Usage Example:** In a fine-tuning script, read the JSON and load like: `from transformers import AutoModel; model = AutoModel.from_pretrained(models[0]['path'])`.
- **Optimization for Your Mac:** All models are tuned for M4 Max GPU (use MLX for speed, PyTorch MPS for parallel batches up to 32+ with 128GB RAM).
- **Fine-Tuning Focus:** Stock tasks like analyzing market news or forecasting trends. For missing data (e.g., incomplete stock country stats), we'll fill with means and log in XLSX reports.

## Model List
### 1. Qwen2.5-0.5B-bf16
- **Name:** qwen2.5-0.5b-bf16
- **Paths:** 
  - **Project Path (for fine-tuning scripts):** `/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/models/qwen2.5-0.5b-bf16`
  - **App Path (for GUI model selector):** `/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/qwen2.5-0.5b-bf16`
- **Description:** A small, fast model (0.5 billion parameters) great for quick experiments on stock text data. Runs efficiently on your Apple Silicon without needing tons of power.
- **File Size:** ~950 MB (bf16 compressed format)
- **Format:** bf16 (memory-efficient for GPU), from Hugging Face's MLX community.
- **Hardware Notes:** Loads in 1-2GB VRAM; perfect for parallel fine-tuning (e.g., batch size 32 on your 128GB setup).
- **Fine-Tuning Params Suggestion:** Use LoRA (low-rank adaptation) with rank 16; libraries: Transformers or MLX. Parallelize with `torch.backends.mps` or `mlx.device('gpu')`.
- **Download Date/Version:** 2025-10-26 / Qwen2.5-0.5B-bf16
- **Data Quality:** 100% complete download. No missing values. Copied to app location for GUI visibility. For stock fine-tuning: Impute country data means, generate PDF plots and XLSX logs.
- **Status:** ✅ Available in GUI model selector

## Adding New Models
1. Download to `models/[model-name]/` (e.g., via Git clone).
2. Add entry to `config/models.json` (see example above).
3. Update this MD file.
4. Verify: Run a quick load test (ask me for a script if needed).

## Data Handling Rules
- **Missing Data:** Fill gaps (e.g., stock metrics per country) with available means; log replacements in `logs/data_imputes.xlsx`.
- **Quality Reports:** After fine-tuning, create `reports/completeness_[date].xlsx` showing % complete by stock/country.
- **Graphs:** Use PDF for visualizations (e.g., training loss curves).
- **No Deletions:** Only add/create—your code stays safe.

For questions or to add more models, just ask!
