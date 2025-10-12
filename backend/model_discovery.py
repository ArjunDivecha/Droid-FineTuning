import os
import json
import sys
from pathlib import Path

def discover_models(search_dir):
    """Discovers MLX models in a given directory."""
    models = []
    base_dir = Path(search_dir)

    if not base_dir.is_dir():
        return []

    for item in base_dir.iterdir():
        if item.is_dir():
            # Check if it's a valid model directory (contains config.json)
            config_path = item / "config.json"
            if config_path.is_file():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        models.append({
                            "name": item.name,
                            "path": str(item),
                            "model_type": config.get("model_type", "unknown"),
                            "vocab_size": config.get("vocab_size", 0)
                        })
                except Exception as e:
                    # If we can't read the config, still add the model with unknown metadata
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "model_type": "unknown",
                        "vocab_size": 0
                    })
    return models

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
        discovered_models = discover_models(search_directory)
        print(json.dumps({"models": discovered_models}))
    else:
        print(json.dumps({"models": [], "error": "No search directory provided"}))
