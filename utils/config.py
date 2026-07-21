import json
import os
from typing import Any, Dict


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Saves a configuration dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def apply_overrides(config: Dict[str, Any], overrides: list[str]) -> None:
    """
    Applies a list of key=value overrides to a nested dictionary using dot notation.
    Example: ["env.num_decks=1", "train.ent_coef=0.01"]
    """
    if not overrides:
        return

    for override in overrides:
        if "=" not in override:
            print(f"Warning: Invalid override format '{override}', expected key=value.")
            continue

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Try to parse value as int, float, or bool
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        else:
            try:
                if "." in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                value = value_str  # Keep as string

        # Traverse dictionary and set value
        current = config
        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
