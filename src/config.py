"""
src/config.py — Config loading & per-model config mapping with deep merge
"""
from __future__ import annotations

import os
import yaml
from copy import deepcopy

# Root of the project (one level up from src/)
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR  = os.path.join(ROOT_DIR, "config")
DEFAULT_CFG = os.path.join(CONFIG_DIR, "default.yaml")
LEGACY_CFG  = os.path.join(ROOT_DIR, "config.yaml")   # backward compat

# ─── Model → config mapping ──────────────────────────────────────────────────
# Checked in order — first substring match wins (case-insensitive).
# Fallback: default.yaml
MODEL_CONFIG_MAP: list[tuple[str, str]] = [
    # Specific / longer keywords first
    ("qwopus",                          "qwopus-coder-9b.yaml"),
    ("jackwrong",                       "qwopus-coder-9b.yaml"),
    ("minicpm",                         "minicpm-v4.yaml"),
    ("huihui-ai",                       "gemma4-huihui-qat-abliterated.yaml"),
    ("huihui-gemma-4-12b-it-qat",      "gemma4-huihui-qat-abliterated.yaml"),
    ("huhui",                           "gemma4-huhui-abliterated.yaml"),
    ("lmstudio-community",              "gemma4-lmstudio-qat.yaml"),
    ("gemma-4-12b-it-qat",             "gemma4-lmstudio-qat.yaml"),
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_default_config() -> dict:
    """Load default.yaml, with fallback to legacy config.yaml."""
    for path in (DEFAULT_CFG, LEGACY_CFG):
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        "No config found — create config/default.yaml"
    )


def load_model_config(model_rel_path: str) -> tuple[dict, str]:
    """
    Return (merged_config_dict, config_filename) for a model.

    Matches MODEL_CONFIG_MAP keywords against the model's relative path.
    Deep-merges model-specific config on top of default.yaml.
    Falls back to default.yaml if no match.
    """
    default_cfg = load_default_config()
    rel_lower = model_rel_path.lower().replace("\\", "/")

    for keyword, cfg_file in MODEL_CONFIG_MAP:
        if keyword.lower() in rel_lower:
            cfg_path = os.path.join(CONFIG_DIR, cfg_file)
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    model_cfg = yaml.safe_load(f) or {}
                return _deep_merge(default_cfg, model_cfg), cfg_file

    return default_cfg, "default.yaml"