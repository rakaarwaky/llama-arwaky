"""
src/config.py — Config loading & per-model config mapping
"""
from __future__ import annotations

import os
import yaml

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


def load_default_config() -> dict:
    """Load default.yaml, with fallback to legacy config.yaml."""
    for path in (DEFAULT_CFG, LEGACY_CFG):
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "No config found — create config/default.yaml"
    )


def load_model_config(model_rel_path: str) -> tuple[dict, str]:
    """
    Return (config_dict, config_filename) for a model.

    Matches MODEL_CONFIG_MAP keywords against the model's relative path.
    Falls back to default.yaml if no match.
    """
    rel_lower = model_rel_path.lower().replace("\\", "/")

    for keyword, cfg_file in MODEL_CONFIG_MAP:
        if keyword.lower() in rel_lower:
            cfg_path = os.path.join(CONFIG_DIR, cfg_file)
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    return yaml.safe_load(f), cfg_file

    return load_default_config(), "default.yaml"
