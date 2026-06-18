"""
src/models.py — GGUF model scanner & mmproj detection
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .config import ROOT_DIR


@dataclass
class ModelEntry:
    path:    str     # absolute path to .gguf
    rel:     str     # relative path from ROOT_DIR
    size_gb: float   # file size in GiB
    name:    str     # short display name  (parent_dir/filename)


def find_models() -> list[ModelEntry]:
    """Recursively scan ROOT_DIR for non-mmproj .gguf files."""
    entries: list[ModelEntry] = []

    skip_dirs = {'.git', 'config', 'src', 'llama-b9656', '__pycache__'}

    for root, dirs, files in os.walk(ROOT_DIR, followlinks=True):
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith('.') and d not in skip_dirs
        )
        for fname in sorted(files):
            if not fname.endswith('.gguf'):
                continue
            if 'mmproj' in fname or 'mmprj' in fname:
                continue
            full_path = os.path.join(root, fname)
            rel       = os.path.relpath(full_path, ROOT_DIR)
            size_gb   = os.path.getsize(full_path) / (1024 ** 3)
            parent    = os.path.basename(os.path.dirname(full_path))
            name      = f"{parent}/{fname}"
            entries.append(ModelEntry(full_path, rel, size_gb, name))

    return entries


def find_mmproj(model_path: str) -> str | None:
    """Return the mmproj .gguf file in the same directory, if any."""
    model_dir = os.path.dirname(model_path)
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith('.gguf') and ('mmproj' in fname or 'mmprj' in fname):
            return os.path.join(model_dir, fname)
    return None
