"""Runtime helpers for reproducibility and status."""

from __future__ import annotations

import random

import numpy as np

try:  # Optional dependency
    import torch  # type: ignore[import-not-found]

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from graphrag_plus.app.config.settings import Settings


def apply_global_seed(seed: int) -> None:
    """Apply deterministic seeds across supported libraries."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def enabled_modules(settings: Settings) -> list[str]:
    """Return human-readable enabled module names."""
    modules: list[str] = []
    if settings.use_graph:
        modules.append("graph")
    if settings.use_vector:
        modules.append("vector")
    if settings.use_gnn:
        modules.append("gnn")
    if settings.enable_calibration and settings.use_calibration:
        modules.append("calibration")
    if settings.use_trust:
        modules.append("trust")
    if settings.enable_contradiction:
        modules.append("contradiction")
    if settings.enable_active_learning:
        modules.append("active_learning")
    return modules


def backend_status(settings: Settings) -> dict[str, str]:
    """Return simple backend status info for CLI display."""
    return {
        "graph": "networkx-json",
        "vector": "tfidf-bm25",
        "llm": "enabled" if settings.llm_enabled else "disabled",
        "cache_dir": str(settings.cache_dir),
    }
