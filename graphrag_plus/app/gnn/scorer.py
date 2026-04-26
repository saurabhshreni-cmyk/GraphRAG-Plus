"""Lightweight graph scorer with optional PyTorch backend.

Torch is an optional dependency. When unavailable, ``GNNScorer`` falls back to
a deterministic, dependency-free linear blend of candidate features so the
pipeline degrades gracefully on lightweight installs.
"""

from __future__ import annotations

from typing import Dict, List

try:  # Optional dependency
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 - torch import can fail in many ways
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


_FEATURE_KEYS = ("semantic_score", "graph_score", "confidence_score", "trust_score")


def _features(candidate: Dict[str, float]) -> List[float]:
    return [float(candidate.get(key, 0.0)) for key in _FEATURE_KEYS]


if _TORCH_AVAILABLE:

    class TinyGraphScorer(nn.Module):  # type: ignore[misc]
        """Simple MLP scorer over candidate features."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.net(x)


class GNNScorer:
    """Wrapper around the tiny graph scorer with a deterministic fallback."""

    def __init__(self) -> None:
        if _TORCH_AVAILABLE:
            self.model: "TinyGraphScorer | None" = TinyGraphScorer()
            self.model.eval()
        else:
            self.model = None

    @property
    def torch_available(self) -> bool:
        return _TORCH_AVAILABLE

    def _fallback_score(self, candidates: List[Dict[str, float]]) -> List[float]:
        # Deterministic blend in [0, 1] when torch is unavailable.
        out: List[float] = []
        for candidate in candidates:
            sem, graph, conf, trust = _features(candidate)
            score = 0.35 * sem + 0.25 * graph + 0.2 * conf + 0.2 * trust
            out.append(max(0.0, min(1.0, score)))
        return out

    def score(self, candidates: List[Dict[str, float]]) -> List[float]:
        """Infer graph-aware scores. Returns [] for empty input."""
        if not candidates:
            return []
        if not _TORCH_AVAILABLE or self.model is None:
            return self._fallback_score(candidates)

        with torch.no_grad():  # type: ignore[union-attr]
            feats = torch.tensor(  # type: ignore[union-attr]
                [_features(c) for c in candidates],
                dtype=torch.float32,  # type: ignore[union-attr]
            )
            preds = self.model(feats).squeeze(-1).tolist()
        return [float(p) for p in preds]
