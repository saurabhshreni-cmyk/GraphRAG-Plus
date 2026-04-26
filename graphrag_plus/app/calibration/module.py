"""Confidence calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from graphrag_plus.app.utils.io_utils import dump_json, load_json


@dataclass
class CalibrationOutput:
    """Calibrated confidence output."""

    raw_confidence: float
    calibrated_confidence: float
    calibration_error: float


class CalibrationModule:
    """Reliability binning + temperature scaling + ECE."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        state = load_json(state_path, default={})
        self.temperature: float = float(state.get("temperature", 1.0))
        self.bin_stats: Dict[str, Dict[str, float]] = state.get("bin_stats", {})

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def calibrate(self, raw_confidence: float) -> CalibrationOutput:
        """Calibrate one confidence score using saved temperature."""
        raw = min(max(raw_confidence, 1e-6), 1 - 1e-6)
        logit = np.log(raw / (1 - raw))
        calibrated = float(self._sigmoid(logit / max(self.temperature, 1e-6)))
        ece = self.expected_calibration_error()
        return CalibrationOutput(raw_confidence=raw_confidence, calibrated_confidence=calibrated, calibration_error=ece)

    def fit_temperature(self, logits: List[float], labels: List[int]) -> float:
        """Fit scalar temperature via grid search."""
        if not logits or len(logits) != len(labels):
            return self.temperature
        best_t = self.temperature
        best_loss = float("inf")
        y = np.array(labels, dtype=np.float32)
        x = np.array(logits, dtype=np.float32)
        for temp in np.linspace(0.5, 3.0, num=50):
            probs = 1.0 / (1.0 + np.exp(-(x / temp)))
            eps = 1e-8
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            if loss < best_loss:
                best_loss = float(loss)
                best_t = float(temp)
        self.temperature = best_t
        self.persist()
        return self.temperature

    def update_reliability(self, confidences: List[float], labels: List[int], bins: int = 10) -> None:
        """Update reliability stats for ECE."""
        if not confidences or len(confidences) != len(labels):
            return
        hist = {str(i): {"count": 0, "conf_sum": 0.0, "acc_sum": 0.0} for i in range(bins)}
        for confidence, label in zip(confidences, labels):
            idx = min(bins - 1, int(confidence * bins))
            key = str(idx)
            hist[key]["count"] += 1
            hist[key]["conf_sum"] += float(confidence)
            hist[key]["acc_sum"] += float(label)
        self.bin_stats = hist
        self.persist()

    def expected_calibration_error(self) -> float:
        """Compute ECE from current bins."""
        total = sum(int(bucket["count"]) for bucket in self.bin_stats.values())
        if total == 0:
            return 0.0
        ece = 0.0
        for bucket in self.bin_stats.values():
            count = int(bucket["count"])
            if count == 0:
                continue
            avg_conf = float(bucket["conf_sum"]) / count
            avg_acc = float(bucket["acc_sum"]) / count
            ece += (count / total) * abs(avg_conf - avg_acc)
        return float(ece)

    def persist(self) -> None:
        """Persist calibration state."""
        dump_json(
            self.state_path,
            {
                "temperature": self.temperature,
                "bin_stats": self.bin_stats,
            },
        )

