"""Active learning queue and correction loop."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from graphrag_plus.app.utils.io_utils import append_jsonl


class ActiveLearningManager:
    """Manage low-confidence/conflict review queue."""

    def __init__(self, queue_path: Path):
        self.queue_path = queue_path

    def enqueue(self, case: dict[str, object]) -> None:
        """Add case to review queue."""
        row = {"timestamp": datetime.now(UTC).isoformat(), **case}
        append_jsonl(self.queue_path, [row])

    def simulate_correction(self, case: dict[str, object]) -> dict[str, object]:
        """Simulated correction for dual-mode AL loop."""
        corrected = dict(case)
        corrected["simulated_corrected"] = True
        confidence_raw = case.get("confidence", 0.5)
        corrected["corrected_confidence"] = min(1.0, float(confidence_raw) + 0.1)  # type: ignore[arg-type]
        return corrected

    def process_cases(self, cases: list[dict[str, object]]) -> list[dict[str, object]]:
        """Queue and return simulated corrections."""
        corrected_rows: list[dict[str, object]] = []
        for case in cases:
            self.enqueue(case)
            corrected_rows.append(self.simulate_correction(case))
        return corrected_rows
