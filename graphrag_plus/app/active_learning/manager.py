"""Active learning queue and correction loop."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from graphrag_plus.app.utils.io_utils import append_jsonl


class ActiveLearningManager:
    """Manage low-confidence/conflict review queue."""

    def __init__(self, queue_path: Path):
        self.queue_path = queue_path

    def enqueue(self, case: Dict[str, object]) -> None:
        """Add case to review queue."""
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **case}
        append_jsonl(self.queue_path, [row])

    def simulate_correction(self, case: Dict[str, object]) -> Dict[str, object]:
        """Simulated correction for dual-mode AL loop."""
        corrected = dict(case)
        corrected["simulated_corrected"] = True
        corrected["corrected_confidence"] = min(1.0, float(case.get("confidence", 0.5)) + 0.1)
        return corrected

    def process_cases(self, cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Queue and return simulated corrections."""
        corrected_rows: List[Dict[str, object]] = []
        for case in cases:
            self.enqueue(case)
            corrected_rows.append(self.simulate_correction(case))
        return corrected_rows

