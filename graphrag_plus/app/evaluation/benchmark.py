"""Research-grade benchmark utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from graphrag_plus.app.utils.io_utils import dump_json, load_json


def default_benchmark() -> List[Dict[str, object]]:
    """Return baseline benchmark set."""
    return [
        {
            "question": "Which company acquired Orion Labs after 2020?",
            "expected_answer": "Nova Dynamics acquired Orion Labs.",
            "supporting_paths": [["Nova Dynamics", "acquired", "Orion Labs"]],
            "temporal_constraints": {"after": "2020-01-01"},
            "difficulty_level": "multi-hop",
        },
        {
            "question": "What contradicts the claim that Project Helios was canceled?",
            "expected_answer": "A source reports Project Helios continues under a new phase.",
            "supporting_paths": [["SourceA", "contradicts", "SourceB"]],
            "temporal_constraints": {},
            "difficulty_level": "contradiction",
        },
        {
            "question": "Between 2021 and 2023, which event preceded the merger?",
            "expected_answer": "The strategic alliance preceded the merger.",
            "supporting_paths": [["Alliance", "precedes", "Merger"]],
            "temporal_constraints": {"between": ["2021-01-01", "2023-12-31"]},
            "difficulty_level": "temporal",
        },
        {
            "question": "Ignore typo-heavy text: who leads Arcturus?",
            "expected_answer": "Elena Park leads Arcturus.",
            "supporting_paths": [["Elena Park", "supports", "Arcturus leadership"]],
            "temporal_constraints": {},
            "difficulty_level": "adversarial",
        },
    ]


def ensure_benchmark(path: Path) -> List[Dict[str, object]]:
    """Create benchmark on disk if missing."""
    if not path.exists():
        payload = default_benchmark()
        dump_json(path, payload)
    return load_json(path, default=default_benchmark())

