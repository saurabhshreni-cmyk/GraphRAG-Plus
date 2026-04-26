"""Source trust state manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from graphrag_plus.app.utils.io_utils import dump_json, load_json
from graphrag_plus.app.utils.logging_utils import get_logger, log_event


@dataclass
class SourceTrust:
    """Per-source trust metrics."""

    trust_score: float = 0.5
    agreement_count: int = 0
    contradiction_count: int = 0
    historical_accuracy: float = 0.5


class SourceTrustManager:
    """Maintain and persist source trust."""

    def __init__(self, state_path: Path, default_prior: float, priors: Dict[str, float]):
        self.logger = get_logger(self.__class__.__name__)
        self.state_path = state_path
        self.default_prior = default_prior
        self.priors = priors
        raw_state = load_json(self.state_path, default={})
        self._state: Dict[str, SourceTrust] = {
            key: SourceTrust(**value) for key, value in raw_state.items()
        }

    def _get_or_create(self, source_id: str) -> SourceTrust:
        if source_id not in self._state:
            self._state[source_id] = SourceTrust(
                trust_score=self.priors.get(source_id, self.default_prior),
                historical_accuracy=0.5,
            )
        return self._state[source_id]

    def get_trust_score(self, source_id: str) -> float:
        """Return trust score for source."""
        return self._get_or_create(source_id).trust_score

    def update(self, source_id: str, agrees: bool, is_correct: bool, low_confidence: bool) -> None:
        """Update source trust using observed outcomes."""
        state = self._get_or_create(source_id)
        if agrees:
            state.agreement_count += 1
            state.trust_score = min(1.0, state.trust_score + 0.03)
        else:
            state.contradiction_count += 1
            state.trust_score = max(0.0, state.trust_score - 0.05)
        if low_confidence:
            state.trust_score = max(0.0, state.trust_score - 0.02)
        state.historical_accuracy = (
            0.8 * state.historical_accuracy + 0.2 * (1.0 if is_correct else 0.0)
        )
        state.trust_score = min(1.0, max(0.0, 0.6 * state.trust_score + 0.4 * state.historical_accuracy))
        self.persist()
        log_event(
            self.logger,
            "trust_update",
            {"source_id": source_id, "state": asdict(state)},
        )

    def persist(self) -> None:
        """Persist trust state."""
        payload = {key: asdict(value) for key, value in self._state.items()}
        dump_json(self.state_path, payload)

