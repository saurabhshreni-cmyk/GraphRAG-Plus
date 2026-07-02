"""Application settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load the project-root .env into the process environment so both the
# GRAPHRAG_-prefixed settings below and the unprefixed integration vars
# (NEO4J_*, OLLAMA_*, EMBEDDING_MODEL) are available everywhere.
load_dotenv()


class Settings(BaseSettings):
    """Runtime settings with safe defaults."""

    model_config = SettingsConfigDict(env_prefix="GRAPHRAG_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("graphrag_plus/data"))
    reports_dir: Path = Field(default=Path("graphrag_plus/data/reports"))
    graph_path: Path = Field(default=Path("graphrag_plus/data/graph.json"))
    chunks_path: Path = Field(default=Path("graphrag_plus/data/chunks.json"))
    corpora_dir: Path = Field(default=Path("graphrag_plus/data/corpora"))
    graph_versions_dir: Path = Field(default=Path("graphrag_plus/data/graph_versions"))
    trust_state_path: Path = Field(default=Path("graphrag_plus/data/trust_state.json"))
    calibration_state_path: Path = Field(default=Path("graphrag_plus/data/calibration_state.json"))
    review_queue_path: Path = Field(default=Path("graphrag_plus/data/review_queue.jsonl"))
    answers_log_path: Path = Field(default=Path("graphrag_plus/data/answers_log.jsonl"))
    run_logs_path: Path = Field(default=Path("graphrag_plus/data/run_logs.jsonl"))
    outputs_dir: Path = Field(default=Path("graphrag_plus/data/outputs"))
    cache_dir: Path = Field(default=Path("graphrag_plus/.cache"))
    temp_dir: Path = Field(default=Path("graphrag_plus/.cache/tmp"))

    chunk_size: int = 400
    chunk_overlap: int = 80
    extraction_threshold: float = 0.6
    answer_threshold: float = 0.7
    high_uncertainty_threshold: float = 0.55

    enable_calibration: bool = True
    enable_contradiction: bool = True
    enable_active_learning: bool = True
    analyst_mode_default: bool = False

    use_gnn: bool = True
    use_calibration: bool = True
    use_graph: bool = True
    use_vector: bool = True
    use_trust: bool = True

    default_trust_prior: float = 0.5
    source_trust_priors: dict[str, float] = Field(default_factory=dict)

    # Scoring weights — semantic dominates so on-topic chunks rank ahead of
    # off-topic ones with similar trust/confidence priors. Bumped further so
    # a strong cosine match can't be outvoted by trust + confidence priors
    # on a barely-relevant chunk.
    scoring_w1_semantic: float = 0.60
    scoring_w2_graph: float = 0.15
    scoring_w3_confidence: float = 0.10
    scoring_w4_trust: float = 0.10
    scoring_w5_uncertainty_penalty: float = 0.05

    random_seed: int = 42
    # LLM generation is on by default in the upgraded stack (Ollama-backed);
    # the generator falls back to extractive answers whenever the daemon is
    # unreachable, so this stays safe on hosts without Ollama.
    llm_enabled: bool = True
    llm_model_name: str = "qwen2.5:3b"

    # --- external integrations (unprefixed env vars from .env) --------------
    neo4j_uri: str = Field(
        default="", validation_alias=AliasChoices("NEO4J_URI", "GRAPHRAG_NEO4J_URI")
    )
    neo4j_username: str = Field(
        default="", validation_alias=AliasChoices("NEO4J_USERNAME", "GRAPHRAG_NEO4J_USERNAME")
    )
    neo4j_password: str = Field(
        default="", validation_alias=AliasChoices("NEO4J_PASSWORD", "GRAPHRAG_NEO4J_PASSWORD")
    )
    ollama_model: str = Field(
        default="qwen2.5:3b", validation_alias=AliasChoices("OLLAMA_MODEL", "GRAPHRAG_OLLAMA_MODEL")
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "GRAPHRAG_OLLAMA_BASE_URL"),
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "GRAPHRAG_EMBEDDING_MODEL"),
    )

    # Optional Postgres/Supabase connection string for durable, shared-across-
    # instances corpus storage. When set, corpora are persisted as JSON blobs
    # in Postgres (local dir is scratch); when empty, the file-based store is
    # used. Read from GRAPHRAG_DATABASE_URL, falling back to a bare
    # DATABASE_URL if present (see get_settings).
    database_url: str = ""


def validate_settings(settings: Settings) -> None:
    """Validate settings and fail fast with clear errors."""
    errors: list[str] = []

    for name in ("extraction_threshold", "answer_threshold", "high_uncertainty_threshold"):
        value = float(getattr(settings, name))
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name} must be in [0, 1], got {value}")

    if settings.chunk_size <= 0:
        errors.append("chunk_size must be > 0")
    if settings.chunk_overlap < 0:
        errors.append("chunk_overlap must be >= 0")
    if settings.chunk_overlap >= settings.chunk_size:
        errors.append("chunk_overlap must be < chunk_size")

    weights = [
        settings.scoring_w1_semantic,
        settings.scoring_w2_graph,
        settings.scoring_w3_confidence,
        settings.scoring_w4_trust,
        settings.scoring_w5_uncertainty_penalty,
    ]
    if any(weight < 0.0 for weight in weights):
        errors.append("scoring weights must be >= 0")
    if sum(weights) <= 0.0:
        errors.append("scoring weights sum must be > 0")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def get_settings() -> Settings:
    """Get settings instance."""
    settings = Settings()

    # Convenience: honour a bare DATABASE_URL (what Supabase/Neon hand out)
    # when the prefixed GRAPHRAG_DATABASE_URL isn't set, so users don't have
    # to rename the variable.
    if not settings.database_url:
        fallback = os.environ.get("DATABASE_URL", "")
        if fallback:
            settings.database_url = fallback

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.graph_versions_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    # If TEMP/TMP are not set (or point somewhere problematic), prefer a writable project-local temp dir.
    os.environ.setdefault("TEMP", str(settings.temp_dir))
    os.environ.setdefault("TMP", str(settings.temp_dir))

    validate_settings(settings)
    return settings
