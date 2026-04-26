# GraphRAG++

Research-grade backend for **graph-aware Retrieval-Augmented Generation** with
hybrid retrieval, uncertainty-aware reranking, source trust modeling,
contradiction handling, graph versioning, active learning, analyst-mode
explanations, and an ablation runner.

> Pure backend. No UI, no notebook integrations.

---

## Architecture

```
                ┌──────────────┐
files / URLs ──▶│  ingestion   │── chunks ─┐
                └──────────────┘           ▼
                                  ┌──────────────────┐
                                  │   extraction     │── entities, relations
                                  └──────────────────┘
                                           │
                                           ▼
                                ┌────────────────────┐
                                │ contradiction (opt)│
                                └────────────────────┘
                                           │
                                           ▼
                                ┌────────────────────┐
                                │  graph store       │  NetworkX-backed,
                                │  + versioning      │  snapshot per ingest
                                └────────────────────┘
                                           │
        question ─────────────────────────▶│
                                ┌────────────────────┐
                                │ retrieval          │  vector + BM25 +
                                │ (hybrid)           │  graph expansion
                                └────────────────────┘
                                           │
                                ┌────────────────────┐
                                │ scoring + GNN(opt) │  weighted blend
                                └────────────────────┘
                                           │
                                ┌────────────────────┐
                                │ calibration + trust│  per-source priors
                                └────────────────────┘
                                           │
                                ┌────────────────────┐
                                │ failure classifier │  abstain / partial /
                                │ + analyst (opt)    │  conflict resolution
                                └────────────────────┘
                                           │
                                           ▼
                                  QueryResponse JSON
```

### Module layout

| Path | Responsibility |
|------|----------------|
| `app/ingestion/` | File + URL loading (with SSRF guard), chunking |
| `app/extraction/` | Entity & relation extraction |
| `app/contradiction/` | Disagreement detection on `(subject, predicate)` claims |
| `app/graph/` | NetworkX-backed graph + versioning manager |
| `app/retrieval/` | Hybrid vector / BM25 / graph expansion |
| `app/scoring/` | Weighted blend over semantic / graph / confidence / trust / uncertainty |
| `app/gnn/` | Optional Torch reranker with deterministic fallback |
| `app/calibration/` | Confidence calibration |
| `app/trust/` | Per-source Beta-style trust priors |
| `app/failure/` | Typed failure modes (`NO_EVIDENCE`, `LOW_CONFIDENCE`, `CONFLICTING_EVIDENCE`, ...) |
| `app/active_learning/` | Review queue for low-confidence / conflict cases |
| `app/analyst/` | Analyst-mode reasoning steps + follow-ups |
| `app/evaluation/` | Benchmark & ablation runners |
| `app/api/` | FastAPI service (`/ingest`, `/query`, `/graph`, `/health`, `/evaluate`, `/metrics`) |
| `app/utils/metrics.py` | Prometheus metrics with no-op fallback |
| `app/pipeline.py` | End-to-end orchestration |

---

## Install

Base install is **lightweight** (no PyTorch). Heavy dependencies are opt-in.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .            # core only
python -m pip install -e .[dev]       # + lint/type/test tooling
python -m pip install -e .[gnn]       # + torch / torch-geometric
python -m pip install -e .[embeddings]  # + sentence-transformers, faiss
python -m pip install -e .[extras]    # + neo4j, chromadb, spacy
```

Optional modules degrade gracefully when their extras aren't installed
(GNN reranker uses a deterministic linear blend; `/metrics` returns a stub
when `prometheus-client` is missing).

---

## Quickstart

```bash
# Ingest sample docs
python -m graphrag_plus.app.cli ingest \
  --files graphrag_plus/data/sample_docs/sample1.txt \
          graphrag_plus/data/sample_docs/sample2.txt

# Query
python -m graphrag_plus.app.cli query \
  --question "Which source contradicts the cancellation claim?" \
  --analyst-mode

# Evaluate / Ablate
python -m graphrag_plus.app.cli evaluate
python -m graphrag_plus.app.cli run_ablation

# API
uvicorn graphrag_plus.app.api.main:app --reload
```

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/health`           | Liveness + capability info |
| POST   | `/ingest`           | Ingest files / URLs |
| POST   | `/query`            | Run hybrid retrieval + answer generation |
| GET    | `/graph/{node_id}`  | Neighborhood for a graph node |
| GET    | `/evaluate`         | Run benchmark stub |
| GET    | `/metrics`          | Prometheus exposition |

Sample requests live in [`examples/api_requests.http`](examples/api_requests.http).

---

## Configuration

`graphrag_plus/app/config/settings.py` — env vars prefixed `GRAPHRAG_`,
loaded from `.env` if present.

Important flags:

- `enable_calibration`, `use_calibration`
- `enable_contradiction`
- `enable_active_learning`
- `use_gnn`, `use_graph`, `use_vector`, `use_trust`
- `answer_threshold`, `high_uncertainty_threshold`
- `random_seed`

Settings are validated at startup (weight ranges, threshold bounds, chunk
overlap < chunk size).

---

## Output guarantees

Every query response includes:

- `query_id`, `graph_version_id`, `answer_state` (`updated` / `stale`)
- `raw_confidence`, `calibrated_confidence`, `calibration_error`
- per-evidence `trust_score`, `semantic_score`, `graph_score`, `final_score`
- `failure_type`, `mitigation_strategy_used`, `explanation`
- conflicting evidence (driven by the **real** contradiction reasoner —
  no substring heuristics)
- artifact under `data/outputs/<query_id>.json`
- structured run log line in `data/run_logs.jsonl`

---

## Observability

- **`/metrics`** exposes Prometheus counters / histograms:
  `graphrag_queries_total`, `graphrag_query_latency_seconds`,
  `graphrag_module_latency_seconds{module="..."}`,
  `graphrag_ingest_total`, `graphrag_stage_errors_total{stage="..."}`.
- **Run logs** (`data/run_logs.jsonl`) record per-query timing for every
  module (planning, retrieval, scoring, generation) and ingestion stages
  (load, chunk, extract, contradiction, graph upsert, index build).

---

## Security

- **SSRF guard** on URL ingestion: scheme allowlist (`http`/`https`),
  resolved-IP block (private / loopback / link-local / multicast / reserved /
  AWS metadata IP), manual redirect re-validation, response size cap (10 MB).
- Secrets via env vars only (`.env.example` provided; `.env` is gitignored).
- All external inputs validated at ingestion boundary.

---

## Reproducibility

- Global seeding for `random`, `numpy`, and `torch` (when installed).
- Default seed configurable via `GRAPHRAG_RANDOM_SEED`.
- Per-query JSON artifacts + JSONL run log enable deterministic replay.

---

## Development

```bash
python -m pip install -e .[dev]

ruff check .
black --check .
mypy graphrag_plus/app
pytest --cov=graphrag_plus/app --cov-report=term-missing
```

Continuous integration runs on push / PR (Python 3.11 & 3.12); see
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

## Troubleshooting

- `pip install -e .` fails on locked-down Windows machines:
  - Always use a venv.
  - Try `python -m pip install . --no-build-isolation`.
  - Last resort: `python -m pip install --user .`
- "No module named torch": expected. Install the GNN extra (`.[gnn]`)
  if you want the Torch-backed reranker; otherwise the deterministic
  fallback is used automatically.
- Quick health check: `python -m graphrag_plus.app.cli health_check`.
