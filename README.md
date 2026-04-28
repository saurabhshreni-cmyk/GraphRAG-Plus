# GraphRAG++

[![CI](https://github.com/saurabhshreni-cmyk/GraphRAG-Plus/actions/workflows/ci.yml/badge.svg)](https://github.com/saurabhshreni-cmyk/GraphRAG-Plus/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A research-grade, production-ready **Graph-aware Retrieval-Augmented
> Generation** backend with hybrid retrieval, source trust modeling,
> confidence calibration, contradiction handling, and graph versioning.

GraphRAG++ ships as a **FastAPI backend** plus an optional **React + Vite
+ Tailwind dashboard** that visualises the knowledge graph and explains,
step by step, how each answer was reached.

---

## Why this project matters

Traditional RAG pipelines have three blind spots that make them hard to
trust in real applications:

1. **They retrieve, but don't reason about *which sources to trust*.**
   GraphRAG++ maintains per-source trust priors that update with every
   contradiction or low-confidence answer.
2. **They emit a confidence number that doesn't mean anything.**
   GraphRAG++ runs a calibration module so the confidence you see actually
   correlates with answer correctness.
3. **They silently paper over conflicting evidence.**
   GraphRAG++ runs a real contradiction reasoner at ingestion, persists
   the disagreement signal, and surfaces conflicting sources at query
   time with explicit resolution rationale.

On top of that you get **graph versioning** (every ingest snapshots the
graph and tags answers with the version they were derived from, so you
can detect when a stored answer goes stale), **active learning queues**
for low-confidence cases, an **analyst mode** that returns reasoning
steps + follow-up questions, and a Prometheus `/metrics` endpoint with
per-stage latency histograms.

---

## Key features

| Feature | What it does |
|---------|--------------|
| **Hybrid retrieval** | Vector + BM25 + graph expansion fused with weighted scoring |
| **Source trust modeling** | Per-source Beta-style priors that update on contradictions and corrections |
| **Confidence calibration** | Maps raw model confidence to a calibrated probability |
| **Contradiction reasoning** | Detects disagreement on `(subject, predicate)` claims, persists the signal, threads it into query responses |
| **Graph versioning** | Snapshots the graph on every ingest; flags answers as `stale` when their supporting nodes change |
| **Failure-mode classifier** | Typed states: `NO_EVIDENCE`, `LOW_CONFIDENCE`, `CONFLICTING_EVIDENCE`, `HIGH_UNCERTAINTY`, `LLM_FAILURE` |
| **Active learning queue** | Routes low-confidence and conflicting cases to a JSONL review queue |
| **Analyst mode** | Returns explicit reasoning steps and follow-up questions for human review |
| **Optional GNN reranker** | Tiny PyTorch MLP over candidate features; deterministic linear-blend fallback when torch isn't installed |
| **SSRF-hardened ingestion** | Scheme allowlist, private/loopback/link-local IP block, manual redirect re-validation, response size cap |
| **Prometheus `/metrics`** | Counters + histograms for queries, ingest, per-module latency, and per-stage errors |
| **Reproducibility** | Global seeding (random / numpy / torch), per-query JSON artifacts, JSONL run logs |

---

## Architecture

```
   files / URLs
        │
        ▼
  ┌──────────────┐         ┌────────────────────┐
  │  ingestion   │── chunks│   extraction       │── entities, relations
  │ (SSRF guard) │────────▶│ (rule-based)       │
  └──────────────┘         └────────────────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │ contradiction      │  per-chunk
                           │ reasoner (opt)     │  conflict map
                           └────────────────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │ graph store        │  NetworkX-backed,
                           │ + versioning       │  snapshot per ingest
                           └────────────────────┘
                                     │
                                     │
   question  ──────────────────────▶ │
                           ┌────────────────────┐
                           │ retrieval (hybrid) │  vector + BM25 +
                           │                    │  graph expansion
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

---

## Installation

GraphRAG++ requires **Python 3.11+**. Heavy dependencies (PyTorch,
sentence-transformers, FAISS, etc.) are **opt-in** so the base install
stays lightweight (~150 MB).

```bash
git clone https://github.com/saurabhshreni-cmyk/GraphRAG-Plus.git
cd GraphRAG-Plus

python -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install -U pip

cd graphrag_plus

python -m pip install -e .            # core (lightweight)
python -m pip install -e .[dev]       # + ruff/black/mypy/pytest-cov
python -m pip install -e .[gnn]       # + torch / torch-geometric
python -m pip install -e .[embeddings]   # + sentence-transformers, faiss
python -m pip install -e .[extras]    # + neo4j, chromadb, spacy
```

Optional modules **degrade gracefully**: the GNN reranker uses a
deterministic linear blend when torch is missing, and `/metrics` returns
a stub when `prometheus-client` is absent.

---

## Usage

### CLI

```bash
# Ingest sample documents
python -m graphrag_plus.app.cli ingest \
  --files graphrag_plus/data/sample_docs/sample1.txt \
          graphrag_plus/data/sample_docs/sample2.txt

# Ask a question (analyst mode shows reasoning steps)
python -m graphrag_plus.app.cli query \
  --question "Which source contradicts the cancellation claim?" \
  --analyst-mode

# Run the benchmark stub
python -m graphrag_plus.app.cli evaluate

# Run the ablation matrix
python -m graphrag_plus.app.cli run_ablation
```

### REST API

```bash
uvicorn graphrag_plus.app.api.main:app --reload
```

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health`           | Liveness + capability info |
| `POST` | `/ingest`           | Ingest local files and/or URLs |
| `POST` | `/query`            | Hybrid retrieval + answer generation |
| `GET`  | `/graph`            | Full graph snapshot (nodes + edges) for visualization |
| `GET`  | `/graph/{node_id}`  | Neighborhood for a graph node |
| `GET`  | `/evaluate`         | Run the benchmark stub |
| `GET`  | `/metrics`          | Prometheus exposition |

CORS origins are controlled by `GRAPHRAG_CORS_ORIGINS` (comma-separated).
Defaults cover the local Vite dev servers (`http://localhost:5173`,
`http://127.0.0.1:5173`).

#### Example: ingest + query

```bash
# Ingest
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "graphrag_plus/data/sample_docs/sample1.txt",
      "graphrag_plus/data/sample_docs/sample2.txt"
    ],
    "urls": []
  }'

# Query
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which source contradicts the cancellation claim?",
    "top_k": 3,
    "analyst_mode": true
  }'
```

Sample response (truncated):

```json
{
  "query_id": "qry_0b191bd16950",
  "answer": "Partial answer: On 2024-02-10, another source contradicts ...",
  "raw_confidence": 0.5,
  "calibrated_confidence": 0.5,
  "calibration_error": 0.0,
  "evidence": [{"id": "doc_e08…_ch_0", "trust_score": 0.5, "final_score": 0.52}],
  "failure_type": "LOW_CONFIDENCE",
  "mitigation_strategy_used": "partial_answer_with_warning",
  "reasoning_steps": [
    "Parsed question and decomposed into subqueries.",
    "Retrieved hybrid evidence from vector, keyword, and graph signals.",
    "Applied uncertainty-aware reranking with trust and confidence."
  ],
  "graph_version_id": "v20260426080950619019",
  "answer_state": "updated"
}
```

More ready-to-run requests live in
[`graphrag_plus/examples/api_requests.http`](graphrag_plus/examples/api_requests.http).

---

## Frontend dashboard

The `frontend/` folder is a **React 18 + Vite 5 + Tailwind 3** SPA that
talks to the FastAPI backend over the `VITE_API_BASE` URL.

Highlights:

- Dark-by-default UI with a polished light mode toggle
- Glassmorphism cards, Framer Motion transitions, micro-interactions
- **Interactive knowledge-graph view** (`react-force-graph-2d`) — zoom,
  pan, hover, color-coded by node type
- **Reasoning Story** panel — five animated steps explaining how the
  answer was reached, with synchronized graph-node highlighting
- Animated calibrated-confidence bar, failure-mode badge, evidence list
- Toast notifications for success/error, loading skeletons, spinner states

Run it locally (with the backend already running on `:8765`):

```bash
cd frontend
cp .env.example .env       # optional — defaults to local backend
npm install
npm run dev                # http://localhost:5173
```

Build for production:

```bash
npm run build              # outputs to frontend/dist
npm run preview            # http://localhost:4173
```

See [`frontend/README.md`](frontend/README.md) for full notes.

---

## Deployment

### Backend — Render / Railway / Fly.io

Any of these PaaS hosts work. The repo's existing layout means the
backend's working directory should be `graphrag_plus/` and the Python
package root should be the parent (`PYTHONPATH=.`).

**Render** (recommended starting point):

1. New → **Web Service** → connect this repo.
2. **Root directory:** `graphrag_plus`
3. **Build command:**
   ```bash
   python -m pip install --upgrade pip && python -m pip install -e .
   ```
4. **Start command:**
   ```bash
   PYTHONPATH=.. uvicorn graphrag_plus.app.api.main:app --host 0.0.0.0 --port $PORT
   ```
5. **Environment variables:**
   - `GRAPHRAG_CORS_ORIGINS=https://<your-vercel-app>.vercel.app`
   - `PYTHON_VERSION=3.12`

**Railway / Fly.io**: same idea — install `-e .`, set `PYTHONPATH=..`,
start uvicorn binding to `$PORT`. On Fly.io, add a `Dockerfile` that
copies the repo and runs the same command.

### Frontend — Vercel

1. **Import Project** → point at this repo, pick the `frontend/` folder
   as the project root (Vercel auto-detects Vite).
2. **Build command:** `npm run build` (default).
3. **Output directory:** `dist` (default).
4. **Environment variables:**
   - `VITE_API_BASE=https://<your-backend-domain>`
5. After deploy, copy the Vercel URL into the backend's
   `GRAPHRAG_CORS_ORIGINS` env var so CORS preflights succeed.

---

## Folder structure

```
GraphRAG-Plus/
├── README.md                   ← you are here
├── LICENSE                     ← MIT
├── .github/workflows/ci.yml    ← lint, format check, mypy, tests with coverage
├── frontend/                   ← React + Vite + Tailwind dashboard
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.js
│   ├── public/favicon.svg
│   └── src/
│       ├── App.jsx             ← layout + state orchestration
│       ├── api.js              ← typed wrapper around FastAPI calls
│       └── components/         ← Header, IngestPanel, QueryBox, ResultCard,
│                                 ConfidenceBar, GraphView, ReasoningStory,
│                                 ThemeToggle, Spinner
└── graphrag_plus/
    ├── pyproject.toml          ← deps, extras, ruff/black/mypy/coverage config
    ├── .env.example
    ├── app/
    │   ├── pipeline.py         ← end-to-end orchestration
    │   ├── api/                ← FastAPI service
    │   ├── cli.py
    │   ├── config/             ← Pydantic settings (env-prefixed GRAPHRAG_*)
    │   ├── ingestion/          ← file + URL loaders (SSRF-guarded), chunker
    │   ├── extraction/         ← entity / relation extraction
    │   ├── contradiction/      ← (subject, predicate) disagreement detection
    │   ├── graph/              ← NetworkX store + versioning manager
    │   ├── retrieval/          ← vector + BM25 + graph expansion
    │   ├── scoring/            ← weighted score blend
    │   ├── gnn/                ← optional torch reranker (with fallback)
    │   ├── calibration/        ← confidence calibration
    │   ├── trust/              ← per-source trust priors
    │   ├── failure/            ← typed failure-mode classifier
    │   ├── active_learning/    ← review queue for low-confidence/conflict
    │   ├── analyst/            ← analyst-mode reasoning + follow-ups
    │   ├── evaluation/         ← benchmark + ablation runners
    │   ├── planning/           ← query planner stub
    │   ├── schemas/            ← Pydantic request/response models
    │   ├── utils/              ← logging, IO, metrics, run logger, runtime
    │   └── tests/              ← pytest suite (21 tests)
    ├── data/
    │   └── sample_docs/        ← shipped samples for the quickstart
    ├── scripts/                ← PowerShell helpers (demo, run_api, etc.)
    └── examples/api_requests.http
```

---

## Tech stack

- **Python 3.11+**
- **FastAPI** + **Uvicorn** — REST surface
- **Pydantic v2** — request/response models, settings
- **NetworkX** — in-memory graph store
- **rank-bm25** + **scikit-learn** — keyword + vector retrieval
- **PyTorch** *(optional)* — GNN reranker
- **prometheus-client** — `/metrics` exposition
- **httpx** + **BeautifulSoup** + **pypdf** — ingestion adapters
- **pytest** + **ruff** + **black** + **mypy** — quality gate
- **GitHub Actions** — CI on Python 3.11 and 3.12

---

## Configuration

Settings live in `graphrag_plus/app/config/settings.py` and are loadable
from environment variables (prefix `GRAPHRAG_`) or a local `.env`.

Notable flags:

- `enable_calibration`, `use_calibration`
- `enable_contradiction`
- `enable_active_learning`
- `use_gnn`, `use_graph`, `use_vector`, `use_trust`
- `answer_threshold`, `high_uncertainty_threshold`
- `random_seed`

Settings are validated at startup (weight ranges, threshold bounds,
`chunk_overlap < chunk_size`).

---

## Observability

Every query response and ingest call records:

- `query_id`, `graph_version_id`, `answer_state` (`updated` / `stale`)
- raw + calibrated confidence + calibration error
- per-evidence `trust_score`, `semantic_score`, `graph_score`, `final_score`
- typed `failure_type` and `mitigation_strategy_used`
- artifact under `data/outputs/<query_id>.json`
- run-log line in `data/run_logs.jsonl` with per-module latency

`GET /metrics` exposes Prometheus counters and histograms:

```
graphrag_queries_total{failure_type="..."}
graphrag_query_latency_seconds_{count,sum,bucket}
graphrag_module_latency_seconds_{count,sum,bucket}{module="..."}
graphrag_ingest_total
graphrag_ingest_documents_total
graphrag_stage_errors_total{stage="..."}
```

---

## Development

```bash
cd graphrag_plus
python -m pip install -e .[dev]

ruff check .
black --check .
mypy app
pytest --cov=app --cov-report=term-missing
```

CI runs on every push / PR (Python 3.11 & 3.12) — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Future improvements

- **Persistent contradiction map** (currently in-process) → `data/contradictions.json` for CLI workflows
- **FAISS-backed retrieval** for >10k chunks (already in `[embeddings]` extra)
- **Neo4j-backed graph store** behind the existing `GraphStore` interface (`[extras]` already pins `neo4j>=5.20`)
- **Async ingest/query** to unblock the FastAPI event loop
- **Auth middleware** (API key / JWT) for non-`/health|/metrics` endpoints
- **Calibration backfill job** with rolling-window compaction
- **Graph drift alerting** (`graphrag_stale_answers_total` Prometheus counter)
- **Hypothesis-driven fuzz tests** for `validate_url`

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## Acknowledgements

Built as a research-oriented exploration of how to make RAG systems
*honest*: about what they retrieved, how confident they are, when their
sources disagree, and when their stored answers go stale.
