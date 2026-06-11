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

GraphRAG++ supports **Python 3.11, 3.12, and 3.13**. Python 3.14 is not
targeted yet because several scientific Python dependencies are still
catching up. Heavy dependencies (PyTorch,
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
a lightweight fallback when `prometheus-client` is absent.

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

# Run the lightweight benchmark
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
| `POST` | `/ingest`           | Ingest files/URLs into a new or existing corpus (optional `corpus_name`) |
| `POST` | `/query`            | Hybrid retrieval + answer generation (optional `corpus_id`) |
| `GET`  | `/corpora`          | List all corpora (newest first) |
| `GET`  | `/corpora/active`   | Currently active corpus |
| `POST` | `/corpora/{id}/select` | Switch the active corpus |
| `DELETE` | `/corpora/{id}`   | Permanently delete a corpus |
| `GET`  | `/graph`            | Graph snapshot (nodes + edges) for visualization |
| `GET`  | `/graph/{node_id}`  | Neighborhood for a graph node |
| `GET`  | `/evaluate`         | Run the lightweight benchmark |
| `GET`  | `/metrics`          | Prometheus exposition |

Each ingest call creates an **isolated corpus** by default (preventing
cross-domain contamination); the corpus's domain is auto-detected from
its text (machine learning, physics, finance, biology, chemistry,
mathematics, computer science, or general).

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

### Resume demo dataset

The repo includes a small curated local demo corpus at
`graphrag_plus/data/corpora/corpus_demo_nova/`, generated from the sample
documents in `graphrag_plus/data/sample_docs/`. It gives the dashboard an
immediate graph to render after startup while keeping generated logs,
query outputs, and scratch corpora out of version control.

Good local demo questions:

- `Which source contradicts the cancellation claim?`
- `What did Nova Dynamics acquire?`
- `How are Orion Labs and Project Helios connected?`

---

## Deployment

### Live demo (Vercel)

| Surface | URL |
|---------|-----|
| Dashboard | https://graphrag-plus-dashboard.vercel.app |
| API | https://graphrag-plus-api.vercel.app (try [`/health`](https://graphrag-plus-api.vercel.app/health)) |

The backend runs as a **Vercel Python serverless function** via
[`api/index.py`](api/index.py): all writable paths are redirected to
`/tmp` through `GRAPHRAG_*` env overrides, and the committed demo corpus
is seeded on boot so queries work immediately. Note that `/tmp` is
ephemeral — corpora ingested on the Vercel demo survive only per
serverless instance. **For durable, shared-across-all-users state, run the
backend on Render with a persistent disk (below) — that's the recommended
production setup.**

To deploy your own copy:

```bash
vercel link --project my-graphrag-api && vercel deploy --prod   # repo root = backend
cd frontend
vercel link --project my-graphrag-dashboard
vercel env add VITE_API_BASE production    # https://<your-api>.vercel.app
vercel deploy --prod
# then on the backend project:
vercel env add GRAPHRAG_CORS_ORIGINS production   # https://<your-dashboard>.vercel.app
vercel deploy --prod
```

### Backend — Render with a persistent disk (recommended for production)

This repo ships a [`render.yaml`](render.yaml) Blueprint that provisions a
web service **with a 1 GB persistent disk** mounted at `/var/data`. Every
writable `GRAPHRAG_*` path points at that disk, so **ingested corpora
survive restarts and deploys and are shared by every user** hitting the
single instance. The demo corpus is seeded onto the disk automatically on
first boot (idempotent — kept across restarts).

**One-click deploy:**

1. Push this repo to GitHub (already done for the canonical copy).
2. Render Dashboard → **New** → **Blueprint** → select this repo. Render
   reads `render.yaml` and creates the service + disk.
3. Pick a plan that supports disks (**Starter** or higher — Render's free
   tier has no persistent disk and spins down on idle).
4. Deploy. The service comes up at
   `https://graphrag-plus-api.onrender.com` (or a name-suffixed variant).
5. Point the dashboard at it and re-deploy the frontend:
   ```bash
   cd frontend
   vercel env rm  VITE_API_BASE production    # remove the old Vercel API base
   vercel env add VITE_API_BASE production    # https://<your-render-url>.onrender.com
   vercel deploy --prod
   ```
6. If your dashboard URL differs from the default, update
   `GRAPHRAG_CORS_ORIGINS` in `render.yaml` (or the Render dashboard) and
   redeploy.

> **Why a disk and not autoscaling?** The corpus store is file-backed
> (`data/corpora/<id>/{meta,graph,chunks}.json`). A single instance with
> one mounted disk gives one writer and one source of truth — exactly what
> this store wants. To scale horizontally later, swap the `GraphStore` /
> `RetrievalService` persistence for Postgres/object storage behind their
> existing interfaces (see *Future improvements*).

**Build / start (handled by `render.yaml`, shown for reference):**

- Root directory: `graphrag_plus`
- Build: `pip install --upgrade pip && pip install -e .`
- Start: `uvicorn graphrag_plus.app.api.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

**Railway / Fly.io**: same idea — install `-e .`, attach a volume, and set
the `GRAPHRAG_*` paths to the mount. On Fly.io, add a `Dockerfile` that
copies the repo and runs the same uvicorn command, with a `[mounts]` block
pointing at the volume.

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
    │   ├── corpus/             ← multi-corpus manager (isolated graphs + indexes)
    │   ├── domain/             ← keyword-frequency domain classifier
    │   ├── retrieval/          ← vector + BM25 + graph expansion
    │   ├── scoring/            ← weighted score blend
    │   ├── gnn/                ← optional torch reranker (with fallback)
    │   ├── calibration/        ← confidence calibration
    │   ├── trust/              ← per-source trust priors
    │   ├── failure/            ← typed failure-mode classifier
    │   ├── active_learning/    ← review queue for low-confidence/conflict
    │   ├── analyst/            ← analyst-mode reasoning + follow-ups
    │   ├── evaluation/         ← benchmark + ablation runners
    │   ├── planning/           ← query intent heuristics
    │   ├── schemas/            ← Pydantic request/response models
    │   ├── utils/              ← logging, IO, metrics, run logger, runtime
    │   └── tests/              ← pytest suite (66 tests)
    ├── data/
    │   └── sample_docs/        ← shipped samples for the quickstart
    ├── scripts/                ← PowerShell helpers (demo, run_api, etc.)
    └── examples/api_requests.http
```

---

## Tech stack

- **Python 3.11, 3.12, or 3.13**
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

### Local pre-trust gate

On Windows, run the local quality gate from the repository root before
trusting answer-quality changes or demoing the app:

```powershell
.\graphrag_plus\scripts\quality_gate.ps1
```

The gate sets `PYTHONPATH`, applies safe Python autofixes (`ruff --fix`
and `black`) only when run with `-Fix`. By default it is verify-only: it
runs the install check, `ruff`, `black --check`, `mypy`, and the backend
pytest suite, then builds the frontend. It does not start, stop, ingest,
reset, or mutate runtime graph data. Pass `-ProbeHealth` when you already
have the backend running and want the gate to check `127.0.0.1:8765/health`.

```powershell
# Optional: allow the gate to format/fix before verification
.\graphrag_plus\scripts\quality_gate.ps1 -Fix

# Optional: include a live backend health probe
.\graphrag_plus\scripts\quality_gate.ps1 -ProbeHealth
```

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
