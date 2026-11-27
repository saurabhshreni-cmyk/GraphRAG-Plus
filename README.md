# GraphRAG++

[![CI](https://github.com/saurabhshreni-cmyk/GraphRAG-Plus/actions/workflows/ci.yml/badge.svg)](https://github.com/saurabhshreni-cmyk/GraphRAG-Plus/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A **trust-aware Graph RAG** system: it builds a knowledge graph from your
> documents, answers questions with hybrid graph + vector + keyword
> retrieval, has a second reasoning model check every answer against the
> evidence, and says so plainly when the documents don't support an answer.

**▶ Live demo: [graphrag-plus-dashboard.vercel.app](https://graphrag-plus-dashboard.vercel.app)**
 · API: [graphrag-plus-api.vercel.app/health](https://graphrag-plus-api.vercel.app/health)

It opens on a small demo corpus. Try the example questions under the Ask box:
*"Was Project Helios cancelled?"* shows contradiction handling, and
*"What is the capital of France?"* shows it refusing a question the documents
can't answer. Or upload your own PDF / DOCX / TXT / MD / HTML file.

The whole stack runs on **free tiers with no credit card**: Vercel, Groq,
Gemini and Neon. See [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md).

---

## Why this project matters

Traditional RAG pipelines have blind spots that make them hard to trust:

1. **They retrieve, but don't reason about *which sources to trust*.**
   GraphRAG++ keeps per-source trust priors that update with every
   contradiction or low-confidence answer.
2. **They emit a confidence number that doesn't mean anything.**
   A calibration module maps raw scores to a probability that tracks
   correctness.
3. **They silently paper over conflicting evidence.**
   A contradiction reasoner runs at ingestion, persists the disagreement,
   and surfaces conflicting sources at query time with a resolution.
4. **They answer anyway.** GraphRAG++ abstains (`NO_EVIDENCE`) when retrieval
   finds nothing relevant, and a reasoning model re-checks every generated
   answer against the evidence before it is shown.

On top of that: **graph versioning** (answers are tagged with the graph
version they came from and flagged `stale` when it changes), **entity
resolution** ("Apple", "Apple Inc" and "Apple Incorporated" become one
node), an **analyst mode** with reasoning steps and follow-up questions, and a
Prometheus `/metrics` endpoint.

---

## How a question is answered

```
 upload / URL
      │
      ▼
 ┌──────────────┐   ┌───────────────────────────┐   ┌─────────────────────┐
 │ ingestion    │──▶│ extraction                │──▶│ knowledge graph     │
 │ PDF/DOCX/... │   │ spaCy NER  +  LLM typed   │   │ + entity resolution │
 │ SSRF-guarded │   │ relations (Groq / Ollama) │   │ + versioning        │
 └──────────────┘   └───────────────────────────┘   └─────────────────────┘
      │                         │                              │
      ▼                         ▼                              │
 ┌──────────────┐   ┌───────────────────────────┐              │
 │ chunks       │──▶│ FAISS embeddings (Gemini  │              │
 │ BM25, TF-IDF │   │ or local bge-large)       │              │
 └──────────────┘   └───────────────────────────┘              │
                                                               │
 question ──▶ hybrid retrieval: BM25 + dense + graph signal ◀──┘
                 │   (relevance floors calibrated per embedding model)
                 ▼
            scoring + trust + calibration ──▶ no evidence? ──▶ abstain
                 │
                 ▼
            draft answer (gpt-oss-20b / local LLM), grounded in the evidence
                 │
                 ▼
            reasoning verifier (gpt-oss-120b / DeepSeek R1) checks the draft
                 │
                 ▼
            answer + calibrated confidence + evidence + conflicts + reasoning trace
```

The same code runs locally and in the cloud; only the model providers change:

| Stage | Cloud (live demo) | Local |
|---|---|---|
| Relation extraction | Groq `openai/gpt-oss-20b` (JSON mode) | Ollama `qwen3.5:4b` |
| Draft answers | Groq `openai/gpt-oss-20b` | Ollama `qwen3.5:4b` |
| Answer verification | Groq `openai/gpt-oss-120b`, high reasoning effort | Ollama `deepseek-r1:8b` |
| Embeddings | Gemini `gemini-embedding-001`, 1024-dim | `BAAI/bge-large-en-v1.5`, 1024-dim |
| Graph signal | In-process graph (entity matches) | Neo4j Personalized PageRank when `NEO4J_*` is set |
| Corpus storage | Neon Postgres, shared by every serverless instance | Local disk |

Every model is optional: without an LLM the system answers extractively;
without embeddings it uses BM25 + TF-IDF + the graph.

---

## Key features

| Feature | What it does |
|---------|--------------|
| **Hybrid retrieval** | BM25 + dense vectors (FAISS) + graph signal, blended with weights; relevance floors calibrated per embedding model ([`calibrate_backend.py`](graphrag_plus/scripts/calibrate_backend.py)) |
| **LLM knowledge graph** | spaCy NER plus LLM-extracted typed relations (`acquired`, `founded_by`, `headquartered_in`, …) |
| **Entity resolution** | Merges aliases and legal-suffix variants, without folding products into their company ("Apple Music" stays separate) |
| **Reasoning verification** | A second, larger model checks each draft against the evidence and can correct it; the UI shows which model checked it and what it looked at |
| **Contradiction reasoning** | Flags disagreement on single-valued facts (`status`, `founded_in`, …), not on multi-valued ones (`based_in`, `acquired`) |
| **Abstention** | Typed failure modes: `NO_EVIDENCE`, `LOW_CONFIDENCE`, `CONFLICTING_EVIDENCE`, `HIGH_UNCERTAINTY`, `LLM_FAILURE` |
| **Source trust + calibration** | Per-source priors that update on contradictions; calibrated confidence |
| **Graph versioning** | Snapshots the graph on every ingest; flags stored answers `stale` when their supporting nodes change |
| **Multi-corpus isolation** | Each upload is its own corpus with its own graph and indexes; the demo corpus is protected from deletion |
| **Free-tier friendly** | Embeddings are stored with each corpus and reused across serverless instances; query embeddings are cached; transient API errors pause rather than disable |
| **Public-demo safety** | Per-client rate limits (queries, ingests, deletes), a 4 MB upload cap, SSRF-guarded URL fetching, and API keys / database URLs redacted from every log line |
| **Observability** | Prometheus `/metrics`, per-stage latency, per-query JSON artifacts, JSONL run logs |

---

## Quick start (local)

Python **3.11 – 3.13** and Node 18+.

```bash
git clone https://github.com/saurabhshreni-cmyk/GraphRAG-Plus.git
cd GraphRAG-Plus
python -m venv venv
. venv/bin/activate                    # Windows: venv\Scripts\activate
pip install -r requirements.txt        # the same bundle the cloud API runs

# API on :8765 (the dashboard's default)
PYTHONPATH=. uvicorn graphrag_plus.app.api.main:app --port 8765
```

```bash
cd frontend
npm install
npm run dev                            # http://localhost:5173
```

Models are picked up automatically:

- **Ollama** running locally → local LLM extraction, answers and DeepSeek R1 verification.
- **`GROQ_API_KEY`** → Groq instead.
- **`GEMINI_API_KEY`** → Gemini embeddings. Install `sentence-transformers` to use local bge-large instead.
- **`GRAPHRAG_DATABASE_URL`** → Postgres storage.
- **`NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`** → Neo4j graph with Personalized PageRank.

Package extras for development (`cd graphrag_plus`):

```bash
pip install -e .[dev]          # pytest, ruff, black, mypy, faiss
pip install -e .[embeddings]   # sentence-transformers + faiss (local bge-large)
pip install -e .[extras]       # neo4j, spacy, ...
pip install -e .[gnn]          # optional torch reranker
```

---

## Deployment (free, no card)

| Layer | Service |
|---|---|
| Dashboard | Vercel project `graphrag-plus-dashboard` (root `frontend/`), deploys from `main` |
| API | Vercel Python project `graphrag-plus-api` (repo root, served from [`index.py`](index.py)), deploys from `main` |
| LLMs | Groq free API |
| Embeddings | Gemini free API (Jina as a fallback) |
| Storage | Neon Postgres free plan |

Step-by-step setup is in **[deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md)**: keys, Neon, calibrating
the relevance floors for your embedding model, and environment variables.

Notes on the serverless design:

- **Storage:** writable paths are redirected to `/tmp`, and Postgres is the source of truth.
  Each corpus (metadata, graph, chunks and its FAISS index) is one row, hydrated on demand by
  whichever instance serves the request.
- **Demo corpus:** seeded on boot, so the site always has something to show.
- **No PyTorch or Neo4j in the cloud bundle:** it stays at about 370 MB, well under Vercel's
  500 MB limit for Python.

---

## REST API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Liveness + whether an LLM is enabled |
| `POST` | `/ingest/file` | Upload one document (multipart, ≤ 4 MB) into a new or existing corpus |
| `POST` | `/ingest` | Ingest URLs (or server-side file paths when self-hosted) |
| `POST` | `/query` | Hybrid retrieval + generated, verified answer (optional `corpus_id`) |
| `GET`  | `/corpora` | List corpora (newest first) with counts and a `deletable` flag |
| `GET`  | `/corpora/active` | The active corpus |
| `POST` | `/corpora/{id}/select` | Switch the active corpus |
| `DELETE` | `/corpora/{id}` | Delete a corpus (the demo corpus is refused) |
| `GET`  | `/corpora/{id}/chunks` | Paginated chunks of a corpus |
| `GET`  | `/graph` | Graph snapshot for visualization (`mode=important\|full`) |
| `GET`  | `/graph/{corpus_id}/full` | Full graph of a corpus |
| `GET`  | `/graph/{corpus_id}/stats` | Node/relationship counts by type, most-connected entities |
| `GET`  | `/graph/{corpus_id}/query-path?q=` | Entities a question traverses (drives the red path in the UI) |
| `GET`  | `/graph/{corpus_id}/entity/{name}` | An entity with its neighbours and source chunks |
| `GET`  | `/graph/{node_id}` | Neighbourhood of a graph node |
| `GET`  | `/evaluate` | Run the lightweight benchmark |
| `GET`  | `/metrics` | Prometheus exposition |

```bash
curl -X POST https://graphrag-plus-api.vercel.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Was Project Helios cancelled?", "corpus_id": "corpus_demo_nova"}'
```

Abridged response:

```json
{
  "answer": "No. The source on 2024-02-10 contradicts the claim that Project Helios was canceled and indicates that it is continuing under Phase II.",
  "generated_by": "llm",
  "verified_by_reasoning": true,
  "reasoning_model": "openai/gpt-oss-120b",
  "calibrated_confidence": 0.75,
  "failure_type": null,
  "evidence": [{"source_id": "doc_sample2", "raw_semantic": 0.79, "raw_bm25": 0.41, "final_score": 0.9}],
  "reasoning_steps": ["Parsed question and decomposed into subqueries.", "..."],
  "graph_version_id": "v20261127045544009648"
}
```

Public-demo rate limits per client (env-tunable):

| Requests | Limit | Variable |
|---|---|---|
| `/query` | 20 per minute | `GRAPHRAG_RATE_QUERY_PER_MIN` |
| `/ingest*` | 6 per 10 minutes | `GRAPHRAG_RATE_INGEST_PER_10MIN` |
| Corpus deletes | 10 per 10 minutes | `GRAPHRAG_RATE_DELETE_PER_10MIN` |

Turn them all off locally with `GRAPHRAG_RATE_LIMIT=0`. More ready-to-run
requests are in
[`graphrag_plus/examples/api_requests.http`](graphrag_plus/examples/api_requests.http).

---

## Dashboard

`frontend/` is a **React 18 + Vite 5 + Tailwind 3** app.

- **Ingest:** drag-and-drop upload or web page URLs. When self-hosted, server
  file paths too.
- **Ask:**
  - example questions tailored to the open corpus
  - LLM / extractive strategy switch
  - a typed failure badge when the system abstains
- **Answer:**
  - calibrated-confidence bar and evidence with per-signal scores
  - a "Verified by …" badge naming the model that checked the answer
  - an expandable view of what it checked
- **Knowledge graph** (`react-force-graph-2d`):
  - search and type filters, fit-to-panel, readable labels at any zoom
  - the question's retrieval path highlighted in red
  - entity details with source chunks
- **Reasoning story:** a step-by-step trace of how the answer was reached,
  synchronised with the graph.
- **Workspaces:** switch between isolated corpora, delete your uploads, and
  see real document / chunk / entity / relation counts.
- **Layout and themes:** works down to phone width; dark and light themes.

---

## Folder structure

```
GraphRAG-Plus/
├── index.py                    ← Vercel entrypoint (FastAPI app)
├── requirements.txt            ← the deployed API bundle
├── vercel.json                 ← function settings (300 s max duration)
├── deploy/DEPLOYMENT.md        ← free deployment guide
├── frontend/                   ← React + Vite + Tailwind dashboard
│   └── src/
│       ├── App.jsx             ← layout + state
│       ├── api.js              ← API client
│       ├── suggestions.js      ← corpus-aware example questions
│       └── components/         ← IngestPanel, QueryBox, ResultCard, GraphView,
│                                 ReasoningStory, CorpusSelector, Header, ...
└── graphrag_plus/
    ├── pyproject.toml          ← deps, extras, ruff/black/mypy/coverage config
    ├── scripts/                ← calibrate_backend.py, check_postgres.py, quality gate
    └── app/
        ├── pipeline.py         ← end-to-end orchestration
        ├── api/                ← FastAPI routes, rate limiting, serverless entry
        ├── ingestion/          ← file + URL loaders (SSRF-guarded), chunker
        ├── extraction/         ← spaCy NER + LLM relation extraction (Ollama / Groq)
        ├── graph/              ← NetworkX + Neo4j stores, entity resolution, versioning
        ├── corpus/             ← multi-corpus manager, Postgres blob store, demo seed
        ├── embeddings/         ← local + remote (Gemini/Jina) embedders, FAISS store
        ├── retrieval/          ← hybrid BM25 + dense + graph retrieval, TF-IDF
        ├── generation/         ← answer generation, LLM clients, reasoning verifier
        ├── contradiction/      ← single-valued-fact disagreement detection
        ├── scoring/ trust/ calibration/ failure/ analyst/ planning/ ...
        ├── utils/              ← logging (with secret redaction), HTTP helpers, IO
        └── tests/              ← pytest suite (254 tests)
```

---

## Tech stack

- **Backend:** Python 3.11–3.13, FastAPI, Pydantic v2
- **Retrieval:** rank-bm25, a NumPy TF-IDF index, FAISS
- **NLP:** spaCy `en_core_web_sm`, plus LLMs via Groq (gpt-oss-20b / 120b) or Ollama (qwen3.5, DeepSeek R1)
- **Embeddings:** Gemini `gemini-embedding-001`, Jina, or local `sentence-transformers` (bge-large)
- **Graph:** NetworkX in-process; Neo4j optional (Personalized PageRank)
- **Storage:** Postgres (Neon) via psycopg 3, or local disk
- **Frontend:** React 18, Vite 5, Tailwind 3, Framer Motion, react-force-graph-2d
- **Hosting:** Vercel (serverless Python + static site), all free tiers
- **Quality:** pytest (254 tests), ruff, black, mypy, and GitHub Actions on Python 3.11 and 3.12

---

## Configuration

Settings live in [`graphrag_plus/app/config/settings.py`](graphrag_plus/app/config/settings.py)
(env prefix `GRAPHRAG_`, or a local `.env`). The cloud-specific variables are:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq LLMs for extraction, answers and verification |
| `GEMINI_API_KEY` / `JINA_API_KEY` | Hosted embeddings |
| `GRAPHRAG_DATABASE_URL` | Postgres for durable, shared corpora |
| `GRAPHRAG_MIN_COSINE_SEMANTIC`, `GRAPHRAG_STRONG_COSINE_SEMANTIC`, `GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD` | Calibrated per embedding model by `calibrate_backend.py` |
| `GRAPHRAG_CORS_ORIGINS` | Allowed dashboard origins |
| `GRAPHRAG_MAX_UPLOAD_BYTES` | Upload cap (default 4 MB) |

The full list is in the [deployment guide](deploy/DEPLOYMENT.md#environment-variable-reference).

---

## Observability

Every query and ingest records:

- the `query_id`, `graph_version_id` and `answer_state`
- raw and calibrated confidence
- per-evidence trust, semantic, BM25, graph and final scores
- the typed `failure_type`
- per-stage latency

`GET /metrics` exposes Prometheus counters and histograms:

```
graphrag_queries_total{failure_type="..."}
graphrag_query_latency_seconds_{count,sum,bucket}
graphrag_module_latency_seconds_{count,sum,bucket}{module="..."}
graphrag_ingest_total
graphrag_stage_errors_total{stage="..."}
```

Logs never contain secrets: a logging filter replaces API keys, bearer
tokens and database URLs with `[REDACTED]`.

---

## Development

```bash
cd graphrag_plus
pip install -e .[dev]

ruff check .
black --check .
mypy app
pytest --cov=app --cov-report=term-missing
```

CI runs all four on every push and PR, on Python 3.11 and 3.12
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Changes reach
`main` through pull requests; Vercel builds a preview of both projects for
each PR and deploys production from `main`.

On Windows, `.\graphrag_plus\scripts\quality_gate.ps1` runs the same checks
plus a frontend build. `-Fix` applies safe autofixes first, and `-ProbeHealth`
also checks a running backend.

---

## Future improvements

- **Personalized PageRank on the in-process graph**, so the cloud demo gets the same multi-hop graph signal as a Neo4j deployment
- **Async ingest** with progress streaming for large documents
- **Auth** (API key / JWT) for write endpoints beyond the current rate limits
- **Calibration backfill** with rolling-window compaction
- **Graph drift alerting** (`graphrag_stale_answers_total`)

---

## License

[MIT](LICENSE): free to use, modify, and distribute.

---

## Acknowledgements

Built as an exploration of how to make RAG systems *honest*: about what they
retrieved, how confident they are, when their sources disagree, and when the
documents simply don't contain the answer.
