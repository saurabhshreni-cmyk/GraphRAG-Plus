# GraphRAG++ — Free Deployment Guide

The whole system — ingestion, the knowledge graph, hybrid retrieval, generated
and reasoning-verified answers — runs publicly on free tiers with **no credit
card**, on two Vercel projects you already have.

| Layer | Service | Notes |
|---|---|---|
| Frontend | Vercel — `graphrag-plus-dashboard` | Static Vite build. Already live; already points at the backend. |
| Backend | Vercel — `graphrag-plus-api` (Python) | FastAPI, served from the repo-root `index.py`. 500 MB bundle, 300 s, 2 GB RAM on Hobby. |
| Draft answers + extraction | **Groq** free API — `openai/gpt-oss-20b` | |
| Answer verification | **Groq** free API — `openai/gpt-oss-120b`, `reasoning_effort=high` | Returns its reasoning trace for the UI panel. |
| Embeddings | **Gemini** free API — `gemini-embedding-001`, 1024-dim | Jina (`JINA_API_KEY`) works as a fallback. |
| Vector index, BM25, TF-IDF, spaCy NER, knowledge graph, entity resolution, PPR | In-process on the backend | Same code as local. |
| Corpus storage | **Neon** Postgres free plan — optional | Without it, uploaded corpora live in `/tmp` and vanish on cold start. |

### Why this shape

- **No PyTorch.** `sentence-transformers` + PyTorch need ~1.5 GB; no free,
  card-free host runs them. The hosted embedding API returns the same
  1024-dim vectors, so FAISS and the rest of retrieval are unchanged.
- **No Neo4j in the public deploy.** Free AuraDB instances are deleted after
  ~30 days idle, which is the wrong failure mode for a portfolio link. The
  graph endpoints serve from the per-corpus NetworkX graph, which now runs the
  same entity resolution as Neo4j. Neo4j still works when `NEO4J_*` is set.
- **Neon over Supabase.** Supabase pauses free projects after 7 days of low
  activity — exactly what a portfolio link does between recruiter visits. The
  storage layer is plain Postgres, so Neon needs no code change.
- **Not Hugging Face Spaces.** Docker Spaces now require a PRO plan to create.

---

## 1. Get the API keys (all free)

**Groq** — <https://console.groq.com> → API Keys → Create. Starts with `gsk_`.
Confirm which models your key can call (the key stays on your machine):

```powershell
curl.exe -s -H "Authorization: Bearer $env:GROQ_API_KEY" https://api.groq.com/openai/v1/models
```

`openai/gpt-oss-20b` and `openai/gpt-oss-120b` should be listed. If your
account differs, set `GROQ_MODEL` / `GROQ_REASONING_MODEL` to models it has.

**Gemini** — <https://aistudio.google.com> → Get API key → Create.
*(Fallback: <https://jina.ai> → API key → `JINA_API_KEY`.)*

## 2. (Recommended) Durable storage — Neon

<https://neon.tech> → New project → copy the **pooled** connection string.
That is `GRAPHRAG_DATABASE_URL`. The app creates its one table itself.

## 3. Calibrate the relevance floors for your embedding backend

The retrieval floors and the entity-merge threshold in the code were measured
on bge-large; Gemini's cosine scores sit on a different scale. Measure them
once — three small embedding calls:

```powershell
$env:GEMINI_API_KEY = "<your key>"
$env:GRAPHRAG_EMBEDDING_BACKEND = "remote"
$env:PYTHONPATH = "."
.\venv\Scripts\python.exe graphrag_plus\scripts\calibrate_backend.py
```

It prints three `GRAPHRAG_*` lines. Add them in step 4. (Run against local
bge-large it reproduces the original measurement, which is how the method was
validated.)

## 4. Configure the backend project

Vercel → **graphrag-plus-api** → Settings → Environment Variables
(Production and Preview):

| Name | Value |
|---|---|
| `GROQ_API_KEY` | from step 1 — **sensitive** |
| `GEMINI_API_KEY` | from step 1 — **sensitive** |
| `GRAPHRAG_DATABASE_URL` | from step 2 — **sensitive**, optional |
| `GRAPHRAG_MIN_COSINE_SEMANTIC` | from step 3 |
| `GRAPHRAG_STRONG_COSINE_SEMANTIC` | from step 3 |
| `GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD` | from step 3 |

Settings → General → **Framework Preset** should read FastAPI (or "Other");
it must not be set to a frontend framework.

## 5. Deploy

Merge the pull request into `main`. Vercel builds production from `main`
automatically; the dashboard already calls
`https://graphrag-plus-api.vercel.app`, and that origin is in the backend's
default CORS list, so the frontend needs no change.

## 6. Verify

```powershell
curl.exe -s https://graphrag-plus-api.vercel.app/health
```

Then open <https://graphrag-plus-dashboard.vercel.app> and walk the demo:

1. The demo corpus graph renders.
2. Upload a PDF (≤ 4 MB) → entities and relations appear.
3. Ask an on-topic question → generated answer, BM25 / Semantic / Graph
   pills, and the verification badge with its reasoning panel.
4. Ask *"What is the capital of France?"* → `NO_EVIDENCE` abstention.

If step 4 returns evidence instead of abstaining, the floors from step 3 are
not applied — check the three variables and redeploy.

---

## How it behaves in production

- **Cold start:** the first request after idle boots the function and spaCy
  (a few seconds). Embeddings are an API call, so there is no model load.
- **Degradation, in order:** no Gemini/Jina key → BM25 + TF-IDF + graph (no
  semantic signal); no Groq key → extractive answers; Groq failure → the
  evidence-grounded draft is served, flagged unverified. Nothing hard-fails.
- **Uploads** are capped at 4 MB (Vercel drops bodies over 4.5 MB at its
  edge); larger files get a clear message to split them.
- **Rate limits** on `/query` (20/min) and `/ingest*` (6 per 10 min) per
  client, protecting your free quotas. Tune with
  `GRAPHRAG_RATE_QUERY_PER_MIN` / `GRAPHRAG_RATE_INGEST_PER_10MIN`; disable
  with `GRAPHRAG_RATE_LIMIT=0`. Limits are per warm instance.
- **Storage:** with `GRAPHRAG_DATABASE_URL`, corpora persist across cold
  starts. Without it, only the committed demo corpus survives.

## Environment variable reference

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Enables generated answers and verification |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Draft-answer model |
| `GROQ_REASONING_MODEL` | `openai/gpt-oss-120b` | Verifier model |
| `GROQ_REASONING_EFFORT` | `high` | `low` / `medium` / `high` |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | Gemini embeddings |
| `JINA_API_KEY` | — | Jina embeddings (used if no Gemini key) |
| `GRAPHRAG_EMBEDDING_BACKEND` | `auto` | `auto` / `local` / `remote` |
| `GRAPHRAG_MIN_COSINE_SEMANTIC` | `0.45` | Semantic relevance floor |
| `GRAPHRAG_STRONG_COSINE_SEMANTIC` | `0.55` | Strong-semantic bar |
| `GRAPHRAG_RESOLVER_EMBEDDING_THRESHOLD` | `0.92` bge / `0.95` other | Entity merge bar |
| `GRAPHRAG_DATABASE_URL` | — | Postgres for durable corpora |
| `GRAPHRAG_CORS_ORIGINS` | dev + dashboard | Comma-separated allowed origins |
| `GRAPHRAG_ALLOW_FILE_PATHS` | `1` locally, `0` hosted | Whether `POST /ingest` may read server-side file paths. Hosted entrypoints default it to `0`; leave it off on any public deployment |
| `GRAPHRAG_MAX_UPLOAD_BYTES` | `4000000` | Upload cap |
| `GRAPHRAG_ENTITY_RESOLUTION` | `1` | `0` disables both resolvers |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | — | Optional real Neo4j |

## Local development

Unchanged: run Ollama and the local models and they are preferred
automatically (local embedder first, Ollama verifier first). Set
`GRAPHRAG_EMBEDDING_BACKEND=remote` to exercise the cloud path from a laptop.

## Alternative: container hosts

The root `Dockerfile` (entrypoint `api/index.py`) runs the same app with the
local PyTorch embedder on any container host with ≥ 2 GB RAM. None of those
is currently free without a card — Google Cloud Run's free tier comes closest
but needs a billing account.
