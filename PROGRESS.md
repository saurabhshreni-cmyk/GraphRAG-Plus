# PROGRESS — Production Upgrade

## Session v2 (2026-07-03) — model + quality upgrades — IN PROGRESS

| Step | Deliverable | Status |
|---|---|---|
| 0 | Health check: 82 tests, /health, Neo4j (674 nodes), Ollama (4 models), live LLM query | ✅ all PASS |
| 1 | Embeddings → BAAI/bge-large-en-v1.5 (1024-dim, BGE query prefix, stale-index guard, old indexes purged) | ✅ committed |
| 2 | Extraction/generation LLM → qwen3.5:4b (think=false, 120s extraction timeout for cold loads) | ✅ committed |
| 3 | DeepSeek R1 8b reasoning verifier (final-answer gate; verified/changed/summary surfaced through API) | ✅ committed |
| 4 | Entity resolution & dedup (string + token-prefix + embedding passes, union-find clusters; live test: Apple/Apple Inc/Apple Incorporated → 1 node, Tim/Timothy Cook → 1 node) | ✅ committed |
| 5 | Personalized PageRank retrieval (RELATES_TO ∪ co-mention adjacency; 50 weighted chunks vs 1 from 2-hop on the AI corpus; 2-hop kept as fallback) | ✅ committed |
| 6 | New API endpoints: /graph/{cid}/full, /query-path, /entity/{name}, /stats, POST /ingest/file (PDF/TXT/MD/DOCX/HTML), /corpora/{cid}/chunks — all smoke-tested 200 | ✅ committed |
| 7 | Frontend upgrade: type-colored graph + filter/search + red query-path pulse + entity side panel (Neo4j-backed), drag-drop upload with stages, DeepSeek R1 badge + signal pills + collapsible provenance; `vite build` clean | ✅ committed |
| 8 | Full E2E v2 — see [E2E_RESULTS_V2.md](E2E_RESULTS_V2.md); includes 5 quality fixes found during testing (BGE floors, full-text context, grounding-aware quality gate, temporal endpoint filter, tunable gen timeout); 82/82 tests | ✅ committed |
| 9 | Merge to main | ⏳ in progress |
| 10 | Final docs | pending |

Session v2 notes:
- Verifier timeout is env-tuned to 150s in `.env` (`GRAPHRAG_VERIFIER_TIMEOUT_S`) — R1-8b thinking takes ~90s on this laptop; the 60s spec default remains the constructor fallback. On timeout the draft answer is returned unchanged with `verified_by_reasoning=false` (graceful).
- Verified live: draft (qwen3.5:4b) → verify (deepseek-r1:8b) → verified=True, changed=False, 93.6s, real reasoning summary captured from Ollama's `thinking` field.
- qwen3.5:4b extraction on the Apple/Beats test text: 7 entities with descriptions + 5 typed relationships, clean Pydantic parse (vs 3 relations from qwen2.5:3b).
- bge-large checks: 1024-dim confirmed; cosine("Apple bought Beats","Apple acquired Beats") = 0.964 > 0.90.

---

# Session v1 — Production Upgrade (spaCy + Ollama + Neo4j + FAISS)

Branch: `feat/production-upgrade` (branched off `main`)
Status: **ALL 9 STEPS COMPLETE** — upgraded system verified end to end.

## What was completed

| Step | Deliverable | Commit | Status |
|---|---|---|---|
| 1 | Pydantic schemas (`Entity`, `Relationship`, `ExtractionResult`, `RetrievalResult`, `AnswerResult`) | `feat: add pydantic schemas for full pipeline` | ✅ |
| 2 | spaCy + Ollama two-stage extraction (regex extractor preserved as fallback) | `feat: replace regex extraction with spaCy + Ollama pipeline` | ✅ |
| 3 | Neo4j AuraDB store: CRUD, batched UNWIND writes, 2-hop traversal, corpus isolation, health check | `feat: add Neo4j graph store with full CRUD and traversal` | ✅ |
| 4 | Lazy `all-MiniLM-L6-v2` embedder + FAISS IndexFlatIP store with save/load | `feat: add sentence-transformers embedder and FAISS vector store` | ✅ |
| 5 | Hybrid retrieval: BM25 0.35 + FAISS semantic 0.40 + Neo4j graph 0.25, TF-IDF/BM25 fallback | `feat: upgrade retrieval to BM25 + FAISS semantic + Neo4j graph hybrid` | ✅ |
| 6 | Ollama generation (qwen2.5:3b, 30s timeout, spec prompt) with extractive fallback + `AnswerResult` | `feat: upgrade generation with Ollama LLM + Pydantic structured output` | ✅ |
| 7 | Pipeline wiring: ingest → Neo4j sync + FAISS build; settings load NEO4J_*/OLLAMA_*/EMBEDDING_MODEL from `.env` | `feat: wire full upgraded pipeline end to end` | ✅ |
| 8 | E2E test: Wikipedia AI article ingest + 3 queries + Neo4j verification — see [E2E_RESULTS.md](E2E_RESULTS.md) | `test: full end to end pipeline test results` | ✅ |
| 9 | This file | `docs: update PROGRESS.md with full session summary` | ✅ |

Regression suite: **82/82 existing tests pass** (LLM paths disabled for determinism).

## What failed and why (honest notes)

- **E2E query 2** ("Who are the key researchers in AI?") returned an extractive fallback with `failure_type=LLM_FAILURE`: the LLM correctly abstained because the retrieved chunks don't enumerate researchers. Designed behavior, not a crash — but retrieval could rank a researcher-dense chunk higher (future work).
- **Relations are sparse** (11 from 270 chunks) because only the LLM stage emits relationships and it's budgeted to 8 chunks per ingest (`GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS`, default 8). Raise it for richer graphs at the cost of ~15s/chunk.
- Nothing was skipped; no step required the 2-attempt abandon rule.

## Files created

- `graphrag_plus/app/models/__init__.py`, `graphrag_plus/app/models/schemas.py`
- `graphrag_plus/app/graph/neo4j_store.py`
- `graphrag_plus/app/embeddings/__init__.py`, `graphrag_plus/app/embeddings/embedder.py`, `graphrag_plus/app/embeddings/faiss_store.py`
- `graphrag_plus/app/extraction/legacy_extractor.py` (the old regex extractor, preserved via `git mv`)
- `E2E_RESULTS.md`, `PROGRESS.md`
- `.env` (project root — gitignored, never committed; holds NEO4J_URI/USERNAME/PASSWORD, OLLAMA_MODEL, OLLAMA_BASE_URL, EMBEDDING_MODEL)

## Files modified

- `graphrag_plus/app/extraction/extractor.py` — full rewrite (spaCy + Ollama; keeps `extract_from_chunks` API)
- `graphrag_plus/app/retrieval/service.py` — hybrid 3-signal retrieval
- `graphrag_plus/app/generation/generator.py` — LLM-first gating, dual abstain tokens, `generate_result()`
- `graphrag_plus/app/generation/llm_clients.py` — .env-driven Ollama client, spec prompt, 30s timeout, default model qwen2.5:3b
- `graphrag_plus/app/config/settings.py` — `llm_enabled=True` default; NEO4J_*/OLLAMA_*/EMBEDDING_MODEL fields via `.env`
- `graphrag_plus/app/pipeline.py` — `_sync_to_neo4j()` on every ingest (batched, corpus-scoped, non-blocking)
- `graphrag_plus/app/api/main.py` — corpus delete also clears its Neo4j partition

Untouched by design: React frontend, `vercel.json`, `api/index.py`, chunking pipeline, BM25 logic.

## How to run locally

```powershell
cd C:\Users\Saurabh\Desktop\GraphRAG
# prerequisites: Ollama running with qwen2.5:3b pulled; .env present at repo root
.\venv\Scripts\python.exe -m uvicorn graphrag_plus.app.api.main:app --port 8000
# frontend (separate terminal):
cd frontend; npm install; npm run dev   # Vite on :5173, CORS pre-configured
```

Smoke checks:
- `GET http://localhost:8000/health`
- `POST /ingest` `{"file_paths": [], "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]}`
- `POST /query` `{"question": "What is artificial intelligence?", "top_k": 5}`

Useful env toggles (all optional): `GRAPHRAG_LLM_ENABLED=false` (extractive only), `GRAPHRAG_LLM_EXTRACTION=0` (spaCy-only extraction), `GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS=20` (deeper relation extraction).

## What still needs to be done / next steps

1. **Merge**: open a PR from `feat/production-upgrade` → `main` after review.
2. **Deployment (Render + Vercel)**:
   - The heavy stack (torch/sentence-transformers/faiss/spacy) does NOT fit Vercel's Python serverless runtime. Split: keep the React frontend on Vercel; deploy the FastAPI backend to Render (Docker or native Python service) with `NEO4J_*` env vars set in Render's dashboard.
   - Ollama is localhost-only — on Render either (a) set `GRAPHRAG_LLM_ENABLED=false` (extractive answers still work), (b) point `OLLAMA_BASE_URL` at a hosted inference endpoint, or (c) switch generation to `ANTHROPIC_API_KEY` (client already implemented).
   - Add a `render.yaml` + backend `requirements-server.txt` including torch/faiss/spacy/sentence-transformers (root `requirements.txt` is Vercel-scoped — do not add heavy deps there).
   - Point the frontend's API base URL at the Render service and add the Vercel domain to `GRAPHRAG_CORS_ORIGINS`.
3. **Quality follow-ups**: raise LLM extraction budget for richer relation graphs; entity-aware reranking for "who" questions; consider `neo4j` vector index as an alternative to local FAISS for multi-instance deployments; rotate the Neo4j password before any public deployment (it was shared in a chat session).
4. **Housekeeping**: `data/graph_versions/`, `data/outputs/` committed runtime artifacts could be gitignored to slim the repo.
