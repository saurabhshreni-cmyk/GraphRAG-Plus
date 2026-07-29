# GraphRAG++ — Production System Documentation

**Status: COMPLETE.** All upgrade phases merged to `main` and pushed (`fa84e04`). 82/82 tests passing. Full E2E verified — see [E2E_RESULTS_V2.md](E2E_RESULTS_V2.md) and [E2E_RESULTS.md](E2E_RESULTS.md).

---

## 1. Complete Architecture

```
                              ┌─────────────────────────────────────────────┐
                              │       React + Vite frontend  (:5173)        │
                              │  drag-drop upload · corpus switcher ·       │
                              │  force-graph viz (typed colors, query-path  │
                              │  trace, entity panel) · answer provenance   │
                              │  (signal pills, DeepSeek R1 badge)          │
                              └──────────────────┬──────────────────────────┘
                                                 │ REST (CORS)
                              ┌──────────────────▼──────────────────────────┐
                              │           FastAPI backend  (:8000)          │
                              └──────────────────┬──────────────────────────┘
                    ┌────────────────────────────┼────────────────────────────┐
              INGEST│                       QUERY│                    EXPLORE  │
                    ▼                            ▼                             ▼
   ┌────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────┐
   │ Loader (PDF/TXT/MD/DOCX/   │  │ Hybrid retrieval             │  │ /graph/{cid}/full │
   │ HTML/URL) → Chunker        │  │  • BM25 (rank-bm25)   0.35   │  │ /query-path       │
   │        ↓                   │  │  • FAISS bge-large    0.40   │  │ /entity/{name}    │
   │ Extraction (2-stage)       │  │  • Neo4j PPR graph    0.25   │  │ /stats            │
   │  • spaCy en_core_web_sm    │  │  BGE-aware relevance floors  │  └──────────────────┘
   │  • qwen3.5:4b (JSON,       │  │        ↓                     │
   │    budgeted 8 chunks)      │  │ Scoring + calibration + trust│
   │        ↓                   │  │        ↓                     │
   │ Stores (all corpus-scoped) │  │ qwen3.5:4b draft answer      │
   │  • Neo4j AuraDB (entities, │  │        ↓                     │
   │    relations, chunks)      │  │ deepseek-r1:8b VERIFICATION  │
   │    + EntityResolver dedup  │  │  (re-reads evidence, confirms│
   │  • FAISS 1024-dim + BM25   │  │   /corrects/abstains; draft  │
   │  • NetworkX (viz backbone) │  │   kept on timeout)           │
   └────────────────────────────┘  │        ↓                     │
                                   │ Answer + confidence +        │
                                   │ provenance + failure_type    │
                                   └──────────────────────────────┘
```

Every stage degrades gracefully: no Ollama → extractive answers; no Neo4j → BM25+FAISS for retrieval **and the graph-exploration endpoints (`/graph/{cid}/full`, `/query-path`, `/entity`, `/stats`) fall back to the per-corpus NetworkX store persisted on every ingest**, so the knowledge-graph visualization works fully offline; no embedding model → TF-IDF; verification timeout → unverified draft. **Neo4j is therefore optional** — the system is fully functional (graph included) with only Ollama + a HuggingFace embedding model, both local.

## 2. Models Used and Why

| Model | Role | Why |
|---|---|---|
| **BAAI/bge-large-en-v1.5** (1024-dim) | Semantic retrieval embeddings | Top-tier MTEB English retrieval model; measured on-topic/off-topic cosine separation 0.68–0.85 vs 0.21–0.36. Query-side instruction prefix applied automatically. |
| **spaCy en_core_web_sm** | Stage-1 NER (every chunk, every query) | Millisecond-fast statistical NER for PERSON/ORG/GPE/DATE/PRODUCT; keeps query-time entity extraction LLM-free. |
| **qwen3.5:4b** (Ollama) | Stage-2 extraction + answer drafts | Markedly better structured-JSON compliance than qwen2.5:3b (Apple/Beats test: 5 typed relationships with descriptions vs 3 bare ones); `think=false` keeps latency bounded. |
| **deepseek-r1:8b** (Ollama) | Final-answer reasoning verifier | Reasoning-class model re-reads evidence and confirms/corrects/abstains — demonstrated live refinement of a draft (E2E v2 query 1). Strictly additive: any failure returns the draft. |

## 3. How to Run Locally (from scratch)

Prerequisites: **Python 3.12/3.13**, **Node 18+**, **Ollama** (`ollama pull qwen3.5:4b && ollama pull deepseek-r1:8b`). **Neo4j AuraDB is optional** — if `NEO4J_*` is unset or unreachable, the graph endpoints serve from the local per-corpus NetworkX store instead (data is identical; it's written on every ingest). Set the `NEO4J_*` vars only if you want the graph mirrored into a real Neo4j instance.

```powershell
git clone https://github.com/saurabhshreni-cmyk/GraphRAG-Plus.git
cd GraphRAG-Plus
python -m venv venv
.\venv\Scripts\pip install spacy sentence-transformers faiss-cpu rank-bm25 neo4j pydantic pydantic-settings python-dotenv pypdf beautifulsoup4 httpx ollama fastapi uvicorn numpy scikit-learn prometheus-client networkx python-dateutil python-docx python-multipart
.\venv\Scripts\python -m spacy download en_core_web_sm
```

Create `.env` in the project root (see `graphrag_plus/.env.example` for every knob):
```
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=<user>
NEO4J_PASSWORD=<password>
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
GRAPHRAG_VERIFIER_TIMEOUT_S=150   # laptop-class hardware
OLLAMA_TIMEOUT_S=90
```

Run:
```powershell
# backend (first run downloads bge-large ~1.3 GB)
.\venv\Scripts\python.exe -m uvicorn graphrag_plus.app.api.main:app --port 8000
# frontend (second terminal)
cd frontend && npm install && npm run dev     # http://localhost:5173
```

## 4. How to Deploy

**Frontend → Vercel**: deploy `frontend/`; set `VITE_API_BASE` to the backend URL. (`vercel.json` / `api/index.py` remain the existing lightweight serverless deployment — untouched.)

**Backend → Render.com** (Python web service):
- Build: install the dependency list above; Start: `python -m uvicorn graphrag_plus.app.api.main:app --host 0.0.0.0 --port $PORT`.
- Set `NEO4J_*` env vars in the Render dashboard (never commit them).
- **Ollama does not run on Render's free tier** → set `GRAPHRAG_LLM_ENABLED=false` (extractive answers, still evidence-grounded) or point `OLLAMA_BASE_URL` at hosted inference / set `ANTHROPIC_API_KEY` (client built in).
- bge-large needs ~2 GB RAM — use a paid instance, or set `EMBEDDING_MODEL=all-MiniLM-L6-v2` for small instances (FAISS dimension adapts automatically).
- Add the Vercel domain to `GRAPHRAG_CORS_ORIGINS`.

## 5. Demo Script (5–7 minutes)

1. **Open the UI** — point out corpus isolation dropdown and health badge.
2. **Drag-drop a PDF** (or paste the Wikipedia Knowledge-graph URL) → stages animate → "93 entities · 22 relations" toast. Say: *"Extraction is two-stage: spaCy NER plus a local qwen3.5 LLM emitting strictly-validated JSON."*
3. **Show the graph** — typed colors, filter buttons, search box. Click "Google" → side panel shows Neo4j neighbours + source chunks. Say: *"Every node is deduplicated — 'Apple', 'Apple Inc' and 'Apple Incorporated' merge into one canonical entity via string + embedding similarity."*
4. **Ask "What is a knowledge graph?"** — traversed nodes pulse red (Personalized PageRank seeds). Answer arrives with confidence bar, BM25/Semantic/Graph pills, and the gold **"Verified by DeepSeek R1"** badge. Expand *"What DeepSeek R1 checked."* Say: *"A reasoning model re-reads the evidence and either confirms, corrects, or abstains — in our E2E run it actually refined the draft."*
5. **Ask "What is the capital of France?"** — system abstains with NO_EVIDENCE. Say: *"It refuses to hallucinate: measured BGE relevance floors gate off-topic queries."*
6. Close on the architecture diagram (section 1).

## 6. API Endpoints

| Endpoint | Method | Purpose | Example |
|---|---|---|---|
| `/health` | GET | Liveness + LLM flag | → `{"status":"ok","llm_enabled":true,...}` |
| `/ingest` | POST | Ingest paths/URLs | `{"urls":["https://en.wikipedia.org/wiki/Knowledge_graph"],"new_corpus":true}` → chunks/entities/relations counts |
| `/ingest/file` | POST | Multipart upload (PDF/TXT/MD/DOCX/HTML) | form: `file`, optional `corpus_id`, `corpus_name` → `{"job_id","corpus_id","status":"completed",...}` |
| `/query` | POST | Ask a question | `{"question":"…","top_k":5,"corpus_id":"corpus_x"}` → answer, confidence, evidence w/ raw signal scores, `verified_by_reasoning`, `reasoning_summary`, `answer_changed_by_reasoning`, `failure_type` |
| `/corpora` · `/corpora/active` · `/corpora/{id}/select` · `/corpora/{id}` (DELETE) | — | Corpus management (delete also clears the Neo4j partition) | |
| `/corpora/{id}/chunks` | GET | Paginated chunk listing | `?page=1&page_size=20` |
| `/graph` · `/graph/{node_id}` | GET | Legacy NetworkX viz snapshot / neighborhood | |
| `/graph/{cid}/full` | GET | Neo4j graph (≤500 nodes by degree) | → `{nodes, edges, stats}` |
| `/graph/{cid}/query-path?q=` | GET | Nodes/edges a query traverses (spaCy-only, fast) | |
| `/graph/{cid}/entity/{name}` | GET | Entity metadata + neighbours + chunks | |
| `/graph/{cid}/stats` | GET | Type histograms, top-10 connected entities | |
| `/metrics` | GET | Prometheus metrics | |

## 7. Known Limitations

- **Model co-residency**: qwen3.5:4b + deepseek-r1:8b don't fit together in 16 GB RAM — draft→verify pays a model swap; R1 occasionally exceeds even the 150 s budget (drafts are kept, flagged unverified). A 24 GB+ machine or hosted inference removes this.
- **LLM extraction is budgeted** (8 chunks/ingest by default) — relations are sparse on long documents; raise `GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS` at ~60-90 s/chunk.
- **Entity resolution canonical-name choice** is "longest wins", which occasionally picks awkward canonicals ("Microsoft Research's" over "Microsoft").
- Verification runs only on LLM drafts, not extractive answers (by design — extractive output is already evidence-bound).
- **Neo4j-optional mode**: when Neo4j is unavailable the graph endpoints serve from the local NetworkX store, which does not run the in-database EntityResolver dedup pass — duplicate surface forms (e.g. "Apple"/"Apple Inc") may appear as separate nodes until a Neo4j instance is attached. Retrieval, PPR graph signal, answers, and the visualization are otherwise unaffected.
- PPR loads the corpus entity graph into Python per query; fine to ~10k entities, needs Neo4j GDS beyond that.
- Wikipedia loader extracts the article's lead sections (~8 k chars), not the full page.

## 8. What Makes This Different From Vanilla RAG (interview talking points)

1. **Three-signal hybrid retrieval** — BM25 + 1024-dim bge-large FAISS + knowledge-graph traversal, with measured, backend-aware relevance floors (not folklore thresholds).
2. **A real knowledge graph in Neo4j** — typed entities/relations extracted by a local LLM into strictly-validated Pydantic JSON, corpus-isolated, browsable through the API and UI.
3. **Entity resolution** — the classic production GraphRAG killer (duplicate nodes) handled with a 3-pass merger (string similarity, token-prefix containment, embedding similarity) over union-find clusters, executed in-database.
4. **Personalized PageRank retrieval** — research-grade graph signal (50 weighted chunks vs 1 from naive 2-hop on the same query in our measurements).
5. **Reasoning-verified answers** — a second, reasoning-class model (DeepSeek R1) audits every LLM answer against the evidence before the user sees it, with full provenance surfaced in the UI.
6. **Honest failure semantics** — NO_EVIDENCE abstention, LLM-failure fallbacks, calibrated confidence, trust scores, contradiction detection: the system says "I don't know" instead of hallucinating.
7. **Local-first and free** — every model runs on a laptop via Ollama + HuggingFace; the only cloud dependency (Neo4j Aura) has a free tier.

---

*Upgrade history: see git log (`feat/production-upgrade` branch, 18 commits) — session v1 built the spaCy/Ollama/Neo4j/FAISS foundation; session v2 upgraded models (bge-large, qwen3.5), added DeepSeek R1 verification, entity resolution, PPR, 6 API endpoints, and the frontend showcase.*
