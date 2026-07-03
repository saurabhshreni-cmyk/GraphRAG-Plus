# End-to-End Test Results — Upgraded GraphRAG Pipeline

Date: 2026-07-02/03 · Branch: `feat/production-upgrade`
Stack under test: spaCy `en_core_web_sm` + Ollama `qwen2.5:3b` extraction → Neo4j AuraDB + NetworkX graphs → BM25 + FAISS (`all-MiniLM-L6-v2`) + Neo4j-traversal hybrid retrieval → Ollama LLM generation with extractive fallback.

## Setup

- FastAPI server: `venv\Scripts\python.exe -m uvicorn graphrag_plus.app.api.main:app --port 8000`
- `/health` → `{"status":"ok","llm_enabled":true,"graph_exists":true}`

## Ingest

`POST /ingest` with `https://en.wikipedia.org/wiki/Artificial_intelligence`:

| metric | value |
|---|---|
| documents | 1 |
| chunks | 270 |
| entities | 689 |
| relations | 11 |
| corpus_id | corpus_9824e636 |
| detected domain | machine_learning |

LLM extraction ran on the first 8 chunks (configurable via `GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS`); spaCy NER covered all 270.

## Queries

### 1. "What is artificial intelligence?" — PASS (LLM)
> Artificial intelligence (AI) can be defined as the computational part of the ability to achieve goals in the world, according to McCarthy. Another perspective on AI is described by Marvin Minsky as the ability to solve hard problems.

confidence=1.000 · generated_by=llm · signals: semantic=1.000, graph=0.987 on top evidence

### 2. "Who are the key researchers in AI?" — PARTIAL (extractive fallback)
> These consist of particular traits or capabilities that researchers expect an intelligent system to display.

confidence=1.000 · generated_by=extractive · failure_type=LLM_FAILURE — the LLM abstained because the retrieved chunks (article intro) don't actually enumerate researchers; the system fell back to extractive output instead of hallucinating names. Honest failure: the retrieved evidence was on-topic for "researchers" lexically but doesn't contain a list of key researchers.

### 3. "What are the main applications of AI?" — PASS (LLM)
> The main applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, and play and analysis in strategy games (e.g., chess and Go).

confidence=1.000 · generated_by=llm · signals: semantic=1.000, graph=0.995 on top evidence

## Neo4j verification

`MATCH (n) RETURN count(n)` → **634**

| breakdown | count |
|---|---|
| Entity nodes (corpus_9824e636) | 364 |
| Chunk nodes (corpus_9824e636) | 270 |
| relationships (MENTIONED_IN + RELATES_TO) | 700 |

Sample entities: AI, learning, reasoning, problem-solving, perception, decision-making…

## Regression suite

`pytest graphrag_plus/app/tests` (LLM paths disabled for determinism): **82 passed, 0 failed**.

## Known behaviors / caveats

- Query 2 shows the designed fallback path, not a crash: LLM abstain → extractive answer + `failure_type=LLM_FAILURE` surfaced in the response.
- LLM extraction budget (default 8 chunks/ingest) bounds Wikipedia-scale ingests; raise `GRAPHRAG_LLM_EXTRACTION_MAX_CHUNKS` for deeper relationship coverage.
- Relations count (11) is low relative to entities because only the LLM stage produces relationships and it ran on 8 of 270 chunks.
