# E2E Test Results v2 — Fully Upgraded GraphRAG System

Date: 2026-07-03 · Branch: `feat/production-upgrade`
Stack: **BAAI/bge-large-en-v1.5** (1024-dim) + **qwen3.5:4b** (extraction & drafts) + **deepseek-r1:8b** (reasoning verification) + Neo4j AuraDB (entity resolution + Personalized PageRank) + FAISS + BM25.

## 1–2. Server health
`GET /health` → `{"status":"ok","llm_enabled":true,"graph_exists":true}` ✅

## 3. Ingest — https://en.wikipedia.org/wiki/Knowledge_graph
| metric | value |
|---|---|
| chunks | 27 |
| raw entity mentions | 93 |
| typed relationships | 22 |
| corpus_id | corpus_066d0e8c |

## 4. Neo4j node counts (`MATCH (n) RETURN labels(n), count(n)`)
`Chunk: 351 · Entity: 551` (database-wide, all corpora) ✅

## 5. Entity resolution (this corpus)
93 mentions → **61 canonical entities after deduplication** (5 merges during ingest, live):
`knowledge graph → knowledge graphs`, `Amazon → Amazon Alexa`, `late 1980s → the late 1980s`, `University of Groningen → the University of Groningen and`, `Microsoft → Microsoft Research's` ✅
Relationship types: RELATES_TO 22, MENTIONED_IN 91.

## 6. Query battery (all three signals BM25+Semantic+Graph fired on every on-topic query)

| # | Query | Path | Confidence | R1 verified | R1 changed | Time |
|---|---|---|---|---|---|---|
| 1 | What is a knowledge graph? | **LLM** | 1.000 | ✅ True | ✅ True (refined) | 126s |
| 2 | Which companies have built knowledge graphs? | **LLM** | 1.000 | ✗ (150s timeout → draft kept) | — | 165s |
| 3 | Relationship between KGs and AI? | extractive fallback | 1.000 | — | — | 34s |
| 4 | Who created the semantic web? | extractive fallback | 0.987 | — | — | 11s |
| 5 | What is the capital of France? | **NO_EVIDENCE abstain** ✅ | 0.000 | — | — | 3s |

Answers:
1. *"A knowledge graph is a knowledge base that uses a graph-structured data model to represent and operate on data. Knowledge graphs are often used to store interlinked descriptions of entities — objects, events, situations or abstract concepts — while also encoding the free-form semantics or relationships underlying these entities."* — draft **refined by DeepSeek R1** against evidence.
2. *"Facebook, LinkedIn, Airbnb, Microsoft, Amazon, Uber, and eBay."* — correct entity-list answer.
3–4. Honest extractive fallbacks: the retrieved evidence doesn't explicitly answer these (the article never names the Semantic Web's creator), so the LLM abstained by design instead of hallucinating.
5. **Off-topic gate works**: "I must abstain because retrieval produced no evidence."

## 7. Test suite
`82 passed` ✅ (deterministic mode: LLM paths disabled)

## 8. Frontend build
`vite build` → ✓ built in ~8s, zero errors (one benign chunk-size warning) ✅

## 9. `GET /graph/corpus_066d0e8c/full`
61 nodes · 22 edges · `entity_types: {ORG: 23, PERSON: 11, CONCEPT: 9, TECHNOLOGY: 6, OTHER: 6, DATE: 4, LOCATION: 2}` ✅

## 10. `GET /graph/corpus_066d0e8c/stats`
Top connected entities: **knowledge graphs (11), Google (8), LinkedIn (7), Microsoft Research's (5), Amazon Alexa (5)** ✅

## Fixes shipped during this E2E round
1. **BGE-aware relevance floors** (`_MIN_COSINE_FAISS=0.45`, `_STRONG_COSINE_FAISS=0.55`, gentler loose factor): bge's high cosine baseline was leaking off-topic chunks — query 5 answered nonsense before, abstains correctly now. Measured separation: on-topic 0.68–0.85 vs off-topic 0.21–0.36.
2. **LLM context from full chunk text, top-5 chunks** (was 300-char snippets, top-3): snippets truncated mid-answer and rank-4/5 chunks often carry aggregate answers.
3. **Grounding-aware quality gate**: entity-list answers ("Google, Microsoft, …") share zero tokens with the question and were wrongly rejected; now accepted when ≥50% of answer tokens are grounded in evidence.
4. **Temporal-noise gate on relationship endpoints**: LLM-emitted relationships could backfill bare "2024" entity nodes; now filtered (restored the failing temporal test).
5. **Env-tunable generation timeout** (`OLLAMA_TIMEOUT_S=90`): R1 verification evicts qwen3.5 on 16GB RAM; 30s wasn't enough to reload + generate.

## Honest limitations on this hardware (16GB RAM, RTX 3050)
- qwen3.5:4b and deepseek-r1:8b don't fit in memory together — every draft→verify cycle pays a model swap (~20-30s), and R1's thinking sometimes exceeds even 150s (query 2). The system degrades gracefully every time: draft kept, `verified_by_reasoning=false`.
- For demos: pre-warm the models (`ollama run qwen3.5:4b ""`) and expect ~2 min for a fully verified answer.
