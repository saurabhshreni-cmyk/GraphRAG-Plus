# GraphRAG++ backend — container image for Hugging Face Spaces (Docker SDK),
# Render, Fly.io, or any PaaS. The free HF CPU tier (16 GB RAM) runs the full
# stack: embeddings + FAISS + spaCy NER + BM25 + a Groq-hosted LLM.
#
# Configure at deploy time (Space / service env vars):
#   GROQ_API_KEY            -> enables generated answers (free at console.groq.com)
#   EMBEDDING_MODEL         -> all-MiniLM-L6-v2 (default, fast) or
#                              BAAI/bge-large-en-v1.5 (higher quality, ~2 GB RAM)
#   GRAPHRAG_CORS_ORIGINS   -> your frontend origin, e.g. https://<app>.vercel.app
#   GRAPHRAG_DATABASE_URL   -> optional Supabase/Postgres URI for durable corpora
FROM python:3.12-slim

# Runtime state and the model cache all live under /tmp, which is writable no
# matter which UID the platform runs the container as (HF Spaces uses uid 1000).
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    GRAPHRAG_LLM_ENABLED=true \
    PORT=7860

WORKDIR /app

# CPU-only torch first so sentence-transformers doesn't pull the CUDA build.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY deploy/requirements-hf.txt ./requirements-hf.txt
RUN pip install --no-cache-dir -r requirements-hf.txt \
 && python -m spacy download en_core_web_sm

# App code + the committed demo corpus (a fresh boot always has a graph to show).
# api/index.py is the ASGI entry: it redirects every writable GRAPHRAG_* path to
# /tmp and seeds the demo corpus before importing the FastAPI app — the same
# battle-tested startup the serverless deploy uses.
COPY api ./api
COPY graphrag_plus ./graphrag_plus

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
