"""Vercel entrypoint for the GraphRAG++ FastAPI backend.

Vercel's Python runtime detects FastAPI from requirements.txt and serves the
whole app from a top-level ``app`` in a root ``index.py``, routing every path
to it natively. No rewrite rule is needed - and one must not be added: Vercel
now routes internal rewrites in backend-framework projects by their
*destination* path, so a catch-all rewrite makes every request look like the
rewrite target and 404.

The real bootstrap (redirecting writable paths to /tmp before import) lives in
``graphrag_plus.app.api.serverless`` so this file and the container
entrypoint, ``api/index.py``, boot identically.
"""

from graphrag_plus.app.api.serverless import app

__all__ = ["app"]
