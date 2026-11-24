"""Container entrypoint for the GraphRAG++ FastAPI backend.

Used by the Dockerfile (``uvicorn api.index:app``) on container hosts. The
Vercel deployment uses the repo-root ``index.py`` instead; both import the
same bootstrap from ``graphrag_plus.app.api.serverless``, which redirects
every writable path to /tmp before the app is imported.
"""

from graphrag_plus.app.api.serverless import app

__all__ = ["app"]
