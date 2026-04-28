# GraphRAG++ — Frontend

Modern React + Vite + Tailwind dashboard for the GraphRAG++ FastAPI backend.

## Stack

- **React 18** + **Vite 5**
- **Tailwind CSS** with custom design tokens, dark/light theming, glass surfaces
- **Framer Motion** for transitions and micro-interactions
- **react-force-graph-2d** for the interactive knowledge-graph view
- **react-hot-toast** for non-blocking feedback

## Run locally

Prereqs: Node 18+, the backend running at `http://127.0.0.1:8765`.

```bash
cd frontend
cp .env.example .env       # optional — defaults to local backend
npm install
npm run dev                # http://localhost:5173
```

Build a production bundle:

```bash
npm run build
npm run preview            # http://localhost:4173
```

## Configuration

| Var             | Default                 | Description                          |
| --------------- | ----------------------- | ------------------------------------ |
| `VITE_API_BASE` | `http://127.0.0.1:8765` | Base URL of the GraphRAG++ backend.  |

## Deploy to Vercel

1. Import this `frontend/` folder as a Vercel project (Framework: Vite).
2. Set `VITE_API_BASE` in **Project Settings → Environment Variables** to the
   public URL of your deployed backend.
3. On the backend, set `GRAPHRAG_CORS_ORIGINS` to the Vercel domain(s).

See [`../README.md`](../README.md) for full deployment notes.
