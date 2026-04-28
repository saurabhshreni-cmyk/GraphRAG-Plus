// Single source of truth for backend calls.
// Base URL is configurable via VITE_API_BASE so deployment to Vercel can
// point at a Render/Railway/Fly URL without code changes.

const BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8765").replace(
  /\/$/,
  "",
);

class ApiError extends Error {
  constructor(message, { status, body } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  let response;
  try {
    response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (err) {
    throw new ApiError(`Network error contacting ${url}: ${err.message}`);
  }
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!response.ok) {
    const detail =
      (data && (data.detail || data.message)) || response.statusText;
    throw new ApiError(`${response.status} ${detail}`, {
      status: response.status,
      body: data,
    });
  }
  return data;
}

export const api = {
  health: () => request("/health"),
  ingest: ({ filePaths = [], urls = [] }) =>
    request("/ingest", {
      method: "POST",
      body: JSON.stringify({ file_paths: filePaths, urls }),
    }),
  query: ({ question, analystMode = true, topK = 5 }) =>
    request("/query", {
      method: "POST",
      body: JSON.stringify({
        question,
        top_k: topK,
        analyst_mode: analystMode,
      }),
    }),
  graph: (limit = 500) => request(`/graph?limit=${limit}`),
};

export { BASE as API_BASE, ApiError };
