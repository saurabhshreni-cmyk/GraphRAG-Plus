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
  ingest: ({
    filePaths = [],
    urls = [],
    corpusId = null,
    newCorpus = true,
    corpusName = null,
  }) => {
    const body = {
      file_paths: filePaths,
      urls,
      new_corpus: newCorpus,
    };
    if (corpusId) body.corpus_id = corpusId;
    if (corpusName) body.corpus_name = corpusName;
    return request("/ingest", { method: "POST", body: JSON.stringify(body) });
  },
  query: ({
    question,
    analystMode = true,
    topK = 5,
    llmEnabled = null,
    corpusId = null,
  }) => {
    const body = { question, top_k: topK, analyst_mode: analystMode };
    if (llmEnabled !== null) body.llm_enabled = llmEnabled;
    if (corpusId) body.corpus_id = corpusId;
    return request("/query", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
  graph: ({
    mode = "important",
    limit = 60,
    fullLimit = 500,
    corpusId = null,
  } = {}) => {
    const params = new URLSearchParams({
      mode,
      limit: String(limit),
      full_limit: String(fullLimit),
    });
    if (corpusId) params.set("corpus_id", corpusId);
    return request(`/graph?${params.toString()}`);
  },
  corpora: () => request("/corpora"),
  activeCorpus: () => request("/corpora/active"),
  selectCorpus: (corpusId) =>
    request(`/corpora/${encodeURIComponent(corpusId)}/select`, {
      method: "POST",
    }),
  deleteCorpus: (corpusId) =>
    request(`/corpora/${encodeURIComponent(corpusId)}`, { method: "DELETE" }),
};

export { BASE as API_BASE, ApiError };
