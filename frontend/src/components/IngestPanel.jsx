import { useRef, useState } from "react";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { api } from "../api.js";
import Spinner from "./Spinner.jsx";

// Convert pasted text into a temp file path on the server is non-trivial,
// so for the "paste text" path we instead POST a data: URL via /ingest? No —
// the backend only accepts file paths or http(s) URLs. To keep parity without
// adding endpoints, this panel supports two input styles:
//   1. File paths visible to the BACKEND host (works locally).
//   2. URLs (validated server-side for SSRF).
// A third inline mode uses the browser's File API + FormData via a future
// /ingest_text endpoint; until then we surface a clear hint.

export default function IngestPanel({ onIngested }) {
  const [filePathsRaw, setFilePathsRaw] = useState("");
  const [urlsRaw, setUrlsRaw] = useState("");
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState(null);
  const dropRef = useRef(null);

  const submit = async () => {
    const filePaths = filePathsRaw
      .split(/\n|,/)
      .map((s) => s.trim())
      .filter(Boolean);
    const urls = urlsRaw
      .split(/\n|,/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (!filePaths.length && !urls.length) {
      toast.error("Provide at least one file path or URL");
      return;
    }
    setBusy(true);
    try {
      const result = await api.ingest({ filePaths, urls });
      setStats(result);
      toast.success(
        `Ingested ${result.documents} doc(s), ${result.entities} entities`,
      );
      onIngested?.(result);
    } catch (err) {
      toast.error(err.message || "Ingest failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.05, duration: 0.45 }}
      className="glass rounded-2xl p-5"
    >
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Ingest</h2>
          <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
            Files (local paths visible to backend) or URLs.
          </p>
        </div>
        <span className="rounded-full border border-white/5 bg-white/[0.02] px-2 py-0.5 text-[10px] uppercase tracking-wider text-ink-400">
          POST /ingest
        </span>
      </header>

      <label className="mb-2 block text-xs font-medium text-ink-300 [html:not(.dark)_&]:text-ink-600">
        File paths
      </label>
      <textarea
        ref={dropRef}
        value={filePathsRaw}
        onChange={(e) => setFilePathsRaw(e.target.value)}
        rows={3}
        placeholder={"C:/path/to/notes.txt\nC:/path/to/paper.pdf"}
        className="thin-scroll w-full resize-y rounded-lg border border-white/5 bg-ink-900/40 px-3 py-2
                   font-mono text-xs text-ink-100 placeholder:text-ink-500
                   focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500/30
                   [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white
                   [html:not(.dark)_&]:text-ink-900 [html:not(.dark)_&]:placeholder:text-ink-400"
      />

      <label className="mt-3 mb-2 block text-xs font-medium text-ink-300 [html:not(.dark)_&]:text-ink-600">
        URLs
      </label>
      <textarea
        value={urlsRaw}
        onChange={(e) => setUrlsRaw(e.target.value)}
        rows={2}
        placeholder="https://example.com/page"
        className="thin-scroll w-full resize-y rounded-lg border border-white/5 bg-ink-900/40 px-3 py-2
                   font-mono text-xs text-ink-100 placeholder:text-ink-500
                   focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500/30
                   [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white
                   [html:not(.dark)_&]:text-ink-900 [html:not(.dark)_&]:placeholder:text-ink-400"
      />

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={submit}
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white
                     shadow-soft transition-all hover:bg-accent-500 active:scale-[0.98]
                     disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? <Spinner /> : <PlusIcon />}
          Ingest
        </button>
        {stats && !busy && (
          <motion.div
            key={stats.graph_version_id}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-3 text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500"
          >
            <Stat label="docs" value={stats.documents} />
            <Stat label="chunks" value={stats.chunks} />
            <Stat label="entities" value={stats.entities} />
            <Stat label="relations" value={stats.relations} />
          </motion.div>
        )}
      </div>
    </motion.section>
  );
}

function Stat({ label, value }) {
  return (
    <span className="rounded-md border border-white/5 bg-white/[0.02] px-2 py-0.5 font-mono text-[11px]">
      <span className="text-accent-400">{value}</span>{" "}
      <span className="text-ink-500">{label}</span>
    </span>
  );
}

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="M7 1.5v11M1.5 7h11"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}
