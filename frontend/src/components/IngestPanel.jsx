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

const UPLOAD_STAGES = [
  "Uploading",
  "Chunking",
  "Extracting",
  "Building Graph",
  "Embedding",
  "Done",
];
const ACCEPTED_TYPES = ".pdf,.txt,.md,.docx,.html,.htm";

export default function IngestPanel({ onIngested, snapshot }) {
  const [filePathsRaw, setFilePathsRaw] = useState("");
  const [urlsRaw, setUrlsRaw] = useState("");
  const [corpusName, setCorpusName] = useState("");
  const [busy, setBusy] = useState(false);
  const dropRef = useRef(null);
  const fileInputRef = useRef(null);
  const [pendingFile, setPendingFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadStage, setUploadStage] = useState(-1); // -1 = idle
  const [uploadResult, setUploadResult] = useState(null);

  const acceptFile = (file) => {
    if (!file) return;
    const ok = ACCEPTED_TYPES.split(",").some((ext) =>
      file.name.toLowerCase().endsWith(ext),
    );
    if (!ok) {
      toast.error(`Unsupported type — allowed: ${ACCEPTED_TYPES}`);
      return;
    }
    setPendingFile(file);
    setUploadResult(null);
  };

  const uploadFile = async () => {
    if (!pendingFile) return;
    setBusy(true);
    setUploadStage(0);
    setUploadResult(null);
    // The backend ingests synchronously; advance the stage indicator on a
    // timer while the request is in flight, then snap to Done on response.
    const timer = setInterval(() => {
      setUploadStage((s) => Math.min(s + 1, UPLOAD_STAGES.length - 2));
    }, 2500);
    try {
      const result = await api.ingestFile({
        file: pendingFile,
        corpusName: corpusName.trim() || pendingFile.name.replace(/\.[^.]+$/, ""),
      });
      setUploadStage(UPLOAD_STAGES.length - 1);
      setUploadResult(result);
      toast.success(
        `Ingested ${result.chunks} chunks · ${result.entities} entities · ${result.relations} relations`,
      );
      setPendingFile(null);
      setCorpusName("");
      onIngested?.(result);
    } catch (err) {
      setUploadStage(-1);
      toast.error(err.message || "Upload failed");
    } finally {
      clearInterval(timer);
      setBusy(false);
    }
  };

  // Derive cumulative graph counts from the live snapshot so this panel
  // reflects the *current* knowledge graph, not the size of the last
  // ingest batch (which was the source of confusing entity counts before).
  const graphStats = (() => {
    const nodes = snapshot?.nodes ?? [];
    const edges = snapshot?.edges ?? [];
    const byType = nodes.reduce((acc, node) => {
      const t = node.node_type || "Other";
      acc[t] = (acc[t] || 0) + 1;
      return acc;
    }, {});
    const documents = byType.Document || 0;
    const chunks = byType.Chunk || 0;
    // Everything that isn't Document / Chunk counts as an entity-like node.
    const entities = Object.entries(byType)
      .filter(([t]) => t !== "Document" && t !== "Chunk")
      .reduce((sum, [, count]) => sum + count, 0);
    const relations = edges.filter(
      (e) => !["contains", "mentions"].includes(e.edge_type),
    ).length;
    return { documents, chunks, entities, relations };
  })();

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
      const result = await api.ingest({
        filePaths,
        urls,
        corpusName: corpusName.trim() || null,
      });
      toast.success(
        `Ingested ${result.documents} doc(s), ${result.entities} entities (this batch)`,
      );
      setCorpusName("");
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

      {/* Drag & drop upload zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          acceptFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        className={`mb-3 flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed px-3 py-5 text-center transition-all ${
          dragOver
            ? "border-accent-500 bg-accent-500/10"
            : "border-white/10 bg-white/[0.02] hover:border-accent-500/40 [html:not(.dark)_&]:border-ink-300"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          className="hidden"
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
        <UploadIcon />
        <p className="text-xs text-ink-300 [html:not(.dark)_&]:text-ink-600">
          Drop a file or click to browse
        </p>
        <p className="text-[10px] text-ink-500">PDF · TXT · MD · DOCX · HTML</p>
        {pendingFile && (
          <p className="mt-1 rounded-md bg-accent-500/10 px-2 py-0.5 font-mono text-[11px] text-accent-300 [html:not(.dark)_&]:text-accent-700">
            {pendingFile.name} · {(pendingFile.size / 1024).toFixed(1)} KB
          </p>
        )}
      </div>

      {pendingFile && (
        <button
          onClick={uploadFile}
          disabled={busy}
          className="mb-3 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-accent-600 px-4 py-2
                     text-sm font-medium text-white shadow-soft transition-all hover:bg-accent-500
                     active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? <Spinner /> : <PlusIcon />}
          Upload &amp; ingest
        </button>
      )}

      {/* Ingestion stage indicator */}
      {uploadStage >= 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-1 text-[10px]">
          {UPLOAD_STAGES.map((stage, i) => (
            <span
              key={stage}
              className={`rounded-full px-2 py-0.5 uppercase tracking-wider transition-all ${
                i < uploadStage
                  ? "bg-emerald-500/15 text-emerald-300 [html:not(.dark)_&]:text-emerald-700"
                  : i === uploadStage
                    ? uploadStage === UPLOAD_STAGES.length - 1
                      ? "bg-emerald-500/25 text-emerald-200 [html:not(.dark)_&]:text-emerald-700"
                      : "animate-pulse bg-accent-500/20 text-accent-300 [html:not(.dark)_&]:text-accent-700"
                    : "text-ink-600"
              }`}
            >
              {stage}
            </span>
          ))}
        </div>
      )}
      {uploadResult && (
        <p className="mb-3 text-[11px] text-emerald-300 [html:not(.dark)_&]:text-emerald-700">
          ✓ {uploadResult.entities} entities · {uploadResult.relations} relations ·{" "}
          {uploadResult.chunks} chunks
        </p>
      )}

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

      <label className="mt-3 mb-2 block text-xs font-medium text-ink-300 [html:not(.dark)_&]:text-ink-600">
        Corpus name <span className="text-ink-500">(optional)</span>
      </label>
      <input
        type="text"
        value={corpusName}
        onChange={(e) => setCorpusName(e.target.value)}
        maxLength={80}
        placeholder="Auto-derived from first source when empty"
        className="w-full rounded-lg border border-white/5 bg-ink-900/40 px-3 py-2
                   text-xs text-ink-100 placeholder:text-ink-500
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
      </div>

      <div className="mt-4 border-t border-white/5 pt-3 [html:not(.dark)_&]:border-ink-200">
        <div className="mb-1.5 text-[10px] uppercase tracking-wider text-ink-500">
          Graph total
        </div>
        <motion.div
          layout
          key={`${graphStats.documents}-${graphStats.entities}-${graphStats.relations}`}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-wrap items-center gap-2 text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500"
        >
          <Stat label="docs" value={graphStats.documents} />
          <Stat label="chunks" value={graphStats.chunks} />
          <Stat label="entities" value={graphStats.entities} />
          <Stat label="relations" value={graphStats.relations} />
        </motion.div>
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

function UploadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
      <path
        d="M9 12V3m0 0L5.5 6.5M9 3l3.5 3.5M3 12v2a1 1 0 001 1h10a1 1 0 001-1v-2"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.7"
      />
    </svg>
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
