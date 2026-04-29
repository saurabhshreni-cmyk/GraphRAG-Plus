import { useState } from "react";
import { motion } from "framer-motion";
import Spinner from "./Spinner.jsx";

const SAMPLE_PROMPTS = [
  "What does GraphRAG++ use to store entities and relations?",
  "What is graph data structure?",
  "What is adjacency matrix?",
];

export default function QueryBox({
  onAsk,
  busy,
  llmMode,
  onLlmModeChange,
  serverLlmEnabled,
}) {
  const [question, setQuestion] = useState("");

  const submit = (e) => {
    e?.preventDefault?.();
    const q = question.trim();
    if (!q || busy) return;
    onAsk(q);
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1, duration: 0.45 }}
      className="glass relative overflow-hidden rounded-2xl p-5"
    >
      {/* Subtle gradient accent in the corner */}
      <div className="pointer-events-none absolute -right-24 -top-24 h-48 w-48 rounded-full bg-accent-500/10 blur-3xl" />

      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Ask</h2>
          <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
            Hybrid retrieval over the knowledge graph + vector + BM25.
          </p>
        </div>
        <span className="rounded-full border border-white/5 bg-white/[0.02] px-2 py-0.5 text-[10px] uppercase tracking-wider text-ink-400">
          POST /query
        </span>
      </header>

      <form onSubmit={submit} className="relative flex flex-col gap-3">
        <div className="relative">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your ingested corpus…"
            disabled={busy}
            className="w-full rounded-xl border border-white/5 bg-ink-900/40 py-3.5 pl-4 pr-32
                       text-sm text-ink-100 placeholder:text-ink-500 transition-all
                       focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500/30
                       disabled:opacity-60
                       [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white
                       [html:not(.dark)_&]:text-ink-900 [html:not(.dark)_&]:placeholder:text-ink-400"
          />
          <button
            type="submit"
            disabled={busy || !question.trim()}
            className="group absolute right-1.5 top-1/2 inline-flex -translate-y-1/2 items-center gap-2
                       rounded-lg bg-gradient-to-br from-accent-500 to-accent-700 px-4 py-2 text-sm
                       font-medium text-white shadow-soft transition-all
                       hover:from-accent-400 hover:to-accent-600 hover:shadow-[0_0_16px_-2px_rgba(91,140,255,0.55)]
                       active:scale-[0.97]
                       disabled:cursor-not-allowed disabled:from-ink-700 disabled:to-ink-800 disabled:opacity-60"
          >
            {busy ? <Spinner /> : <ArrowRight />}
            {busy ? "Thinking" : "Ask"}
          </button>
        </div>

        {/* Sample prompts + LLM toggle */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            {SAMPLE_PROMPTS.map((p) => (
              <button
                type="button"
                key={p}
                onClick={() => setQuestion(p)}
                disabled={busy}
                className="rounded-full border border-white/5 bg-white/[0.02] px-3 py-1
                           text-xs text-ink-400 transition-all
                           hover:border-accent-500/40 hover:bg-accent-500/5 hover:text-ink-100
                           disabled:opacity-50
                           [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:text-ink-600
                           [html:not(.dark)_&]:hover:text-ink-900"
              >
                {p}
              </button>
            ))}
          </div>
          <LlmModePicker
            mode={llmMode}
            onChange={onLlmModeChange}
            serverEnabled={serverLlmEnabled}
            disabled={busy}
          />
        </div>
      </form>
    </motion.section>
  );
}

function LlmModePicker({ mode, onChange, serverEnabled, disabled }) {
  // Three states: "auto" (use server default), "llm" (force on), "extractive" (force off)
  const options = [
    { value: "auto", label: "Auto" },
    { value: "llm", label: "LLM" },
    { value: "extractive", label: "Extract" },
  ];
  const serverLabel = serverEnabled ? "LLM on" : "LLM off";
  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-white/5 bg-ink-900/40 p-0.5
                 text-[11px] [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white"
      role="radiogroup"
      aria-label="Answer strategy"
    >
      <span
        className="px-2 text-ink-500 [html:not(.dark)_&]:text-ink-500"
        title={`Server default: ${serverLabel}`}
      >
        Strategy
      </span>
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={mode === opt.value}
          disabled={disabled}
          onClick={() => onChange(opt.value)}
          className={`relative rounded-full px-3 py-1 transition-all ${
            mode === opt.value
              ? "bg-accent-500/20 text-ink-50 shadow-[0_0_12px_-3px_rgba(91,140,255,0.6)] [html:not(.dark)_&]:text-ink-900"
              : "text-ink-400 hover:text-ink-100 [html:not(.dark)_&]:text-ink-500 [html:not(.dark)_&]:hover:text-ink-900"
          } disabled:opacity-50`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function ArrowRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="M2 7h10m0 0L8 3m4 4-4 4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
