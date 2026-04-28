import { useState } from "react";
import { motion } from "framer-motion";
import Spinner from "./Spinner.jsx";

const SAMPLE_PROMPTS = [
  "What does GraphRAG++ use to store entities and relations?",
  "How does the trust manager work?",
  "What is the contradiction reasoner?",
];

export default function QueryBox({ onAsk, busy }) {
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
      className="glass rounded-2xl p-5"
    >
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
      <form onSubmit={submit} className="flex flex-col gap-3">
        <div className="relative">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your ingested corpus…"
            className="w-full rounded-xl border border-white/5 bg-ink-900/40 py-3.5 pl-4 pr-32
                       text-sm text-ink-100 placeholder:text-ink-500
                       focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500/30
                       [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white
                       [html:not(.dark)_&]:text-ink-900 [html:not(.dark)_&]:placeholder:text-ink-400"
          />
          <button
            type="submit"
            disabled={busy || !question.trim()}
            className="absolute right-1.5 top-1/2 inline-flex -translate-y-1/2 items-center gap-2
                       rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white shadow-soft
                       transition-all hover:bg-accent-500 active:scale-[0.98]
                       disabled:cursor-not-allowed disabled:opacity-60"
          >
            {busy ? <Spinner /> : <ArrowRight />}
            {busy ? "Thinking" : "Ask"}
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_PROMPTS.map((p) => (
            <button
              type="button"
              key={p}
              onClick={() => setQuestion(p)}
              className="rounded-full border border-white/5 bg-white/[0.02] px-3 py-1
                         text-xs text-ink-400 transition-all hover:border-accent-500/40 hover:text-ink-100
                         [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:text-ink-600
                         [html:not(.dark)_&]:hover:text-ink-900"
            >
              {p}
            </button>
          ))}
        </div>
      </form>
    </motion.section>
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
