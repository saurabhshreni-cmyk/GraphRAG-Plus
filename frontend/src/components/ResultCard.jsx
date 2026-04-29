import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import toast from "react-hot-toast";
import ConfidenceBar from "./ConfidenceBar.jsx";

const FAILURE_LABELS = {
  LOW_CONFIDENCE: { tone: "amber", label: "Low confidence" },
  CONTRADICTION: { tone: "rose", label: "Contradiction" },
  CONFLICTING_EVIDENCE: { tone: "rose", label: "Conflicting" },
  AMBIGUOUS: { tone: "violet", label: "Ambiguous" },
  STALE: { tone: "sky", label: "Stale" },
  HIGH_UNCERTAINTY: { tone: "amber", label: "Uncertain" },
  NO_EVIDENCE: { tone: "rose", label: "No evidence" },
  LLM_FAILURE: { tone: "amber", label: "LLM fallback" },
};

const TONE_CLASSES = {
  amber: "border-amber-400/30 bg-amber-500/10 text-amber-200",
  rose: "border-rose-400/30 bg-rose-500/10 text-rose-200",
  violet: "border-violet-400/30 bg-violet-500/10 text-violet-200",
  sky: "border-sky-400/30 bg-sky-500/10 text-sky-200",
  emerald: "border-emerald-400/30 bg-emerald-500/10 text-emerald-200",
  accent: "border-accent-400/30 bg-accent-500/10 text-accent-400",
};

export default function ResultCard({ result, busy, onEvidenceHover }) {
  const [copied, setCopied] = useState(false);

  const copyAnswer = async () => {
    if (!result?.answer) return;
    try {
      await navigator.clipboard.writeText(result.answer);
      setCopied(true);
      toast.success("Answer copied");
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Could not copy to clipboard");
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15, duration: 0.45 }}
      className="glass relative flex h-full min-h-[260px] flex-col overflow-hidden rounded-2xl p-5"
    >
      <div className="pointer-events-none absolute -left-20 -bottom-20 h-44 w-44 rounded-full bg-emerald-500/5 blur-3xl" />

      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold tracking-tight">Answer</h2>
        <div className="flex items-center gap-2">
          {result && (
            <GenerationBadge
              generatedBy={result.generated_by}
              usedLlm={result.used_llm}
            />
          )}
          {result?.failure_type ? (
            <FailureBadge type={result.failure_type} />
          ) : result ? (
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${TONE_CLASSES.emerald}`}
            >
              Answered
            </span>
          ) : (
            <span className="rounded-full border border-white/5 bg-white/[0.02] px-2 py-0.5 text-[10px] uppercase tracking-wider text-ink-400">
              Idle
            </span>
          )}
        </div>
      </header>

      <AnimatePresence mode="wait">
        {busy && !result && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-1 flex-col gap-3"
          >
            <SkeletonLine width="92%" />
            <SkeletonLine width="84%" />
            <SkeletonLine width="60%" />
          </motion.div>
        )}

        {result && (
          <motion.div
            key={result.query_id || "result"}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex flex-1 flex-col gap-4"
          >
            <div className="group/answer relative">
              <p className="whitespace-pre-wrap pr-9 text-sm leading-relaxed text-ink-100 [html:not(.dark)_&]:text-ink-800">
                {result.answer}
              </p>
              <button
                type="button"
                onClick={copyAnswer}
                aria-label="Copy answer"
                className="absolute right-0 top-0 rounded-md border border-white/5 bg-white/[0.02]
                           p-1.5 opacity-0 transition-all hover:border-accent-500/40 hover:text-ink-100
                           group-hover/answer:opacity-100 focus-visible:opacity-100
                           [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white"
                title="Copy answer to clipboard"
              >
                {copied ? <CheckIcon /> : <CopyIcon />}
              </button>
            </div>

            <ConfidenceBar
              calibrated={result.calibrated_confidence}
              raw={result.raw_confidence}
              error={result.calibration_error}
            />

            {!!result.evidence?.length && (
              <div>
                <h3 className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-ink-400">
                  Evidence
                  <span className="text-ink-500/70 text-[10px] normal-case tracking-normal">
                    (hover to highlight in graph)
                  </span>
                </h3>
                <ul className="thin-scroll max-h-56 space-y-2 overflow-auto pr-1">
                  {result.evidence.map((ev) => (
                    <li
                      key={ev.id}
                      onMouseEnter={() => onEvidenceHover?.([ev.id, ev.source_id])}
                      onMouseLeave={() => onEvidenceHover?.([])}
                      className="cursor-default rounded-lg border border-white/5 bg-white/[0.02] p-3 text-xs leading-relaxed
                                 transition-all hover:border-accent-500/40 hover:bg-accent-500/[0.05]
                                 [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white"
                    >
                      <div className="mb-1 flex items-center justify-between gap-2 text-[10px] uppercase tracking-wider text-ink-400">
                        <span className="truncate font-mono">{ev.source_id}</span>
                        <span className="shrink-0">
                          score{" "}
                          <span className="text-accent-400">
                            {ev.final_score.toFixed(2)}
                          </span>
                        </span>
                      </div>
                      <p className="text-ink-200 [html:not(.dark)_&]:text-ink-700">
                        {ev.snippet}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </motion.div>
        )}

        {!result && !busy && (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-1 items-center justify-center text-center"
          >
            <p className="max-w-xs text-sm text-ink-400 [html:not(.dark)_&]:text-ink-500">
              Ingest a corpus, then ask a question. The answer, calibrated
              confidence, and supporting evidence will appear here.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  );
}

function GenerationBadge({ generatedBy, usedLlm }) {
  const isLlm = usedLlm || generatedBy === "llm";
  const tone = isLlm ? "accent" : "sky";
  const label = isLlm ? "LLM" : "Extractive";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${TONE_CLASSES[tone]}`}
      title={isLlm ? "Answer composed by the LLM" : "Answer extracted from evidence sentences"}
    >
      {isLlm ? <SparkleIcon /> : <DocIcon />}
      {label}
    </span>
  );
}

function FailureBadge({ type }) {
  const meta = FAILURE_LABELS[type] || { tone: "rose", label: type };
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${TONE_CLASSES[meta.tone]}`}
    >
      {meta.label}
    </span>
  );
}

function SkeletonLine({ width }) {
  return (
    <div
      className="bar-shimmer h-3 animate-shimmer rounded-md bg-white/[0.04]"
      style={{ width }}
    />
  );
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden>
      <rect x="4" y="4" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3 3h7v1H4v7H3V3z" fill="currentColor" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M3 8l3 3 7-7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function SparkleIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path
        d="M6 1v4M6 7v4M1 6h4M7 6h4"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
function DocIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden>
      <rect x="2.5" y="1.5" width="7" height="9" rx="1" stroke="currentColor" strokeWidth="1.2" />
      <path d="M4 4h4M4 6h4M4 8h2.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
    </svg>
  );
}
