import { AnimatePresence, motion } from "framer-motion";
import ConfidenceBar from "./ConfidenceBar.jsx";

const FAILURE_LABELS = {
  LOW_CONFIDENCE: { tone: "amber", label: "Low confidence" },
  CONTRADICTION: { tone: "rose", label: "Contradiction" },
  AMBIGUOUS: { tone: "violet", label: "Ambiguous" },
  STALE: { tone: "sky", label: "Stale" },
};

const TONE_CLASSES = {
  amber: "border-amber-400/30 bg-amber-500/10 text-amber-200",
  rose: "border-rose-400/30 bg-rose-500/10 text-rose-200",
  violet: "border-violet-400/30 bg-violet-500/10 text-violet-200",
  sky: "border-sky-400/30 bg-sky-500/10 text-sky-200",
  emerald: "border-emerald-400/30 bg-emerald-500/10 text-emerald-200",
};

export default function ResultCard({ result, busy }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15, duration: 0.45 }}
      className="glass flex h-full min-h-[260px] flex-col rounded-2xl p-5"
    >
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">Answer</h2>
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
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-100 [html:not(.dark)_&]:text-ink-800">
              {result.answer}
            </p>

            <ConfidenceBar
              calibrated={result.calibrated_confidence}
              raw={result.raw_confidence}
              error={result.calibration_error}
            />

            {!!result.evidence?.length && (
              <div>
                <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-400">
                  Evidence
                </h3>
                <ul className="thin-scroll max-h-56 space-y-2 overflow-auto pr-1">
                  {result.evidence.map((ev) => (
                    <li
                      key={ev.id}
                      className="rounded-lg border border-white/5 bg-white/[0.02] p-3 text-xs leading-relaxed
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
