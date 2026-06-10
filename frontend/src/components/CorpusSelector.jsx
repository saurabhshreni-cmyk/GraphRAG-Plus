import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";

const DOMAIN_BADGE = {
  machine_learning: { color: "#7aa7ff", label: "ML" },
  physics: { color: "#a78bfa", label: "Physics" },
  finance: { color: "#06d6a0", label: "Finance" },
  biology: { color: "#34d399", label: "Bio" },
  chemistry: { color: "#fb923c", label: "Chem" },
  mathematics: { color: "#fbbf24", label: "Math" },
  computer_science: { color: "#60a5fa", label: "CS" },
  general: { color: "#9aa6b8", label: "General" },
};

/**
 * Corpus selector + manager. Lets the user pick which isolated corpus the
 * query and graph viz operate against, prevents cross-domain mixing.
 *
 * @param {{
 *   corpora: Array<object>,
 *   activeCorpusId: string|null,
 *   onSelect: (corpusId: string) => void,
 *   onRefresh: () => void,
 *   onDelete: (corpusId: string) => void,
 * }} props
 */
export default function CorpusSelector({
  corpora,
  activeCorpusId,
  onSelect,
  onRefresh,
  onDelete,
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    function onClickOutside(e) {
      if (!containerRef.current?.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  const active = corpora.find((c) => c.corpus_id === activeCorpusId);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => {
          setOpen((v) => !v);
          if (!open) onRefresh?.();
        }}
        className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs
                   transition-all hover:border-accent-500/40 hover:bg-white/[0.06]
                   [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white/80
                   [html:not(.dark)_&]:hover:bg-white"
        title="Switch active corpus"
      >
        <span className="text-[10px] uppercase tracking-wider text-ink-400 [html:not(.dark)_&]:text-ink-500">
          Corpus
        </span>
        {active ? <DomainBadge domain={active.domain} /> : null}
        <span className="max-w-[180px] truncate font-medium">
          {active?.name || "(none)"}
        </span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          className={`transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          <path
            d="M2 4L5 7L8 4"
            stroke="currentColor"
            strokeWidth="1.4"
            fill="none"
            strokeLinecap="round"
          />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 z-30 mt-2 w-80 max-h-[60vh] overflow-y-auto rounded-xl
                       border border-white/10 bg-ink-900/96 p-2 shadow-2xl backdrop-blur-md
                       [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white/96"
          >
            {corpora.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
                No corpora yet — ingest a document to create one.
              </div>
            ) : (
              <ul className="space-y-1">
                {corpora.map((c) => {
                  const isActive = c.corpus_id === activeCorpusId;
                  return (
                    <li key={c.corpus_id}>
                      <div
                        className={`flex items-start gap-2 rounded-md px-2 py-2 text-xs transition-colors ${
                          isActive
                            ? "bg-accent-500/10 ring-1 ring-accent-500/40"
                            : "hover:bg-white/[0.04] [html:not(.dark)_&]:hover:bg-ink-100"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            onSelect?.(c.corpus_id);
                            setOpen(false);
                          }}
                          className="flex flex-1 flex-col items-start gap-1 text-left"
                        >
                          <div className="flex items-center gap-2">
                            <DomainBadge domain={c.domain} />
                            <span className="line-clamp-1 font-medium">
                              {c.name}
                            </span>
                          </div>
                          <div className="flex items-center gap-3 text-[10px] text-ink-500">
                            <span>{c.document_count}d</span>
                            <span>{c.chunk_count}ch</span>
                            <span>{c.entity_count}e</span>
                          </div>
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (
                              window.confirm(
                                `Delete corpus "${c.name}"? This is irreversible.`,
                              )
                            ) {
                              onDelete?.(c.corpus_id);
                            }
                          }}
                          className="rounded p-1 text-ink-500 opacity-60 hover:bg-rose-500/15 hover:text-rose-300 hover:opacity-100"
                          title="Delete corpus"
                          aria-label={`Delete ${c.name}`}
                        >
                          <svg
                            width="12"
                            height="12"
                            viewBox="0 0 12 12"
                            fill="none"
                            aria-hidden
                          >
                            <path
                              d="M3 3L9 9M9 3L3 9"
                              stroke="currentColor"
                              strokeWidth="1.4"
                              strokeLinecap="round"
                            />
                          </svg>
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function DomainBadge({ domain }) {
  const meta = DOMAIN_BADGE[domain] || DOMAIN_BADGE.general;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider"
      style={{
        background: `${meta.color}1f`,
        color: meta.color,
        boxShadow: `inset 0 0 0 1px ${meta.color}33`,
      }}
    >
      {meta.label}
    </span>
  );
}
