import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

// Build a five-step reasoning narrative from the QueryResponse + graph snapshot.
// Each step exposes a `highlight` set of node ids that the GraphView pulses.
export function buildStorySteps(result, snapshot) {
  if (!result) return [];
  const evidence = result.evidence || [];
  const evidenceIds = evidence.map((e) => e.id);
  const sourceIds = evidence.map((e) => e.source_id).filter(Boolean);
  const docNodeIds = (snapshot?.nodes || [])
    .filter(
      (n) =>
        n.node_type === "Document" &&
        sourceIds.some((sid) => sid === n.id || (n.id || "").startsWith(sid)),
    )
    .map((n) => n.id);
  const entityIds = (snapshot?.nodes || [])
    .filter((n) => n.node_type && n.node_type !== "Document" && n.node_type !== "Chunk")
    .map((n) => n.id)
    .slice(0, 12);
  const pathIds = (result.evidence_paths || []).flat().filter(Boolean);

  return [
    {
      id: "retrieve",
      title: "Retrieved documents",
      detail:
        evidence.length > 0
          ? `Pulled ${evidence.length} candidate chunk${evidence.length === 1 ? "" : "s"} via hybrid retrieval (vector + BM25 + graph).`
          : "No supporting chunks were retrieved.",
      highlight: [...docNodeIds, ...evidenceIds],
    },
    {
      id: "extract",
      title: "Extracted entities",
      detail: entityIds.length
        ? `Linked the question to ${entityIds.length} entities already in the graph.`
        : "No entities have been extracted yet for this corpus.",
      highlight: entityIds,
    },
    {
      id: "traverse",
      title: "Traversed graph relationships",
      detail: pathIds.length
        ? `Walked ${result.evidence_paths.length} evidence path${result.evidence_paths.length === 1 ? "" : "s"} across the knowledge graph.`
        : "No explicit graph paths were available — falling back to text similarity.",
      highlight: pathIds,
    },
    {
      id: "rank",
      title: "Ranked evidence",
      detail:
        evidence.length > 0
          ? `Scored each candidate (semantic, graph, trust, uncertainty). Top score: ${(evidence[0]?.final_score ?? 0).toFixed(2)}.`
          : "Nothing to rank.",
      highlight: evidenceIds,
    },
    {
      id: "answer",
      title: "Final answer generated",
      detail: result.failure_type
        ? `Returned a ${result.failure_type.toLowerCase().replace("_", " ")} answer (calibrated confidence ${(result.calibrated_confidence * 100).toFixed(0)}%).`
        : `Returned answer with calibrated confidence ${(result.calibrated_confidence * 100).toFixed(0)}%.`,
      highlight: [...docNodeIds, ...evidenceIds],
    },
  ];
}

export default function ReasoningStory({
  steps,
  enabled,
  onToggle,
  onStepHighlight,
}) {
  const [active, setActive] = useState(0);

  // Auto-advance through steps when enabled and steps exist.
  useEffect(() => {
    if (!enabled || steps.length === 0) return;
    setActive(0);
    onStepHighlight?.(steps[0]?.highlight || []);
    const interval = setInterval(() => {
      setActive((i) => {
        const next = (i + 1) % steps.length;
        onStepHighlight?.(steps[next]?.highlight || []);
        return next;
      });
    }, 2400);
    return () => clearInterval(interval);
  }, [enabled, steps, onStepHighlight]);

  // When toggled off, clear highlights.
  useEffect(() => {
    if (!enabled) onStepHighlight?.([]);
  }, [enabled, onStepHighlight]);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25, duration: 0.45 }}
      className="glass rounded-2xl p-5"
    >
      <header className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Reasoning story</h2>
          <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
            How GraphRAG++ reached this answer.
          </p>
        </div>
        <ToggleSwitch checked={enabled} onChange={onToggle} label="Show reasoning path" />
      </header>

      <ol className="relative space-y-3 pl-6">
        <span className="absolute left-2 top-1 bottom-1 w-px bg-white/8 [html:not(.dark)_&]:bg-ink-200" />
        <AnimatePresence initial={false}>
          {steps.length === 0 && (
            <motion.li
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-sm text-ink-400 [html:not(.dark)_&]:text-ink-500"
            >
              Ask a question to see the reasoning trace.
            </motion.li>
          )}
        </AnimatePresence>
        {steps.map((step, idx) => {
          const isActive = enabled && idx === active;
          const reached = enabled && idx <= active;
          return (
            <motion.li
              key={step.id}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.05 * idx }}
              onMouseEnter={() => onStepHighlight?.(step.highlight)}
              className="relative"
            >
              <span
                className={`absolute -left-[18px] top-1.5 flex h-3.5 w-3.5 items-center justify-center rounded-full border ${
                  isActive
                    ? "border-accent-400 bg-accent-500 shadow-[0_0_14px_2px_rgba(91,140,255,0.7)]"
                    : reached
                      ? "border-accent-500/40 bg-accent-600/40"
                      : "border-white/20 bg-ink-800"
                }`}
              />
              <p
                className={`text-sm font-medium ${
                  isActive ? "text-ink-50 [html:not(.dark)_&]:text-ink-900" : "text-ink-200 [html:not(.dark)_&]:text-ink-700"
                }`}
              >
                {idx + 1}. {step.title}
              </p>
              <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
                {step.detail}
              </p>
            </motion.li>
          );
        })}
      </ol>
    </motion.section>
  );
}

function ToggleSwitch({ checked, onChange, label }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-300 [html:not(.dark)_&]:text-ink-600">
      <span>{label}</span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full border transition-colors ${
          checked
            ? "border-accent-500/50 bg-accent-500/30"
            : "border-white/10 bg-ink-900/60 [html:not(.dark)_&]:bg-ink-200"
        }`}
        aria-pressed={checked}
      >
        <motion.span
          layout
          transition={{ type: "spring", stiffness: 500, damping: 38 }}
          className={`absolute h-3.5 w-3.5 rounded-full ${
            checked ? "left-[18px] bg-accent-400" : "left-1 bg-ink-300"
          }`}
        />
      </button>
    </label>
  );
}
