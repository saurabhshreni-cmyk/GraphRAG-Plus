import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { motion } from "framer-motion";

const NODE_PALETTE = {
  Document: "#7aa7ff",
  Chunk: "#9aa6b8",
  Entity: "#ffd166",
  Concept: "#f78fb3",
  Phrase: "#c4b5fd",
  Person: "#06d6a0",
  Organization: "#ef476f",
  default: "#9aa6b8",
};

const EDGE_PALETTE = {
  contains: "rgba(122,167,255,0.45)",
  mentions: "rgba(255,209,102,0.45)",
  supports: "rgba(6,214,160,0.7)",
  contradicts: "rgba(239,71,111,0.75)",
  is_a: "rgba(196,181,253,0.7)",
  default: "rgba(255,255,255,0.18)",
};

const LEGEND = [
  ["Document", NODE_PALETTE.Document],
  ["Chunk", NODE_PALETTE.Chunk],
  ["Entity", NODE_PALETTE.Entity],
  ["Concept", NODE_PALETTE.Concept],
  ["Phrase", NODE_PALETTE.Phrase],
];

export default function GraphView({ snapshot, highlights, theme }) {
  const fgRef = useRef(null);
  const containerRef = useRef(null);
  const [size, setSize] = useState({ w: 600, h: 480 });
  const [hovered, setHovered] = useState(null);

  // Convert backend snapshot to react-force-graph shape.
  const data = useMemo(() => {
    if (!snapshot) return { nodes: [], links: [] };
    const nodes = (snapshot.nodes || []).map((n) => ({
      id: n.id,
      label: n.label || n.id,
      type: n.node_type || "default",
      raw: n,
    }));
    const ids = new Set(nodes.map((n) => n.id));
    const links = (snapshot.edges || [])
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        type: e.edge_type || "default",
        predicate: e.predicate || e.edge_type || "",
        raw: e,
      }));
    return { nodes, links };
  }, [snapshot]);

  // Track container size for responsive force graph.
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const apply = () => {
      const rect = el.getBoundingClientRect();
      setSize({
        w: Math.max(320, rect.width),
        h: Math.max(360, rect.height),
      });
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const highlightSet = useMemo(() => new Set(highlights || []), [highlights]);
  const isLight = theme === "light";
  const labelColor = isLight ? "#171b28" : "#e6ebf2";
  const isHighlighted = (id) => highlightSet.has(id);

  return (
    <motion.section
      initial={{ opacity: 0, scale: 0.99 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.2, duration: 0.5 }}
      className="glass relative flex h-[520px] flex-col overflow-hidden rounded-2xl"
    >
      <header className="flex items-center justify-between gap-3 border-b border-white/5 px-5 py-3 [html:not(.dark)_&]:border-ink-200">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Knowledge graph</h2>
          <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
            {data.nodes.length} nodes · {data.links.length} edges · scroll to zoom · drag to pan
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => fgRef.current?.zoomToFit?.(400, 60)}
            className="rounded-md border border-white/5 bg-white/[0.02] px-2 py-1 text-[10px] uppercase
                       tracking-wider text-ink-400 transition-all hover:border-accent-500/40 hover:text-ink-100
                       [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:text-ink-500
                       [html:not(.dark)_&]:hover:text-ink-900"
            title="Re-fit graph to viewport"
          >
            Fit
          </button>
          <Legend />
        </div>
      </header>

      <div ref={containerRef} className="relative flex-1">
        {data.nodes.length === 0 ? (
          <EmptyState />
        ) : (
          <ForceGraph2D
            ref={fgRef}
            graphData={data}
            width={size.w}
            height={size.h}
            backgroundColor="rgba(0,0,0,0)"
            cooldownTicks={120}
            nodeRelSize={5}
            linkColor={(link) => EDGE_PALETTE[link.type] || EDGE_PALETTE.default}
            linkWidth={(link) => {
              const lit =
                isHighlighted(link.source.id ?? link.source) ||
                isHighlighted(link.target.id ?? link.target);
              return lit ? 2 : 0.6;
            }}
            linkDirectionalParticles={(link) =>
              isHighlighted(link.source.id ?? link.source) ||
              isHighlighted(link.target.id ?? link.target)
                ? 3
                : 0
            }
            linkDirectionalParticleSpeed={0.012}
            // Edge label rendering — only shown when the edge is highlighted
            // OR the user is hovering a connected node, so the canvas stays clean.
            linkCanvasObjectMode={() => "after"}
            linkCanvasObject={(link, ctx, globalScale) => {
              const lit =
                isHighlighted(link.source.id ?? link.source) ||
                isHighlighted(link.target.id ?? link.target);
              const involvesHovered =
                hovered &&
                ((link.source.id ?? link.source) === hovered.id ||
                  (link.target.id ?? link.target) === hovered.id);
              if (!lit && !involvesHovered) return;
              if (typeof link.source.x !== "number") return;
              const midX = (link.source.x + link.target.x) / 2;
              const midY = (link.source.y + link.target.y) / 2;
              const label = (link.predicate || link.type || "").slice(0, 22);
              if (!label) return;
              ctx.font = `${10 / globalScale}px InterVariable, sans-serif`;
              const textWidth = ctx.measureText(label).width;
              const padX = 4 / globalScale;
              const padY = 2 / globalScale;
              ctx.fillStyle = isLight ? "rgba(255,255,255,0.92)" : "rgba(12,15,26,0.85)";
              ctx.fillRect(
                midX - textWidth / 2 - padX,
                midY - 6 / globalScale - padY,
                textWidth + padX * 2,
                12 / globalScale + padY * 2,
              );
              ctx.fillStyle = labelColor;
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(label, midX, midY);
            }}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const color = NODE_PALETTE[node.type] || NODE_PALETTE.default;
              const isHi = highlightSet.has(node.id);
              const isHover = hovered?.id === node.id;
              const r = isHi ? 7 : isHover ? 6 : 4.5;
              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.shadowColor = isHi || isHover ? color : "transparent";
              ctx.shadowBlur = isHi ? 16 : isHover ? 10 : 0;
              ctx.fill();
              ctx.shadowBlur = 0;
              if (isHi || isHover || globalScale > 1.4) {
                const label = (node.label || node.id).slice(0, 28);
                ctx.font = `${10 / globalScale}px InterVariable, sans-serif`;
                ctx.fillStyle = labelColor;
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.fillText(label, node.x, node.y + r + 2);
              }
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              ctx.beginPath();
              ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            onNodeHover={(node) => {
              setHovered(node || null);
              if (containerRef.current) {
                containerRef.current.style.cursor = node ? "pointer" : "grab";
              }
            }}
          />
        )}

        {/* Floating tooltip for hovered node */}
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="pointer-events-none absolute left-3 top-3 max-w-xs rounded-xl
                       border border-white/10 bg-ink-900/90 px-3 py-2 text-xs shadow-lg
                       backdrop-blur [html:not(.dark)_&]:border-ink-200
                       [html:not(.dark)_&]:bg-white/95 [html:not(.dark)_&]:text-ink-800"
          >
            <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wider text-ink-400">
              <span
                className="h-2 w-2 rounded-full"
                style={{
                  background: NODE_PALETTE[hovered.type] || NODE_PALETTE.default,
                }}
              />
              {hovered.type}
            </div>
            <div className="font-medium text-ink-100 [html:not(.dark)_&]:text-ink-900">
              {hovered.label}
            </div>
            <div className="mt-1 font-mono text-[10px] text-ink-500">{hovered.id}</div>
          </motion.div>
        )}
      </div>
    </motion.section>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-3 text-[10px] uppercase tracking-wider text-ink-400">
      {LEGEND.map(([label, color]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: color, boxShadow: `0 0 8px ${color}55` }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
      <svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden>
        <circle cx="14" cy="20" r="4" fill="#7aa7ff" opacity="0.7" />
        <circle cx="42" cy="14" r="3" fill="#ffd166" opacity="0.7" />
        <circle cx="44" cy="40" r="4" fill="#06d6a0" opacity="0.7" />
        <circle cx="18" cy="44" r="3" fill="#ef476f" opacity="0.7" />
        <g stroke="rgba(255,255,255,0.18)" strokeWidth="1">
          <line x1="14" y1="20" x2="42" y2="14" />
          <line x1="42" y1="14" x2="44" y2="40" />
          <line x1="44" y1="40" x2="18" y2="44" />
          <line x1="18" y1="44" x2="14" y2="20" />
          <line x1="14" y1="20" x2="44" y2="40" />
        </g>
      </svg>
      <p className="max-w-sm text-sm text-ink-400 [html:not(.dark)_&]:text-ink-500">
        The graph is empty. Ingest documents from the left panel to populate
        nodes and relationships.
      </p>
    </div>
  );
}
