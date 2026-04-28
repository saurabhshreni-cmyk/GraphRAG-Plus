import { useEffect, useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { motion } from "framer-motion";

const NODE_PALETTE = {
  Document: "#7aa7ff",
  Chunk: "#8b9bb4",
  Entity: "#ffd166",
  Concept: "#f78fb3",
  Person: "#06d6a0",
  Organization: "#ef476f",
  default: "#9aa6b8",
};

const EDGE_PALETTE = {
  contains: "rgba(122,167,255,0.55)",
  mentions: "rgba(255,209,102,0.55)",
  supports: "rgba(6,214,160,0.7)",
  contradicts: "rgba(239,71,111,0.75)",
  default: "rgba(255,255,255,0.18)",
};

export default function GraphView({ snapshot, highlights, theme }) {
  const fgRef = useRef(null);
  const containerRef = useRef(null);
  const sizeRef = useRef({ w: 600, h: 480 });

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
      sizeRef.current = {
        w: Math.max(320, rect.width),
        h: Math.max(360, rect.height),
      };
      // Force a re-render via fg ref by zooming to fit.
      fgRef.current?.zoomToFit?.(400, 60);
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // When highlights change, briefly center on the first highlighted node.
  useEffect(() => {
    if (!highlights?.length || !data.nodes.length || !fgRef.current) return;
    const target = data.nodes.find((n) => highlights.includes(n.id));
    if (!target) return;
    const fg = fgRef.current;
    if (typeof target.x === "number" && typeof target.y === "number") {
      fg.centerAt(target.x, target.y, 800);
      fg.zoom(2.5, 800);
    }
  }, [highlights, data.nodes]);

  const highlightSet = useMemo(() => new Set(highlights || []), [highlights]);

  const isLight = theme === "light";
  const labelColor = isLight ? "#171b28" : "#e6ebf2";

  return (
    <motion.section
      initial={{ opacity: 0, scale: 0.99 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.2, duration: 0.5 }}
      className="glass relative flex h-[520px] flex-col overflow-hidden rounded-2xl"
    >
      <header className="flex items-center justify-between border-b border-white/5 px-5 py-3 [html:not(.dark)_&]:border-ink-200">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Knowledge graph</h2>
          <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
            {data.nodes.length} nodes · {data.links.length} edges · scroll to zoom
          </p>
        </div>
        <Legend />
      </header>
      <div ref={containerRef} className="relative flex-1">
        {data.nodes.length === 0 ? (
          <EmptyState />
        ) : (
          <ForceGraph2D
            ref={fgRef}
            graphData={data}
            width={sizeRef.current.w}
            height={sizeRef.current.h}
            backgroundColor="rgba(0,0,0,0)"
            cooldownTicks={120}
            nodeRelSize={5}
            linkColor={(link) =>
              EDGE_PALETTE[link.type] || EDGE_PALETTE.default
            }
            linkWidth={(link) => (highlightSet.has(link.source.id) || highlightSet.has(link.target.id) ? 2 : 0.6)}
            linkDirectionalParticles={(link) =>
              highlightSet.has(link.source.id) || highlightSet.has(link.target.id) ? 3 : 0
            }
            linkDirectionalParticleSpeed={0.012}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const color = NODE_PALETTE[node.type] || NODE_PALETTE.default;
              const isHi = highlightSet.has(node.id);
              const r = isHi ? 7 : 4.5;
              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.shadowColor = isHi ? color : "transparent";
              ctx.shadowBlur = isHi ? 14 : 0;
              ctx.fill();
              ctx.shadowBlur = 0;
              if (isHi || globalScale > 1.4) {
                const label = (node.label || node.id).slice(0, 26);
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
              if (containerRef.current) {
                containerRef.current.style.cursor = node ? "pointer" : "grab";
              }
            }}
          />
        )}
      </div>
    </motion.section>
  );
}

function Legend() {
  const items = [
    ["Document", NODE_PALETTE.Document],
    ["Chunk", NODE_PALETTE.Chunk],
    ["Entity", NODE_PALETTE.Entity],
  ];
  return (
    <div className="flex items-center gap-3 text-[10px] uppercase tracking-wider text-ink-400">
      {items.map(([label, color]) => (
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
