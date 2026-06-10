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
  mentions: "rgba(255,209,102,0.30)",
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

// Node radius is scaled by degree so high-degree nodes (the most connected
// concepts) read as the visual anchors of the graph.
const MIN_NODE_RADIUS = 4;
const MAX_NODE_RADIUS = 14;
function radiusForDegree(degree, maxDegree) {
  if (!maxDegree || maxDegree <= 0) return MIN_NODE_RADIUS;
  const t = Math.min(1, Math.sqrt(degree / maxDegree));
  return MIN_NODE_RADIUS + t * (MAX_NODE_RADIUS - MIN_NODE_RADIUS);
}

export default function GraphView({
  snapshot,
  highlights,
  theme,
  mode = "important",
  onModeChange,
}) {
  const fgRef = useRef(null);
  const containerRef = useRef(null);
  const [size, setSize] = useState({ w: 600, h: 480 });
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);

  // Convert backend snapshot to react-force-graph shape, including degree.
  const data = useMemo(() => {
    if (!snapshot) return { nodes: [], links: [], maxDegree: 0 };
    const nodes = (snapshot.nodes || []).map((n) => ({
      id: n.id,
      label: n.label || n.id,
      type: n.node_type || "default",
      degree: n.degree || 0,
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
    const maxDegree = nodes.reduce((m, n) => Math.max(m, n.degree), 0);
    return { nodes, links, maxDegree };
  }, [snapshot]);

  // Connected-node lookup so hover highlights an entire neighborhood.
  const adjacency = useMemo(() => {
    const map = new Map();
    for (const link of data.links) {
      const s = typeof link.source === "object" ? link.source.id : link.source;
      const t = typeof link.target === "object" ? link.target.id : link.target;
      if (!map.has(s)) map.set(s, new Set());
      if (!map.has(t)) map.set(t, new Set());
      map.get(s).add(t);
      map.get(t).add(s);
    }
    return map;
  }, [data]);

  // Tune force layout for spacing and stability once nodes load.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    // Stronger repulsion + longer link distance keeps clusters separated.
    fg.d3Force("charge")?.strength(-180).distanceMax(420);
    fg.d3Force("link")
      ?.distance((link) => {
        // High-degree nodes pull their neighborhood tighter; low-degree
        // ones float further so labels don't overlap.
        const deg = Math.max(link.source.degree || 0, link.target.degree || 0);
        return 50 + Math.min(40, deg * 2);
      })
      .strength(0.25);
    fg.d3Force("center")?.strength(0.05);
  }, [data]);

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

  // When a node is hovered, also highlight its neighbors.
  const hoverNeighborhood = useMemo(() => {
    if (!hovered) return new Set();
    const ns = new Set([hovered.id]);
    const adj = adjacency.get(hovered.id);
    if (adj) for (const id of adj) ns.add(id);
    return ns;
  }, [hovered, adjacency]);

  const isLight = theme === "light";
  const labelColor = isLight ? "#171b28" : "#e6ebf2";
  const isHighlighted = (id) => highlightSet.has(id);
  const isInHoverHood = (id) => hoverNeighborhood.has(id);

  return (
    <motion.section
      initial={{ opacity: 0, scale: 0.99 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.2, duration: 0.5 }}
      className="glass relative flex h-[520px] flex-col overflow-hidden rounded-2xl"
    >
      <header className="flex items-center justify-between gap-3 border-b border-white/5 px-5 py-3 [html:not(.dark)_&]:border-ink-200">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Knowledge graph
          </h2>
          <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
            {data.nodes.length} nodes · {data.links.length} edges ·{" "}
            {mode === "important" ? "top by degree" : "full graph"} · scroll to
            zoom · drag to pan
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onChange={onModeChange} />
          <button
            type="button"
            onClick={() => fgRef.current?.zoomToFit?.(500, 80)}
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
            graphData={{ nodes: data.nodes, links: data.links }}
            width={size.w}
            height={size.h}
            backgroundColor="rgba(0,0,0,0)"
            cooldownTicks={140}
            nodeRelSize={5}
            // Smoother zoom & pan
            enablePointerInteraction
            enableZoomInteraction
            enablePanInteraction
            minZoom={0.3}
            maxZoom={6}
            linkColor={(link) => EDGE_PALETTE[link.type] || EDGE_PALETTE.default}
            linkWidth={(link) => {
              const sId = link.source.id ?? link.source;
              const tId = link.target.id ?? link.target;
              const lit = isHighlighted(sId) || isHighlighted(tId);
              const inHood = isInHoverHood(sId) && isInHoverHood(tId);
              return lit ? 2.4 : inHood ? 1.6 : 0.5;
            }}
            linkDirectionalParticles={(link) => {
              const sId = link.source.id ?? link.source;
              const tId = link.target.id ?? link.target;
              return isHighlighted(sId) || isHighlighted(tId) ? 3 : 0;
            }}
            linkDirectionalParticleSpeed={0.012}
            linkCanvasObjectMode={() => "after"}
            linkCanvasObject={(link, ctx, globalScale) => {
              const sId = link.source.id ?? link.source;
              const tId = link.target.id ?? link.target;
              const lit = isHighlighted(sId) || isHighlighted(tId);
              const involvesHovered =
                hovered && (sId === hovered.id || tId === hovered.id);
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
              ctx.fillStyle = isLight
                ? "rgba(255,255,255,0.92)"
                : "rgba(12,15,26,0.85)";
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
              const isSel = selected?.id === node.id;
              const inHood = isInHoverHood(node.id) && !isHover;

              const baseR = radiusForDegree(node.degree, data.maxDegree);
              const r = isHi
                ? baseR * 1.35 + 1.5
                : isSel
                  ? baseR * 1.25
                  : isHover
                    ? baseR * 1.2
                    : inHood
                      ? baseR * 1.05
                      : baseR;

              // Soft halo for prominent / highlighted / selected nodes.
              if (isHi || isHover || isSel) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r + 6, 0, 2 * Math.PI);
                ctx.fillStyle = `${color}22`;
                ctx.fill();
              }

              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.shadowColor = isHi || isHover || isSel ? color : "transparent";
              ctx.shadowBlur = isHi ? 18 : isHover || isSel ? 12 : 0;
              ctx.fill();
              ctx.shadowBlur = 0;

              // Selection ring
              if (isSel) {
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5 / globalScale;
                ctx.stroke();
              }

              // Show labels only for nodes that are: highlighted, hovered,
              // selected, in hover-neighborhood, or "important" by degree
              // (top tier by node radius). Cuts label clutter at low zoom.
              const isImportant = baseR >= MIN_NODE_RADIUS + 4;
              const showLabel =
                isHi ||
                isHover ||
                isSel ||
                inHood ||
                isImportant ||
                globalScale > 1.6;
              if (showLabel) {
                const label = (node.label || node.id).slice(0, 28);
                ctx.font = `${
                  isHi || isHover || isSel ? 11 : 10
                }px InterVariable, sans-serif`;
                ctx.fillStyle = labelColor;
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.fillText(label, node.x, node.y + r + 2);
              }
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              const baseR = radiusForDegree(node.degree, data.maxDegree);
              ctx.beginPath();
              ctx.arc(node.x, node.y, baseR + 4, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            onNodeHover={(node) => {
              setHovered(node || null);
              if (containerRef.current) {
                containerRef.current.style.cursor = node ? "pointer" : "grab";
              }
            }}
            onNodeClick={(node) => {
              setSelected(node || null);
            }}
            onBackgroundClick={() => setSelected(null)}
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
              {hovered.type} · degree {hovered.degree || 0}
            </div>
            <div className="font-medium text-ink-100 [html:not(.dark)_&]:text-ink-900">
              {hovered.label}
            </div>
            <div className="mt-1 font-mono text-[10px] text-ink-500">
              {hovered.id}
            </div>
          </motion.div>
        )}

        {/* Persistent details panel for the selected node */}
        {selected && (
          <motion.div
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            className="absolute right-3 top-3 max-w-xs rounded-xl border border-white/10
                       bg-ink-900/95 px-3 py-3 text-xs shadow-xl backdrop-blur
                       [html:not(.dark)_&]:border-ink-200
                       [html:not(.dark)_&]:bg-white/95 [html:not(.dark)_&]:text-ink-800"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-ink-400">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{
                    background:
                      NODE_PALETTE[selected.type] || NODE_PALETTE.default,
                  }}
                />
                {selected.type}
              </div>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="text-[10px] text-ink-500 hover:text-ink-100
                           [html:not(.dark)_&]:hover:text-ink-900"
              >
                close
              </button>
            </div>
            <div className="mt-2 text-sm font-medium text-ink-100 [html:not(.dark)_&]:text-ink-900">
              {selected.label}
            </div>
            <div className="mt-1 font-mono text-[10px] text-ink-500">
              {selected.id}
            </div>
            <div className="mt-2 text-[11px] text-ink-400 [html:not(.dark)_&]:text-ink-500">
              degree:{" "}
              <span className="font-mono">{selected.degree || 0}</span> ·
              connections:{" "}
              <span className="font-mono">
                {adjacency.get(selected.id)?.size ?? 0}
              </span>
            </div>
          </motion.div>
        )}
      </div>
    </motion.section>
  );
}

function ModeToggle({ mode, onChange }) {
  const opts = [
    { id: "important", label: "Top" },
    { id: "full", label: "Full" },
  ];
  return (
    <div className="flex items-center gap-1 rounded-md border border-white/5 bg-white/[0.02] p-0.5
                    [html:not(.dark)_&]:border-ink-200">
      {opts.map((o) => {
        const active = mode === o.id;
        return (
          <button
            key={o.id}
            type="button"
            onClick={() => onChange?.(o.id)}
            className={`rounded px-2 py-0.5 text-[10px] uppercase tracking-wider transition-all ${
              active
                ? "bg-accent-500/15 text-accent-300 [html:not(.dark)_&]:text-accent-700"
                : "text-ink-400 hover:text-ink-100 [html:not(.dark)_&]:text-ink-500 [html:not(.dark)_&]:hover:text-ink-900"
            }`}
            title={
              o.id === "important"
                ? "Show top 80 nodes by degree"
                : "Show the entire graph"
            }
          >
            {o.label}
          </button>
        );
      })}
    </div>
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
