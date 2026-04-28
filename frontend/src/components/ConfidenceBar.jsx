import { motion } from "framer-motion";

export default function ConfidenceBar({ calibrated = 0, raw = 0, error = 0 }) {
  const pct = Math.max(0, Math.min(1, calibrated)) * 100;
  const tone =
    pct >= 75 ? "emerald" : pct >= 50 ? "accent" : pct >= 25 ? "amber" : "rose";
  const fill =
    tone === "emerald"
      ? "from-emerald-400 to-emerald-600"
      : tone === "accent"
        ? "from-accent-400 to-accent-600"
        : tone === "amber"
          ? "from-amber-400 to-amber-600"
          : "from-rose-400 to-rose-600";

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-[11px] text-ink-400 [html:not(.dark)_&]:text-ink-500">
        <span className="uppercase tracking-wider">Calibrated confidence</span>
        <span className="font-mono">
          {(calibrated * 100).toFixed(1)}%
          <span className="ml-2 text-ink-500">
            (raw {(raw * 100).toFixed(1)}%, err {error.toFixed(2)})
          </span>
        </span>
      </div>
      <div className="relative h-2.5 overflow-hidden rounded-full bg-white/[0.05] [html:not(.dark)_&]:bg-ink-100">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className={`h-full rounded-full bg-gradient-to-r ${fill} shadow-[0_0_18px_-2px_rgba(91,140,255,0.55)]`}
        />
      </div>
    </div>
  );
}
