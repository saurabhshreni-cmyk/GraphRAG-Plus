import { motion } from "framer-motion";
import ThemeToggle from "./ThemeToggle.jsx";

export default function Header({ healthStatus, theme, onToggleTheme }) {
  const dotClass =
    healthStatus === "ok"
      ? "bg-emerald-400"
      : healthStatus === "checking"
        ? "bg-amber-400"
        : "bg-rose-400";
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="sticky top-0 z-30 border-b border-white/5 bg-ink-950/70 backdrop-blur-xl
                 dark:border-white/5
                 [html:not(.dark)_&]:bg-white/70 [html:not(.dark)_&]:border-ink-200"
    >
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ rotate: -8, scale: 0.9 }}
            animate={{ rotate: 0, scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 18 }}
          >
            <LogoMark />
          </motion.div>
          <div className="leading-tight">
            <h1 className="text-base font-semibold tracking-tight">
              <span className="bg-gradient-to-br from-ink-50 to-ink-300 bg-clip-text text-transparent
                               [html:not(.dark)_&]:from-ink-900 [html:not(.dark)_&]:to-ink-600">
                GraphRAG
              </span>
              <span className="bg-gradient-to-br from-accent-400 to-accent-600 bg-clip-text text-transparent">
                ++
              </span>
            </h1>
            <p className="text-xs text-ink-400 [html:not(.dark)_&]:text-ink-500">
              Trust-aware Graph Retrieval Augmented Generation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span
            className="flex items-center gap-2 rounded-full border border-white/5 px-3 py-1.5
                       text-xs text-ink-300 [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:text-ink-600"
            title={`API health: ${healthStatus}`}
          >
            <span className={`h-2 w-2 rounded-full animate-pulse-soft ${dotClass}`} />
            {healthStatus === "ok"
              ? "Backend online"
              : healthStatus === "checking"
                ? "Checking…"
                : "Backend offline"}
          </span>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </div>
    </motion.header>
  );
}

function LogoMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#7aa7ff" />
          <stop offset="1" stopColor="#3358c8" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="url(#lg)" />
      <g fill="none" stroke="white" strokeWidth="1.4" strokeLinecap="round">
        <line x1="9" y1="11" x2="22" y2="9" />
        <line x1="22" y1="9" x2="23" y2="22" />
        <line x1="23" y1="22" x2="11" y2="23" />
        <line x1="11" y1="23" x2="9" y2="11" />
        <line x1="9" y1="11" x2="23" y2="22" />
        <circle cx="9" cy="11" r="2.4" fill="white" />
        <circle cx="22" cy="9" r="2" fill="white" />
        <circle cx="23" cy="22" r="2.4" fill="white" />
        <circle cx="11" cy="23" r="2" fill="white" />
      </g>
    </svg>
  );
}
