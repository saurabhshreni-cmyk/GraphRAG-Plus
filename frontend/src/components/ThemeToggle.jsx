import { motion } from "framer-motion";

export default function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
      className="relative inline-flex h-9 w-16 items-center rounded-full border border-white/10
                 bg-ink-900/60 transition-colors hover:bg-ink-900
                 [html:not(.dark)_&]:border-ink-200 [html:not(.dark)_&]:bg-white"
    >
      <motion.span
        layout
        transition={{ type: "spring", stiffness: 500, damping: 38 }}
        className={`absolute h-7 w-7 rounded-full shadow-soft
          ${isDark ? "left-1 bg-gradient-to-br from-accent-500 to-accent-700" : "left-8 bg-gradient-to-br from-amber-300 to-amber-500"}`}
      />
      <span className="ml-2 text-[10px] uppercase tracking-wider text-ink-400">
        {isDark ? "" : "DK"}
      </span>
      <span className="ml-auto mr-2 text-[10px] uppercase tracking-wider text-ink-400 [html:not(.dark)_&]:text-ink-500">
        {isDark ? "LT" : ""}
      </span>
    </button>
  );
}
