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
      <span className="ml-2 text-ink-400" aria-hidden>
        {isDark ? null : <MoonIcon />}
      </span>
      <span
        className="ml-auto mr-2 text-ink-400 [html:not(.dark)_&]:text-ink-500"
        aria-hidden
      >
        {isDark ? <SunIcon /> : null}
      </span>
    </button>
  );
}

function SunIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <path
        d="M13.5 9.6A5.5 5.5 0 0 1 6.4 2.5a5.5 5.5 0 1 0 7.1 7.1Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}
