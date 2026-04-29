import { useCallback, useEffect, useMemo, useState } from "react";
import toast from "react-hot-toast";
import Header from "./components/Header.jsx";
import IngestPanel from "./components/IngestPanel.jsx";
import QueryBox from "./components/QueryBox.jsx";
import ResultCard from "./components/ResultCard.jsx";
import GraphView from "./components/GraphView.jsx";
import ReasoningStory, { buildStorySteps } from "./components/ReasoningStory.jsx";
import { api } from "./api.js";

const THEME_KEY = "graphrag.theme";
const LLM_MODE_KEY = "graphrag.llmMode"; // "auto" | "llm" | "extractive"

export default function App() {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    return window.localStorage.getItem(THEME_KEY) || "dark";
  });
  const [llmMode, setLlmMode] = useState(() => {
    if (typeof window === "undefined") return "auto";
    return window.localStorage.getItem(LLM_MODE_KEY) || "auto";
  });

  const [healthStatus, setHealthStatus] = useState("checking");
  const [serverLlmEnabled, setServerLlmEnabled] = useState(false);
  const [snapshot, setSnapshot] = useState({ nodes: [], edges: [] });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [storyOn, setStoryOn] = useState(true);
  const [storyHighlights, setStoryHighlights] = useState([]);
  const [evidenceHighlights, setEvidenceHighlights] = useState([]);

  // Apply theme to <html> and persist.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(LLM_MODE_KEY, llmMode);
  }, [llmMode]);

  const refreshGraph = useCallback(async () => {
    try {
      const snap = await api.graph(500);
      setSnapshot(snap);
    } catch (err) {
      // Non-fatal — graph viz just stays empty.
      console.warn("graph fetch failed", err);
    }
  }, []);

  // Initial health + graph fetch + periodic re-poll so the badge stays honest.
  useEffect(() => {
    let cancelled = false;
    const checkHealth = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealthStatus(h.status === "ok" ? "ok" : "down");
        setServerLlmEnabled(Boolean(h.llm_enabled));
      } catch {
        if (!cancelled) setHealthStatus("down");
      }
    };
    (async () => {
      await checkHealth();
      if (!cancelled) refreshGraph();
    })();
    const interval = setInterval(checkHealth, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [refreshGraph]);

  const handleIngested = useCallback(() => {
    refreshGraph();
  }, [refreshGraph]);

  const handleAsk = useCallback(
    async (question) => {
      setBusy(true);
      setResult(null);
      setEvidenceHighlights([]);
      try {
        const llmEnabled =
          llmMode === "llm" ? true : llmMode === "extractive" ? false : null;
        const r = await api.query({ question, analystMode: true, llmEnabled });
        setResult(r);
      } catch (err) {
        toast.error(err.message || "Query failed");
      } finally {
        setBusy(false);
      }
    },
    [llmMode],
  );

  const steps = useMemo(() => buildStorySteps(result, snapshot), [result, snapshot]);

  // Combined highlights: evidence-hover takes precedence over reasoning-story.
  const activeHighlights =
    evidenceHighlights.length > 0 ? evidenceHighlights : storyHighlights;

  return (
    <div className="app-shell min-h-screen text-ink-100 [html:not(.dark)_&]:bg-ink-50 [html:not(.dark)_&]:text-ink-900">
      <Header
        healthStatus={healthStatus}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      />

      <main className="mx-auto grid max-w-[1600px] gap-5 px-4 py-6 sm:px-6 md:grid-cols-2 lg:grid-cols-12">
        {/* LEFT PANEL: ingest + reasoning story */}
        <div className="flex flex-col gap-5 md:col-span-2 lg:col-span-3">
          <IngestPanel onIngested={handleIngested} snapshot={snapshot} />
          <ReasoningStory
            steps={steps}
            enabled={storyOn && steps.length > 0}
            onToggle={setStoryOn}
            onStepHighlight={setStoryHighlights}
          />
        </div>

        {/* CENTER: query + graph */}
        <div className="flex flex-col gap-5 md:col-span-2 lg:col-span-6">
          <QueryBox
            onAsk={handleAsk}
            busy={busy}
            llmMode={llmMode}
            onLlmModeChange={setLlmMode}
            serverLlmEnabled={serverLlmEnabled}
          />
          <GraphView
            snapshot={snapshot}
            highlights={activeHighlights}
            theme={theme}
          />
        </div>

        {/* RIGHT PANEL: result */}
        <div className="md:col-span-2 lg:col-span-3">
          <ResultCard
            result={result}
            busy={busy}
            onEvidenceHover={setEvidenceHighlights}
          />
        </div>
      </main>

      <footer className="mx-auto max-w-[1600px] px-6 pb-8 pt-2 text-xs text-ink-500 [html:not(.dark)_&]:text-ink-400">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p>
            GraphRAG++ · trust-aware graph retrieval, calibration &amp; contradiction
            handling — FastAPI backend · React + Vite + Tailwind frontend.
          </p>
          <p className="font-mono text-[10px] text-ink-500/80">
            server LLM: {serverLlmEnabled ? "on" : "off"} · view: {llmMode}
          </p>
        </div>
      </footer>
    </div>
  );
}
