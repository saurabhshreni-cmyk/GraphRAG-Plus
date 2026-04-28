import { useCallback, useEffect, useState } from "react";
import toast from "react-hot-toast";
import Header from "./components/Header.jsx";
import IngestPanel from "./components/IngestPanel.jsx";
import QueryBox from "./components/QueryBox.jsx";
import ResultCard from "./components/ResultCard.jsx";
import GraphView from "./components/GraphView.jsx";
import ReasoningStory, { buildStorySteps } from "./components/ReasoningStory.jsx";
import { api } from "./api.js";

const THEME_KEY = "graphrag.theme";

export default function App() {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    return window.localStorage.getItem(THEME_KEY) || "dark";
  });
  const [healthStatus, setHealthStatus] = useState("checking");
  const [snapshot, setSnapshot] = useState({ nodes: [], edges: [] });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [storyOn, setStoryOn] = useState(true);
  const [highlights, setHighlights] = useState([]);

  // Apply theme to <html> and persist.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  // Initial health + graph fetch.
  const refreshGraph = useCallback(async () => {
    try {
      const snap = await api.graph(500);
      setSnapshot(snap);
    } catch (err) {
      // Non-fatal — graph viz just stays empty.
      console.warn("graph fetch failed", err);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await api.health();
        if (!cancelled) setHealthStatus(h.status === "ok" ? "ok" : "down");
      } catch {
        if (!cancelled) setHealthStatus("down");
      }
      if (!cancelled) refreshGraph();
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshGraph]);

  const handleIngested = useCallback(() => {
    refreshGraph();
  }, [refreshGraph]);

  const handleAsk = useCallback(
    async (question) => {
      setBusy(true);
      setResult(null);
      try {
        const r = await api.query({ question, analystMode: true });
        setResult(r);
        // Refresh graph in case ingestion changed it (cheap call).
        refreshGraph();
      } catch (err) {
        toast.error(err.message || "Query failed");
      } finally {
        setBusy(false);
      }
    },
    [refreshGraph],
  );

  const steps = buildStorySteps(result, snapshot);

  return (
    <div className="app-shell min-h-screen text-ink-100 [html:not(.dark)_&]:bg-ink-50 [html:not(.dark)_&]:text-ink-900">
      <Header
        healthStatus={healthStatus}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      />

      <main className="mx-auto grid max-w-[1600px] gap-5 px-6 py-6 lg:grid-cols-12">
        {/* LEFT PANEL: ingest + reasoning story */}
        <div className="flex flex-col gap-5 lg:col-span-3">
          <IngestPanel onIngested={handleIngested} />
          <ReasoningStory
            steps={steps}
            enabled={storyOn && steps.length > 0}
            onToggle={setStoryOn}
            onStepHighlight={setHighlights}
          />
        </div>

        {/* CENTER: query + graph */}
        <div className="flex flex-col gap-5 lg:col-span-6">
          <QueryBox onAsk={handleAsk} busy={busy} />
          <GraphView snapshot={snapshot} highlights={highlights} theme={theme} />
        </div>

        {/* RIGHT PANEL: result */}
        <div className="lg:col-span-3">
          <ResultCard result={result} busy={busy} />
        </div>
      </main>

      <footer className="mx-auto max-w-[1600px] px-6 pb-8 pt-2 text-xs text-ink-500 [html:not(.dark)_&]:text-ink-400">
        <p>
          GraphRAG++ · trust-aware graph retrieval, calibration, and contradiction
          handling — backend FastAPI · frontend React + Vite + Tailwind.
        </p>
      </footer>
    </div>
  );
}
