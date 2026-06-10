import { useCallback, useEffect, useMemo, useState } from "react";
import toast from "react-hot-toast";
import Header from "./components/Header.jsx";
import IngestPanel from "./components/IngestPanel.jsx";
import QueryBox from "./components/QueryBox.jsx";
import ResultCard from "./components/ResultCard.jsx";
import GraphView from "./components/GraphView.jsx";
import CorpusSelector from "./components/CorpusSelector.jsx";
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
  // "important" caps to top-N nodes by degree; "full" returns the entire graph.
  const [graphMode, setGraphMode] = useState("important");
  // Corpus isolation: list of all corpora + the active one. Each ingestion
  // creates a new isolated corpus; queries operate on the active one.
  const [corpora, setCorpora] = useState([]);
  const [activeCorpusId, setActiveCorpusId] = useState(null);

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
      const snap = await api.graph({
        mode: graphMode,
        limit: 60,
        corpusId: activeCorpusId,
      });
      setSnapshot(snap);
    } catch (err) {
      // Non-fatal — graph viz just stays empty.
      console.warn("graph fetch failed", err);
    }
  }, [graphMode, activeCorpusId]);

  const refreshCorpora = useCallback(async () => {
    try {
      const list = await api.corpora();
      setCorpora(list || []);
      if (!activeCorpusId && list && list.length > 0) {
        // Initial selection: pick the most recently created corpus that's
        // backed by real data so we never default to the empty bootstrap one.
        const meaningful = list.find((c) => c.entity_count > 0) || list[0];
        setActiveCorpusId(meaningful.corpus_id);
      }
    } catch (err) {
      console.warn("corpora fetch failed", err);
    }
  }, [activeCorpusId]);

  const handleSelectCorpus = useCallback(
    async (corpusId) => {
      try {
        await api.selectCorpus(corpusId);
        setActiveCorpusId(corpusId);
        toast.success("Corpus switched");
      } catch (err) {
        toast.error(`Could not switch corpus: ${err.message}`);
      }
    },
    [],
  );

  const handleDeleteCorpus = useCallback(
    async (corpusId) => {
      try {
        await api.deleteCorpus(corpusId);
        toast.success("Corpus deleted");
        if (corpusId === activeCorpusId) setActiveCorpusId(null);
        await refreshCorpora();
      } catch (err) {
        toast.error(`Delete failed: ${err.message}`);
      }
    },
    [activeCorpusId, refreshCorpora],
  );

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
      if (!cancelled) {
        await refreshCorpora();
        refreshGraph();
      }
    })();
    const interval = setInterval(checkHealth, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [refreshGraph, refreshCorpora]);

  // Refresh graph whenever active corpus changes so the viz reflects the
  // currently selected isolated graph (no cross-corpus mixing).
  useEffect(() => {
    if (activeCorpusId) refreshGraph();
  }, [activeCorpusId, refreshGraph]);

  const handleIngested = useCallback(
    (response) => {
      // Each ingest creates a new isolated corpus by default. Switch to it.
      if (response?.corpus_id) {
        setActiveCorpusId(response.corpus_id);
      }
      refreshCorpora();
      refreshGraph();
    },
    [refreshCorpora, refreshGraph],
  );

  const handleAsk = useCallback(
    async (question) => {
      setBusy(true);
      setResult(null);
      setEvidenceHighlights([]);
      try {
        const llmEnabled =
          llmMode === "llm" ? true : llmMode === "extractive" ? false : null;
        const r = await api.query({
          question,
          analystMode: true,
          llmEnabled,
          corpusId: activeCorpusId,
        });
        setResult(r);
      } catch (err) {
        toast.error(err.message || "Query failed");
      } finally {
        setBusy(false);
      }
    },
    [llmMode, activeCorpusId],
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

      <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-3 px-4 pt-4 sm:px-6">
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase tracking-wider text-ink-400 [html:not(.dark)_&]:text-ink-500">
            Active workspace
          </span>
          <CorpusSelector
            corpora={corpora}
            activeCorpusId={activeCorpusId}
            onSelect={handleSelectCorpus}
            onRefresh={refreshCorpora}
            onDelete={handleDeleteCorpus}
          />
        </div>
        <span className="text-[10px] text-ink-500/80 [html:not(.dark)_&]:text-ink-400">
          {corpora.length} corpus{corpora.length === 1 ? "" : "es"} · isolated
        </span>
      </div>

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
            mode={graphMode}
            onModeChange={setGraphMode}
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
