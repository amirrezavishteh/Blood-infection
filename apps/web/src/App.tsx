import { useEffect, useState } from "react";
import { api, ApiError, setUser } from "./api";
import { AlertReview } from "./components/AlertReview";
import { DatasetManager } from "./components/DatasetManager";
import { ExperimentReport } from "./components/ExperimentReport";
import { ReplayWorkspace } from "./components/ReplayWorkspace";

type Tab = "replay" | "alerts" | "datasets" | "experiments";
const TABS: [Tab, string][] = [["replay", "Replay"], ["alerts", "Alert review"], ["datasets", "Datasets"], ["experiments", "Experiments"]];

function stored(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function store(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

export function App() {
  const [tab, setTab] = useState<Tab>(() => (stored("sepsis.tab", "replay") as Tab));
  const [replayId, setReplayId] = useState<string | null>(() => stored("sepsis.replay", "") || null);
  const [user, setUserState] = useState(() => stored("sepsis.user", "local-user"));
  const [disclaimer, setDisclaimer] = useState("Research simulator — not for patient care.");
  const [ready, setReady] = useState<{ ready: boolean; checks: Record<string, unknown> } | null>(null);
  const [needToken, setNeedToken] = useState(false);
  const [tokenInput, setTokenInput] = useState("");

  useEffect(() => { setUser(user); store("sepsis.user", user); }, [user]);
  useEffect(() => { store("sepsis.tab", tab); }, [tab]);
  useEffect(() => { store("sepsis.replay", replayId ?? ""); }, [replayId]);
  useEffect(() => {
    api.meta().then((m) => setDisclaimer(m.disclaimer)).catch((e) => {
      // The server was started with SEPSIS_API_TOKEN: ask for it once, keep it in this browser only.
      if (e instanceof ApiError && e.status === 401) setNeedToken(true);
    });
    const check = () => api.ready().then(setReady).catch(() => setReady(null));
    check();
    const t = setInterval(check, 10000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Sepsis early-warning research simulator</h1>
        <span className="spacer" />
        {ready && (
          <span className="badge" title={JSON.stringify(ready.checks)}>
            <span className="dot" style={{ background: ready.ready ? "var(--status-good)" : "var(--status-warning)" }} />
            {ready.ready ? "✓ ready" : `⚠ ${Object.entries(ready.checks).filter(([k, v]) => ["database", "worker", "model"].includes(k) && !v).map(([k]) => k).join(", ") || "not ready"} unavailable`}
          </span>
        )}
        <label className="field" style={{ flexDirection: "row", alignItems: "center" }}>
          Reviewer
          <input value={user} onChange={(e) => setUserState(e.target.value.replace(/[^\w.@-]/g, "").slice(0, 64) || "local-user")} style={{ width: 130 }} />
        </label>
      </header>
      <div className="banner" role="note">{disclaimer}</div>
      {needToken && (
        <form className="card row" style={{ marginTop: 12 }}
              onSubmit={(e) => { e.preventDefault(); store("sepsis.token", tokenInput.trim()); location.reload(); }}>
          <span>This server requires an access token.</span>
          <input type="password" aria-label="API token" value={tokenInput} onChange={(e) => setTokenInput(e.target.value)}
                 placeholder="API token" style={{ border: "1px solid var(--border)", borderRadius: 6, padding: "6px 8px", background: "var(--surface-1)" }} />
          <button className="btn primary" type="submit" disabled={!tokenInput.trim()}>Save token</button>
          <span className="small muted">Stored in this browser only.</span>
        </form>
      )}
      <nav className="tabs" role="tablist">
        {TABS.map(([k, label]) => (
          <button key={k} role="tab" className="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>
        ))}
      </nav>
      {tab === "replay" && <ReplayWorkspace replayId={replayId} onSelectReplay={setReplayId} />}
      {tab === "alerts" && <AlertReview replayId={replayId} />}
      {tab === "datasets" && <DatasetManager />}
      {tab === "experiments" && <ExperimentReport />}
    </div>
  );
}
