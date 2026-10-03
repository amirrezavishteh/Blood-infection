import { useEffect, useState } from "react";
import { api, type AlertRow, type Replay } from "../api";

export function AlertReview({ replayId }: { replayId: string | null }) {
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [filter, setFilter] = useState<"all" | "open" | "acknowledged">("open");
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    if (!replayId) return;
    api.replay(replayId).then(setReplay).catch((e) => setError(e.message));
    api.alerts(replayId).then(setAlerts).catch((e) => setError(e.message));
  };
  useEffect(load, [replayId]);
  useEffect(() => {
    if (replay?.status !== "playing") return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [replay?.status]);

  async function ack(a: AlertRow) {
    try {
      await api.acknowledge(a.id, notes[a.id] ?? "");
      load();
    } catch (e: any) { setError(e.message); }
  }

  if (!replayId) return <div className="card muted">Select or create a replay in the Replay tab first.</div>;
  const shown = alerts.filter((a) => filter === "all" || a.state === filter);
  return (
    <div className="card stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Simulated alerts — {replayId}</h2>
        <div className="row">
          {(["open", "acknowledged", "all"] as const).map((f) => (
            <button key={f} className="btn" aria-pressed={filter === f} style={filter === f ? { borderColor: "var(--accent)" } : undefined}
                    onClick={() => setFilter(f)}>{f} ({alerts.filter((a) => f === "all" || a.state === f).length})</button>
          ))}
        </div>
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        Acknowledgement records simulated workflow activity only. It is not an outcome label, it does not retrain the model and it
        does not change when later alerts fire. Alerts are shown only up to the replay clock{replay ? ` (hour ${replay.clock})` : ""}.
      </p>
      {error && <div className="error">{error}</div>}
      <div className="table-wrap">
        <table>
          <thead><tr><th>Record</th><th className="num">Trigger hour</th><th className="num">Score</th><th>State</th><th>Annotation</th><th></th></tr></thead>
          <tbody>
            {shown.map((a) => (
              <tr key={a.id}>
                <td className="mono">{a.stay_id.replace("physionet2019:", "")}</td>
                <td className="num">{a.trigger_hour}</td>
                <td className="num">{a.score.toFixed(3)}</td>
                <td>
                  {a.state === "open"
                    ? <span className="badge"><span className="dot" style={{ background: "var(--status-critical)" }} />! open</span>
                    : <span className="badge"><span className="dot" style={{ background: "var(--status-good)" }} />✓ acknowledged by {a.reviewed_by}</span>}
                </td>
                <td>
                  {a.state === "open"
                    ? <input aria-label={`annotation for alert ${a.id}`} value={notes[a.id] ?? ""} placeholder="optional note"
                             onChange={(e) => setNotes({ ...notes, [a.id]: e.target.value })}
                             style={{ border: "1px solid var(--border)", borderRadius: 6, padding: "4px 6px", background: "var(--surface-1)", width: "100%" }} />
                    : <span className="small">{a.annotation || <span className="muted">—</span>}</span>}
                </td>
                <td>{a.state === "open" && <button className="btn" onClick={() => ack(a)}>Acknowledge</button>}</td>
              </tr>
            ))}
            {!shown.length && <tr><td colSpan={6} className="muted">No {filter === "all" ? "" : filter} alerts.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
