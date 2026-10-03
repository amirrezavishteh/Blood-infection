import { useEffect, useState } from "react";
import { api, type Replay, type Retrospective, type Timeline } from "../api";
import { ScoreChart } from "../charts/ScoreChart";
import { SmallLine } from "../charts/SmallLine";

const VITALS = ["HR", "MAP", "SBP", "Resp", "Temp", "O2Sat"];
const LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "pH", "HCO3", "BUN", "Glucose", "FiO2"];

export function PatientTimeline({ replay, stayId, refreshKey }: { replay: Replay; stayId: string; refreshKey: number }) {
  const [tl, setTl] = useState<Timeline | null>(null);
  const [retro, setRetro] = useState<Retrospective | null>(null);
  const [showRetro, setShowRetro] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.timeline(replay.id, stayId).then((t) => { setTl(t); setError(null); }).catch((e) => setError(e.message));
  }, [replay.id, stayId, refreshKey]);
  useEffect(() => { setRetro(null); setShowRetro(false); }, [replay.id, stayId]);
  useEffect(() => {
    if (showRetro && !retro && replay.status === "finished") {
      api.retrospective(replay.id, stayId).then(setRetro).catch((e) => setError(e.message));
    }
  }, [showRetro, replay.status]);

  if (error) return <div className="card error">{error}</div>;
  if (!tl) return <div className="card muted">Loading…</div>;

  const latest = tl.scores[tl.scores.length - 1];
  const xMax = Math.max(tl.through_hour, 1);
  const unavailable = tl.scores.filter((s) => s.score === null).length;
  return (
    <div className="stack">
      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <h2>{tl.stay_id}</h2>
            <div className="small muted">
              Age {tl.demographics.Age ?? "—"} · Gender (source code) {tl.demographics.Gender_source_code ?? "—"} ·
              {tl.record_ended ? `record ended at hour ${tl.through_hour}` : `visible through simulated hour ${tl.through_hour}`}
            </div>
          </div>
          <div className="row">
            {latest && (
              latest.score === null
                ? <span className="badge"><span className="dot" style={{ background: "var(--status-warning)" }} />⚠ data unavailable</span>
                : <span className="badge">score {latest.score.toFixed(3)} · {latest.policy_state}</span>
            )}
            <button className="btn" disabled={replay.status !== "finished"} onClick={() => setShowRetro((v) => !v)}
                    title={replay.status !== "finished" ? "Labels are hidden until the replay finishes" : ""}>
              {showRetro ? "Hide" : "Show"} retrospective labels
            </button>
          </div>
        </div>
        <ScoreChart points={tl.scores} threshold={tl.threshold} alertHours={tl.alerts.map((a) => a.trigger_hour)}
                    xMax={showRetro && retro ? Math.max(xMax, retro.labels.length - 1) : xMax}
                    labels={showRetro ? retro?.labels : null} onsetProxy={showRetro ? retro?.onset_proxy_hour : null} />
        <div className="small muted">
          {unavailable > 0 && <>{unavailable} hour(s) without enough valid vital signs are shown as unavailable, never as low risk. </>}
          Target <code>{tl.target_id}</code> · model <code>{tl.model_version}</code> · policy <code>{tl.policy_version}</code> ·
          schema <code>{tl.feature_schema_version}</code>
        </div>
        {showRetro && retro && (
          <div className="small">
            Retrospective view: label start hour {retro.label_start_hour ?? "none"}, onset proxy {retro.onset_proxy_hour ?? "none"}
            {retro.left_censored ? " (positive from first row: onset unknown)" : ""}. {retro.note}
          </div>
        )}
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: "minmax(280px, 1fr) minmax(280px, 1fr)" }}>
        <Contributions tl={tl} />
        <div className="card stack">
          <h2>Latest labs</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Measurement</th><th className="num">Latest</th><th>Unit</th><th className="num">Age of value</th></tr></thead>
              <tbody>
                {LABS.map((v) => {
                  const pts = tl.observations[v];
                  const f = tl.freshness[v];
                  return (
                    <tr key={v}>
                      <td>{v}</td>
                      <td className="num">{pts ? pts[pts.length - 1].value : <span className="muted">not measured</span>}</td>
                      <td className="muted small">{pts ? tl.units[v] : ""}</td>
                      <td className="num small">{f ? `${f.hours_since} h` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid-auto">
        {VITALS.map((v) => (
          <SmallLine key={v} title={v} unit={tl.units[v]} points={tl.observations[v] ?? []} xMax={xMax} freshness={tl.freshness[v]} />
        ))}
      </div>
    </div>
  );
}

function Contributions({ tl }: { tl: Timeline }) {
  const c = tl.contributions;
  const max = c ? Math.max(...c.map((x) => Math.abs(x.contribution)), 1e-6) : 1;
  return (
    <div className="card stack">
      <h2>Feature contributions at hour {tl.scores.length ? tl.scores[tl.scores.length - 1].hour : "—"}</h2>
      {!c && <div className="muted small">No contributions: the latest hour was not scored.</div>}
      {c && (
        <div className="contrib" role="table" aria-label="feature contributions">
          {c.map((x) => {
            const w = (Math.abs(x.contribution) / max) * 50;
            const pos = x.contribution >= 0;
            return (
              <div key={x.feature} style={{ display: "contents" }} role="row">
                <span title={x.feature} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{x.feature}</span>
                <span className="bar-track" aria-hidden>
                  <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--grid)" }} />
                  <span className="bar" style={{
                    left: pos ? "50%" : `${50 - w}%`, width: `${w}%`,
                    background: pos ? "var(--div-pos, #e34948)" : "var(--series-1)",
                  }} />
                </span>
                <span className="mono">{pos ? "+" : "−"}{Math.abs(x.contribution).toFixed(3)} {x.missing ? "(missing)" : x.value != null ? `@ ${+x.value.toFixed(2)}` : ""}</span>
              </div>
            );
          })}
        </div>
      )}
      <div className="legend">
        <span className="key"><span className="swatch" style={{ background: "var(--div-pos, #e34948)" }} />raises score</span>
        <span className="key"><span className="swatch" style={{ background: "var(--series-1)" }} />lowers score</span>
      </div>
      <div className="small muted">{tl.contributions_note}</div>
    </div>
  );
}
