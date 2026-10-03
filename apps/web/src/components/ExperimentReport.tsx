import { useEffect, useState } from "react";
import { api, type EvalFull, type Explain, type RunDetail, type RunSummary } from "../api";
import { ReliabilityChart } from "../charts/ReliabilityChart";

const f3 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3));
const ci = (c?: number[] | null) => (c ? `[${c[0].toFixed(3)}, ${c[1].toFixed(3)}]` : "");
const SPLITS = ["a_validation", "a_test", "b_external"];

export function ExperimentReport() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [defaultRun, setDefaultRun] = useState<string | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.experiments().then((r) => {
      setRuns(r.runs);
      setDefaultRun(r.default_model_version);
      setSel((s) => s ?? r.default_model_version ?? r.runs[r.runs.length - 1]?.run_id ?? null);
    }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (sel) api.experiment(sel).then(setDetail).catch((e) => setError(e.message));
  }, [sel]);

  function exportJson() {
    if (!detail) return;
    const blob = new Blob([JSON.stringify(detail, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${detail.run_id}-report.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div className="stack">
      <div className="card stack">
        <h2>Experiments</h2>
        {error && <div className="error">{error}</div>}
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Run</th><th>Model</th><th>Features</th><th className="num">Val AUPRC</th>
                <th className="num">A test AUROC</th><th className="num">A test utility</th>
                <th className="num">B AUROC</th><th className="num">B utility</th><th className="num">B early sens.</th><th className="num">B alerts / 100 pd</th></tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const a = r.evaluations.a_test, b = r.evaluations.b_external;
                return (
                  <tr key={r.run_id} onClick={() => setSel(r.run_id)} style={{ cursor: "pointer", background: sel === r.run_id ? "var(--series-1-wash)" : undefined }}>
                    <td className="mono">{r.run_id}{r.run_id === defaultRun ? " ★" : ""}</td>
                    <td>{r.model_kind}</td><td>{r.feature_set}</td>
                    <td className="num">{f3(r.validation_auprc)}</td>
                    <td className="num">{f3(a?.auroc)}</td><td className="num">{f3(a?.utility)}</td>
                    <td className="num">{f3(b?.auroc)}</td><td className="num">{f3(b?.utility)}</td>
                    <td className="num">{f3(b?.early_sensitivity)}</td>
                    <td className="num">{b?.alerts_per_100_patient_days == null ? "—" : b.alerts_per_100_patient_days.toFixed(1)}</td>
                  </tr>
                );
              })}
              {!runs.length && <tr><td colSpan={10} className="muted">No runs yet. Train with <code>python -m sepsis train</code>.</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="small muted">★ = model used for new replays (promotion manifest, else the latest calibrated run with a frozen policy).</div>
      </div>
      {detail && <RunDetailView d={detail} onExport={exportJson} />}
    </div>
  );
}

function RunDetailView({ d, onExport }: { d: RunDetail; onExport: () => void }) {
  const evals = SPLITS.filter((s) => d.evaluations[s]).map((s) => [s, d.evaluations[s]] as [string, EvalFull]);
  const rel = [
    d.evaluations.a_test && { name: "A internal test", color: "var(--series-1)", bins: d.evaluations.a_test.hourly.reliability },
    d.evaluations.b_external && { name: "B external hospital", color: "var(--series-2)", bins: d.evaluations.b_external.hourly.reliability },
  ].filter(Boolean) as { name: string; color: string; bins: any[] }[];
  return (
    <div className="stack">
      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Run {d.run_id}</h2>
          <button className="btn" onClick={onExport}>Export metrics + configuration (JSON)</button>
        </div>
        <div className="small muted">
          {d.model_kind} · feature set {d.feature_set} · target <code>{d.target_id}</code> · split <code>{String(d.provenance.split_hash)}</code> ·
          code <code>{String(d.provenance.code_revision).slice(0, 12)}</code> · calibration {d.calibration ? `sigmoid on ${String(d.calibration.fitted_on)}` : "none"}
        </div>
        {!evals.length && <div className="muted">Not evaluated yet.</div>}
        {evals.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Split</th><th className="num">Records</th><th className="num">AUROC (95% CI)</th><th className="num">AUPRC (95% CI)</th>
                <th className="num">Brier</th><th className="num">Utility (95% CI)</th><th className="num">Official utility</th><th className="num">Coverage</th><th className="num" title="How many times this run was scored on this split; above 1 means a repeated look">Look</th></tr></thead>
              <tbody>
                {evals.map(([s, e]) => (
                  <tr key={s}>
                    <td>{s}</td><td className="num">{e.records.toLocaleString()}</td>
                    <td className="num">{f3(e.hourly.auroc)} <span className="muted small">{ci(e.hourly.auroc_ci)}</span></td>
                    <td className="num">{f3(e.hourly.auprc)} <span className="muted small">{ci(e.hourly.auprc_ci)}</span></td>
                    <td className="num">{f3(e.hourly.brier)}</td>
                    <td className="num">{f3(e.benchmark.utility)} <span className="muted small">{ci(e.benchmark.utility_ci)}</span></td>
                    <td className="num">{e.official ? f3(e.official.Utility as number) : "—"}</td>
                    <td className="num">{(e.coverage.coverage * 100).toFixed(1)}%</td>
                    <td className="num">{e.evaluation_number_for_split ?? "—"}{(e.evaluation_number_for_split ?? 1) > 1 ? " ⚠" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: "minmax(300px, 440px) 1fr" }}>
        <div className="card stack">
          <h2>Calibration</h2>
          {rel.length ? <ReliabilityChart series={rel} /> : <div className="muted">No evaluation yet.</div>}
          <div className="small muted">Calibration was fitted on A calibration records and is re-checked under hospital transfer.</div>
        </div>
        <div className="card stack">
          <h2>Alert policy — burden trade-off (A validation)</h2>
          {d.policy_search ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th className="num">Budget / 100 pd</th><th>Feasible</th><th className="num">Threshold</th><th className="num">Consec.</th><th className="num">Cooldown</th>
                  <th className="num">Alerts / 100 pd</th><th className="num">Early sens.</th><th className="num">≥6 h warning</th><th className="num">Precision</th></tr></thead>
                <tbody>
                  {Object.entries(d.policy_search.by_budget).map(([b, v]) => (
                    <tr key={b} style={+b === d.policy_search!.primary_budget ? { fontWeight: 600 } : undefined}>
                      <td className="num">{b}{+b === d.policy_search!.primary_budget ? " (primary)" : ""}</td>
                      <td>{v.feasible ? "yes" : "infeasible"}</td>
                      <td className="num">{v.feasible ? (v.threshold as number).toFixed(3) : "—"}</td>
                      <td className="num">{v.feasible ? String(v.consecutive) : "—"}</td>
                      <td className="num">{v.feasible ? `${v.cooldown_h} h` : "—"}</td>
                      <td className="num">{v.feasible ? (v.alerts_per_100_patient_days as number).toFixed(2) : "—"}</td>
                      <td className="num">{v.feasible ? f3(v.early_sensitivity as number) : "—"}</td>
                      <td className="num">{v.feasible ? f3(v.six_hour_warning_rate as number) : "—"}</td>
                      <td className="num">{v.feasible ? f3(v.alert_precision as number) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="muted">No policy selected yet.</div>}
          <div className="small muted">Budgets are simulated research comparison points, not clinically approved limits.</div>
        </div>
      </div>

      {d.explain && <ExplainView x={d.explain} />}
      {evals.filter(([s]) => s !== "a_validation").map(([s, e]) => <SplitDetail key={s} split={s} e={e} />)}
    </div>
  );
}

function SplitDetail({ split, e }: { split: string; e: EvalFull }) {
  const a = e.alerts as Record<string, any>;
  return (
    <div className="card stack">
      <h2>{split === "b_external" ? "B external hospital" : "A internal test"} — alerts, subgroups, references</h2>
      <div className="kpis">
        <Kpi label="Early event sensitivity" value={f3(a.early_sensitivity)} sub={`${a.detected_early} detected · ${a.missed} missed`} />
        <Kpi label="≥6 h warning" value={f3(a.six_hour_warning_rate)} />
        <Kpi label="Median lead time" value={a.lead_time_median_h == null ? "—" : `${a.lead_time_median_h.toFixed(1)} h`}
             sub={a.lead_time_iqr_h ? `IQR ${a.lead_time_iqr_h[0]}–${a.lead_time_iqr_h[1]} h` : undefined} />
        <Kpi label="Alert precision" value={f3(a.alert_precision)} sub={`pre-onset ${a.matched_pre_onset} · late ${a.matched_late}`} />
        <Kpi label="Alerts / 100 patient-days" value={a.alerts_per_100_patient_days?.toFixed(2)} sub={`unmatched ${a.unmatched_per_100_patient_days?.toFixed(2)}`} />
        <Kpi label="False alerts (non-septic) / 100 pd" value={a.false_alerts_nonseptic_per_100_patient_days?.toFixed(2)} />
      </div>
      <div className="small muted">
        {a.septic_onset_evaluable_stays} onset-evaluable septic records; {a.left_censored_positive_stays} positive-from-first-row records
        excluded from lead-time summaries. Benchmark threshold {e.benchmark.threshold.toFixed(4)}; {e.benchmark.fallback_rows} unavailable hours used the fixed fallback.
      </div>
      <div className="grid-2" style={{ gridTemplateColumns: "1fr minmax(240px, 360px)" }}>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Subgroup</th><th className="num">Records</th><th className="num">Septic</th><th className="num">AUROC</th><th className="num">95% CI</th><th></th></tr></thead>
            <tbody>
              {e.subgroups.map((g) => (
                <tr key={g.subgroup}><td>{g.subgroup}</td><td className="num">{g.stays.toLocaleString()}</td><td className="num">{g.septic_stays}</td>
                  <td className="num">{f3(g.auroc)}</td><td className="num small">{ci(g.auroc_ci)}</td><td className="small">{g.unstable ? "⚠ unstable (few events)" : ""}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Reference score</th><th className="num">AUROC</th><th className="num">AUPRC</th></tr></thead>
            <tbody>
              <tr><td><strong>Model</strong></td><td className="num">{f3(e.hourly.auroc)}</td><td className="num">{f3(e.hourly.auprc)}</td></tr>
              {Object.entries(e.baselines).map(([k, v]) => (
                <tr key={k}><td>{k.replace("_", " ")}</td><td className="num">{f3(v.auroc)}</td><td className="num">{f3(v.auprc)}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="small muted">Partial rule scores from the latest values; GCS and culture data are not available in this dataset.</div>
        </div>
      </div>
    </div>
  );
}

function ExplainView({ x }: { x: Explain }) {
  const max = Math.max(...x.top_features.map((f) => f.importance_share), 1e-9);
  return (
    <div className="card stack">
      <h2>What the model relies on, and how it shifts at hospital B</h2>
      <div className="kpis">
        {Object.entries(x.family_share).map(([fam, share]) => (
          <div key={fam} className="kpi"><div className="label">{fam}</div><div className="value">{(share * 100).toFixed(0)}%</div></div>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Feature</th><th>Family</th><th style={{ width: "30%" }}>Share of importance</th>
            <th className="num" title="Standardised mean difference, hospital B minus A training">Shift (SMD)</th>
            <th className="num">Missing A</th><th className="num">Missing B</th></tr></thead>
          <tbody>
            {x.top_features.slice(0, 15).map((f) => (
              <tr key={f.feature}>
                <td className="mono">{f.feature}</td><td className="small">{f.family}</td>
                <td>
                  <span style={{ display: "inline-block", height: 10, borderRadius: 2, verticalAlign: "middle",
                                 width: `${(f.importance_share / max) * 80}%`, background: "var(--series-1)",
                                 opacity: f.family === "observation pattern" ? 0.45 : 1 }} />
                  <span className="small"> {(f.importance_share * 100).toFixed(1)}%</span>
                </td>
                <td className="num">{f.smd_b_vs_a == null ? "—" : `${f.smd_b_vs_a > 0 ? "+" : ""}${f.smd_b_vs_a.toFixed(2)}`}{f.smd_b_vs_a != null && Math.abs(f.smd_b_vs_a) >= 0.5 ? " ⚠" : ""}</td>
                <td className="num">{(f.missing_a_train * 100).toFixed(0)}%</td>
                <td className="num">{(f.missing_b * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="small muted">
        Mean absolute contribution on {x.sample_rows.toLocaleString()} A-validation hours. Observation-pattern features (lighter bars) can
        encode local charting habits rather than physiology. ⚠ marks a large shift (|SMD| ≥ 0.5). Hospital B outcomes are not used here.
      </div>
    </div>
  );
}

function Kpi({ label, value, sub }: { label: string; value?: string; sub?: string }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="value">{value ?? "—"}</div>
      {sub && <div className="small muted">{sub}</div>}
    </div>
  );
}
