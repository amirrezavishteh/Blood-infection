import { useEffect, useState } from "react";
import { api, type Dataset } from "../api";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function DatasetManager({ onChange }: { onChange?: () => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [quality, setQuality] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => api.datasets().then(setDatasets).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (!datasets.some((d) => d.status === "pending" || d.status === "validating")) return;
    const t = setInterval(() => { load(); onChange?.(); }, 1500);
    return () => clearInterval(t);
  }, [datasets]);
  useEffect(() => {
    if (selected == null) return;
    api.quality(selected).then((d) => setQuality(d)).catch((e) => setError(e.message));
  }, [selected, datasets]);

  async function importPhysionet() {
    setError(null);
    try {
      const r = await api.importDataset("physionet2019");
      setSelected(r.dataset.id);
      await load();
    } catch (e: any) { setError(e.message); }
  }

  const q = quality?.quality;
  return (
    <div className="stack">
      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Datasets</h2>
          <button className="btn primary" onClick={importPhysionet}>Import PhysioNet 2019 (registered source)</button>
        </div>
        <p className="muted small" style={{ margin: 0 }}>
          Only registered server-side sources can be imported; validation verifies output checksums, record counts and the
          frozen split file. Source: PhysioNet/CinC Challenge 2019 v1.0.0, CC BY 4.0.
        </p>
        {error && <div className="error">{error}</div>}
        <div className="table-wrap">
          <table>
            <thead><tr><th>ID</th><th>Name</th><th>Source</th><th>Version</th><th>Status</th><th>Imported</th></tr></thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id} onClick={() => setSelected(d.id)} style={{ cursor: "pointer", background: selected === d.id ? "var(--series-1-wash)" : undefined }}>
                  <td>{d.id}</td><td>{d.name}</td><td>{d.source}</td><td>{d.dataset_version ?? "—"}</td>
                  <td><StatusBadge status={d.status} />{d.error && <div className="error small">{d.error}</div>}</td>
                  <td className="small muted">{new Date(d.created_at).toLocaleString()}</td>
                </tr>
              ))}
              {!datasets.length && <tr><td colSpan={6} className="muted">No datasets imported yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
      {q?.hospitals && (
        <div className="card stack">
          <h2>Quality and provenance — dataset {quality.id}</h2>
          {q.synthetic && (
            <div className="banner" role="alert">
              ⚠ Synthetic fixture: simulated timelines in PhysioNet format, not patient data. Metrics computed on it say nothing about real performance.
            </div>
          )}
          <div className="table-wrap">
            <table>
              <thead><tr><th>Hospital</th><th className="num">Records</th><th className="num">Hourly rows</th><th className="num">Patient-days</th>
                <th className="num">Ever-positive records</th><th className="num">Positive from first row</th><th className="num">Positive-hour prevalence</th><th className="num">Median hours</th></tr></thead>
              <tbody>
                {Object.entries<any>(q.hospitals).map(([h, v]) => (
                  <tr key={h}><td>{h}</td><td className="num">{v.stays.toLocaleString()}</td><td className="num">{v.rows.toLocaleString()}</td>
                    <td className="num">{Math.round(v.patient_days).toLocaleString()}</td><td className="num">{v.ever_positive_stays.toLocaleString()}</td>
                    <td className="num">{v.left_censored_positive_stays}</td><td className="num">{pct(v.positive_hour_prevalence)}</td><td className="num">{v.median_hours}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small">
            Splits ({q.split_hash}): {Object.entries<number>(q.splits).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(" · ")}
          </div>
          <div className="small muted">Invalid files: {q.invalid_files.length} · raw manifest sha256 <code>{String(q.raw_manifest_sha256 ?? "n/a").slice(0, 16)}</code> · {q.license}</div>
          <h3>Missingness (fraction of hourly rows without a value)</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Variable</th>{Object.keys(q.hospitals).map((h) => <th key={h} className="num">Hospital {h}</th>)}</tr></thead>
              <tbody>
                {Object.keys(q.hospitals[Object.keys(q.hospitals)[0]].missing_fraction).map((v) => (
                  <tr key={v}><td>{v}</td>{Object.keys(q.hospitals).map((h) => <td key={h} className="num">{pct(q.hospitals[h].missing_fraction[v])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, [string, string]> = {
    ready: ["var(--status-good)", "✓"], failed: ["var(--status-critical)", "✕"],
    pending: ["var(--status-warning)", "…"], validating: ["var(--status-warning)", "…"],
    playing: ["var(--status-good)", "▶"], paused: ["var(--text-muted)", "Ⅱ"], finished: ["var(--text-muted)", "■"],
  };
  const [color, icon] = map[status] ?? ["var(--text-muted)", "•"];
  return <span className="badge"><span className="dot" style={{ background: color }} aria-hidden />{icon} {status}</span>;
}
