import { useEffect, useState } from "react";
import { api, ApiError, type Dataset, type PatientRow, type Replay } from "../api";
import { StatusBadge } from "./DatasetManager";
import { PatientTimeline } from "./PatientTimeline";

export function ReplayWorkspace({ replayId, onSelectReplay }: { replayId: string | null; onSelectReplay: (id: string | null) => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [replays, setReplays] = useState<Replay[]>([]);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [patients, setPatients] = useState<PatientRow[]>([]);
  const [stay, setStay] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({ split: "a_test", n_stays: 12, seed: 0, enrich: true, enrich_fraction: 0.5, speed: 2 });

  useEffect(() => {
    api.datasets().then(setDatasets).catch(() => undefined);
    api.replays().then(setReplays).catch(() => undefined);
  }, [replayId]);

  async function refresh() {
    if (!replayId) return;
    try {
      const r = await api.patients(replayId);
      setReplay(r.replay);
      setReplays((list) => list.map((x) => (x.id === r.replay.id ? r.replay : x)));
      setPatients(r.patients);
      setTick((t) => t + 1);
      if (!stay || !r.patients.some((p) => p.stay_id === stay)) setStay(r.patients[0]?.stay_id ?? null);
    } catch (e: any) {
      if (e instanceof ApiError && e.status === 404) { setReplay(null); onSelectReplay(null); return; } // stale remembered id
      setError(e.message);
    }
  }
  useEffect(() => { setStay(null); refresh(); }, [replayId]);
  useEffect(() => {
    if (replay?.status !== "playing") return;
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [replay?.status, replayId, stay]);

  const ready = datasets.find((d) => d.status === "ready" && d.source === "physionet2019");

  async function create() {
    if (!ready) return;
    setBusy(true); setError(null);
    try {
      const r = await api.createReplay({
        dataset_id: ready.id, split: form.split, n_stays: form.n_stays, seed: form.seed,
        enrich_septic_fraction: form.enrich ? form.enrich_fraction : null, hours_per_second: form.speed,
      });
      setReplays((list) => [r, ...list]);
      onSelectReplay(r.id);
    } catch (e: any) { setError(e.message); } finally { setBusy(false); }
  }

  async function act(fn: () => Promise<unknown>, poll = false) {
    setBusy(true); setError(null);
    try {
      await fn();
      if (poll) await waitForClock();
      await refresh();
    } catch (e: any) { setError(e.message); } finally { setBusy(false); }
  }

  async function waitForClock() {
    if (!replay) return;
    const target = replay.clock + 1;
    for (let i = 0; i < 40; i++) {
      const r = await api.replay(replay.id);
      if (r.clock >= target) return;
      await new Promise((res) => setTimeout(res, 250));
    }
    throw new Error("The worker has not processed the step yet. Is the worker running?");
  }

  const progress = replay ? Math.max(0, (replay.clock + 1) / (replay.max_hour + 1)) : 0;
  // Stable order (replay position) so the list never jumps under the cursor while playing.
  const sorted = patients;

  return (
    <div className="stack">
      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Replay workspace</h2>
          <label className="field" style={{ minWidth: 260 }}>
            Replay
            <select value={replayId ?? ""} onChange={(e) => e.target.value && onSelectReplay(e.target.value)}>
              <option value="">— select a replay —</option>
              {replays.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.split} · {r.status} · hour {r.clock}</option>)}
            </select>
          </label>
        </div>
        <details open={!replayId}>
          <summary className="small" style={{ cursor: "pointer" }}>New replay of held-out records</summary>
          <div className="row" style={{ marginTop: 10, alignItems: "flex-end" }}>
            <label className="field">Split
              <select value={form.split} onChange={(e) => setForm({ ...form, split: e.target.value })}>
                <option value="a_test">a_test (hospital A internal test)</option>
                <option value="b_external">b_external (hospital B)</option>
              </select>
            </label>
            <label className="field">Records<input type="number" min={1} max={100} value={form.n_stays} onChange={(e) => setForm({ ...form, n_stays: +e.target.value })} style={{ width: 80 }} /></label>
            <label className="field">Seed<input type="number" value={form.seed} onChange={(e) => setForm({ ...form, seed: +e.target.value })} style={{ width: 80 }} /></label>
            <label className="field">Hours / second<input type="number" min={0.5} max={20} step={0.5} value={form.speed} onChange={(e) => setForm({ ...form, speed: +e.target.value })} style={{ width: 90 }} /></label>
            <label className="field" title="Selection only: labels choose which records to include; they stay hidden from scoring and from the active replay.">
              <span><input type="checkbox" checked={form.enrich} onChange={(e) => setForm({ ...form, enrich: e.target.checked })} /> Enrich septic records</span>
              <input type="number" min={0} max={1} step={0.1} value={form.enrich_fraction} disabled={!form.enrich}
                     onChange={(e) => setForm({ ...form, enrich_fraction: +e.target.value })} style={{ width: 90 }} />
            </label>
            <button className="btn primary" disabled={!ready || busy} onClick={create}>Create replay</button>
          </div>
          {!ready && <div className="small error" style={{ marginTop: 6 }}>Import and validate the PhysioNet dataset first (Datasets tab).</div>}
        </details>
        {replay && (
          <div className="row">
            <StatusBadge status={replay.status} />
            <span className="small" style={{ fontVariantNumeric: "tabular-nums" }}>
              Simulated ICU hour <strong>{replay.clock < 0 ? "–" : replay.clock}</strong> / {replay.max_hour}
            </span>
            <div className="progress" aria-label="replay progress"><div style={{ width: `${progress * 100}%` }} /></div>
            <button className="btn" disabled={busy || replay.status !== "paused"} onClick={() => act(() => api.step(replay.id, replay.clock + 1), true)}>Step +1 h</button>
            {replay.status === "playing"
              ? <button className="btn" disabled={busy} onClick={() => act(() => api.pause(replay.id))}>Pause</button>
              : <button className="btn" disabled={busy || replay.status === "finished"} onClick={() => act(() => api.play(replay.id))}>Play ({replay.hours_per_second} h/s)</button>}
            <button className="btn" disabled={busy} onClick={() => act(async () => onSelectReplay((await api.reset(replay.id)).id))}>Reset (new replay id)</button>
          </div>
        )}
        {replay && (
          <div className="small muted">
            {replay.split} · model <code>{replay.model_version}</code> · policy <code>{replay.policy_version}</code> (threshold {replay.policy.threshold.toFixed(3)},
            {" "}{replay.policy.consecutive} consecutive h, cooldown {replay.policy.cooldown_h} h){replay.parent_replay_id ? ` · reset of ${replay.parent_replay_id}` : ""}
          </div>
        )}
        {error && <div className="error">{error}</div>}
      </div>

      {replay && (
        <div className="grid-2">
          <div className="card stack">
            <h2>Records ({patients.length})</h2>
            <div className="patient-list" role="list">
              {sorted.map((p) => (
                <button key={p.stay_id} role="listitem" className="patient" aria-current={p.stay_id === stay} onClick={() => setStay(p.stay_id)}>
                  <span className="mono">{p.stay_id.replace("physionet2019:", "")}</span>
                  <span>
                    {p.latest_status === "data_unavailable"
                      ? <span className="badge"><span className="dot" style={{ background: "var(--status-warning)" }} />⚠ unavailable</span>
                      : p.latest_score != null ? <span className="mono">{p.latest_score.toFixed(3)}</span> : <span className="muted small">not started</span>}
                  </span>
                  <span className="meta">
                    {p.visible_hours} h visible{p.ended ? " · record ended" : ""}
                    {p.alerts > 0 && <> · <span style={{ color: "var(--text-primary)" }}>● {p.alerts} alert{p.alerts > 1 ? "s" : ""}{p.open_alerts ? ` (${p.open_alerts} open)` : ""}</span></>}
                  </span>
                </button>
              ))}
            </div>
          </div>
          {stay ? <PatientTimeline replay={replay} stayId={stay} refreshKey={tick} /> : <div className="card muted">Select a record.</div>}
        </div>
      )}
    </div>
  );
}
