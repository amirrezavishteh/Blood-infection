// Typed client for the FastAPI backend (same origin, or proxied by Vite in dev).

export type Dataset = {
  id: number; name: string; source: string; dataset_version: string | null;
  status: string; error: string | null; created_at: string;
};

export type Policy = { threshold: number; consecutive: number; cooldown_h: number; policy_version: string };

export type Replay = {
  id: string; dataset_id: number; split: string; target_id: string; model_version: string;
  policy_version: string; policy: Policy; feature_schema_version: string; clock: number;
  max_hour: number; status: "paused" | "playing" | "finished" | "failed"; hours_per_second: number;
  parent_replay_id: string | null; created_by: string; created_at: string;
};

export type PatientRow = {
  stay_id: string; visible_hours: number; active: boolean; ended: boolean;
  latest_hour: number | null; latest_score: number | null; latest_status: string | null;
  policy_state: string | null; alerts: number; open_alerts: number;
};

export type AlertRow = {
  id: number; replay_id: string; stay_id: string; trigger_hour: number; score: number;
  state: "open" | "acknowledged"; reviewed_by: string | null; reviewed_at: string | null;
  annotation: string | null; model_version: string; policy_version: string;
};

export type Contribution = { feature: string; contribution: number; value: number | null; missing: boolean | null };

export type Timeline = {
  replay_id: string; stay_id: string; clock: number; through_hour: number; record_ended: boolean; target_id: string;
  model_version: string; policy_version: string; feature_schema_version: string;
  demographics: { Age: number | null; Gender_source_code: number | null };
  units: Record<string, string>;
  observations: Record<string, { hour: number; value: number }[]>;
  freshness: Record<string, { last_hour: number; hours_since: number }>;
  scores: { hour: number; score: number | null; status: string; policy_state: string }[];
  contributions: Contribution[] | null; contributions_note: string;
  alerts: AlertRow[]; threshold: number; disclaimer: string;
};

export type Retrospective = {
  stay_id: string; labels: number[]; label_start_hour: number | null;
  onset_proxy_hour: number | null; left_censored: boolean; note: string;
};

export type EvalSummary = {
  auroc: number | null; auprc: number | null; utility: number | null;
  early_sensitivity: number | null; alerts_per_100_patient_days: number | null; records: number;
};

export type RunSummary = {
  run_id: string; model_kind: string; target_id: string; feature_set: string; calibrated: boolean;
  validation_auprc: number | null; alert_policy: (Policy & { budget_alerts_per_100_patient_days?: number }) | null;
  benchmark_threshold: number | null; evaluations: Record<string, EvalSummary>;
  provenance: Record<string, unknown>;
};

export type Reliability = { bin_lower: number; bin_upper: number; n: number; mean_predicted: number; observed_rate: number };

export type EvalFull = {
  split: string; records: number;
  hourly: { auroc: number; auprc: number; brier: number; positive_hour_prevalence: number;
            auroc_ci?: number[]; auprc_ci?: number[]; reliability: Reliability[]; rows_scored: number };
  benchmark: { utility: number; utility_ci?: number[]; threshold: number; fallback_rows: number };
  official?: Record<string, number | string>;
  alerts: Record<string, number | number[] | null>;
  coverage: { coverage: number; scored_hours: number; scheduled_hours: number };
  subgroups: { subgroup: string; stays: number; septic_stays: number; auroc: number | null;
               auroc_ci?: number[]; unstable: boolean }[];
  baselines: Record<string, { auroc: number; auprc: number }>;
};

export type RunDetail = Omit<RunSummary, "evaluations"> & {
  evaluations: Record<string, EvalFull>;
  policy_search?: { primary_budget: number; by_budget: Record<string, Record<string, number | boolean | string>> };
  selected_params?: Record<string, unknown>;
  calibration?: Record<string, unknown>;
};

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

let currentUser = "local-user";
export function setUser(u: string) { currentUser = u; }

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = typeof localStorage !== "undefined" ? safeGet("sepsis.token") : null;
  const headers: Record<string, string> = { "Content-Type": "application/json", "X-User": currentUser };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { ...init, headers: { ...headers, ...(init.headers as Record<string, string>) } });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON error */ }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

function safeGet(k: string): string | null {
  try { return localStorage.getItem(k); } catch { return null; }
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  meta: () => request<{ disclaimer: string; default_model_version: string | null; held_out_splits: string[] }>("/v1/meta"),
  ready: () => fetch("/health/ready").then((r) => r.json() as Promise<{ ready: boolean; checks: Record<string, unknown> }>),
  datasets: () => request<Dataset[]>("/v1/datasets"),
  importDataset: (source: string) => post<{ dataset: Dataset; job_id: number }>("/v1/datasets/import", { source }),
  quality: (id: number) => request<Dataset & { quality: any }>(`/v1/datasets/${id}/quality`),
  replays: () => request<Replay[]>("/v1/replays"),
  createReplay: (body: Record<string, unknown>) => post<Replay>("/v1/replays", body),
  replay: (id: string) => request<Replay>(`/v1/replays/${id}`),
  step: (id: string, targetHour: number) => post<{ job_id: number; target_hour: number }>(`/v1/replays/${id}/step`, { target_hour: targetHour }),
  play: (id: string) => post<Replay>(`/v1/replays/${id}/play`),
  pause: (id: string) => post<Replay>(`/v1/replays/${id}/pause`),
  reset: (id: string) => post<Replay>(`/v1/replays/${id}/reset`),
  patients: (id: string) => request<{ replay: Replay; patients: PatientRow[] }>(`/v1/replays/${id}/patients`),
  // The client never asks beyond the replay clock; the server enforces it too (403).
  timeline: (id: string, stay: string) => request<Timeline>(`/v1/replays/${id}/patients/${encodeURIComponent(stay)}/timeline`),
  retrospective: (id: string, stay: string) => request<Retrospective>(`/v1/replays/${id}/patients/${encodeURIComponent(stay)}/retrospective`),
  alerts: (id: string) => request<AlertRow[]>(`/v1/replays/${id}/alerts`),
  acknowledge: (alertId: number, annotation: string) => post<AlertRow>(`/v1/alerts/${alertId}/acknowledge`, { annotation }),
  experiments: () => request<{ runs: RunSummary[]; default_model_version: string | null }>("/v1/experiments"),
  experiment: (id: string) => request<RunDetail>(`/v1/experiments/${encodeURIComponent(id)}`),
};
