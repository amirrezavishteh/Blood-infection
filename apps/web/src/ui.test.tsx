import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ScoreChart } from "./charts/ScoreChart";
import { PatientTimeline } from "./components/PatientTimeline";
import type { Replay, Timeline } from "./api";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const replay: Replay = {
  id: "rp-1", dataset_id: 1, split: "a_test", target_id: "challenge2019_state", model_version: "m1",
  policy_version: "policy-x", policy: { threshold: 0.5, consecutive: 1, cooldown_h: 0, policy_version: "policy-x" },
  feature_schema_version: "causal-v1-abc", clock: 3, max_hour: 10, status: "paused", hours_per_second: 2,
  parent_replay_id: null, created_by: "u", created_at: new Date().toISOString(),
};

const timeline: Timeline = {
  replay_id: "rp-1", stay_id: "physionet2019:A:p000001", clock: 3, through_hour: 3, record_ended: false, target_id: "challenge2019_state",
  model_version: "m1", policy_version: "policy-x", feature_schema_version: "causal-v1-abc",
  demographics: { Age: 70, Gender_source_code: 1 }, units: { HR: "beats/min" },
  observations: { HR: [{ hour: 1, value: 90 }, { hour: 2, value: 95 }, { hour: 3, value: 101 }] },
  freshness: { HR: { last_hour: 3, hours_since: 0 } },
  scores: [{ hour: 0, score: null, status: "data_unavailable", policy_state: "data_unavailable" },
           { hour: 1, score: 0.1, status: "scored", policy_state: "monitoring" },
           { hour: 2, score: 0.4, status: "scored", policy_state: "monitoring" },
           { hour: 3, score: 0.6, status: "scored", policy_state: "alerted" }],
  contributions: [{ feature: "HR__last", contribution: 0.4, value: 101, missing: false },
                  { feature: "Lactate__last", contribution: -0.1, value: null, missing: true }],
  contributions_note: "associations only", alerts: [{ id: 1, replay_id: "rp-1", stay_id: "physionet2019:A:p000001",
    trigger_hour: 3, score: 0.6, state: "open", reviewed_by: null, reviewed_at: null, annotation: null, model_version: "m1", policy_version: "policy-x" }],
  threshold: 0.5, disclaimer: "research only",
};

describe("ScoreChart", () => {
  it("renders unavailable hours as bands, not as low scores", () => {
    const { container } = render(<ScoreChart points={timeline.scores} threshold={0.5} alertHours={[3]} xMax={3} />);
    const path = container.querySelector("path.series")!;
    expect(path.getAttribute("d")!.split("L").length).toBe(3); // hours 1-3 only; hour 0 is a gap
    expect(screen.getByText("data unavailable")).toBeTruthy();
    expect(screen.getAllByText("alert").length).toBe(2); // marker label + legend
  });
});

describe("PatientTimeline", () => {
  it("never requests retrospective labels while the replay is active", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      new Response(JSON.stringify(timeline), { status: 200, headers: { "Content-Type": "application/json" } }) as any);
    render(<PatientTimeline replay={replay} stayId={timeline.stay_id} refreshKey={0} />);
    await waitFor(() => expect(screen.getByText("physionet2019:A:p000001")).toBeTruthy());
    const button = screen.getByText(/retrospective labels/) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls.every((u) => !u.includes("retrospective") && !u.includes("until_hour"))).toBe(true);
    expect(screen.getByText(/never as low risk/)).toBeTruthy();
    expect(screen.getByText(/HR__last/)).toBeTruthy();
  });
});
