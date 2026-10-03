import { useState } from "react";
import type { Reliability } from "../api";

type Series = { name: string; color: string; bins: Reliability[] };
const W = 360, H = 300, M = { top: 12, right: 12, bottom: 36, left: 44 };

/** Observed rate vs mean predicted score per bin, against the identity line. */
export function ReliabilityChart({ series }: { series: Series[] }) {
  const [hover, setHover] = useState<{ s: string; b: Reliability; cx: number; cy: number } | null>(null);
  const iw = W - M.left - M.right, ih = H - M.top - M.bottom;
  const all = series.flatMap((s) => s.bins.flatMap((b) => [b.mean_predicted, b.observed_rate]));
  const max = Math.min(1, Math.max(0.1, ...all) * 1.1);
  const x = (v: number) => M.left + (v / max) * iw;
  const y = (v: number) => M.top + (1 - v / max) * ih;
  const ticks = [0, max / 4, max / 2, (3 * max) / 4, max];
  return (
    <div className="chart-wrap" style={{ maxWidth: 420 }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reliability diagram">
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text x={M.left - 6} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
            <text x={x(t)} y={H - 20} textAnchor="middle">{t.toFixed(2)}</text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} stroke="var(--text-muted)" strokeWidth={1} />
        <text x={W / 2} y={H - 4} textAnchor="middle">mean predicted score</text>
        <text transform={`translate(12 ${H / 2}) rotate(-90)`} textAnchor="middle">observed positive rate</text>
        {series.map((s) => (
          <g key={s.name}>
            <path d={s.bins.map((b, i) => `${i ? "L" : "M"}${x(b.mean_predicted)},${y(b.observed_rate)}`).join("")}
                  fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
            {s.bins.map((b) => (
              <circle key={b.bin_lower} cx={x(b.mean_predicted)} cy={y(b.observed_rate)} r={4.5} fill={s.color}
                      stroke="var(--surface-1)" strokeWidth={2}
                      onMouseEnter={() => setHover({ s: s.name, b, cx: x(b.mean_predicted), cy: y(b.observed_rate) })}
                      onMouseLeave={() => setHover(null)} style={{ cursor: "default" }} />
            ))}
          </g>
        ))}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: `${(hover.cx / W) * 100}%`, top: `${(hover.cy / H) * 100}%` }}>
          <div><strong>{hover.s}</strong> · bin {hover.b.bin_lower.toFixed(1)}–{hover.b.bin_upper.toFixed(1)}</div>
          <div>predicted {hover.b.mean_predicted.toFixed(3)} · observed {hover.b.observed_rate.toFixed(3)}</div>
          <div>{hover.b.n.toLocaleString()} hours</div>
        </div>
      )}
      <div className="legend">
        {series.map((s) => (
          <span key={s.name} className="key"><span className="line" style={{ background: s.color }} />{s.name}</span>
        ))}
        <span className="key"><span className="line" style={{ background: "var(--text-muted)", height: 1 }} />perfect calibration</span>
      </div>
    </div>
  );
}
