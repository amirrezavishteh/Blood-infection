import { useRef, useState } from "react";

type Props = {
  title: string;
  unit?: string;
  points: { hour: number; value: number }[];
  xMax: number;
  freshness?: { hours_since: number };
};

const W = 260, H = 96, M = { top: 8, right: 10, bottom: 18, left: 34 };

/** One measurement over the visible hours (small multiple; single series, no legend box). */
export function SmallLine({ title, unit, points, xMax, freshness }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ left: number; p: { hour: number; value: number } } | null>(null);
  const iw = W - M.left - M.right, ih = H - M.top - M.bottom;
  const vals = points.map((p) => p.value);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.1;
  lo -= pad; hi += pad;
  const dm = Math.max(xMax, 1);
  const x = (h: number) => M.left + (h / dm) * iw;
  const y = (v: number) => M.top + (1 - (v - lo) / (hi - lo)) * ih;
  const last = points[points.length - 1];

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const r = ref.current?.getBoundingClientRect();
    if (!r || !points.length) return;
    const h = ((e.clientX - r.left) / r.width * W - M.left) / iw * dm;
    const p = points.reduce((a, b) => (Math.abs(b.hour - h) < Math.abs(a.hour - h) ? b : a));
    setHover({ left: (x(p.hour) / W) * r.width, p });
  }

  return (
    <div className="card" style={{ padding: 10 }}>
      <div className="row" style={{ justifyContent: "space-between", gap: 6 }}>
        <h3>{title}{unit ? <span className="muted small"> ({unit})</span> : null}</h3>
        <span className="small" style={{ fontVariantNumeric: "tabular-nums" }}>
          {last ? last.value.toFixed(last.value < 10 ? 2 : 0) : "—"}
          {freshness ? <span className="muted"> · {freshness.hours_since} h ago</span> : null}
        </span>
      </div>
      <div className="chart-wrap">
        <svg ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label={`${title}: ${points.length} observations`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {[lo + pad, hi - pad].map((t, i) => (
            <g key={i}>
              <line className="gridline" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
              <text x={M.left - 4} y={y(t) + 4} textAnchor="end">{t.toFixed(Math.abs(t) < 10 ? 1 : 0)}</text>
            </g>
          ))}
          <text x={M.left} y={H - 4}>0</text>
          <text x={W - M.right} y={H - 4} textAnchor="end">{xMax} h</text>
          {points.length > 1 && (
            <path className="series" d={points.map((p, i) => `${i ? "L" : "M"}${x(p.hour)},${y(p.value)}`).join("")} />
          )}
          {points.length <= 24 && points.map((p) => (
            <circle key={p.hour} cx={x(p.hour)} cy={y(p.value)} r={3} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={1.5} />
          ))}
          {points.length === 0 && <text x={W / 2} y={H / 2} textAnchor="middle">not measured yet</text>}
        </svg>
        {hover && (
          <div className="tooltip" style={{ left: hover.left + 6, top: 0 }}>
            Hour {hover.p.hour}: {hover.p.value}{unit ? ` ${unit}` : ""}
          </div>
        )}
      </div>
    </div>
  );
}
