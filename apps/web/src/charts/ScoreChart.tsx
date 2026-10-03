import { useMemo, useRef, useState } from "react";

export type ScorePoint = { hour: number; score: number | null; status: string };

type Props = {
  points: ScorePoint[];
  threshold: number;
  alertHours: number[];
  xMax: number;
  labels?: number[] | null; // retrospective only
  onsetProxy?: number | null; // retrospective only
  height?: number;
};

const M = { top: 12, right: 16, bottom: 26, left: 40 };

/** Research score trajectory. Null scores are gaps (never drawn as low risk) with a shaded band. */
export function ScoreChart({ points, threshold, alertHours, xMax, labels, onsetProxy, height = 220 }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ x: number; p: ScorePoint } | null>(null);
  const width = 720;
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const domainMax = Math.max(xMax, 1);
  const x = (h: number) => M.left + (h / domainMax) * iw;
  const y = (s: number) => M.top + (1 - s) * ih;

  const segments = useMemo(() => {
    const segs: ScorePoint[][] = [];
    let cur: ScorePoint[] = [];
    for (const p of points) {
      if (p.score === null) { if (cur.length) segs.push(cur); cur = []; } else cur.push(p);
    }
    if (cur.length) segs.push(cur);
    return segs;
  }, [points]);

  const unavailable = points.filter((p) => p.score === null);
  const step = Math.max(1, Math.ceil(domainMax / 12));
  const xticks = Array.from({ length: Math.floor(domainMax / step) + 1 }, (_, i) => i * step);
  const labelStart = labels ? labels.findIndex((v) => v === 1) : -1;
  const bandW = iw / domainMax;

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const svg = ref.current;
    if (!svg || !points.length) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const h = Math.round(((px - M.left) / iw) * domainMax);
    const p = points.find((q) => q.hour === h);
    setHover(p ? { x: (x(p.hour) / width) * rect.width, p } : null);
  }

  return (
    <div className="chart-wrap">
      <svg ref={ref} className="chart" viewBox={`0 0 ${width} ${height}`} role="img"
           aria-label={`Research score by ICU hour; threshold ${threshold.toFixed(3)}`}
           onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="gridline" x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} />
            <text x={M.left - 6} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
          </g>
        ))}
        {xticks.map((t) => (
          <text key={t} x={x(t)} y={height - 8} textAnchor="middle">{t}</text>
        ))}
        {labelStart >= 0 && (
          <rect x={x(labelStart) - bandW / 2} y={M.top} width={Math.max(0, x(labels!.length - 1) - x(labelStart) + bandW)}
                height={ih} fill="var(--label-band)" />
        )}
        {unavailable.map((p) => (
          <rect key={`u${p.hour}`} x={x(p.hour) - bandW / 2} y={M.top} width={bandW} height={ih} fill="var(--unavailable-band)" />
        ))}
        <line className="threshold" x1={M.left} x2={width - M.right} y1={y(threshold)} y2={y(threshold)} />
        <text x={width - M.right} y={y(threshold) - 4} textAnchor="end">alert threshold</text>
        {onsetProxy != null && onsetProxy <= domainMax && (
          <g>
            <line x1={x(onsetProxy)} x2={x(onsetProxy)} y1={M.top} y2={M.top + ih} stroke="var(--status-critical)" strokeWidth={1} />
            <text x={x(onsetProxy) + 4} y={M.top + 10}>onset proxy</text>
          </g>
        )}
        {segments.map((seg, i) =>
          seg.length === 1 ? null : (
            <path key={i} className="series" d={seg.map((p, j) => `${j ? "L" : "M"}${x(p.hour)},${y(p.score!)}`).join("")} />
          ))}
        {segments.filter((s) => s.length === 1).map((s) => (
          <circle key={`s${s[0].hour}`} cx={x(s[0].hour)} cy={y(s[0].score!)} r={3} fill="var(--series-1)" />
        ))}
        {alertHours.map((h) => {
          const p = points.find((q) => q.hour === h);
          if (!p || p.score === null) return null;
          return (
            <g key={`a${h}`}>
              <circle cx={x(h)} cy={y(p.score)} r={6} fill="var(--status-critical)" stroke="var(--surface-1)" strokeWidth={2} />
              <text x={x(h)} y={y(p.score) - 10} textAnchor="middle" style={{ fill: "var(--text-primary)" }}>alert</text>
            </g>
          );
        })}
        {points.length > 0 && (() => {
          const last = [...points].reverse().find((p) => p.score !== null);
          return last ? <circle cx={x(last.hour)} cy={y(last.score!)} r={4} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} /> : null;
        })()}
        {hover && <line className="cursor" x1={x(hover.p.hour)} x2={x(hover.p.hour)} y1={M.top} y2={M.top + ih} />}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 8, top: 8 }}>
          <div><strong>Hour {hover.p.hour}</strong></div>
          <div>{hover.p.score === null ? "Data unavailable (not low risk)" : `Score ${hover.p.score.toFixed(3)}`}</div>
          {alertHours.includes(hover.p.hour) && <div>Alert emitted</div>}
          {labels && <div>Benchmark label {labels[hover.p.hour] ?? "–"}</div>}
        </div>
      )}
      <div className="legend" style={{ marginTop: 6 }}>
        <span className="key"><span className="line" style={{ background: "var(--series-1)" }} />research score (calibrated)</span>
        <span className="key"><span className="line" style={{ background: "var(--text-secondary)", height: 1 }} />threshold</span>
        <span className="key"><span className="dot badge-dot" style={{ width: 8, height: 8, borderRadius: 4, background: "var(--status-critical)", display: "inline-block" }} />alert</span>
        <span className="key"><span className="swatch" style={{ background: "var(--unavailable-band)" }} />data unavailable</span>
        {labels && <span className="key"><span className="swatch" style={{ background: "var(--label-band)" }} />benchmark label positive (onset − 6 h onward)</span>}
      </div>
    </div>
  );
}
