import { useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { useElementWidth } from "../hooks/useElementWidth";
import { labelIndexes, linePath, nearestIndex, niceTicks, scaleLinear } from "../lib/chart";

export interface LineSeries {
  id: string;
  label: string;
  /** CSS colour, normally a var(--series-n) token. */
  color: string;
  values: number[];
}

interface Props {
  /** One label per point: used on the x axis and as the tooltip title. */
  xLabels: string[];
  series: LineSeries[];
  /** A dashed horizontal line, e.g. a savings target. */
  reference?: { label: string; value: number };
  yTick: (v: number) => string;
  tooltipValue: (v: number) => string;
  ariaLabel: string;
  height?: number;
  /** Start the y axis at zero (amounts) instead of fitting the data (prices). */
  zeroBased?: boolean;
}

const M = { top: 14, right: 58, bottom: 28, left: 6 };
const TOOLTIP_W = 170;

/**
 * Responsive SVG line chart: 2px lines on recessive gridlines, one y axis,
 * crosshair + tooltip on hover/touch, arrow keys move the crosshair when focused.
 * Two or more series get a legend; a single series is named by the surrounding title.
 */
export default function LineChart({ xLabels, series, reference, yTick, tooltipValue, ariaLabel, height = 240, zeroBased }: Props) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const [active, setActive] = useState<number | null>(null);

  const n = xLabels.length;
  const all = series.flatMap((s) => s.values).concat(reference ? [reference.value] : []);
  if (n === 0 || all.length === 0) return null;

  const ticks = niceTicks(zeroBased ? Math.min(0, ...all) : Math.min(...all), Math.max(...all), 4);
  const innerW = Math.max(width - M.left - M.right, 10);
  const innerH = height - M.top - M.bottom;
  const x0 = M.left;
  const step = n > 1 ? innerW / (n - 1) : 0;
  const x = (i: number) => x0 + i * step;
  const y = scaleLinear(ticks[0], ticks[ticks.length - 1], M.top + innerH, M.top);
  const axisIdx = labelIndexes(n, width < 480 ? 3 : 5);

  const onPointer = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    setActive(nearestIndex(px, x0, step, n));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const move = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    if (move) {
      e.preventDefault();
      setActive((a) => Math.max(0, Math.min(n - 1, (a ?? n - 1) + (a === null ? 0 : move))));
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      setActive(e.key === "Home" ? 0 : n - 1);
    } else if (e.key === "Escape") {
      setActive(null);
    }
  };

  const tipLeft = active === null ? 0 : Math.max(0, Math.min(width - TOOLTIP_W, x(active) - TOOLTIP_W / 2));

  return (
    <div>
      {series.length > 1 && (
        <ul className="ui-legend">
          {series.map((s) => (
            <li key={s.id}>
              <span className="ui-swatch" style={{ background: s.color }} />
              {s.label}
            </li>
          ))}
          {reference && (
            <li>
              <span className="ui-swatch" style={{ background: "repeating-linear-gradient(90deg, var(--muted) 0 3px, transparent 3px 5px)" }} />
              {reference.label}
            </li>
          )}
        </ul>
      )}

      <div
        ref={wrapRef}
        className="ui-chart"
        tabIndex={0}
        role="img"
        aria-label={ariaLabel}
        onKeyDown={onKeyDown}
        onBlur={() => setActive(null)}
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          height={height}
          onPointerMove={onPointer}
          onPointerDown={onPointer}
          onPointerLeave={() => setActive(null)}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line className={t === ticks[0] ? "baseline" : "grid"} x1={M.left} x2={M.left + innerW} y1={y(t)} y2={y(t)} />
              <text className="tick" x={M.left + innerW + 8} y={y(t)} dy="0.32em">
                {yTick(t)}
              </text>
            </g>
          ))}

          {axisIdx.map((i, k) => (
            <text
              key={i}
              className="tick"
              x={x(i)}
              y={height - 8}
              textAnchor={k === 0 ? "start" : k === axisIdx.length - 1 ? "end" : "middle"}
            >
              {xLabels[i]}
            </text>
          ))}

          {reference && (
            <line
              x1={M.left}
              x2={M.left + innerW}
              y1={y(reference.value)}
              y2={y(reference.value)}
              stroke="var(--muted)"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
          )}

          {series.map((s) => (
            <path key={s.id} className="series" stroke={s.color} d={linePath(s.values.map((v, i) => [x(i), y(v)]))} />
          ))}

          {active !== null && (
            <g>
              <line className="crosshair" x1={x(active)} x2={x(active)} y1={M.top} y2={M.top + innerH} />
              {series.map((s) => (
                <circle key={s.id} className="marker" cx={x(active)} cy={y(s.values[active])} r={4.5} fill={s.color} />
              ))}
            </g>
          )}
        </svg>

        {active !== null && (
          <div className="ui-tooltip" style={{ left: tipLeft, width: TOOLTIP_W }} aria-hidden="true">
            <div className="ui-tooltip-title">{xLabels[active]}</div>
            {series.map((s) => (
              <div key={s.id} className="ui-tooltip-row">
                <span className="ui-swatch" style={{ background: s.color }} />
                {s.label}
                <strong>{tooltipValue(s.values[active])}</strong>
              </div>
            ))}
            {reference && (
              <div className="ui-tooltip-row">
                <span className="ui-swatch" style={{ background: "var(--muted)" }} />
                {reference.label}
                <strong>{tooltipValue(reference.value)}</strong>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
