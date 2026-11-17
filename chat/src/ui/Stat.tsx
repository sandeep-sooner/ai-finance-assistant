import type { ReactNode } from "react";

/** One headline number: label, value, optional context line. */
export default function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="ui-stat">
      <span className="ui-stat-label">{label}</span>
      <span className="ui-stat-value">{value}</span>
      {sub && <span className="ui-stat-sub">{sub}</span>}
    </div>
  );
}
