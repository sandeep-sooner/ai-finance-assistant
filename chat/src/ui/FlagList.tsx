import type { PortfolioFlag } from "../types/api";
import { InfoIcon, WarnIcon } from "./icons";

/** Warnings and notes. Each carries an icon and a text label, never colour alone. */
export default function FlagList({ flags, hide = [] }: { flags: PortfolioFlag[]; hide?: string[] }) {
  const shown = flags.filter((f) => !hide.includes(f.code));
  if (!shown.length) return null;
  return (
    <ul className="ui-flags">
      {shown.map((f) => (
        <li key={f.code + f.message} className="ui-flag" data-level={f.level}>
          {f.level === "warning" ? <WarnIcon /> : <InfoIcon />}
          <span>
            <strong>{f.level === "warning" ? "Worth a look: " : "Note: "}</strong>
            {f.message}
          </span>
        </li>
      ))}
    </ul>
  );
}
