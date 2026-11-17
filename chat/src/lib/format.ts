// Number and date formatting shared by every tab. Signs are always explicit,
// so gains and losses never rely on colour alone.

const cache = new Map<string, Intl.NumberFormat>();

function nf(key: string, options: Intl.NumberFormatOptions): Intl.NumberFormat {
  let f = cache.get(key);
  if (!f) {
    f = new Intl.NumberFormat("en-US", options);
    cache.set(key, f);
  }
  return f;
}

/** $12,345 (no cents). */
export function money(x: number, currency = "USD"): string {
  return nf(`m0-${currency}`, { style: "currency", currency, maximumFractionDigits: 0 }).format(x);
}

/** $123.45 (always cents). */
export function price(x: number, currency = "USD"): string {
  return nf(`m2-${currency}`, { style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(x);
}

/** $1.2M, $45K, $900. */
export function compactMoney(x: number, currency = "USD"): string {
  return nf(`mc-${currency}`, { style: "currency", currency, notation: "compact", maximumFractionDigits: 1 }).format(x);
}

const MINUS = "−";

/** +$1,234 / −$1,234. */
export function signedMoney(x: number, currency = "USD"): string {
  return `${x >= 0 ? "+" : MINUS}${money(Math.abs(x), currency)}`;
}

/** +1.23 / −1.23 (price change, two decimals). */
export function signedNumber(x: number): string {
  return `${x >= 0 ? "+" : MINUS}${Math.abs(x).toFixed(2)}`;
}

/** 0.4123 -> "41%"; small values keep one decimal. */
export function pct(fraction: number): string {
  const v = fraction * 100;
  return `${v.toFixed(v !== 0 && Math.abs(v) < 1 ? 1 : 0)}%`;
}

/** 0.0123 -> "+1.23%". */
export function signedPct(fraction: number, digits = 2): string {
  return `${fraction >= 0 ? "+" : MINUS}${Math.abs(fraction * 100).toFixed(digits)}%`;
}

function toDate(iso: string): Date {
  return new Date(iso.length === 10 ? `${iso}T00:00:00` : iso);
}

/** "Nov 13", or "Nov 13, 2024" when the year differs from now. */
export function shortDate(iso: string): string {
  const d = toDate(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", ...(sameYear ? {} : { year: "numeric" }) });
}

/** "Nov 2024": for long chart axes. */
export function monthYear(iso: string): string {
  const d = toDate(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

/** "3:42 PM" today, otherwise a short date. */
export function timeOrDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }) : shortDate(iso);
}

/** "5 min ago", "2 h ago", or a date. */
export function relativeTime(iso: string): string {
  const d = new Date(iso);
  const s = (Date.now() - d.getTime()) / 1000;
  if (Number.isNaN(s)) return "";
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86_400) return `${Math.floor(s / 3600)} h ago`;
  return shortDate(iso);
}
