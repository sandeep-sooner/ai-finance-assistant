import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, analyzePortfolio, getSamplePortfolios } from "../api/client";
import { useInsightStream } from "../hooks/useInsightStream";
import { uuid } from "../hooks/useSession";
import { money, pct, price, signedMoney, signedPct } from "../lib/format";
import type {
  AllocationSlice,
  Holding,
  HoldingResult,
  PortfolioAnalysis,
  PortfolioRequest,
  SamplePortfolio,
} from "../types/api";
import FlagList from "../ui/FlagList";
import { CloseIcon, InfoIcon, PlusIcon } from "../ui/icons";
import InsightCard from "../ui/InsightCard";
import Stat from "../ui/Stat";
import "../ui/panel.css";
import "./portfolio.css";

const STORAGE_KEY = "finance-assistant-portfolio";
const MAX_ROWS = 50;
const SYMBOL_RE = /^[A-Za-z][A-Za-z0-9.-]{0,9}$/;

interface Row {
  id: string;
  symbol: string;
  shares: string;
  avgCost: string;
}

const EXAMPLE: Omit<Row, "id">[] = [
  { symbol: "AAPL", shares: "40", avgCost: "150" },
  { symbol: "MSFT", shares: "10", avgCost: "310" },
  { symbol: "VTI", shares: "25", avgCost: "220" },
  { symbol: "BND", shares: "30", avgCost: "" },
];

const emptyRow = (): Row => ({ id: uuid(), symbol: "", shares: "", avgCost: "" });

function loadRows(): Row[] {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (Array.isArray(saved) && saved.length) {
      return saved.slice(0, MAX_ROWS).map((r) => ({
        id: uuid(),
        symbol: String(r.symbol ?? ""),
        shares: String(r.shares ?? ""),
        avgCost: String(r.avgCost ?? ""),
      }));
    }
  } catch {
    /* storage blocked or corrupt: start empty */
  }
  return [emptyRow()];
}

function saveRows(rows: Row[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rows.map(({ symbol, shares, avgCost }) => ({ symbol, shares, avgCost }))));
  } catch {
    /* storage blocked */
  }
}

const isBlank = (r: Row) => !r.symbol.trim() && !r.shares.trim() && !r.avgCost.trim();

/** Per-row problems keyed by row id. Blank rows are ignored. */
function validate(rows: Row[]): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const r of rows) {
    if (isBlank(r)) continue;
    const shares = Number(r.shares);
    const cost = Number(r.avgCost);
    if (!SYMBOL_RE.test(r.symbol.trim())) errors[r.id] = "Enter a ticker such as AAPL.";
    else if (!r.shares.trim() || !Number.isFinite(shares) || shares <= 0) errors[r.id] = "Shares must be more than 0.";
    else if (r.avgCost.trim() && (!Number.isFinite(cost) || cost < 0)) errors[r.id] = "Average cost must be 0 or more.";
  }
  return errors;
}

function toHoldings(rows: Row[]): Holding[] {
  return rows
    .filter((r) => !isBlank(r))
    .map((r) => ({
      symbol: r.symbol.trim().toUpperCase(),
      shares: Number(r.shares),
      avg_cost: r.avgCost.trim() ? Number(r.avgCost) : null,
    }));
}

/* ---------- Result pieces ---------- */

/** Single-series horizontal bars: one hue, sorted, every bar direct-labelled. */
function AllocationBars({ title, slices }: { title: string; slices: AllocationSlice[] }) {
  const max = Math.max(...slices.map((s) => s.weight), 0.0001);
  return (
    <figure className="pf-bars">
      <figcaption>{title}</figcaption>
      <ul>
        {slices.map((s) => (
          <li key={s.name} title={`${s.name}: ${money(s.value)} (${pct(s.weight)})`}>
            <span className="pf-bar-name">{s.name}</span>
            <span className="pf-bar-track" aria-hidden="true">
              <span className="pf-bar-fill" style={{ width: `${(s.weight / max) * 100}%` }} />
            </span>
            <span className="pf-bar-pct">{pct(s.weight)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

const SOURCE_BADGE: Record<string, { label: string; title: string } | undefined> = {
  cache: { label: "earlier", title: "From an earlier fetch: the data source did not respond" },
  sample: { label: "sample", title: "Illustrative sample price, not a real quote" },
  cost: { label: "at cost", title: "No market price: valued at your average cost" },
};

function PriceCell({ h }: { h: HoldingResult }) {
  if (h.price == null) return <span className="ui-muted">No price</span>;
  const badge = SOURCE_BADGE[h.price_source];
  return (
    <>
      {price(h.price)}
      {badge && (
        <span className="ui-badge" title={badge.title} data-tone={h.price_source === "cost" ? undefined : "warn"}>
          {badge.label}
        </span>
      )}
    </>
  );
}

function HoldingsTable({ holdings }: { holdings: HoldingResult[] }) {
  return (
    <div className="ui-table-wrap">
      <table className="ui-table">
        <caption className="ui-sr">Holdings, largest first</caption>
        <thead>
          <tr>
            <th scope="col">Symbol</th>
            <th scope="col">Type</th>
            <th scope="col" className="ui-right">Shares</th>
            <th scope="col" className="ui-right">Price</th>
            <th scope="col" className="ui-right">Value</th>
            <th scope="col" className="ui-right">Weight</th>
            <th scope="col" className="ui-right">Gain / loss</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h) => (
            <tr key={h.symbol}>
              <th scope="row">{h.symbol}</th>
              <td>
                {h.asset_class}
                <span className="ui-muted pf-sector">{h.sector}</span>
              </td>
              <td className="ui-right">{h.shares.toLocaleString("en-US")}</td>
              <td className="ui-right">
                <PriceCell h={h} />
              </td>
              <td className="ui-right">{money(h.value)}</td>
              <td className="ui-right">{pct(h.weight)}</td>
              <td className="ui-right">
                {h.gain != null && h.gain_pct != null ? (
                  <span className={h.gain >= 0 ? "ui-pos" : "ui-neg"}>
                    {signedMoney(h.gain)} <span className="ui-muted">({signedPct(h.gain_pct, 1)})</span>
                  </span>
                ) : (
                  <span className="ui-muted">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A 0-100 score drawn as a ring, with a word or number in the middle. Colour is never the only cue:
    the caption says the same thing in words. */
function ScoreCard({
  label,
  score,
  display,
  tone,
  caption,
}: {
  label: string;
  score: number;
  display?: string;
  tone: "good" | "mid" | "low";
  caption: string;
}) {
  const r = 26;
  const circumference = 2 * Math.PI * r;
  const filled = (Math.max(0, Math.min(score, 100)) / 100) * circumference;
  return (
    <div className="ui-stat pf-score" data-tone={tone}>
      <svg className="pf-ring" viewBox="0 0 64 64" role="img" aria-label={`${label}: ${score} out of 100`}>
        <circle className="pf-ring-track" cx="32" cy="32" r={r} />
        <circle
          className="pf-ring-value"
          cx="32"
          cy="32"
          r={r}
          strokeDasharray={`${filled} ${circumference - filled}`}
          transform="rotate(-90 32 32)"
        />
        <text x="32" y="37" textAnchor="middle" className="pf-ring-text">
          {score}
        </text>
      </svg>
      <div className="pf-score-text">
        <span className="ui-stat-label">{label}</span>
        <span className="ui-stat-value">{display ?? `${score} / 100`}</span>
        <span className="ui-stat-sub">{caption}</span>
      </div>
    </div>
  );
}

/* ---------- Main component ---------- */

export default function PortfolioTab() {
  const [rows, setRows] = useState<Row[]>(loadRows);
  const [showErrors, setShowErrors] = useState(false);
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null);
  const [analyzedKey, setAnalyzedKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const insights = useInsightStream<PortfolioRequest>("/api/portfolio/insights/stream");
  const resultsRef = useRef<HTMLDivElement>(null);
  const analyzeAbort = useRef<AbortController | null>(null);
  const [samples, setSamples] = useState<SamplePortfolio[]>([]);

  // Sample portfolios are a convenience: if the request fails the picker simply is not shown.
  useEffect(() => {
    const controller = new AbortController();
    getSamplePortfolios(controller.signal)
      .then((r) => setSamples(r.portfolios))
      .catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(() => saveRows(rows), [rows]);
  useEffect(() => () => analyzeAbort.current?.abort(), []);

  const errors = useMemo(() => validate(rows), [rows]);
  const holdings = useMemo(() => toHoldings(rows), [rows]);
  const stale = analysis !== null && JSON.stringify(holdings) !== analyzedKey;
  const errorCount = Object.keys(errors).length;

  const update = (id: string, field: keyof Omit<Row, "id">, value: string) =>
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, [field]: value } : r)));
  const remove = (id: string) => setRows((prev) => (prev.length === 1 ? [emptyRow()] : prev.filter((r) => r.id !== id)));
  const addRow = () => setRows((prev) => (prev.length >= MAX_ROWS ? prev : [...prev, emptyRow()]));
  const loadExample = () => {
    setRows(EXAMPLE.map((r) => ({ ...r, id: uuid() })));
    setShowErrors(false);
  };
  const loadSample = (id: string) => {
    const sample = samples.find((s) => s.id === id);
    if (!sample) return;
    setRows(
      sample.holdings.map((h) => ({
        id: uuid(),
        symbol: h.symbol,
        shares: String(h.shares),
        avgCost: h.avg_cost != null ? String(h.avg_cost) : "",
      }))
    );
    setShowErrors(false);
  };
  const clearAll = () => {
    setRows([emptyRow()]);
    setShowErrors(false);
    setAnalysis(null);
    setRequestError(null);
    insights.clear();
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setShowErrors(true);
    if (errorCount || !holdings.length || loading) return;

    analyzeAbort.current?.abort();
    const controller = new AbortController();
    analyzeAbort.current = controller;
    setLoading(true);
    setRequestError(null);
    try {
      const result = await analyzePortfolio({ holdings }, controller.signal);
      setAnalysis(result);
      setAnalyzedKey(JSON.stringify(holdings));
      void insights.start({ holdings });
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setRequestError(err instanceof ApiError ? err.message : "Something went wrong.");
      }
    } finally {
      if (analyzeAbort.current === controller) {
        analyzeAbort.current = null;
        setLoading(false);
      }
    }
  };

  return (
    <section className="panel" aria-label="Portfolio">
      <div className="panel-inner">
        <header className="panel-intro">
          <h2>Portfolio check-up</h2>
          <p>Enter what you hold to see how it is spread out, and get a plain-language explanation.</p>
        </header>

        <form className="ui-card" onSubmit={onSubmit} noValidate>
          <div className="ui-card-head">
            <h3>Your holdings</h3>
            <div className="pf-tools">
              {samples.length > 0 && (
                <select
                  className="ui-btn pf-sample"
                  aria-label="Load a sample portfolio"
                  value=""
                  onChange={(e) => loadSample(e.target.value)}
                >
                  <option value="">Sample portfolios…</option>
                  {samples.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              )}
              <button type="button" className="ui-btn" onClick={loadExample}>
                Load example
              </button>
              <button type="button" className="ui-btn" onClick={clearAll}>
                Clear
              </button>
            </div>
          </div>

          <div className="pf-grid" role="group" aria-label="Holdings">
            <div className="pf-grid-head" aria-hidden="true">
              <span>Ticker</span>
              <span>Shares</span>
              <span>Avg cost (optional)</span>
              <span />
            </div>
            {rows.map((r, i) => {
              const err = showErrors ? errors[r.id] : undefined;
              const errId = `pf-err-${r.id}`;
              return (
                <div key={r.id} className="pf-row">
                  <input
                    className="ui-input pf-symbol"
                    value={r.symbol}
                    onChange={(e) => update(r.id, "symbol", e.target.value.toUpperCase())}
                    placeholder="AAPL"
                    aria-label={`Ticker, row ${i + 1}`}
                    aria-invalid={!!err}
                    aria-describedby={err ? errId : undefined}
                    maxLength={10}
                    autoCapitalize="characters"
                    spellCheck={false}
                  />
                  <input
                    className="ui-input"
                    value={r.shares}
                    onChange={(e) => update(r.id, "shares", e.target.value)}
                    placeholder="10"
                    inputMode="decimal"
                    aria-label={`Shares, row ${i + 1}`}
                    aria-invalid={!!err}
                  />
                  <div className="ui-affix pf-cost" data-prefix>
                    <span aria-hidden="true">$</span>
                    <input
                      className="ui-input"
                      value={r.avgCost}
                      onChange={(e) => update(r.id, "avgCost", e.target.value)}
                      placeholder="150.00"
                      inputMode="decimal"
                      aria-label={`Average cost per share, row ${i + 1}, optional`}
                      aria-invalid={!!err}
                    />
                  </div>
                  <button type="button" className="ui-icon-btn" onClick={() => remove(r.id)} aria-label={`Remove row ${i + 1}`}>
                    <CloseIcon />
                  </button>
                  {err && (
                    <p id={errId} className="pf-row-error">
                      {err}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          <div className="pf-foot">
            <button type="button" className="ui-btn" onClick={addRow} disabled={rows.length >= MAX_ROWS}>
              <PlusIcon />
              Add holding
            </button>
            <button type="submit" className="ui-btn-primary" disabled={loading}>
              {loading ? "Analyzing…" : analysis && !stale ? "Analyze again" : "Analyze portfolio"}
            </button>
          </div>

          {showErrors && !holdings.length && <p className="ui-error">Add at least one holding.</p>}
          {showErrors && errorCount > 0 && (
            <p className="ui-error" role="alert">
              Fix {errorCount === 1 ? "the highlighted row" : `the ${errorCount} highlighted rows`} first.
            </p>
          )}
          {requestError && (
            <p className="ui-error" role="alert">
              {requestError}
            </p>
          )}
        </form>

        {analysis && (
          <div ref={resultsRef} className="pf-results">
            {stale && (
              <p className="ui-banner">
                <InfoIcon />
                Your holdings changed. Press Analyze to update these results.
              </p>
            )}

            <div className="ui-stats">
              <Stat label="Total value" value={money(analysis.total_value)} sub={`${analysis.holdings.length} holdings`} />
              <Stat
                label="Gain / loss"
                value={
                  analysis.total_gain != null ? (
                    <span className={analysis.total_gain >= 0 ? "ui-pos" : "ui-neg"}>{signedMoney(analysis.total_gain)}</span>
                  ) : (
                    "—"
                  )
                }
                sub={
                  analysis.total_gain_pct != null
                    ? `${signedPct(analysis.total_gain_pct, 1)} on holdings with a cost`
                    : "Add average costs to see this"
                }
              />
              <Stat
                label="Largest position"
                value={analysis.largest_symbol ?? "—"}
                sub={analysis.largest_symbol ? `${pct(analysis.largest_weight)} of the portfolio` : undefined}
              />
              <Stat
                label="Effective holdings"
                value={analysis.effective_holdings.toFixed(1)}
                sub={`behaves like this many equal positions (of ${analysis.holdings.length})`}
              />
            </div>

            <div className="pf-scores">
              <ScoreCard
                label="Diversification"
                score={analysis.diversification_score}
                tone={analysis.diversification_score >= 70 ? "good" : analysis.diversification_score >= 40 ? "mid" : "low"}
                caption={
                  analysis.diversification_score >= 70
                    ? "Spread across many holdings"
                    : analysis.diversification_score >= 40
                      ? "Some concentration"
                      : "Concentrated in a few positions"
                }
              />
              <ScoreCard
                label="Risk level"
                score={analysis.risk_score}
                display={analysis.risk_level}
                tone={analysis.risk_score < 35 ? "good" : analysis.risk_score < 60 ? "mid" : "low"}
                caption={`${pct(analysis.equity_weight)} in stocks and stock funds`}
              />
              <div className="ui-stat pf-fees">
                <span className="ui-stat-label">Fund costs</span>
                <span className="ui-stat-value ui-num">
                  {analysis.expense_ratio != null ? `${(analysis.expense_ratio * 100).toFixed(2)}%` : "—"}
                </span>
                <span className="ui-stat-sub">
                  {analysis.annual_fund_cost != null
                    ? `about ${money(analysis.annual_fund_cost)} a year in fund expenses`
                    : "No funds with a known expense ratio"}
                </span>
              </div>
            </div>

            <FlagList flags={analysis.flags} />

            <div className="ui-card pf-alloc">
              <AllocationBars title="By asset type" slices={analysis.by_asset_class} />
              <AllocationBars title="By sector" slices={analysis.by_sector} />
            </div>

            <div className="ui-card">
              <h3>Holdings</h3>
              <HoldingsTable holdings={analysis.holdings} />
              <p className="ui-note">{analysis.price_note}</p>
            </div>

            <InsightCard
              title="What this means"
              agentLabel="Portfolio agent"
              text={insights.text}
              status={insights.status}
              error={insights.error}
              onStop={insights.stop}
              onRegenerate={insights.regenerate}
            />
          </div>
        )}
      </div>
    </section>
  );
}
