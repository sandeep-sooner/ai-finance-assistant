import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { getMarketOverview, getNews, getPriceHistory, getQuotes, getSymbolDetail } from "../api/client";
import { useApi } from "../hooks/useApi";
import { DISCLAIMER } from "../lib/constants";
import { compactMoney, monthYear, price, relativeTime, shortDate, signedNumber, signedPct, timeOrDate } from "../lib/format";
import type { HistoryPeriod, NewsItem, PriceHistory, Quote, SymbolDetail } from "../types/api";
import { CloseIcon, ExternalIcon, RefreshIcon, SearchIcon, StarIcon, WarnIcon } from "../ui/icons";
import LineChart from "../ui/LineChart";
import Segmented from "../ui/Segmented";
import "../ui/panel.css";
import "./market.css";

const STORAGE_KEY = "finance-assistant-market";
const REFRESH_MS = 60_000;
const MAX_WATCH = 20;
const SYMBOL_RE = /^[A-Za-z^][A-Za-z0-9.\-=^]{0,11}$/;
const PERIODS = [
  { value: "1mo", label: "1M" },
  { value: "6mo", label: "6M" },
  { value: "1y", label: "1Y" },
  { value: "5y", label: "5Y" },
] as const;

interface Saved {
  symbol: string;
  period: HistoryPeriod;
  watchlist: string[];
}

const DEFAULTS: Saved = { symbol: "SPY", period: "6mo", watchlist: ["AAPL", "MSFT", "NVDA"] };

function load(): Saved {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (s && typeof s === "object") {
      return {
        symbol: typeof s.symbol === "string" && SYMBOL_RE.test(s.symbol) ? s.symbol : DEFAULTS.symbol,
        period: PERIODS.some((p) => p.value === s.period) ? s.period : DEFAULTS.period,
        watchlist: Array.isArray(s.watchlist) ? s.watchlist.filter((x: unknown) => typeof x === "string").slice(0, MAX_WATCH) : DEFAULTS.watchlist,
      };
    }
  } catch {
    /* storage blocked or corrupt */
  }
  return DEFAULTS;
}

/** Text + glyph + colour, so direction never depends on colour alone. */
function Change({ q, showAbs = true }: { q: Quote; showAbs?: boolean }) {
  const up = q.change >= 0;
  return (
    <span className={`mk-change ${up ? "ui-pos" : "ui-neg"}`}>
      <span aria-hidden="true">{up ? "▲" : "▼"}</span>
      {showAbs && <span>{signedNumber(q.change)}</span>}
      <span>{showAbs ? `(${signedPct(q.change_pct)})` : signedPct(q.change_pct)}</span>
      <span className="ui-sr">{up ? "up" : "down"} today</span>
    </span>
  );
}

function DataBadge({ q }: { q: Quote }) {
  if (q.is_sample)
    return (
      <span className="ui-badge" data-tone="warn" title="Illustrative price, not market data">
        sample
      </span>
    );
  if (q.is_stale)
    return (
      <span className="ui-badge" data-tone="warn" title="The data source did not respond; showing an earlier price">
        earlier
      </span>
    );
  return null;
}

function QuoteTile({ q, selected, onSelect }: { q: Quote; selected: boolean; onSelect: (s: string) => void }) {
  return (
    <button type="button" className="mk-tile" aria-pressed={selected} onClick={() => onSelect(q.symbol)}>
      <span className="mk-tile-head">
        <strong>{q.symbol}</strong>
        <DataBadge q={q} />
      </span>
      <span className="mk-tile-name">{q.name ?? " "}</span>
      <span className="mk-tile-price">{price(q.price, q.currency)}</span>
      <Change q={q} showAbs={false} />
    </button>
  );
}

function TileSkeletons({ count }: { count: number }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="mk-tile ui-skeleton" aria-hidden="true" />
      ))}
    </>
  );
}

/* ---------- Detail ---------- */

function periodChange(h: PriceHistory | undefined): number | null {
  if (!h || h.points.length < 2) return null;
  return h.points[h.points.length - 1].close / h.points[0].close - 1;
}

function Detail({
  detail,
  history,
  historyLoading,
  historyError,
  period,
  onPeriod,
  watched,
  onToggleWatch,
}: {
  detail: SymbolDetail;
  history: PriceHistory | undefined;
  historyLoading: boolean;
  historyError: string | null;
  period: HistoryPeriod;
  onPeriod: (p: HistoryPeriod) => void;
  watched: boolean;
  onToggleWatch: () => void;
}) {
  const { quote: q, profile } = detail;
  const change = periodChange(history);
  const periodLabel = PERIODS.find((p) => p.value === period)?.label ?? period;
  const labelFor = period === "5y" || period === "1y" ? monthYear : shortDate;
  const points = history?.points ?? [];

  return (
    <section className="ui-card mk-detail" aria-label={`${q.symbol} details`}>
      <div className="mk-detail-head">
        <div className="mk-detail-title">
          <h3>
            {q.symbol}
            <span className="ui-muted mk-detail-name">{q.name ?? profile.name ?? ""}</span>
          </h3>
          <span className="ui-muted mk-detail-class">
            {profile.asset_class}
            {profile.sector !== "Unknown" && profile.sector !== profile.asset_class ? ` · ${profile.sector}` : ""}
          </span>
        </div>
        <button
          type="button"
          className="ui-icon-btn"
          aria-pressed={watched}
          aria-label={watched ? `Remove ${q.symbol} from watchlist` : `Add ${q.symbol} to watchlist`}
          title={watched ? "In your watchlist" : "Add to watchlist"}
          onClick={onToggleWatch}
        >
          <StarIcon filled={watched} />
        </button>
      </div>

      <div className="mk-price-row">
        <span className="mk-price">{price(q.price, q.currency)}</span>
        <Change q={q} />
        <DataBadge q={q} />
      </div>
      <p className="mk-asof ui-muted">
        {q.is_sample
          ? "Sample price for offline use, not market data."
          : `As of ${timeOrDate(q.fetched_at)} · Yahoo Finance · may be delayed about 15 minutes`}
      </p>

      <div className="mk-chart-head">
        <h4>
          Price, {periodLabel}
          {change !== null && (
            <span className={change >= 0 ? "ui-pos" : "ui-neg"}>
              {" "}
              {signedPct(change, 1)}
            </span>
          )}
        </h4>
        <Segmented label="Chart period" options={PERIODS} value={period} onChange={onPeriod} />
      </div>

      <div className="mk-chart" aria-busy={historyLoading}>
        {points.length > 1 ? (
          <LineChart
            xLabels={points.map((p) => labelFor(p.date))}
            series={[{ id: "close", label: "Close", color: "var(--series-1)", values: points.map((p) => p.close) }]}
            yTick={(v) => (v >= 1000 ? compactMoney(v, q.currency) : price(v, q.currency).replace(/\.00$/, ""))}
            tooltipValue={(v) => price(v, q.currency)}
            ariaLabel={`${q.symbol} closing price over ${periodLabel}: from ${price(points[0].close)} on ${shortDate(points[0].date)} to ${price(points[points.length - 1].close)} on ${shortDate(points[points.length - 1].date)}${change !== null ? `, ${signedPct(change, 1)}` : ""}.`}
          />
        ) : historyError ? (
          <p className="ui-error">{historyError}</p>
        ) : (
          <div className="ui-skeleton mk-chart-skeleton" aria-hidden="true" />
        )}
      </div>
      {history?.is_stale && <p className="ui-note">Chart is from an earlier fetch; the data source did not respond.</p>}

      <dl className="mk-facts">
        <div>
          <dt>Previous close</dt>
          <dd>{q.previous_close != null ? price(q.previous_close, q.currency) : "—"}</dd>
        </div>
        <div>
          <dt>Day change</dt>
          <dd>
            <Change q={q} />
          </dd>
        </div>
        <div>
          <dt>52-week range</dt>
          <dd>
            {q.year_low != null && q.year_high != null ? `${price(q.year_low, q.currency)} – ${price(q.year_high, q.currency)}` : "—"}
          </dd>
        </div>
        <div>
          <dt>Fund or company</dt>
          <dd>{profile.diversified ? "Diversified fund" : profile.asset_class}</dd>
        </div>
      </dl>
    </section>
  );
}

function NewsList({ items }: { items: NewsItem[] }) {
  if (!items.length) return <p className="ui-muted">No recent headlines.</p>;
  return (
    <ul className="mk-news">
      {items.map((n, i) => (
        <li key={`${n.title}-${i}`}>
          {n.url ? (
            <a href={n.url} target="_blank" rel="noreferrer">
              {n.title}
              <ExternalIcon />
            </a>
          ) : (
            <span>{n.title}</span>
          )}
          <span className="ui-muted mk-news-meta">
            {[n.publisher, n.published_at ? relativeTime(n.published_at) : null].filter(Boolean).join(" · ")}
          </span>
        </li>
      ))}
    </ul>
  );
}

/* ---------- Main component ---------- */

export default function MarketTab({ active }: { active: boolean }) {
  const [saved] = useState(load);
  const [symbol, setSymbol] = useState(saved.symbol);
  const [period, setPeriod] = useState<HistoryPeriod>(saved.period);
  const [watchlist, setWatchlist] = useState<string[]>(saved.watchlist);
  const [query, setQuery] = useState("");
  const [queryError, setQueryError] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ symbol, period, watchlist }));
    } catch {
      /* storage blocked */
    }
  }, [symbol, period, watchlist]);

  const poll = { refreshMs: REFRESH_MS, refreshEnabled: active };
  const overview = useApi("overview", getMarketOverview, poll);
  const watch = useApi(watchlist.length ? `watch:${watchlist.join(",")}` : null, (s) => getQuotes(watchlist, s), poll);
  const detail = useApi(`detail:${symbol}`, (s) => getSymbolDetail(symbol, s), poll);
  const history = useApi(`history:${symbol}:${period}`, (s) => getPriceHistory(symbol, period, s));
  const news = useApi(`news:${symbol}`, (s) => getNews(symbol, 6, s));

  const detailIsCurrent = detail.dataKey === `detail:${symbol}`;
  const historyIsCurrent = history.dataKey === `history:${symbol}:${period}`;
  const watched = watchlist.includes(symbol);

  const select = (s: string) => {
    setSymbol(s);
    setQueryError(null);
  };

  const onSearch = (e: FormEvent) => {
    e.preventDefault();
    const s = query.trim().toUpperCase().replace(/\./g, "-");
    if (!SYMBOL_RE.test(s)) {
      setQueryError("Enter a ticker such as AAPL or VTI.");
      return;
    }
    select(s);
    setQuery("");
  };

  const toggleWatch = () =>
    setWatchlist((w) => (w.includes(symbol) ? w.filter((s) => s !== symbol) : [...w, symbol].slice(0, MAX_WATCH)));

  const refreshAll = () => {
    overview.reload();
    watch.reload();
    detail.reload();
    history.reload();
    news.reload();
  };

  const updatedAt = overview.data?.quotes[0]?.fetched_at;
  const watchQuotes = new Map((watch.data?.quotes ?? []).map((q) => [q.symbol, q]));

  return (
    <section className="panel" aria-label="Market">
      <div className="panel-inner">
        <header className="mk-header">
          <div className="panel-intro">
            <h2>Markets</h2>
            <p>Prices from Yahoo Finance. Look up a ticker to see its chart and recent headlines.</p>
          </div>
          <form className="mk-search" onSubmit={onSearch} role="search">
            <div className="ui-affix" data-prefix>
              <span aria-hidden="true">
                <SearchIcon />
              </span>
              <input
                className="ui-input"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setQueryError(null);
                }}
                placeholder="Ticker, e.g. AAPL"
                aria-label="Look up a ticker"
                aria-invalid={!!queryError}
                maxLength={12}
                autoCapitalize="characters"
                spellCheck={false}
              />
            </div>
            <button type="submit" className="ui-btn-primary">
              Look up
            </button>
          </form>
        </header>
        {queryError && (
          <p className="ui-error" role="alert">
            {queryError}
          </p>
        )}

        <section aria-label="Market overview">
          <div className="mk-section-head">
            <h3>Overview</h3>
            <span className="ui-muted mk-updated">
              {updatedAt && `Updated ${timeOrDate(updatedAt)}`}
              <button type="button" className="ui-icon-btn" onClick={refreshAll} aria-label="Refresh prices" title="Refresh">
                <RefreshIcon />
              </button>
            </span>
          </div>
          {overview.error && !overview.data ? (
            <p className="ui-banner" data-tone="warn" role="alert">
              <WarnIcon />
              {overview.error}
              <button type="button" className="ui-btn" onClick={overview.reload}>
                Retry
              </button>
            </p>
          ) : (
            <div className="mk-tiles">
              {overview.data ? (
                overview.data.quotes.map((q) => <QuoteTile key={q.symbol} q={q} selected={q.symbol === symbol} onSelect={select} />)
              ) : (
                <TileSkeletons count={7} />
              )}
            </div>
          )}
        </section>

        <div className="mk-layout">
          <div className="mk-main">
            {detail.error && !detailIsCurrent ? (
              <section className="ui-card">
                <p className="ui-banner" data-tone="warn" role="alert">
                  <WarnIcon />
                  {detail.error}
                </p>
              </section>
            ) : detail.data && detailIsCurrent ? (
              <Detail
                detail={detail.data}
                history={historyIsCurrent ? history.data : undefined}
                historyLoading={history.loading}
                historyError={history.error}
                period={period}
                onPeriod={setPeriod}
                watched={watched}
                onToggleWatch={toggleWatch}
              />
            ) : (
              <section className="ui-card" aria-busy="true">
                <div className="ui-skeleton mk-detail-skeleton" aria-hidden="true" />
              </section>
            )}

            <section className="ui-card" aria-label={`Headlines for ${symbol}`}>
              <h3 className="mk-card-title">Headlines: {symbol}</h3>
              {news.error && news.dataKey !== `news:${symbol}` ? (
                <p className="ui-muted">{news.error}</p>
              ) : news.data && news.dataKey === `news:${symbol}` ? (
                <NewsList items={news.data.items} />
              ) : (
                <div className="ui-skeleton mk-news-skeleton" aria-hidden="true" />
              )}
            </section>
          </div>

          <aside className="ui-card mk-watch" aria-label="Watchlist">
            <h3 className="mk-card-title">Watchlist</h3>
            {watchlist.length === 0 ? (
              <p className="ui-muted">Use the star on a ticker to add it here.</p>
            ) : (
              <ul className="mk-watch-list">
                {watchlist.map((s) => {
                  const q = watchQuotes.get(s);
                  const err = watch.data?.errors.find((e) => e.symbol === s);
                  return (
                    <li key={s}>
                      <button type="button" className="mk-watch-row" aria-pressed={s === symbol} onClick={() => select(s)}>
                        <span className="mk-watch-sym">
                          <strong>{s}</strong>
                          <span className="ui-muted">{q?.name ?? (err ? "Not found" : "")}</span>
                        </span>
                        {q ? (
                          <span className="mk-watch-px">
                            {price(q.price, q.currency)}
                            <Change q={q} showAbs={false} />
                          </span>
                        ) : (
                          <span className="ui-muted">{err ? "—" : "…"}</span>
                        )}
                      </button>
                      <button
                        type="button"
                        className="ui-icon-btn"
                        aria-label={`Remove ${s} from watchlist`}
                        onClick={() => setWatchlist((w) => w.filter((x) => x !== s))}
                      >
                        <CloseIcon />
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </aside>
        </div>

        <p className="ui-disclaimer">{DISCLAIMER} Market data may be delayed and is not suitable for trading.</p>
      </div>
    </section>
  );
}
