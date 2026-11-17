// TypeScript mirrors of the backend Pydantic models (src/core/schemas.py).
// Keep these in sync with the backend, or generate them from FastAPI's
// OpenAPI schema with openapi-typescript so the contract never drifts.

/** Agent node names in the LangGraph workflow. */
export type AgentName =
  | "finance_qa"
  | "portfolio"
  | "market"
  | "goal_planning"
  | "news"
  | "tax"
  | "guardrail"
  | "fallback";

/** How much jargon an answer may assume. */
export type Level = "beginner" | "intermediate";

export interface ChatRequest {
  session_id: string;
  /** 1 to 2000 characters, matching Field(min_length=1, max_length=2000). */
  message: string;
  level?: Level;
}

export interface Source {
  title: string;
  url?: string | null;
  category: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  agent_used: string;
  sources: Source[];
  disclaimer: string;
}

export interface Holding {
  symbol: string;
  shares: number;
  avg_cost?: number | null;
}

/** A price from GET /api/market/*. */
export interface Quote {
  symbol: string;
  name?: string | null;
  price: number;
  previous_close?: number | null;
  change: number;
  /** Fraction: 0.012 means +1.2%. */
  change_pct: number;
  currency: string;
  year_high?: number | null;
  year_low?: number | null;
  /** ISO 8601 datetime when the provider returned this price. */
  fetched_at: string;
  source: "yfinance" | "cache" | "sample";
  /** Served from cache after a provider failure. */
  is_stale: boolean;
  /** Illustrative price, not market data. */
  is_sample: boolean;
}

/** Shape returned by the global exception handler. */
export interface ApiErrorBody {
  error_code: string;
  message: string;
  details?: { loc: (string | number)[]; msg: string; type: string }[];
}

/* ---------- Server-sent events from POST /api/chat/stream ----------
   event: agent    data: "finance_qa"            (or ["finance_qa","market"])
   event: token    data: "partial text"          (JSON-encoded string)
   event: sources  data: [Source, ...]
   event: done     data: { sources?, agent_used?, disclaimer? }  ({} is fine)
   event: error    data: "message" or { error_code, message }          */

export type StreamEventName = "agent" | "token" | "sources" | "meta" | "done" | "error";

/** One LLM gateway call: which model answered, why, and what it took (src/core/gateway.py CallTrace). */
export interface CallTrace {
  agent: string;
  tier: "fast" | "smart" | "";
  model: string;
  score: number;
  reasons: string[];
  cache_hit: boolean;
  retries: number;
  fallbacks: string[];
  continued: boolean;
  degraded: boolean;
  latency_ms: number;
  first_token_ms: number | null;
  tokens_in: number;
  tokens_out: number;
}

/** `meta` event: how the answer was produced. */
export interface StreamMeta {
  agents: string[];
  route: { plan: string[]; method: string };
  calls: CallTrace[];
  disclaimer: string;
  flags: string[];
}

export interface StreamDonePayload {
  sources?: Source[];
  agent_used?: string | string[];
  disclaimer?: string;
}

/* ---------- Portfolio (mirrors PortfolioRequest / PortfolioAnalysis) ---------- */

export interface PortfolioRequest {
  /** 1 to 50 holdings. */
  holdings: Holding[];
}

export type PriceSource = "live" | "cache" | "sample" | "cost" | "missing";

export interface HoldingResult {
  symbol: string;
  shares: number;
  price: number | null;
  price_source: PriceSource;
  value: number;
  /** 0..1 share of total value. */
  weight: number;
  avg_cost: number | null;
  cost_basis: number | null;
  gain: number | null;
  gain_pct: number | null;
  asset_class: string;
  sector: string;
  /** Yearly fund cost as a fraction; null for single stocks. */
  expense_ratio?: number | null;
}

export interface AllocationSlice {
  name: string;
  value: number;
  weight: number;
}

export interface PortfolioFlag {
  level: "info" | "warning";
  code: string;
  message: string;
}

export interface PortfolioAnalysis {
  total_value: number;
  total_cost: number | null;
  total_gain: number | null;
  total_gain_pct: number | null;
  largest_symbol: string | null;
  largest_weight: number;
  top3_weight: number;
  /** 1 / sum(weight^2): how many equal-sized positions the portfolio behaves like. */
  effective_holdings: number;
  holdings: HoldingResult[];
  by_asset_class: AllocationSlice[];
  by_sector: AllocationSlice[];
  flags: PortfolioFlag[];
  price_note: string;
  /** 0 (one stock) to 100 (broadly diversified). */
  diversification_score: number;
  risk_score: number;
  risk_level: "Low" | "Moderate" | "High" | "Very high" | "Unknown";
  equity_weight: number;
  expense_ratio: number | null;
  annual_fund_cost: number | null;
}

export interface SamplePortfolio {
  id: string;
  name: string;
  description: string;
  holdings: Holding[];
}

/* ---------- System (GET /api/system/metrics) ---------- */

export interface SystemMetrics {
  uptime_seconds: number;
  llm: {
    llm_calls: number;
    cache_hits: number;
    cache_hit_rate: number;
    by_tier: Record<string, number>;
    by_model: Record<string, number>;
    by_agent: Record<string, number>;
    retries: number;
    fallbacks: number;
    continuations: number;
    degraded_answers: number;
    latency_ms: { p50: number; p95: number };
    first_token_ms: { p50: number; p95: number };
    estimated_cost_usd: number;
    estimated_cost_all_smart_usd: number;
    estimated_savings_usd: number;
    recent: CallTrace[];
  };
  gateway: { models: { name: string; tier: string; circuit_open: boolean }[]; threshold: number; retries: number };
  response_cache: { enabled: boolean; ttl_seconds: number; entries: number; hits: number; misses: number; hit_rate: number };
  retrieval: {
    chunks: number;
    articles?: number;
    categories?: string[];
    embedder: string;
    backend?: string;
    query_cache_hits?: number;
    query_cache_misses?: number;
    query_cache_hit_rate?: number;
    dense_failures?: number;
  };
  market: {
    provider: string;
    chain: string[];
    provider_switches: number;
    cached_entries: number;
    cache_hits: number;
    provider_calls: number;
    retries: number;
    provider_errors: number;
    stale_served: number;
    sample_served: number;
  };
}

export interface RoutePreview {
  tier: "fast" | "smart";
  score: number;
  reasons: string[];
  threshold: number;
  model_chain: string[];
}

/* ---------- Market data ---------- */

export interface Profile {
  symbol: string;
  name?: string | null;
  asset_class: string;
  sector: string;
  diversified: boolean;
}

export interface QuoteError {
  symbol: string;
  error_code: string;
  message: string;
}

export interface QuotesResponse {
  quotes: Quote[];
  errors: QuoteError[];
}

export interface SymbolDetail {
  quote: Quote;
  profile: Profile;
}

export type HistoryPeriod = "1mo" | "6mo" | "1y" | "5y";

export interface PricePoint {
  /** ISO date. */
  date: string;
  close: number;
}

export interface PriceHistory {
  symbol: string;
  period: HistoryPeriod;
  points: PricePoint[];
  source: string;
  is_stale: boolean;
  is_sample: boolean;
}

export interface NewsItem {
  title: string;
  publisher?: string | null;
  url?: string | null;
  published_at?: string | null;
  symbol?: string | null;
}

export interface NewsResponse {
  symbol: string | null;
  items: NewsItem[];
}

/* ---------- Goal planning (mirrors GoalRequest / GoalPlan) ---------- */

export type RiskProfile = "conservative" | "balanced" | "growth";

export interface GoalRequest {
  name: string;
  /** In today's money. */
  target_amount: number;
  /** 1 to 60. */
  years: number;
  current_savings: number;
  monthly_contribution: number;
  risk_profile: RiskProfile;
  /** Fraction, e.g. 0.03. */
  inflation_rate: number;
  /** Overrides the risk profile's assumed return when set. */
  annual_return?: number | null;
}

export interface ProjectionPoint {
  year: number;
  balance: number;
  contributed: number;
}

export interface GoalScenario {
  risk_profile: RiskProfile;
  annual_return: number;
  projected_value: number;
  required_monthly: number;
  on_track: boolean;
}

/** Flags share PortfolioFlag's shape. */
export type Flag = PortfolioFlag;

export interface GoalPlan {
  name: string;
  years: number;
  annual_return: number;
  inflation_rate: number;
  target_today: number;
  target_future: number;
  projected_value: number;
  projected_value_today: number;
  /** projected - target_future; negative means a shortfall. */
  gap: number;
  progress: number;
  on_track: boolean;
  required_monthly: number;
  total_contributed: number;
  growth: number;
  projection: ProjectionPoint[];
  scenarios: GoalScenario[];
  flags: Flag[];
  assumptions_note: string;
}
