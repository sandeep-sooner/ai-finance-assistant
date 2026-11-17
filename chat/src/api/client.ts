// One typed wrapper around fetch: base URL, timeouts and friendly errors.
import type {
  ApiErrorBody,
  ChatRequest,
  GoalPlan,
  GoalRequest,
  HistoryPeriod,
  NewsResponse,
  PortfolioAnalysis,
  PortfolioRequest,
  PriceHistory,
  QuotesResponse,
  RoutePreview,
  SamplePortfolio,
  SymbolDetail,
  SystemMetrics,
} from "../types/api";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");
const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code = "unknown_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function apiUrl(path: string): string {
  return `${BASE_URL}${path}`;
}

// Turn an error response into a message a user can act on.
async function errorFromResponse(res: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> & { detail?: unknown } = {};
  try {
    body = await res.json();
  } catch {
    /* not JSON */
  }

  if (res.status === 429) {
    return new ApiError("Too many requests. Wait a moment and try again.", 429, body.error_code ?? "rate_limited");
  }
  if (res.status === 503) {
    return new ApiError(
      body.message ?? "The service is degraded right now. Some data may be cached.",
      503,
      body.error_code ?? "service_unavailable"
    );
  }
  if (res.status === 422) {
    return new ApiError(body.message ?? "The request was not valid. Check your input and try again.", 422, "validation_error");
  }
  return new ApiError(body.message ?? `Request failed (${res.status}).`, res.status, body.error_code);
}

export function networkError(): ApiError {
  return new ApiError("Can't reach the server. Check your connection and that the backend is running.", 0, "network_error");
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  timeoutMs?: number;
}

/** JSON request with a timeout. Throws ApiError. */
export async function request<T>(path: string, { body, timeoutMs = DEFAULT_TIMEOUT_MS, signal, headers, ...init }: RequestOptions = {}): Promise<T> {
  const timeout = AbortSignal.timeout(timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: { Accept: "application/json", ...(body !== undefined && { "Content-Type": "application/json" }), ...headers },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: combined,
    });
  } catch (err) {
    if (timeout.aborted) throw new ApiError("The server took too long to respond.", 0, "timeout");
    if ((err as Error).name === "AbortError") throw err;
    throw networkError();
  }

  if (!res.ok) throw await errorFromResponse(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** POSTs JSON and returns the response whose body is an SSE stream. */
export async function openStream(path: string, payload: unknown, signal: AbortSignal): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw networkError();
  }
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError("The server sent an empty response.", res.status, "empty_body");
  return res;
}

/** Opens POST /api/chat/stream. */
export function openChatStream(payload: ChatRequest, signal: AbortSignal): Promise<Response> {
  return openStream("/api/chat/stream", payload, signal);
}

/** POST /api/portfolio/analyze: computed metrics, no LLM. */
export function analyzePortfolio(payload: PortfolioRequest, signal?: AbortSignal): Promise<PortfolioAnalysis> {
  return request<PortfolioAnalysis>("/api/portfolio/analyze", { method: "POST", body: payload, signal });
}

/** GET /api/portfolio/samples: illustrative portfolios to load with one click. */
export function getSamplePortfolios(signal?: AbortSignal): Promise<{ portfolios: SamplePortfolio[] }> {
  return request<{ portfolios: SamplePortfolio[] }>("/api/portfolio/samples", { signal });
}

/* ---------- System ---------- */

/** GET /api/system/metrics: model routing, cache, fallbacks, latency and cost. */
export function getSystemMetrics(signal?: AbortSignal): Promise<SystemMetrics> {
  return request<SystemMetrics>("/api/system/metrics", { signal, timeoutMs: 10_000 });
}

/** GET /api/system/route-preview: which model tier a question would use, without calling a model. */
export function getRoutePreview(q: string, agent: string, signal?: AbortSignal): Promise<RoutePreview> {
  return request<RoutePreview>(`/api/system/route-preview?q=${encodeURIComponent(q)}&agent=${encodeURIComponent(agent)}`, { signal });
}

/* ---------- Market data ---------- */

const enc = encodeURIComponent;

export function getMarketOverview(signal?: AbortSignal): Promise<QuotesResponse> {
  return request<QuotesResponse>("/api/market/overview", { signal });
}

export function getQuotes(symbols: string[], signal?: AbortSignal): Promise<QuotesResponse> {
  return request<QuotesResponse>(`/api/market/quotes?symbols=${enc(symbols.join(","))}`, { signal });
}

export function getSymbolDetail(symbol: string, signal?: AbortSignal): Promise<SymbolDetail> {
  return request<SymbolDetail>(`/api/market/symbols/${enc(symbol)}`, { signal });
}

export function getPriceHistory(symbol: string, period: HistoryPeriod, signal?: AbortSignal): Promise<PriceHistory> {
  return request<PriceHistory>(`/api/market/history/${enc(symbol)}?period=${period}`, { signal });
}

export function getNews(symbol: string | null, limit = 8, signal?: AbortSignal): Promise<NewsResponse> {
  const q = symbol ? `symbol=${enc(symbol)}&limit=${limit}` : `limit=${limit}`;
  return request<NewsResponse>(`/api/market/news?${q}`, { signal });
}

/* ---------- Goal planning ---------- */

/** POST /api/goals/plan: deterministic projection, no LLM. */
export function planGoal(payload: GoalRequest, signal?: AbortSignal): Promise<GoalPlan> {
  return request<GoalPlan>("/api/goals/plan", { method: "POST", body: payload, signal });
}

/** DELETE /api/sessions/{id}: clears the server-side conversation memory. */
export function deleteSession(sessionId: string): Promise<void> {
  return request<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE", timeoutMs: 10_000 });
}
