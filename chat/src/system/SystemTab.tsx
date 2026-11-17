import { useState } from "react";
import type { FormEvent } from "react";
import { ApiError, getRoutePreview, getSystemMetrics } from "../api/client";
import { useApi } from "../hooks/useApi";
import { AGENT_LABELS } from "../lib/constants";
import type { CallTrace, RoutePreview, SystemMetrics } from "../types/api";
import Stat from "../ui/Stat";
import "../ui/panel.css";
import "./system.css";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const usd = (x: number) => (x > 0 && x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`);
const ms = (x: number) => (x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`);
const shortModel = (name: string) => name.split(":").pop() ?? name;

const AGENTS = ["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"] as const;
const PREVIEW_EXAMPLES = [
  "What is an ETF?",
  "Compare index funds and active funds: what are the trade-offs, and why do costs matter over 30 years?",
];

/** Two-part bar: share of calls answered by the fast model versus the smart one. */
function TierSplit({ fast, smart }: { fast: number; smart: number }) {
  const total = fast + smart;
  if (!total) return <p className="ui-note sys-empty">No model calls yet. Ask the assistant something, then come back.</p>;
  const fastShare = fast / total;
  return (
    <div className="sys-split">
      <div className="sys-split-bar" role="img" aria-label={`${pct(fastShare)} fast model, ${pct(1 - fastShare)} smart model`}>
        {fast > 0 && <span className="sys-split-fast" style={{ flexGrow: fast }} />}
        {smart > 0 && <span className="sys-split-smart" style={{ flexGrow: smart }} />}
      </div>
      <div className="sys-split-legend">
        <span>
          <i className="sys-swatch sys-split-fast" /> Fast model <b className="ui-num">{fast}</b> ({pct(fastShare)})
        </span>
        <span>
          <i className="sys-swatch sys-split-smart" /> Smart model <b className="ui-num">{smart}</b> ({pct(1 - fastShare)})
        </span>
      </div>
    </div>
  );
}

function outcome(c: CallTrace): { label: string; tone: string } {
  if (c.degraded) return { label: "Offline fallback", tone: "warn" };
  if (c.cache_hit) return { label: "Cache hit", tone: "ok" };
  if (c.continued) return { label: "Continued by backup", tone: "warn" };
  if (c.fallbacks.length) return { label: "Switched model", tone: "warn" };
  if (c.retries) return { label: `Retried ×${c.retries}`, tone: "warn" };
  return { label: "OK", tone: "plain" };
}

function RecentCalls({ calls }: { calls: CallTrace[] }) {
  if (!calls.length) return null;
  return (
    <div className="ui-card">
      <div className="ui-card-head">
        <h3>Recent model calls</h3>
        <span className="ui-muted sys-small">newest first</span>
      </div>
      <div className="ui-table-wrap">
        <table className="ui-table sys-table">
          <thead>
            <tr>
              <th>Agent</th>
              <th>Tier</th>
              <th>Model</th>
              <th className="ui-right">Complexity</th>
              <th className="ui-right">First token</th>
              <th className="ui-right">Total</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {calls.map((c, i) => {
              const o = outcome(c);
              return (
                <tr key={i}>
                  <td>{AGENT_LABELS[c.agent] ?? c.agent ?? "—"}</td>
                  <td>
                    <span className="sys-tier" data-tier={c.tier}>
                      {c.tier || "—"}
                    </span>
                  </td>
                  <td className="sys-model">{shortModel(c.model)}</td>
                  <td className="ui-right ui-num" title={c.reasons.join("\n")}>
                    {c.score.toFixed(2)}
                  </td>
                  <td className="ui-right ui-num">{c.first_token_ms != null ? ms(c.first_token_ms) : "—"}</td>
                  <td className="ui-right ui-num">{ms(c.latency_ms)}</td>
                  <td>
                    <span className="sys-outcome" data-tone={o.tone}>
                      {o.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="ui-note">Hover a complexity score to see the signals that produced it.</p>
    </div>
  );
}

/** Try the model router: type a question and see the tier, the score and each signal behind it. */
function RouterPlayground({ threshold }: { threshold: number }) {
  const [question, setQuestion] = useState(PREVIEW_EXAMPLES[1]);
  const [agent, setAgent] = useState<string>("finance_qa");
  const [result, setResult] = useState<RoutePreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async (q: string, a: string) => {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await getRoutePreview(q.trim(), a));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not preview the route.");
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void run(question, agent);
  };

  return (
    <div className="ui-card">
      <div className="ui-card-head">
        <h3>Model router playground</h3>
        <span className="ui-muted sys-small">no model is called</span>
      </div>
      <form className="sys-form" onSubmit={onSubmit}>
        <input
          className="ui-input sys-q"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          aria-label="Question to route"
          maxLength={500}
          placeholder="Type a question"
        />
        <select className="ui-input sys-agent" value={agent} onChange={(e) => setAgent(e.target.value)} aria-label="Agent">
          {AGENTS.map((a) => (
            <option key={a} value={a}>
              {AGENT_LABELS[a]}
            </option>
          ))}
        </select>
        <button type="submit" className="ui-btn ui-btn-primary" disabled={busy || !question.trim()}>
          Route it
        </button>
      </form>
      <div className="sys-examples">
        {PREVIEW_EXAMPLES.map((q) => (
          <button
            key={q}
            type="button"
            className="ui-btn sys-example"
            onClick={() => {
              setQuestion(q);
              void run(q, agent);
            }}
          >
            {q.length > 46 ? `${q.slice(0, 46)}…` : q}
          </button>
        ))}
      </div>

      {error && <p className="ui-error">{error}</p>}

      {result && (
        <div className="sys-result">
          <div className="sys-verdict" data-tier={result.tier}>
            <span className="sys-verdict-tier">{result.tier === "smart" ? "Smart model" : "Fast model"}</span>
            <span className="sys-verdict-model">{shortModel(result.model_chain[0] ?? "")}</span>
          </div>
          <div className="sys-meter">
            <div className="sys-meter-track">
              <span className="sys-meter-fill" data-tier={result.tier} style={{ width: `${Math.min(result.score, 1) * 100}%` }} />
              <span className="sys-meter-mark" style={{ left: `${threshold * 100}%` }} aria-hidden="true" />
            </div>
            <div className="sys-meter-labels ui-num">
              <span>complexity {result.score.toFixed(2)}</span>
              <span>smart at {threshold.toFixed(2)} and above</span>
            </div>
          </div>
          <ul className="sys-reasons">
            {result.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
          <p className="ui-note">
            If this model fails: {result.model_chain.slice(1).map(shortModel).join(" → ") || "no other model configured"} → offline
            answer from the knowledge base.
          </p>
        </div>
      )}
    </div>
  );
}

export default function SystemTab({ active }: { active: boolean }) {
  // Poll only while the tab is visible.
  const { data, error } = useApi<SystemMetrics>(active ? "metrics" : null, (signal) => getSystemMetrics(signal), {
    refreshMs: 5000,
    refreshEnabled: active,
  });

  const llm = data?.llm;
  const fast = llm?.by_tier.fast ?? 0;
  const smart = llm?.by_tier.smart ?? 0;

  return (
    <section className="panel" aria-label="System">
      <div className="panel-inner">
        <div className="panel-intro">
          <h2>System</h2>
          <p>How answers are produced: which model was used, what the cache saved, and what happened when something failed.</p>
        </div>

        {error && !data && (
          <p className="ui-banner" data-tone="warn">
            {error}
          </p>
        )}

        {data && llm && (
          <>
            <div className="ui-stats">
              <Stat label="Model calls" value={llm.llm_calls} sub={`${llm.cache_hits} answered from cache`} />
              <Stat
                label="Cache hit rate"
                value={pct(llm.cache_hit_rate)}
                sub={`${data.response_cache.entries} answers stored · ${Math.round(data.response_cache.ttl_seconds / 60)} min TTL`}
              />
              <Stat
                label="Estimated cost"
                value={usd(llm.estimated_cost_usd)}
                sub={`${usd(llm.estimated_savings_usd)} saved vs. always using the smart model`}
              />
              <Stat
                label="Response time"
                value={llm.llm_calls ? ms(llm.latency_ms.p50) : "—"}
                sub={llm.llm_calls ? `p95 ${ms(llm.latency_ms.p95)} · first token ${ms(llm.first_token_ms.p50)}` : "median, per model call"}
              />
            </div>

            <div className="sys-grid">
              <div className="ui-card">
                <div className="ui-card-head">
                  <h3>Model router</h3>
                  <span className="ui-muted sys-small">easy → fast, hard → smart</span>
                </div>
                <TierSplit fast={fast} smart={smart} />
                <ul className="sys-chain">
                  {data.gateway.models.map((m, i) => (
                    <li key={m.name}>
                      <span className="sys-chain-n">{i + 1}</span>
                      <span className="sys-model">{m.name}</span>
                      <span className="sys-tier" data-tier={m.tier}>
                        {m.tier}
                      </span>
                      <span className="sys-outcome" data-tone={m.circuit_open ? "warn" : "ok"}>
                        {m.circuit_open ? "Circuit open" : "Healthy"}
                      </span>
                    </li>
                  ))}
                  <li>
                    <span className="sys-chain-n">{data.gateway.models.length + 1}</span>
                    <span className="sys-model">offline answer from knowledge base</span>
                    <span className="sys-outcome" data-tone="plain">
                      Last resort
                    </span>
                  </li>
                </ul>
              </div>

              <div className="ui-card">
                <div className="ui-card-head">
                  <h3>Fallback system</h3>
                  <span className="ui-muted sys-small">users never see an error</span>
                </div>
                <dl className="sys-kv">
                  <dt>Retries (same model, with backoff)</dt>
                  <dd className="ui-num">{llm.retries}</dd>
                  <dt>Switched to another model</dt>
                  <dd className="ui-num">{llm.fallbacks}</dd>
                  <dt>Answers finished by a backup model</dt>
                  <dd className="ui-num">{llm.continuations}</dd>
                  <dt>Offline answers (all models down)</dt>
                  <dd className="ui-num">{llm.degraded_answers}</dd>
                  <dt>Market data served from stale cache</dt>
                  <dd className="ui-num">{data.market.stale_served}</dd>
                  <dt>Market provider switches</dt>
                  <dd className="ui-num">{data.market.provider_switches}</dd>
                </dl>
              </div>

              <div className="ui-card">
                <div className="ui-card-head">
                  <h3>Knowledge base (RAG)</h3>
                  <span className="ui-muted sys-small">{data.retrieval.backend ?? "off"}</span>
                </div>
                <dl className="sys-kv">
                  <dt>Articles and glossary terms</dt>
                  <dd className="ui-num">{data.retrieval.articles ?? 0}</dd>
                  <dt>Indexed chunks</dt>
                  <dd className="ui-num">{data.retrieval.chunks}</dd>
                  <dt>Embedding model</dt>
                  <dd>{data.retrieval.embedder}</dd>
                  <dt>Query cache hit rate</dt>
                  <dd className="ui-num">{pct(data.retrieval.query_cache_hit_rate ?? 0)}</dd>
                </dl>
              </div>

              <div className="ui-card">
                <div className="ui-card-head">
                  <h3>Market data</h3>
                  <span className="ui-muted sys-small">{data.market.chain.join(" → ")}</span>
                </div>
                <dl className="sys-kv">
                  <dt>Provider calls</dt>
                  <dd className="ui-num">{data.market.provider_calls}</dd>
                  <dt>Served from cache</dt>
                  <dd className="ui-num">{data.market.cache_hits}</dd>
                  <dt>Provider errors</dt>
                  <dd className="ui-num">{data.market.provider_errors}</dd>
                  <dt>Cached entries</dt>
                  <dd className="ui-num">{data.market.cached_entries}</dd>
                </dl>
              </div>
            </div>

            <RouterPlayground threshold={data.gateway.threshold} />
            <RecentCalls calls={llm.recent} />
          </>
        )}

        {!data && !error && <div className="ui-skeleton sys-skeleton" aria-label="Loading system metrics" />}
      </div>
    </section>
  );
}
