import { useEffect, useMemo, useState } from "react";
import { planGoal } from "../api/client";
import { useApi } from "../hooks/useApi";
import { useDebounced } from "../hooks/useDebounced";
import { useInsightStream } from "../hooks/useInsightStream";
import { compactMoney, money, pct, signedMoney } from "../lib/format";
import type { GoalPlan, GoalRequest, RiskProfile } from "../types/api";
import FlagList from "../ui/FlagList";
import { CheckCircleIcon, InfoIcon, WarnIcon } from "../ui/icons";
import InsightCard from "../ui/InsightCard";
import LineChart from "../ui/LineChart";
import Segmented from "../ui/Segmented";
import Stat from "../ui/Stat";
import "../ui/panel.css";
import "./goals.css";

const STORAGE_KEY = "finance-assistant-goal";

// Mirrors RETURN_ASSUMPTIONS in src/goals/planner.py (shown as labels only; the server does the math).
const MIXES: readonly { value: RiskProfile; label: string; hint: string }[] = [
  { value: "conservative", label: "Conservative", hint: "Mostly bonds and cash · assumes 4% a year" },
  { value: "balanced", label: "Balanced", hint: "Stocks and bonds · assumes 6% a year" },
  { value: "growth", label: "Growth", hint: "Mostly stocks · assumes 8% a year" },
];

interface Form {
  name: string;
  target: string;
  years: string;
  saved: string;
  monthly: string;
  mix: RiskProfile;
  inflation: string;
  customReturn: string;
}

const DEFAULT_FORM: Form = {
  name: "House down payment",
  target: "60000",
  years: "8",
  saved: "5000",
  monthly: "400",
  mix: "balanced",
  inflation: "3",
  customReturn: "",
};

function loadForm(): Form {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (s && typeof s === "object") {
      const merged = { ...DEFAULT_FORM, ...s };
      if (!MIXES.some((m) => m.value === merged.mix)) merged.mix = DEFAULT_FORM.mix;
      return merged;
    }
  } catch {
    /* storage blocked or corrupt */
  }
  return DEFAULT_FORM;
}

type Errors = Partial<Record<keyof Form, string>>;

const num = (s: string) => (s.trim() === "" ? NaN : Number(s.replace(/[,\s$]/g, "")));

/** Validate the form; returns the API request when valid. Mirrors GoalRequest's server-side limits. */
function toRequest(f: Form): { req: GoalRequest | null; errors: Errors } {
  const errors: Errors = {};
  const target = num(f.target);
  const years = num(f.years);
  const saved = f.saved.trim() ? num(f.saved) : 0;
  const monthly = f.monthly.trim() ? num(f.monthly) : 0;
  const inflation = f.inflation.trim() ? num(f.inflation) : 3;
  const custom = f.customReturn.trim() ? num(f.customReturn) : null;

  if (!(target > 0)) errors.target = "Enter an amount above 0.";
  else if (target > 1e10) errors.target = "That amount is too large.";
  if (!Number.isInteger(years) || years < 1 || years > 60) errors.years = "Use whole years from 1 to 60.";
  if (!(saved >= 0)) errors.saved = "Use 0 or more.";
  if (!(monthly >= 0)) errors.monthly = "Use 0 or more.";
  if (!(inflation >= 0 && inflation <= 15)) errors.inflation = "Use 0 to 15%.";
  if (custom !== null && !(custom >= 0 && custom <= 20)) errors.customReturn = "Use 0 to 20%.";

  if (Object.keys(errors).length) return { req: null, errors };
  return {
    errors,
    req: {
      name: f.name.trim().slice(0, 60) || "My goal",
      target_amount: target,
      years,
      current_savings: saved,
      monthly_contribution: monthly,
      risk_profile: f.mix,
      inflation_rate: inflation / 100,
      annual_return: custom === null ? null : custom / 100,
    },
  };
}

/* ---------- Pieces ---------- */

function MoneyField({
  label,
  value,
  onChange,
  error,
  hint,
  id,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  error?: string;
  hint?: string;
  id: string;
}) {
  return (
    <label className="ui-field" htmlFor={id}>
      <span>{label}</span>
      <span className="ui-affix" data-prefix>
        <span aria-hidden="true">$</span>
        <input
          id={id}
          className="ui-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          inputMode="decimal"
          aria-invalid={!!error}
          aria-describedby={error || hint ? `${id}-msg` : undefined}
        />
      </span>
      {(error || hint) && (
        <span id={`${id}-msg`} className={error ? "gl-field-error" : "ui-field-hint"}>
          {error ?? hint}
        </span>
      )}
    </label>
  );
}

function StatusBanner({ plan }: { plan: GoalPlan }) {
  return plan.on_track ? (
    <p className="ui-banner gl-status" data-tone="ok">
      <CheckCircleIcon />
      <span>
        <strong>On track.</strong> Projected {money(plan.projected_value)} against a target of {money(plan.target_future)} in{" "}
        {plan.years} {plan.years === 1 ? "year" : "years"}.
      </span>
    </p>
  ) : (
    <p className="ui-banner gl-status" data-tone="warn">
      <WarnIcon />
      <span>
        <strong>Short by {money(-plan.gap)}.</strong> Saving about {money(plan.required_monthly)} a month would reach{" "}
        {money(plan.target_future)} under these assumptions.
      </span>
    </p>
  );
}

function ScenarioTable({ plan, current }: { plan: GoalPlan; current: RiskProfile }) {
  return (
    <div className="ui-table-wrap">
      <table className="ui-table">
        <caption className="ui-sr">Projection under each investment mix</caption>
        <thead>
          <tr>
            <th scope="col">Investment mix</th>
            <th scope="col" className="ui-right">Assumed return</th>
            <th scope="col" className="ui-right">Projected</th>
            <th scope="col" className="ui-right">Monthly needed</th>
            <th scope="col">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {plan.scenarios.map((s) => (
            <tr key={s.risk_profile} aria-current={s.risk_profile === current ? "true" : undefined} className="gl-scenario">
              <th scope="row">
                {MIXES.find((m) => m.value === s.risk_profile)?.label}
                {s.risk_profile === current && <span className="ui-badge">selected</span>}
              </th>
              <td className="ui-right">{pct(s.annual_return)} a year</td>
              <td className="ui-right">{money(s.projected_value)}</td>
              <td className="ui-right">{money(s.required_monthly)}</td>
              <td>
                {s.on_track ? (
                  <span className="gl-outcome ui-pos">
                    <CheckCircleIcon /> On track
                  </span>
                ) : (
                  <span className="gl-outcome gl-short">
                    <WarnIcon /> Short
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- Main component ---------- */

export default function GoalsTab() {
  const [form, setForm] = useState<Form>(loadForm);
  const { req, errors } = useMemo(() => toRequest(form), [form]);
  const reqKey = req ? JSON.stringify(req) : null;
  const debouncedKey = useDebounced(reqKey, 350);
  const plan = useApi(debouncedKey, (signal) => planGoal(JSON.parse(debouncedKey!) as GoalRequest, signal));
  const insights = useInsightStream<GoalRequest>("/api/goals/insights/stream");
  const [explainedKey, setExplainedKey] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
    } catch {
      /* storage blocked */
    }
  }, [form]);

  const set = <K extends keyof Form>(key: K) => (value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));
  const data = plan.data;
  const explanationStale = insights.status !== "idle" && explainedKey !== reqKey;

  const explain = () => {
    if (!req) return;
    setExplainedKey(reqKey);
    void insights.start(req);
  };

  const yearLabel = (y: number) => (y === 0 ? "Now" : `Year ${y}`);

  return (
    <section className="panel" aria-label="Goals">
      <div className="panel-inner">
        <header className="panel-intro">
          <h2>Goal planner</h2>
          <p>See whether your savings plan reaches a goal, and how much the investment mix changes the outcome.</p>
        </header>

        <form className="ui-card gl-form" onSubmit={(e) => e.preventDefault()} noValidate>
          <label className="ui-field gl-name" htmlFor="gl-name">
            <span>Goal</span>
            <input
              id="gl-name"
              className="ui-input"
              value={form.name}
              onChange={(e) => set("name")(e.target.value)}
              maxLength={60}
              placeholder="e.g. House down payment"
            />
          </label>
          <MoneyField id="gl-target" label="Target, in today's money" value={form.target} onChange={set("target")} error={errors.target} />
          <label className="ui-field" htmlFor="gl-years">
            <span>Years from now</span>
            <input
              id="gl-years"
              className="ui-input"
              value={form.years}
              onChange={(e) => set("years")(e.target.value)}
              inputMode="numeric"
              aria-invalid={!!errors.years}
              aria-describedby={errors.years ? "gl-years-msg" : undefined}
            />
            {errors.years && (
              <span id="gl-years-msg" className="gl-field-error">
                {errors.years}
              </span>
            )}
          </label>
          <MoneyField id="gl-saved" label="Saved so far" value={form.saved} onChange={set("saved")} error={errors.saved} />
          <MoneyField id="gl-monthly" label="Monthly saving" value={form.monthly} onChange={set("monthly")} error={errors.monthly} />

          <div className="ui-field gl-mix">
            <span id="gl-mix-label">Investment mix</span>
            <Segmented label="Investment mix" options={MIXES} value={form.mix} onChange={set("mix")} />
            <span className="ui-field-hint">{MIXES.find((m) => m.value === form.mix)?.hint}</span>
          </div>

          <details className="gl-advanced">
            <summary>Assumptions</summary>
            <div className="gl-advanced-grid">
              <label className="ui-field" htmlFor="gl-inflation">
                <span>Inflation per year</span>
                <span className="ui-affix" data-suffix>
                  <input
                    id="gl-inflation"
                    className="ui-input"
                    value={form.inflation}
                    onChange={(e) => set("inflation")(e.target.value)}
                    inputMode="decimal"
                    aria-invalid={!!errors.inflation}
                  />
                  <span aria-hidden="true">%</span>
                </span>
                {errors.inflation && <span className="gl-field-error">{errors.inflation}</span>}
              </label>
              <label className="ui-field" htmlFor="gl-return">
                <span>Custom return per year (optional)</span>
                <span className="ui-affix" data-suffix>
                  <input
                    id="gl-return"
                    className="ui-input"
                    value={form.customReturn}
                    onChange={(e) => set("customReturn")(e.target.value)}
                    inputMode="decimal"
                    placeholder="uses the mix"
                    aria-invalid={!!errors.customReturn}
                  />
                  <span aria-hidden="true">%</span>
                </span>
                {errors.customReturn && <span className="gl-field-error">{errors.customReturn}</span>}
              </label>
            </div>
          </details>
        </form>

        {plan.error && (
          <p className="ui-banner" data-tone="warn" role="alert">
            <WarnIcon />
            {plan.error}
            <button type="button" className="ui-btn" onClick={plan.reload}>
              Retry
            </button>
          </p>
        )}

        {data && (
          <div className="gl-results" aria-busy={plan.loading}>
            {!req && (
              <p className="ui-banner">
                <InfoIcon />
                Fix the highlighted fields to update these results.
              </p>
            )}
            <StatusBanner plan={data} />

            <div className="ui-stats">
              <Stat
                label={`Target in ${data.years} ${data.years === 1 ? "year" : "years"}`}
                value={money(data.target_future)}
                sub={`${money(data.target_today)} today, grown by ${pct(data.inflation_rate)} inflation`}
              />
              <Stat label="Projected value" value={money(data.projected_value)} sub={`≈ ${money(data.projected_value_today)} in today's money`} />
              <Stat
                label="Monthly needed"
                value={money(data.required_monthly)}
                sub={`you plan ${money(num(form.monthly) || 0)} a month`}
              />
              <Stat
                label="From investment growth"
                value={signedMoney(data.growth)}
                sub={`on ${money(data.total_contributed)} saved in total`}
              />
            </div>

            <section className="ui-card" aria-label="Projection chart">
              <div className="ui-card-head">
                <h3>{data.name}: projected balance</h3>
                <span className="ui-muted gl-assume">{pct(data.annual_return)} a year assumed</span>
              </div>
              <LineChart
                xLabels={data.projection.map((p) => yearLabel(p.year))}
                series={[
                  { id: "balance", label: "Projected balance", color: "var(--series-1)", values: data.projection.map((p) => p.balance) },
                  { id: "saved", label: "Total saved", color: "var(--series-2)", values: data.projection.map((p) => p.contributed) },
                ]}
                reference={{ label: "Target", value: data.target_future }}
                yTick={(v) => compactMoney(v)}
                tooltipValue={(v) => money(v)}
                zeroBased
                height={260}
                ariaLabel={`Projected balance grows from ${money(data.projection[0].balance)} now to ${money(data.projected_value)} in year ${data.years}, against a target of ${money(data.target_future)}. You would save ${money(data.total_contributed)} in total.`}
              />
              <p className="ui-note">{data.assumptions_note}</p>
            </section>

            <FlagList flags={data.flags} hide={["on_track", "shortfall"]} />

            <section className="ui-card">
              <h3 className="gl-card-title">Same plan, different investment mixes</h3>
              <ScenarioTable plan={data} current={form.mix} />
            </section>

            {insights.status === "idle" ? (
              <div className="ui-card gl-explain">
                <div>
                  <h3>Want this explained?</h3>
                  <p className="ui-muted">The goal planning agent walks through what these numbers mean for you.</p>
                </div>
                <button type="button" className="ui-btn-primary" onClick={explain} disabled={!req}>
                  Explain my plan
                </button>
              </div>
            ) : (
              <>
                {explanationStale && (
                  <p className="ui-banner">
                    <InfoIcon />
                    This explanation is for your earlier numbers.
                    <button type="button" className="ui-btn" onClick={explain} disabled={!req}>
                      Update
                    </button>
                  </p>
                )}
                <InsightCard
                  title="What this means"
                  agentLabel="Goal planning agent"
                  text={insights.text}
                  status={insights.status}
                  error={insights.error}
                  onStop={insights.stop}
                  onRegenerate={explain}
                />
              </>
            )}
          </div>
        )}

        {!data && !plan.error && (
          <div className="ui-card" aria-busy="true">
            <div className="ui-skeleton gl-skeleton" aria-hidden="true" />
          </div>
        )}
      </div>
    </section>
  );
}
