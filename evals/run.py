"""Eval runner and deploy gate.

    python -m evals.run                      # offline suites: free, deterministic, ~5 s
    python -m evals.run --suite retrieval    # one suite
    python -m evals.run --live               # also run the real-model suite (needs an API key)
    python -m evals.run --update-baseline    # accept the current scores as the new baseline

The run FAILS (exit code 1) if any metric is
  * below its absolute floor in evals/thresholds.yaml, or
  * more than `regression_tolerance` below the committed baseline (evals/baseline.json).

The second rule is what catches regressions: a change that drops retrieval hit@3 from
0.98 to 0.91 still clears a 0.85 floor, but it is a real regression and blocks the deploy.
Reports are written to evals/reports/latest.json and latest.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
BASELINE = HERE / "baseline.json"
THRESHOLDS = HERE / "thresholds.yaml"
REPORTS = HERE / "reports"


def load_thresholds() -> dict:
    import yaml

    return yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8")) or {}


def load_baseline() -> dict:
    try:
        return json.loads(BASELINE.read_text(encoding="utf-8")).get("metrics", {})
    except (OSError, ValueError):
        return {}


def evaluate(results, thresholds: dict, baseline: dict) -> list[dict]:
    """One row per metric: value, floor, baseline and a verdict."""
    tolerance = float(thresholds.get("regression_tolerance", 0.02))
    floors = thresholds.get("floors", {})
    rows = []
    for res in results:
        for metric, value in res.metrics.items():
            key = f"{res.name}.{metric}"
            floor = floors.get(res.name, {}).get(metric, thresholds.get("default_floor", 0.0))
            base = baseline.get(key)
            status = "pass"
            if value < floor:
                status = "below_floor"
            elif base is not None and value < base - tolerance:
                status = "regression"
            rows.append({"metric": key, "value": value, "floor": floor, "baseline": base, "status": status})
    return rows


def render_markdown(rows: list[dict], results, elapsed: float, passed: bool) -> str:
    lines = [
        f"# Eval report: {'PASS' if passed else 'FAIL'}",
        "",
        f"Run at {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"{sum(r.cases for r in results)} cases · {elapsed:.1f}s",
        "",
        "| Metric | Value | Floor | Baseline | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        base = "-" if r["baseline"] is None else f"{r['baseline']:.3f}"
        mark = {"pass": "pass", "below_floor": "**BELOW FLOOR**", "regression": "**REGRESSION**"}[r["status"]]
        lines.append(f"| {r['metric']} | {r['value']:.3f} | {r['floor']:.2f} | {base} | {mark} |")
    failing = [(res.name, f) for res in results for f in res.failures]
    if failing:
        lines += ["", "## Failing cases", ""] + [f"- `{name}` {text}" for name, text in failing]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suites and gate on regressions.")
    parser.add_argument("--suite", action="append", help="run only this suite (repeatable)")
    parser.add_argument("--live", action="store_true", help="also run suites that call real models")
    parser.add_argument("--update-baseline", action="store_true", help="save current scores as the baseline")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.live:  # offline suites must never touch the network, whatever .env says
        os.environ["LLM_PROVIDER"] = "fake"
        os.environ["MARKET_DATA_PROVIDER"] = "sample"
        os.environ["RAG_EMBEDDER"] = "local"
    os.environ.setdefault("LOG_LEVEL", "WARNING")

    import logging

    logging.disable(logging.WARNING)  # scripted failures are expected; keep the report readable
    from evals.suites import LIVE, OFFLINE

    suites = dict(OFFLINE) | (LIVE if args.live else {})
    if args.suite:
        unknown = set(args.suite) - set(suites)
        if unknown:
            parser.error(f"unknown suite(s): {', '.join(sorted(unknown))}; choose from {', '.join(suites)}")
        suites = {k: v for k, v in suites.items() if k in args.suite}

    started = time.perf_counter()
    results = []
    for name, fn in suites.items():
        t = time.perf_counter()
        res = fn()
        results.append(res)
        if not args.quiet:
            summary = "  ".join(f"{k}={v:.3f}" for k, v in res.metrics.items())
            print(f"{name:16} {res.cases:3d} cases  {time.perf_counter() - t:5.1f}s  {summary}")
    elapsed = time.perf_counter() - started

    rows = evaluate(results, load_thresholds(), load_baseline())
    bad = [r for r in rows if r["status"] != "pass"]
    skipped = [res.name for res in results if not res.metrics]
    passed = not bad

    REPORTS.mkdir(exist_ok=True)
    report = {
        "passed": passed, "elapsed_seconds": round(elapsed, 2), "rows": rows,
        "failures": {res.name: res.failures for res in results if res.failures},
    }
    (REPORTS / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS / "latest.md").write_text(render_markdown(rows, results, elapsed, passed), encoding="utf-8")

    if not args.quiet:
        for res in results:
            for failure in res.failures[:8]:
                print(f"  - [{res.name}] {failure}")
        for r in bad:
            base = "" if r["baseline"] is None else f", baseline {r['baseline']:.3f}"
            print(f"FAIL {r['metric']}: {r['value']:.3f} ({r['status']}; floor {r['floor']:.2f}{base})")
        if skipped:
            print(f"skipped: {', '.join(skipped)}")
        print(f"\n{'PASS' if passed else 'FAIL'}: {len(rows) - len(bad)}/{len(rows)} metrics ok in {elapsed:.1f}s "
              f"(report: evals/reports/latest.md)")

    if args.update_baseline:
        if not passed and any(r["status"] == "below_floor" for r in bad):
            print("refusing to save a baseline that is below the absolute floors")
            return 1
        merged = load_baseline() | {r["metric"]: r["value"] for r in rows}
        BASELINE.write_text(json.dumps({"metrics": dict(sorted(merged.items()))}, indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated ({len(merged)} metrics)")
        return 0
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
