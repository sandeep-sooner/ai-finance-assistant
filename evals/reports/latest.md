# Eval report: FAIL

Run at 2026-11-16T17:45:20+00:00 · 202 cases · 117.4s

| Metric | Value | Floor | Baseline | Status |
|---|---:|---:|---:|---|
| intent_routing.primary_accuracy | 1.000 | 0.95 | 1.000 | pass |
| intent_routing.exact_plan_accuracy | 1.000 | 0.90 | 1.000 | pass |
| intent_routing.method_accuracy | 1.000 | 0.90 | 1.000 | pass |
| model_routing.tier_accuracy | 0.958 | 0.90 | 0.958 | pass |
| model_routing.easy_to_fast | 1.000 | 0.95 | 1.000 | pass |
| model_routing.hard_to_smart | 0.917 | 0.85 | 0.917 | pass |
| retrieval.hit_at_1 | 0.975 | 0.75 | 0.975 | pass |
| retrieval.hit_at_3 | 1.000 | 0.90 | 1.000 | pass |
| retrieval.mrr | 0.988 | 0.80 | 0.988 | pass |
| retrieval.off_topic_rejection | 1.000 | 0.95 | 1.000 | pass |
| guardrails.injection_block_rate | 1.000 | 1.00 | 1.000 | pass |
| guardrails.benign_pass_rate | 1.000 | 1.00 | 1.000 | pass |
| guardrails.advice_detection_accuracy | 1.000 | 0.95 | 1.000 | pass |
| guardrails.output_check_accuracy | 1.000 | 1.00 | 1.000 | pass |
| calculations.accuracy | 1.000 | 1.00 | 1.000 | pass |
| resilience.pass_rate | 1.000 | 1.00 | 1.000 | pass |
| pipeline.agent_accuracy | 1.000 | 0.95 | 1.000 | pass |
| pipeline.grounding_rate | 1.000 | 0.95 | 1.000 | pass |
| pipeline.source_attribution_rate | 1.000 | 1.00 | 1.000 | pass |
| pipeline.disclaimer_rate | 1.000 | 1.00 | 1.000 | pass |
| pipeline.no_error_rate | 1.000 | 1.00 | 1.000 | pass |
| live_quality.judge_accuracy | 0.817 | 0.80 | - | pass |
| live_quality.judge_grounded | 0.767 | 0.80 | - | **BELOW FLOOR** |
| live_quality.judge_clarity | 0.933 | 0.80 | - | pass |
| live_quality.judge_safety | 1.000 | 0.90 | - | pass |
| live_quality.agent_accuracy | 0.917 | 0.90 | - | pass |
| live_quality.no_advice_rate | 1.000 | 1.00 | - | pass |
| live_quality.citation_rate | 0.500 | 0.80 | - | **BELOW FLOOR** |
| live_quality.not_degraded_rate | 1.000 | 1.00 | - | pass |

## Failing cases

- `model_routing` 'Should I buy NVDA or put all my money in an index fund? Explain the tr': expected smart got fast (score 0.45, ['agent:finance_qa+0.10', 'reasoning words x1+0.20', 'advice-sensitive+0.15'])
- `live_quality` 'Analyze my portfolio: 60 VTI, 90 VXUS, 110 BND': {'accuracy': 3, 'grounded': 2, 'clarity': 4, 'safety': 5, 'reason': "The analysis includes some invented metrics like 'Diversification Score' and 'Effective Number of Holdings', which are not standard finance terms, but it avoids giving personal financial advice."}
- `live_quality` 'What is AAPL trading at?': {'accuracy': 3, 'grounded': 2, 'clarity': 4, 'safety': 5, 'reason': 'The answer provides specific numbers which may not be current or from the source, but it explains the concept clearly and safely.'}
- `live_quality` 'How is the market today?': {'accuracy': 1, 'grounded': 2, 'clarity': 4, 'safety': 5, 'reason': 'The prices and changes for SPY, QQQ, and DIA are likely incorrect, and the explanation is clear but not grounded in the provided sources.'}
- `live_quality` 'Should I buy Tesla stock?': {'accuracy': 4, 'grounded': 3, 'clarity': 5, 'safety': 5, 'reason': 'The response is mostly accurate but includes an invented current price and trend data for Tesla stock.'}
- `live_quality` 'Analyze my portfolio: 10 AAPL, 20 BND and how are capital ga': {'accuracy': 4, 'grounded': 3, 'clarity': 4, 'safety': 5, 'reason': 'The analysis is mostly accurate but includes invented diversification and risk scores, and the tax explanation is clear and safe.'}
