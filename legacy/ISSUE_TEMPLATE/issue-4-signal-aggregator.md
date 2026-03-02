---
name: "\U0001F4C8 Signal Aggregator"
about: Track the multi-horizon probability aggregator feature
title: "Issue 4 - Signal Aggregator"
labels: enhancement
assignees: ''
---

## Summary
Implement a signal aggregator that converts multi-horizon probabilities into a single trading decision and expose `get_latest_signal` for realtime and exit watchdog loops.

## Acceptance Criteria
- `aggregate_signal` combines `(h,t) -> prob` maps with weighting or majority vote
- `get_latest_signal` returns the latest aggregated signal or `None` when data/model missing
- Realtime loop and exit watchdog integrate the aggregator
- Configuration options `strategy.enter_threshold`, `strategy.aggregator_method`, and `strategy.weight_fn` documented in `README.md`

## Testing
- `python - <<'PY'
from csp.strategy.aggregator import aggregate_signal
prob_map = {(2,0.2):0.78,(4,0.5):0.81,(8,1.0):0.62,(16,0.5):0.88}
print(aggregate_signal(prob_map, enter_threshold=0.75, method="max_weighted"))
PY`
- `PYTHONPATH=. python - <<'PY'
from scripts.realtime_loop import run_once
run_once('csp/configs/strategy.yaml')
PY`

