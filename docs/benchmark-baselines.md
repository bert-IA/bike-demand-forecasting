# Preliminary Benchmark — Baseline Models

## Objective and protocol

Compare point forecasts from three models on daily bike counts:

- last observed value;
- weekly seasonal naive;
- additive seasonal exponential smoothing with a seven-day cycle and no trend.

The backtest uses 13 validation cutoffs, from March 23 to September 7, 2026,
with a 14-day forecast horizon. The last validation window ends on
September 21, 2026. The final test period remains reserved.

The evaluated observations and seasonal references were verified to be
identical across the three models. Each series-cutoff pair has equal weight
in the averages below.

## Validation results

| Model | Mean MASE | Median MASE |
|---|---:|---:|
| Additive exponential smoothing | 1.094 | 0.891 |
| Seasonal naive | 1.210 | 1.045 |
| Last value | 1.339 | 1.284 |

MASE was calculable for all 1,305 series-cutoff pairs; 1,186 had all
14 forecast days available for evaluation. On these complete horizons,
mean MASE was 1.128, 1.259 and 1.407 respectively: the ranking was unchanged.

The mean MASE of exponential smoothing was lower than that of seasonal naive
on 10 of the 13 cutoffs. In pairwise comparisons, exponential smoothing
performed better on 862 pairs, seasonal naive on 441, with 2 ties.

On the last cutoff, mean MASE was 0.439 for exponential smoothing, 0.515
for seasonal naive and 0.993 for last value, across 104 pairs with
14 evaluable days.

## Limitations and next steps

- These results are preliminary: performance varies across periods and
  series-cutoff pairs are not independent.
- Exponential smoothing produced 36 negative forecasts out of 18,270,
  across 9 series-cutoff pairs. The raw forecasts were retained in the scores.
- Smoothing parameters were estimated automatically. Grid search may be
  explored later using an appropriate time-based tuning protocol.
- Prediction intervals, execution costs and the final test period remain
  to be evaluated. No champion has been selected at this stage.
- This summary reports an interactive exploration; a reproducible execution
  workflow remains to be formalized.
