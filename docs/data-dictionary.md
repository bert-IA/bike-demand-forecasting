# Data dictionary: series preparation

Three tables produced by `forecasting.data.prepare_series`, in this order:

1. `compute_quality`: measures each series (quality table).
2. `select_series`: decides which series are kept (selection table).
3. `build_daily_grid`: builds the gap-free daily grid of the kept series.

## 1. Quality table

Output of `compute_quality`: one row per counter
(`series_id`), computed from `bike_daily.parquet` (one row per `series_id` and day).
The function only **measures**; keeping or discarding a series is decided elsewhere.

## Reference conventions

Every column is measured against one of two references, and its name says which:

| Marker in the name | Reference |
|---|---|
| `_inside`, `inside_` | The series window: `first_timestamp` to `last_timestamp`, bounds included |
| `_at_start`, `_at_end`, `dataset_` | The dataset window: earliest to latest day of the whole dataset |

A day is *absent* when the series has no row for it. A zero is an observed value
(`y == 0`), not an absence.

## Columns

| Column | Type | Nullable | Reference | Description |
|---|---|---|---|---|
| `series_id` | String | no | - | Counter identifier (unique key). |
| `n_observations` | UInt32 | no | series | Number of rows (days with a measurement). |
| `first_timestamp` | Datetime(us) | no | series | First observed day. |
| `last_timestamp` | Datetime(us) | no | series | Last observed day. |
| `zero_days` | UInt32 | no | series | Days with `y == 0`, including leading zeros. |
| `first_positive_timestamp` | Datetime(us) | yes | series | First day with `y > 0`. Null means the counter never recorded a positive value (dead counter). |
| `calendar_span_days` | Int64 | no | series | Calendar days from `first_timestamp` to `last_timestamp`, bounds included. |
| `missing_days_inside` | Int64 | no | series | Absent days between `first_timestamp` and `last_timestamp`: `calendar_span_days - n_observations`. |
| `inside_coverage` | Float64 | no | series | `n_observations / calendar_span_days`. Reliability while the counter is running (0 to 1). |
| `days_missing_at_start` | Int64 | no | dataset | Days between the dataset start and `first_timestamp` (late start). |
| `days_missing_at_end` | Int64 | no | dataset | Days between `last_timestamp` and the dataset end (early stop). |
| `dataset_coverage` | Float64 | no | dataset | `n_observations / full_span_days`. Share of the dataset window covered (0 to 1). |
| `days_before_first_positive` | Int64 | yes | series | Days between `first_timestamp` and `first_positive_timestamp` (leading zeros). Null for a dead counter. |
| `longest_gap` | Int64 | no | series | Longest run of consecutive absent days; 0 if none. |
| `longest_zero_run` | UInt32 | no | series | Longest run of consecutive zero days from `first_positive_timestamp` onwards. An absent day breaks a run. 0 if none, and 0 for a dead counter. |

## Identity

For every series, with `full_span_days` the number of days of the dataset window:

```
days_missing_at_start + missing_days_inside + days_missing_at_end + n_observations = full_span_days
```

This is checked in `tests/test_prepare_series.py`.

## Observed on the real dataset

Dataset window 2025-09-25 to 2026-10-24 (`full_span_days = 395`), 113 series.

| Column | Min | Max | Nulls |
|---|---|---|---|
| `n_observations` | 147 | 392 | 0 |
| `zero_days` | 0 | 321 | 0 |
| `missing_days_inside` | 0 | 27 | 0 |
| `days_missing_at_start` | 0 | 1 | 0 |
| `days_missing_at_end` | 0 | 247 | 0 |
| `days_before_first_positive` | 0 | 180 | 1 |
| `longest_gap` | 0 | 24 | 0 |
| `longest_zero_run` | 0 | 23 | 0 |
| `dataset_coverage` | 0.372 | 0.992 | 0 |

## Known limits

- A partial day (a few hours of data) appears as a low value, not as an absence.
- `longest_zero_run` also counts zeros at the end of a series; trailing zeros are not
  separated from mid-series zeros.
- `longest_zero_run` is 0 for a dead counter because it has no active part; check
  `first_positive_timestamp` for that case.

## 2. Selection table

Output of `select_series(quality, *, max_gap_days, max_days_missing_at_end, min_useful_days)`:
one row per series, sorted by `series_id`. Thresholds default to `forecasting.config`
(`MAX_FILLABLE_GAP_DAYS`, `MAX_DAYS_MISSING_AT_END`, `MIN_USEFUL_DAYS`) and all comparisons
are strict.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `series_id` | String | no | Counter identifier. |
| `dead` | Boolean | no | `first_positive_timestamp` is null: never a positive value. |
| `early_stop` | Boolean | no | `days_missing_at_end > max_days_missing_at_end`. |
| `large_gap` | Boolean | no | `longest_gap > max_gap_days`: too long to interpolate. |
| `too_short` | Boolean | no | Useful days `< min_useful_days`, with useful days = `calendar_span_days - days_before_first_positive`. False for a dead counter (already flagged `dead`). |
| `kept` | Boolean | no | True only if none of the four reasons applies. |

A series can have several reasons. Real dataset: 105 kept, 8 discarded (1 dead, 4 early stop,
3 large gap; two early-stop series are also too short).

## 3. Daily grid

Output of `build_daily_grid(daily, selection)`: one row per kept series and calendar day,
sorted by `series_id`, `timestamp`.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `series_id` | String | no | Counter identifier. |
| `timestamp` | Datetime(us) | no | Day. |
| `y` | Float64 | no | Observed count, or linear interpolation when `is_imputed`. |
| `is_imputed` | Boolean | no | The day was absent from the source and `y` was reconstructed. Never evaluated in backtests. |
| `is_long_zero_run` | Boolean | no | Observed zero belonging to a run of more than `LONG_ZERO_RUN_DAYS` consecutive observed zeros. Imputed days never count. |

Rules:

- Each series starts at its first positive day (leading zeros are cut) and ends at its own
  `last_timestamp` (no extrapolation).
- `is_imputed` is computed before interpolation; interpolation is per series.
- An observed zero is kept and is not imputed.

Real dataset: 41,152 rows, 105 series, 347 imputed days, 51 days flagged `is_long_zero_run`
(3 series: 23, 14 and 14 days).
