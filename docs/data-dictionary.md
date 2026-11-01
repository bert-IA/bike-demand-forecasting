# Data dictionary: series quality table

Output of `forecasting.data.prepare_series.compute_quality`: one row per counter
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
