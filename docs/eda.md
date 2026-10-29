# Exploratory Data Analysis — Paris Bike Counts

## Purpose

The dataset spans approximately 13 months. This window comes from the source,
not from the ingestion code: the current Paris Open Data dataset is a sliding
window of about 13 months, and no date filter is applied. Yearly archives
(2016-2025) are published separately and can extend the history in a later
phase, after checking schema and counter-identifier consistency. Counter
coverage also varies across dates.

The detailed analysis is available in
[`notebooks/01_eda_bike_counts.ipynb`](../notebooks/01_eda_bike_counts.ipynb).

## Dataset

The source is the Paris Open Data bike-counter export. Hourly Parquet records
were converted to daily counts in the `Europe/Paris` timezone.

- Daily observations: **43,338**
- Counters: **113**
- Date range: **2025-09-25 to 2026-10-24**
- Observed days per counter: median **391**, mean **383.5**, range **147–392**

The number of observed days varies across counters. A shorter series may reflect
a later start, missing dates, or both.

## Data quality and coverage

The raw hourly dataset contains **1,389 null values out of 1,037,729 rows**
(**0.134%**). No `NaN` values were found. Nulls represent missing values and
must not be treated as observed zero counts.

Three counters with long zero-count periods were inspected:

- `100003098-101003098` has 321 observed daily rows, all with a daily count of
  zero, from 2025-09-26 to 2026-08-12. The hourly records inspected contain
  recorded zeros rather than nulls, although some dates have incomplete or
  missing hourly timestamps. This is evidence of a prolonged zero-count
  period, not proof of a counter failure.
- `300030116-353354957` has 180 consecutive zero-count days before positive
  counts appear. Three calendar dates are absent within its overall observed
  period.
- `300030921-353361113` has 54 consecutive zero-count days before positive
  counts appear. Three calendar dates are absent within its overall observed
  period.

These cases are targeted findings, not a general exclusion rule for all
counters. Missing dates, missing hourly timestamps, null values, and observed
zeros are distinct situations.

## Distribution and counter scales

The distribution of mean daily counts is right-skewed: most counters have
lower average counts, while a smaller number have substantially higher
averages. Counters therefore operate at different traffic scales.

High daily counts are candidates for investigation, not automatically errors.
No observations were removed solely because they were large. The daily sum
across counters is also affected by changing counter coverage and is used only
as an exploratory visualization, not as a fixed-network demand series.

## Temporal patterns

### Weekly profile

After normalizing each counter by its own mean, the median relative profile
across 112 counters is:

| Day | Median relative count |
|---|---:|
| Monday | 1.083 |
| Tuesday | 1.170 |
| Wednesday | 1.114 |
| Thursday | 1.148 |
| Friday | 1.005 |
| Saturday | 0.785 |
| Sunday | 0.674 |

The observed profile is lower on weekends, especially Sunday. This is a
descriptive pattern in the observed counter network; it does not establish a
cause or imply that every counter has the same pattern.

### Monthly profile and annual seasonality

The monthly profile varies, with higher relative counts observed in September
and lower values in January and December. These results are exploratory:
approximately 13 months of data provide only one annual cycle, and most months
are represented in only one year. Annual seasonality cannot be validated from
this dataset alone. Weather, holidays, trend, and year effects cannot be
separated using these counts alone.

## Implications for the forecasting POC

- Include a **weekly seasonal baseline** in the model comparison.
- Evaluate models with time-ordered backtesting; the descriptive EDA is not a
  substitute for out-of-sample evaluation.
- Do not replace missing observations with zero by default.
- Do not remove high counts automatically.
- Treat the all-zero counter as a candidate for exclusion from the forecasting
  benchmark because it has no observed variation over its recorded period;
  document this as a POC scope decision, not as a confirmed diagnosis of
  equipment failure.
- Treat annual seasonality as unvalidated until a longer, comparable history
  is available.

## Limitations

The dataset spans approximately 13 months, and counter coverage varies across
dates. The analysis is descriptive and does not establish causal effects of
weather, holidays, commuting, or leisure activity. The results describe
observations at the included counters, not a complete measure of all cycling
activity in Paris.

## Reproducibility

The EDA notebook contains the code and detailed outputs. Key charts are
available in [`figures/`](figures/). The notebook should run from top to bottom
with the project environment before the report is updated.
