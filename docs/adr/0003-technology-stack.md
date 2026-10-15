# ADR-0003: Technology stack

- Status: Accepted
- Date: 2026-10-15
- Deciders: project owner

## Context

The project is a production-oriented forecasting platform: a rigorous model
benchmark, a batch forecasting pipeline, a read API and a French-language
dashboard. The default deployment target is Scenario A (see ADR-0001), and
models are selected following ADR-0002. The stack must be reproducible,
maintainable by one person, and light enough for a small server.

## Decision

### Data and modelling

| Component | Choice | Alternative rejected | Reason |
|---|---|---|---|
| Language / packaging | Python 3.12, uv, lockfile | conda, poetry, pip-tools | One fast tool for interpreter, venv and lockfile; simple in Docker and CI |
| Dataframes / analytics | Polars, DuckDB | pandas only | Speed and memory on ~1M hourly rows; SQL on files without a server |
| Data validation | Pandera | none, Great Expectations | Lightweight data contract (`series_id, timestamp, y, covariates`) |
| Statistical models | statsforecast | statsmodels only, pmdarima | Fast baselines, ETS, ARIMA, native intervals, scales to many series |
| Gradient boosting | LightGBM via mlforecast (XGBoost, CatBoost for comparison) | single-series GBM | Global model across ~150 series; strong reference (ADR-0002) |
| Deep learning | neuralforecast (N-HiTS, TFT), PyTorch in an optional group | hand-written PyTorch, DeepAR | Maintained implementations; kept out of CI and production images |
| Foundation model | Chronos (optional) | none | Zero-shot reference, not a priority |
| Tuning | Optuna | GridSearchCV | Time-series aware search; GridSearchCV was a flaw of the initial project |
| Experiment tracking | MLflow with registry aliases | Weights & Biases, files | Self-hosted, free, aliases `champion-research` / `champion-prod-<scenario>` |

### Application and operations

| Component | Choice | Alternative rejected | Reason |
|---|---|---|---|
| Serving | Batch forecasts plus read API | online inference first | Daily horizon, <= 10 users; far cheaper, fits Scenario A |
| API | FastAPI, Pydantic, OpenAPI | Flask, Django | Typed contracts, generated OpenAPI used by the front end |
| Database | PostgreSQL (SQLite for demo), SQLAlchemy, Alembic | MongoDB, files only | Relational data, migrations, standard hosting |
| Front end | Next.js, TypeScript, Tailwind, shadcn/ui, ECharts | Streamlit, Dash | Professional dashboard, owner knows React; types generated from OpenAPI |
| Containers | Docker, Compose profiles `dev`, `prod-light`, `prod-full` | Kubernetes | Matches the scenarios; Kubernetes is over-engineered here |
| Reverse proxy | Caddy | Nginx, Traefik | Automatic HTTPS, minimal configuration |
| CI/CD | GitHub Actions, GHCR | GitLab CI, Jenkins | Repository already on GitHub; free for public repositories |
| Quality | Ruff, mypy strict, pytest, pre-commit | flake8 + black | Already in place; one fast tool for lint and format |
| Scheduling | cron in v1 | Prefect, Airflow | Enough for one nightly batch; Prefect is a possible later step |
| Weather / calendar | Open-Meteo, French calendar | paid APIs | No key; non-commercial licence to be re-checked before commercial use |

## Alternatives considered (global)

- **Managed cloud ML platform**: less control and a recurring cost, out of scope
  for a portfolio proof of concept.
- **All-in-one notebook plus Streamlit**: fast to build, but not representative
  of a production architecture.

## Consequences

- Deep-learning dependencies are isolated in an optional group, so CI and the
  Scenario A image stay small.
- Training runs on GPU through WSL2 (CUDA); production inference runs on CPU.
- Each choice can be replaced by a new ADR without touching the others,
  because the layers communicate through contracts (data schema, OpenAPI).
- Several tools are young and move fast; versions are pinned by the lockfile
  and upgrades go through CI.
