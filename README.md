# bike-demand-forecasting

Production-oriented time series forecasting platform on Paris bike counters:
rigorous model benchmark (baselines, GBM, deep learning) and a decision-support dashboard.

> Status: work in progress (milestone M0 - foundations).

## Goals
- Compare model families with a frozen, leakage-free rolling-origin protocol.
- Select a research champion and a deployable champion per hardware scenario.
- Serve forecasts through an API and a French-language dashboard.

## Data sources
- Paris open data bike counters, Open-Meteo weather, French calendar.

## Repository layout
See `docs/` for architecture decision records (ADR).

## License
MIT
