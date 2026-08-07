# Apila Sports

Backtest-first sports prediction & betting-edge validation platform. See [`docs/sports-prediction-proposal-v2.md`](docs/sports-prediction-proposal-v2.md) for the project proposal.

## Status

**M0 — point-in-time feature store (basketball first): in progress.**

`apila/` holds the store: a single `team_game_logs` table plus a
`PointInTimeStore` that only ever answers "team rating as of date D" by
filtering on `game_date < D`. There's no separate ratings table to
accidentally read past its valid date. See `tests/test_point_in_time.py`
for the leakage tests, including an adversarial one that checks the result
against what it *would* be if a same-day game had leaked in.

## Setup

```bash
pip install -e .[dev]
pytest
```

## Ingesting NBA data

```bash
pip install -e .[ingest]
python scripts/ingest_nba_games.py --seasons 2021-22 2022-23 2023-24
```

Pulls from `stats.nba.com` via `nba_api` — needs network access to that
host, which isn't available from every environment (e.g. this repo's own
dev sandbox can't reach it). Run it somewhere that can.
