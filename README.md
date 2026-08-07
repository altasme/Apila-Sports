# Apila Sports

Backtest-first sports prediction & betting-edge validation platform. See [`docs/sports-prediction-proposal-v2.md`](docs/sports-prediction-proposal-v2.md) for the project proposal.

## Status

**M0 — point-in-time feature store (basketball first): done.**
**M1 — historical closing odds + devig: done.**
**M2 — prediction engine + fitted probability mapping: done.**

`apila/` holds the store: `team_game_logs` for stats, `closing_odds` for
the market benchmark. `PointInTimeStore` only ever answers "team rating as
of date D" by filtering on `game_date < D` — there's no separate ratings
table to accidentally read past its valid date. See
`tests/test_point_in_time.py` for the leakage tests, including an
adversarial one that checks the result against what it *would* be if a
same-day game had leaked in.

`apila/odds.py` converts American odds to devigged implied probability
(`devig_moneyline`), and `PointInTimeStore.market_probability(date, home,
away)` joins a historical game to its closing line — the number every
model prediction has to beat. See `tests/test_odds.py` and
`tests/test_market_probability.py`.

`apila/rating.py` combines season and recent-form point differential into
a composite team rating (weights are priors, not truth — see the module
docstring). `apila/calibration.py` fits `ProbabilityMapping` — rating diff
→ win probability — via logistic regression on labeled `(rating_diff,
home_win)` pairs, then freezes it; nothing hand-tunes this curve.
`apila/prediction.py`'s `PredictionEngine` combines a store and a frozen
mapping into `.predict(home, away, as_of)`. See `tests/test_rating.py`,
`tests/test_calibration.py`, `tests/test_prediction.py`.

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

## Ingesting closing odds

```bash
python scripts/ingest_closing_odds.py path/to/odds.csv --source kaggle-nba-odds
```

There's no free, reliable historical-odds API, so this takes a CSV
(`game_date, home_team_abbr, away_team_abbr, home_moneyline,
away_moneyline`) sourced from wherever you land per proposal section 4.2 —
a paid API export, a public dataset, or a manually assembled file. Team
abbreviations must match what's in `team_game_logs` or the join in
`market_probability()` won't find the game.

## Fitting the probability mapping

```bash
python scripts/fit_probability_mapping.py --before 2023-10-01 \
    --engine-version v1.0 --out apila/mappings/v1_0.json
```

Walks every game in `team_game_logs`, computes each team's rating as-of
that game's date (so a game can never leak into its own training label),
and fits/freezes the mapping on games strictly before `--before`. Point
`--before` at the start of whatever season you're holding out — see
proposal section 4.7. Don't re-run this for a given engine version once
you start evaluating against held-out data.
