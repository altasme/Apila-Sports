# Apila Sports

Backtest-first sports prediction & betting-edge validation platform. See [`docs/sports-prediction-proposal-v2.md`](docs/sports-prediction-proposal-v2.md) for the project proposal.

## Status

**M0 — point-in-time feature store (basketball first): done.**
**M1 — historical closing odds + devig: done.**
**M2 — prediction engine + fitted probability mapping: done.**
**M3 — betting simulation + baselines: done.**

`apila/` holds the store: `team_game_logs` for stats, `closing_odds` for
the market benchmark. `PointInTimeStore` only ever answers "team rating as
of date D" by filtering on `game_date < D` — there's no separate ratings
table to accidentally read past its valid date. See
`tests/test_point_in_time.py` for the leakage tests, including an
adversarial one that checks the result against what it *would* be if a
same-day game had leaked in.

`apila/odds.py` converts American odds to devigged implied probability
(`devig_moneyline`), and `PointInTimeStore.market_probability(date, home,
away, sport)` joins a historical game to its closing line — the number
every model prediction has to beat. See `tests/test_odds.py` and
`tests/test_market_probability.py`.

`apila/rating.py` combines season and recent-form point differential into
a composite team rating (weights are priors, not truth — see the module
docstring). `apila/calibration.py` fits `ProbabilityMapping` — rating diff
→ win probability — via logistic regression on labeled `(rating_diff,
home_win)` pairs, then freezes it; nothing hand-tunes this curve.
`apila/prediction.py`'s `PredictionEngine` combines a store and a frozen
mapping into `.predict(home, away, as_of)`. See `tests/test_rating.py`,
`tests/test_calibration.py`, `tests/test_prediction.py`.

`apila/baselines.py` fits the "better record" baseline with the same
logistic machinery as the real engine (never hand-drawn) and computes the
"home always" baseline as the training set's actual home-win rate.
`apila/metrics.py` scores every candidate — model, baselines, and the
devigged market — on accuracy, Brier score, and log loss, so a model can't
look good just by picking favorites (proposal section 3).
`apila/simulate.py` runs the flat-stake betting simulation: bet a side
only when its model probability beats its break-even price, track ROI
overall and by confidence tier (`apila/tiers.py`). `apila/backtest.py`
assembles the per-game records (rating diff, actual outcome, closing
odds) that both the fitting script and the backtest script walk.
`scripts/run_backtest.py` ties it all together and prints the report. See
`tests/test_baselines.py`, `tests/test_metrics.py`, `tests/test_tiers.py`,
`tests/test_simulate.py`, `tests/test_backtest.py`.

**CLV is deliberately not reported.** The store only ever holds the
closing price (M1 ingests closing odds specifically), and CLV requires
comparing an entry price against that close — there's no entry price in a
backtest built this way. Real CLV tracking is a Phase 1 concern, once
predictions lock live before a game's market actually closes.

## Multi-sport support

The schema and every store/rating/prediction method are sport-scoped:
`team_id` is only unique *within* a sport, so `sport` is a required
argument almost everywhere (`store.team_rating_asof(team_id, sport,
as_of)`, etc.) — this is deliberate, not an oversight, since silently
mixing one sport's games into another's rating would be a much worse bug
than a required parameter.

Two market shapes are supported:
- **Two-outcome** (basketball, MLB moneyline): `wl` is `W`/`L`,
  `ClosingOdds.draw_moneyline` is `NULL`, `PredictionEngine` +
  `ProbabilityMapping` produce a single home-win probability.
- **Three-outcome** (soccer 1X2): `wl` can be `W`/`D`/`L`,
  `ClosingOdds.draw_moneyline` is populated, `ThreeWayPredictionEngine` +
  `ThreeWayProbabilityMapping` (multinomial logistic regression) produce
  `(home, draw, away)` probabilities. `apila/odds.py`'s `devig_multiplicative`
  generalizes to any number of outcomes, and `apila/simulate.py`'s betting
  sim takes a game as a list of sides, so it's N-way from the start —
  basketball just happens to pass two.

Test fixtures cover both: `tests/fixtures/sample_game_logs.csv` /
`sample_closing_odds.csv` (`sport="nba"`, no draws) and
`sample_soccer_matches.csv` / `sample_soccer_odds.csv` (`sport="soccer"`,
includes draws and all three outcomes). The `store` fixture in
`tests/conftest.py` loads both into one store to exercise sport isolation.

## Setup

```bash
pip install -e .[dev]
pytest
```

## Ingesting NBA data

```bash
pip install -e .[ingest]
python scripts/ingest_nba_games.py --seasons 2021 2022 2023 --api-key YOUR_KEY
```

Pulls from [balldontlie.io](https://app.balldontlie.io) (free API key
required). This originally used `stats.nba.com` via `nba_api`, but that
API blocks or silently hangs on requests from cloud/datacenter IPs (AWS,
GCP, Azure) as anti-scraping protection — it doesn't error, it just times
out, which makes it unusable from GitHub Codespaces, Actions, or most
cloud sandboxes (confirmed against this repo's own dev sandbox and a real
Codespaces run). balldontlie works from those environments.

The tradeoff: balldontlie's `games` endpoint only has date, teams, and
final score — no team box score stats (FGM/FGA/REB/AST/etc.), so those
columns stay `NULL`. That's fine for everything currently computed here
(`apila/rating.py` only uses points and win/loss) but limits future
shooting-stat features from this source. Tags every row `sport="nba"`.

Note `--seasons` takes season-*start* years now (`2023` = the 2023-24
season), not the `"2023-24"` string format the old stats.nba.com version
used. Team ids also come from balldontlie's own scheme — don't mix in
rows from an old nba_api-based ingestion for the same `sport="nba"`, or
the same real-world team will get split across two different `team_id`s.

Confirmed working end-to-end (auth header, field names, cursor pagination
all correct) against a real Codespaces run. The free tier is rate-limited
to 5 requests/minute, so a full ~1,230-game season (13 pages) takes a few
minutes — the script paces itself at ~13s between pages and backs off
further on a 429 rather than failing. That's expected, not stuck.

There's no equivalent live-ingestion script for soccer yet — the data
source is undetermined (see proposal section 4.2's options), so soccer
fixtures for now are the hand-built test data in `tests/fixtures/`. The
store, rating, calibration, and simulation code are already sport-generic
and ready for it once a source is picked.

## Ingesting closing odds

```bash
python scripts/ingest_closing_odds.py path/to/odds.csv --sport nba --source kaggle-nba-odds
python scripts/ingest_closing_odds.py path/to/1x2.csv --sport soccer --source my-soccer-odds
```

There's no free, reliable historical-odds API, so this takes a CSV
(`game_date, home_team_abbr, away_team_abbr, home_moneyline,
away_moneyline`, plus optional `draw_moneyline` for a three-outcome
market) sourced from wherever you land per proposal section 4.2 — a paid
API export, a public dataset, or a manually assembled file. Team
abbreviations must match what's in `team_game_logs` for the same
`--sport` or the join in `market_probability()` won't find the game.

## Fitting the probability mapping

```bash
python scripts/fit_probability_mapping.py --sport nba --before 2023-10-01 \
    --engine-version v1.0 --out apila/mappings/nba_v1_0.json

python scripts/fit_probability_mapping.py --sport soccer --before 2023-10-01 \
    --three-way --engine-version v1.0 --out apila/mappings/soccer_v1_0.json
```

Walks every game for that sport, computes each team's rating as-of that
game's date (so a game can never leak into its own training label), and
fits/freezes the mapping on games strictly before `--before`. Point
`--before` at the start of whatever season you're holding out — see
proposal section 4.7. Don't re-run this for a given engine version once
you start evaluating against held-out data. `--three-way` fits the
soccer-shaped multinomial mapping instead of the binary one.

## Running a backtest

```bash
python scripts/run_backtest.py --sport nba --before 2023-10-01 \
    --mapping apila/mappings/nba_v1_0.json --stake 1.0

python scripts/run_backtest.py --sport soccer --before 2023-10-01 \
    --three-way --mapping apila/mappings/soccer_v1_0.json
```

Evaluates only the holdout set (games on or after `--before`); fits the
naive baselines on the train split with the same cutoff so nothing leaks.
Prints accuracy/Brier/log loss for the model against `home_always`,
`better_record`, and the devigged market, then runs the flat-stake betting
simulation and reports ROI overall and by confidence tier. This is
proposal section 4.9's metrics dashboard, and section 4.10's Go/No-Go gate
is just reading these numbers honestly: ROI ≤ 0, or the model losing to
the naive baselines, means stop.
