"""Run the Phase 0 backtest (M3): baselines, betting simulation, metrics.

This is proposal section 4.9's metrics dashboard as script output. Pass a
mapping already frozen by fit_probability_mapping.py on games strictly
before --before; this script evaluates only the holdout set (games on or
after --before) and fits the two naive baselines (home-always,
better-record) on the identical train/holdout split so nothing here leaks
either.

Two-outcome sports (basketball, MLB moneyline) use accuracy/Brier/log
loss and a binary betting sim. Pass --three-way for soccer to use the
multiclass equivalents and a three-side (home/draw/away) sim instead.

CLV note: this script does NOT report closing-line value. The store only
ever holds one price per game -- the closing line itself (see M1) -- and
CLV requires comparing an entry price against that close. Genuine CLV
tracking is a Phase 1 concern, once predictions lock before a game's
market actually closes.

Usage:
    python scripts/run_backtest.py --sport nba --before 2023-10-01 \
        --mapping apila/mappings/nba_v1_0.json --stake 1.0
    python scripts/run_backtest.py --sport soccer --before 2023-10-01 \
        --three-way --mapping apila/mappings/soccer_v1_0.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apila.backtest import build_game_records  # noqa: E402
from apila.baselines import (  # noqa: E402
    fit_better_record_baseline,
    fit_better_record_baseline_three_way,
    home_win_rate,
    outcome_frequency_baseline,
)
from apila.calibration import ProbabilityMapping, ThreeWayProbabilityMapping  # noqa: E402
from apila.db import get_engine, get_session  # noqa: E402
from apila.metrics import (  # noqa: E402
    accuracy,
    brier_score,
    log_loss,
    multiclass_accuracy,
    multiclass_brier_score,
    multiclass_log_loss,
)
from apila.odds import devig_moneyline  # noqa: E402
from apila.simulate import GameBet, SideBet, simulate_flat_stake, summarize, summarize_by_tier  # noqa: E402
from apila.store import PointInTimeStore  # noqa: E402
from apila.tiers import confidence_tier  # noqa: E402

TIER_ORDER = ("high", "moderate", "low")


def run_binary_backtest(store: PointInTimeStore, sport: str, before, mapping_path: str, stake: float) -> None:
    train = build_game_records(store, sport, until=before)
    holdout = build_game_records(store, sport, since=before)

    print(f"\n=== {sport} backtest (binary): {len(train)} train / {len(holdout)} holdout games ===\n")
    if not holdout:
        print("No holdout games with both a rating and closing odds -- nothing to evaluate.")
        return

    mapping = ProbabilityMapping.from_json(mapping_path)

    train_outcomes = [r.outcome for r in train]
    home_rate = home_win_rate(train_outcomes) if train_outcomes else 0.5
    record_mapping = None
    if train and len({o == "H" for o in train_outcomes}) > 1:
        record_mapping = fit_better_record_baseline(
            [r.record_diff for r in train],
            [r.outcome == "H" for r in train],
            trained_on=f"{sport} games before {before}",
        )

    outcomes: list[bool] = []
    model_probs: list[float] = []
    home_always_probs: list[float] = []
    better_record_probs: list[float] = []
    market_probs: list[float] = []
    games: list[GameBet] = []

    for r in holdout:
        home_win = r.outcome == "H"
        outcomes.append(home_win)

        model_prob = mapping.predict_proba(r.rating_diff)
        model_probs.append(model_prob)
        home_always_probs.append(home_rate)
        better_record_probs.append(
            record_mapping.predict_proba(r.record_diff) if record_mapping else 0.5
        )
        market_home, _ = devig_moneyline(r.home_moneyline, r.away_moneyline)
        market_probs.append(market_home)

        games.append(
            GameBet(
                game_id=r.game_id,
                game_date=r.game_date,
                sides=[
                    SideBet("home", model_prob, r.home_moneyline, home_win),
                    SideBet("away", 1 - model_prob, r.away_moneyline, not home_win),
                ],
            )
        )

    _print_score_table(
        {
            "model": model_probs,
            "home_always": home_always_probs,
            "better_record": better_record_probs,
            "market (devigged)": market_probs,
        },
        outcomes,
    )
    _run_and_print_sim(games, stake)


def run_three_way_backtest(store: PointInTimeStore, sport: str, before, mapping_path: str, stake: float) -> None:
    train = build_game_records(store, sport, until=before)
    holdout = build_game_records(store, sport, since=before)

    print(f"\n=== {sport} backtest (three-way): {len(train)} train / {len(holdout)} holdout games ===\n")
    if not holdout:
        print("No holdout games with both a rating and closing odds -- nothing to evaluate.")
        return

    mapping = ThreeWayProbabilityMapping.from_json(mapping_path)

    train_outcomes = [r.outcome for r in train]
    naive_freq = outcome_frequency_baseline(train_outcomes) if train_outcomes else (1 / 3, 1 / 3, 1 / 3)
    record_mapping = None
    if train and {"H", "D", "A"} <= set(train_outcomes):
        record_mapping = fit_better_record_baseline_three_way(
            [r.record_diff for r in train],
            train_outcomes,
            trained_on=f"{sport} games before {before}",
        )

    outcome_index = {"H": 0, "D": 1, "A": 2}
    outcomes: list[int] = []
    model_probs: list[tuple[float, float, float]] = []
    naive_probs: list[tuple[float, float, float]] = []
    better_record_probs: list[tuple[float, float, float]] = []
    games: list[GameBet] = []

    for r in holdout:
        outcomes.append(outcome_index[r.outcome])

        home_p, draw_p, away_p = mapping.predict_proba(r.rating_diff)
        model_probs.append((home_p, draw_p, away_p))
        naive_probs.append(naive_freq)
        better_record_probs.append(
            record_mapping.predict_proba(r.record_diff) if record_mapping else (1 / 3, 1 / 3, 1 / 3)
        )

        games.append(
            GameBet(
                game_id=r.game_id,
                game_date=r.game_date,
                sides=[
                    SideBet("home", home_p, r.home_moneyline, r.outcome == "H"),
                    SideBet("draw", draw_p, r.draw_moneyline, r.outcome == "D"),
                    SideBet("away", away_p, r.away_moneyline, r.outcome == "A"),
                ],
            )
        )

    print(f"{'':20}{'accuracy':>10}{'brier':>10}{'log_loss':>10}")
    for name, probs in [
        ("model", model_probs),
        ("outcome_freq", naive_probs),
        ("better_record", better_record_probs),
    ]:
        print(
            f"{name:20}{multiclass_accuracy(probs, outcomes):>10.3f}"
            f"{multiclass_brier_score(probs, outcomes):>10.4f}"
            f"{multiclass_log_loss(probs, outcomes):>10.4f}"
        )

    _run_and_print_sim(games, stake)


def _print_score_table(probs_by_name: dict[str, list[float]], outcomes: list[bool]) -> None:
    print(f"{'':20}{'accuracy':>10}{'brier':>10}{'log_loss':>10}")
    for name, probs in probs_by_name.items():
        print(
            f"{name:20}{accuracy(probs, outcomes):>10.3f}"
            f"{brier_score(probs, outcomes):>10.4f}{log_loss(probs, outcomes):>10.4f}"
        )


def _run_and_print_sim(games: list[GameBet], stake: float) -> None:
    outcomes = simulate_flat_stake(games, stake=stake)
    summary = summarize(outcomes)
    print(
        f"\nBetting sim: {summary.n_bets} bets, staked {summary.total_staked:.1f}, "
        f"profit {summary.total_profit:+.2f}, ROI {summary.roi:+.1%}"
    )

    by_tier = summarize_by_tier(outcomes, lambda o: confidence_tier(o.model_prob))
    for tier in TIER_ORDER:
        if tier in by_tier:
            s = by_tier[tier]
            print(f"  {tier:10} {s.n_bets:4} bets  ROI {s.roi:+.1%}")

    print(
        "\nNote: CLV is not reported here -- the store only holds the "
        "closing price (see M1), and CLV needs an entry price to compare "
        "it against. That's a Phase 1 concern once predictions lock live."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sport", required=True, help='e.g. "nba", "soccer"')
    parser.add_argument("--before", required=True, help="Holdout starts here (ISO date)")
    parser.add_argument("--three-way", action="store_true", help="Use the multiclass path (soccer)")
    parser.add_argument("--mapping", required=True, help="Path to a frozen mapping JSON")
    parser.add_argument("--stake", type=float, default=1.0, help="Flat stake per bet")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    before = datetime.strptime(args.before, "%Y-%m-%d").date()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    if args.three_way:
        run_three_way_backtest(store, args.sport, before, args.mapping, args.stake)
    else:
        run_binary_backtest(store, args.sport, before, args.mapping, args.stake)


if __name__ == "__main__":
    main()
