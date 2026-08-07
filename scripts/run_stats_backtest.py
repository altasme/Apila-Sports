"""Stats-only backtest: prediction quality against naive baselines, no odds required.

This is a deliberately narrower cousin of run_backtest.py. It answers "is
the engine's rating-diff -> probability mapping actually predictive?" --
accuracy, Brier score, and log loss against the home_always and
better_record baselines -- without touching market odds at all.

What it does NOT answer: whether the engine makes money. That needs
closing odds (market comparison, devigging, ROI, the actual betting
simulation) -- see run_backtest.py. Proposal section 2's real success
criteria require the market comparison; this script is a lighter,
odds-free checkpoint for when odds coverage doesn't reach the games you
want to evaluate, not a substitute for the real gate.

Usage:
    python scripts/run_stats_backtest.py --sport nba --before 2025-10-01 \
        --mapping apila/mappings/nba_v1_0.json
    python scripts/run_stats_backtest.py --sport soccer --before 2025-10-01 \
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
from apila.store import PointInTimeStore  # noqa: E402


def run_binary(store: PointInTimeStore, sport: str, before, mapping_path: str) -> None:
    train = build_game_records(store, sport, until=before, require_odds=False)
    holdout = build_game_records(store, sport, since=before, require_odds=False)

    print(f"\n=== {sport} stats-only backtest (binary): {len(train)} train / {len(holdout)} holdout games ===\n")
    if not holdout:
        print("No holdout games with rating history -- nothing to evaluate.")
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

    outcomes = [r.outcome == "H" for r in holdout]
    model_probs = [mapping.predict_proba(r.rating_diff) for r in holdout]
    home_always_probs = [home_rate] * len(holdout)
    better_record_probs = [
        record_mapping.predict_proba(r.record_diff) if record_mapping else 0.5 for r in holdout
    ]

    print(f"{'':20}{'accuracy':>10}{'brier':>10}{'log_loss':>10}")
    for name, probs in [
        ("model", model_probs),
        ("home_always", home_always_probs),
        ("better_record", better_record_probs),
    ]:
        print(
            f"{name:20}{accuracy(probs, outcomes):>10.3f}"
            f"{brier_score(probs, outcomes):>10.4f}{log_loss(probs, outcomes):>10.4f}"
        )

    with_odds = sum(1 for r in holdout if r.has_odds)
    print(f"\n({with_odds}/{len(holdout)} holdout games also had matched closing odds)")


def run_three_way(store: PointInTimeStore, sport: str, before, mapping_path: str) -> None:
    train = build_game_records(store, sport, until=before, require_odds=False)
    holdout = build_game_records(store, sport, since=before, require_odds=False)

    print(f"\n=== {sport} stats-only backtest (three-way): {len(train)} train / {len(holdout)} holdout games ===\n")
    if not holdout:
        print("No holdout games with rating history -- nothing to evaluate.")
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
    outcomes = [outcome_index[r.outcome] for r in holdout]
    model_probs = [mapping.predict_proba(r.rating_diff) for r in holdout]
    naive_probs = [naive_freq] * len(holdout)
    better_record_probs = [
        record_mapping.predict_proba(r.record_diff) if record_mapping else (1 / 3, 1 / 3, 1 / 3)
        for r in holdout
    ]

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

    with_odds = sum(1 for r in holdout if r.has_odds)
    print(f"\n({with_odds}/{len(holdout)} holdout games also had matched closing odds)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sport", required=True, help='e.g. "nba", "soccer"')
    parser.add_argument("--before", required=True, help="Holdout starts here (ISO date)")
    parser.add_argument("--three-way", action="store_true", help="Use the multiclass path (soccer)")
    parser.add_argument("--mapping", required=True, help="Path to a frozen mapping JSON")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    before = datetime.strptime(args.before, "%Y-%m-%d").date()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    if args.three_way:
        run_three_way(store, args.sport, before, args.mapping)
    else:
        run_binary(store, args.sport, before, args.mapping)


if __name__ == "__main__":
    main()
