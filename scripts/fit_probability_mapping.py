"""Fit and freeze the rating-diff -> win-probability mapping (M2).

Builds training examples by walking every game in team_game_logs for a
given sport: for a game on date D, the home/away rating is computed as-of
D (games strictly before D only -- see apila.store.PointInTimeStore), and
the label is the actual outcome. That's the walk-forward discipline the
proposal requires (section 4.7): a game's own rating can never leak into
its own label.

Pass --before the start of whatever season you're holding out. Fit here,
freeze the result, and don't re-run this script for that engine version
once you start evaluating -- retuning on eval data is how v1 manufactured
fake improvement.

Two-outcome sports (basketball, MLB moneyline) fit a binary
ProbabilityMapping. Pass --three-way for soccer, where a draw is a real
third outcome, to fit a ThreeWayProbabilityMapping instead.

Usage:
    python scripts/fit_probability_mapping.py --sport nba --before 2025-10-01 \
        --engine-version v1.0 --out apila/mappings/nba_v1_0.json
    python scripts/fit_probability_mapping.py --sport soccer --before 2025-10-01 \
        --three-way --engine-version v1.0 --out apila/mappings/soccer_v1_0.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from apila.calibration import fit_probability_mapping, fit_three_way_probability_mapping  # noqa: E402
from apila.db import get_engine, get_session  # noqa: E402
from apila.models import TeamGameLog  # noqa: E402
from apila.rating import rating_diff  # noqa: E402
from apila.store import PointInTimeStore  # noqa: E402


def build_training_examples(
    store: PointInTimeStore, sport: str, before: date
) -> tuple[list[float], list[str]]:
    """Returns (rating_diffs, outcomes), outcomes each "H"/"D"/"A" from the
    home team's perspective. Two-outcome sports simply never produce "D".
    """
    home_rows = (
        store.session.execute(
            select(TeamGameLog)
            .where(TeamGameLog.sport == sport)
            .where(TeamGameLog.is_home.is_(True))
            .where(TeamGameLog.game_date < before)
            .order_by(TeamGameLog.game_date)
        )
        .scalars()
        .all()
    )

    outcome_map = {"W": "H", "D": "D", "L": "A"}

    diffs: list[float] = []
    outcomes: list[str] = []
    for row in home_rows:
        away = store.session.execute(
            select(TeamGameLog).where(
                TeamGameLog.game_id == row.game_id,
                TeamGameLog.sport == sport,
                TeamGameLog.is_home.is_(False),
            )
        ).scalar_one_or_none()
        if away is None:
            continue

        diff = rating_diff(store, row.team_id, away.team_id, sport, row.game_date)
        if diff is None:
            continue  # not enough prior history yet for one of the teams

        diffs.append(diff)
        outcomes.append(outcome_map[row.wl])

    return diffs, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sport", required=True, help='e.g. "nba", "soccer"')
    parser.add_argument(
        "--before", required=True, help="Only use games strictly before this ISO date (train set)"
    )
    parser.add_argument(
        "--three-way",
        action="store_true",
        help="Fit a home/draw/away multinomial mapping instead of binary home-win",
    )
    parser.add_argument("--engine-version", required=True, help="e.g. v1.0")
    parser.add_argument("--out", required=True, help="Where to write the frozen mapping JSON")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    before = datetime.strptime(args.before, "%Y-%m-%d").date()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    diffs, outcomes = build_training_examples(store, args.sport, before)
    print(f"built {len(diffs)} training examples from {args.sport} games before {before}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.three_way:
        mapping = fit_three_way_probability_mapping(
            diffs,
            outcomes,
            engine_version=args.engine_version,
            trained_on=f"{args.sport} games before {before}",
        )
        mapping.to_json(args.out)
        print(
            f"froze three-way mapping to {args.out}: "
            f"home=({mapping.home_coef:.4f}, {mapping.home_intercept:.4f}) "
            f"draw=({mapping.draw_coef:.4f}, {mapping.draw_intercept:.4f}) "
            f"away=({mapping.away_coef:.4f}, {mapping.away_intercept:.4f})"
        )
    else:
        wins = [o == "H" for o in outcomes]
        mapping = fit_probability_mapping(
            diffs,
            wins,
            engine_version=args.engine_version,
            trained_on=f"{args.sport} games before {before}",
        )
        mapping.to_json(args.out)
        print(f"froze mapping to {args.out}: coef={mapping.coef:.4f} intercept={mapping.intercept:.4f}")


if __name__ == "__main__":
    main()
