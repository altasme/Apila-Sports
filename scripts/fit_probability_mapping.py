"""Fit and freeze the rating-diff -> win-probability mapping (M2).

Builds training examples by walking every game in team_game_logs: for a
game on date D, the home/away rating is computed as-of D (games strictly
before D only -- see apila.store.PointInTimeStore), and the label is
whether the home team won. That's the walk-forward discipline the
proposal requires (section 4.7): a game's own rating can never leak into
its own label.

Pass --before the start of whatever season you're holding out. Fit here,
freeze the result, and don't re-run this script for that engine version
once you start evaluating -- retuning on eval data is how v1 manufactured
fake improvement.

Usage:
    python scripts/fit_probability_mapping.py --before 2023-10-01 \
        --engine-version v1.0 --out apila/mappings/v1_0.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from apila.calibration import fit_probability_mapping  # noqa: E402
from apila.db import get_engine, get_session  # noqa: E402
from apila.models import TeamGameLog  # noqa: E402
from apila.rating import rating_diff  # noqa: E402
from apila.store import PointInTimeStore  # noqa: E402


def build_training_examples(
    store: PointInTimeStore, before: date
) -> tuple[list[float], list[bool]]:
    home_rows = (
        store.session.execute(
            select(TeamGameLog)
            .where(TeamGameLog.is_home.is_(True))
            .where(TeamGameLog.game_date < before)
            .order_by(TeamGameLog.game_date)
        )
        .scalars()
        .all()
    )

    diffs: list[float] = []
    wins: list[bool] = []
    for row in home_rows:
        away = store.session.execute(
            select(TeamGameLog).where(
                TeamGameLog.game_id == row.game_id,
                TeamGameLog.is_home.is_(False),
            )
        ).scalar_one_or_none()
        if away is None:
            continue

        diff = rating_diff(store, row.team_id, away.team_id, row.game_date)
        if diff is None:
            continue  # not enough prior history yet for one of the teams

        diffs.append(diff)
        wins.append(row.wl == "W")

    return diffs, wins


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--before", required=True, help="Only use games strictly before this ISO date (train set)"
    )
    parser.add_argument("--engine-version", required=True, help="e.g. v1.0")
    parser.add_argument("--out", required=True, help="Where to write the frozen mapping JSON")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    before = datetime.strptime(args.before, "%Y-%m-%d").date()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    diffs, wins = build_training_examples(store, before)
    print(f"built {len(diffs)} training examples from games before {before}")

    mapping = fit_probability_mapping(
        diffs,
        wins,
        engine_version=args.engine_version,
        trained_on=f"games before {before}",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_json(args.out)
    print(f"froze mapping to {args.out}: coef={mapping.coef:.4f} intercept={mapping.intercept:.4f}")


if __name__ == "__main__":
    main()
