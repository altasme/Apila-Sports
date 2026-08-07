"""Ingest historical closing odds into the point-in-time store.

There's no single reliable free API for historical closing odds (see
proposal section 4.2), so this script deliberately takes a plain CSV
rather than calling a specific provider. Point it at whatever you've
sourced -- a paid historical-odds API export, a public dataset (e.g. a
Kaggle NBA odds dump), or a manually assembled file.

Required columns:
    game_date, home_team_abbr, away_team_abbr, home_moneyline, away_moneyline

`game_date` must be an ISO date (YYYY-MM-DD) and team abbreviations must
match whatever's already in team_game_logs (see ingest_nba_games.py) or
the join in PointInTimeStore.market_probability() won't find a match.
Moneylines are American odds (e.g. -150, +130).

Usage:
    python scripts/ingest_closing_odds.py path/to/odds.csv --source kaggle-nba-odds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apila.db import get_engine, get_session  # noqa: E402
from apila.store import PointInTimeStore  # noqa: E402

REQUIRED_COLUMNS = [
    "game_date",
    "home_team_abbr",
    "away_team_abbr",
    "home_moneyline",
    "away_moneyline",
]


def load_csv(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df["game_date"] = df["game_date"].dt.date
    df["home_moneyline"] = df["home_moneyline"].astype(int)
    df["away_moneyline"] = df["away_moneyline"].astype(int)
    df["source"] = source
    return df[REQUIRED_COLUMNS + ["source"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="CSV file of closing odds")
    parser.add_argument("--source", required=True, help="Label for where this data came from")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    df = load_csv(args.csv_path, args.source)
    n = store.ingest_odds(df)
    print(f"ingested {n} closing-odds rows from {args.csv_path} (source={args.source})")


if __name__ == "__main__":
    main()
