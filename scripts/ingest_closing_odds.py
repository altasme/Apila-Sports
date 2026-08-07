"""Ingest historical closing odds into the point-in-time store.

There's no single reliable free API for historical closing odds (see
proposal section 4.2), so this script deliberately takes a plain CSV
rather than calling a specific provider. Point it at whatever you've
sourced -- a paid historical-odds API export, a public dataset (e.g. a
Kaggle odds dump), or a manually assembled file.

Required columns:
    game_date, home_team_abbr, away_team_abbr, home_moneyline, away_moneyline

Optional column:
    draw_moneyline -- include this for a three-outcome market (soccer
    1X2). Leave it out entirely for two-outcome markets (basketball, MLB
    moneyline); a column full of blanks is NOT the same thing (blanks
    read as NaN, not "no draw market") and will be rejected.

`game_date` must be an ISO date (YYYY-MM-DD) and team abbreviations must
match whatever's already in team_game_logs for the same --sport (see
ingest_nba_games.py) or the join in PointInTimeStore.market_probability()
won't find a match. Moneylines are American odds (e.g. -150, +130).

Usage:
    python scripts/ingest_closing_odds.py path/to/odds.csv --sport nba --source kaggle-nba-odds
    python scripts/ingest_closing_odds.py path/to/1x2.csv --sport soccer --source my-soccer-odds
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


def load_csv(path: Path, sport: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    has_draw = "draw_moneyline" in df.columns
    if has_draw and df["draw_moneyline"].isna().any():
        raise ValueError(
            "draw_moneyline column is present but has blank rows -- either every "
            "row needs a price (three-outcome market) or the column shouldn't be "
            "there at all (two-outcome market)"
        )

    df["game_date"] = df["game_date"].dt.date
    df["home_moneyline"] = df["home_moneyline"].astype(int)
    df["away_moneyline"] = df["away_moneyline"].astype(int)
    df["sport"] = sport
    df["source"] = source

    columns = REQUIRED_COLUMNS + ["sport", "source"]
    if has_draw:
        df["draw_moneyline"] = df["draw_moneyline"].astype(int)
        columns.append("draw_moneyline")
    return df[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="CSV file of closing odds")
    parser.add_argument("--sport", required=True, help='e.g. "nba", "soccer"')
    parser.add_argument("--source", required=True, help="Label for where this data came from")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    df = load_csv(args.csv_path, args.sport, args.source)
    n = store.ingest_odds(df)
    print(f"ingested {n} closing-odds rows from {args.csv_path} (sport={args.sport}, source={args.source})")


if __name__ == "__main__":
    main()
