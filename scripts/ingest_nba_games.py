"""Ingest historical NBA team game logs into the point-in-time store.

Pulls from stats.nba.com via the `nba_api` package (install with
`pip install -e .[ingest]`). Requires network access to stats.nba.com,
which this repo's dev sandbox does not have -- run this from a machine
that can reach it.

Usage:
    python scripts/ingest_nba_games.py --seasons 2021-22 2022-23 2023-24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apila.db import get_engine, get_session  # noqa: E402
from apila.store import PointInTimeStore  # noqa: E402

COLUMN_MAP = {
    "GAME_ID": "game_id",
    "TEAM_ID": "team_id",
    "SEASON_ID": "season",
    "GAME_DATE": "game_date",
    "TEAM_ABBREVIATION": "team_abbr",
    "MATCHUP": "matchup",
    "WL": "wl",
    "PTS": "pts",
    "PLUS_MINUS": "plus_minus",
    "FGM": "fgm",
    "FGA": "fga",
    "FG_PCT": "fg_pct",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FG3_PCT": "fg3_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT_PCT": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "PF": "pf",
}


def fetch_season(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
    )
    return log.get_data_frames()[0]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["is_home"] = ~df["matchup"].str.contains("@")
    df["opponent_abbr"] = df["matchup"].str.extract(r"(?:@|vs\.)\s*(\w+)$")
    return df.drop(columns=["matchup"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", required=True, help="e.g. 2021-22 2022-23")
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    for season in args.seasons:
        print(f"Fetching {season}...")
        raw = fetch_season(season)
        clean = transform(raw)
        n = store.ingest(clean)
        print(f"  ingested {n} team-game rows")


if __name__ == "__main__":
    main()
