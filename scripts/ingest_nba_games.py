"""Ingest historical NBA games from balldontlie.io into the point-in-time store.

Originally pulled from stats.nba.com via `nba_api`, but that API
aggressively blocks or silently hangs requests from cloud/datacenter IPs
(AWS, GCP, Azure) as anti-scraping protection -- it doesn't just fail,
it times out, which makes it useless from GitHub Codespaces, Actions, or
most cloud sandboxes. balldontlie.io works fine from those environments.

The tradeoff: balldontlie's `games` endpoint only gives date, teams, and
final score -- no team box score stats (FGM/FGA/REB/AST/etc.). Those
columns stay NULL when ingested from here. That's fine for everything
this repo currently computes (apila/rating.py only uses points and
win/loss), but it means shooting-stat features aren't available from this
source without a different provider.

Requires a free API key from https://app.balldontlie.io (their v1 API is
key-gated). Pass it via --api-key or the BALLDONTLIE_API_KEY env var.

The free tier is rate-limited to 5 requests/minute, so pulling a full
~1,230-game season (13 pages at 100/page) takes a few minutes -- the
script paces itself at ~13s between pages and backs off further on a 429
rather than failing. This is expected; a season isn't stuck just because
it's slow.

Also note: team ids here are balldontlie's own scheme, not stats.nba.com's.
Don't mix rows from the old nba_api-based ingestion into the same
sport="nba" data -- the same team would get two different team_ids and
silently split its rating history across them.

Usage:
    python scripts/ingest_nba_games.py --seasons 2021 2022 2023 --api-key YOUR_KEY
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apila.db import get_engine, get_session  # noqa: E402
from apila.store import PointInTimeStore  # noqa: E402

API_BASE = "https://api.balldontlie.io/v1"
REQUEST_DELAY_SECONDS = 13  # free tier is 5 req/min -- 12s is the floor, pad a bit
RATE_LIMIT_BACKOFF_SECONDS = 20  # fallback wait on a 429 with no Retry-After header


def fetch_season_games(season: int, api_key: str) -> list[dict]:
    """balldontlie's `season` param is the year a season STARTED --
    season=2023 means the 2023-24 season. Paginates via cursor until
    exhausted.
    """
    games: list[dict] = []
    cursor: int | None = None
    headers = {"Authorization": api_key}

    while True:
        params: dict = {"seasons[]": season, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor

        payload = _get_with_retry(f"{API_BASE}/games", headers, params)
        games.extend(payload["data"])

        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(REQUEST_DELAY_SECONDS)

    return games


def _get_with_retry(url: str, headers: dict, params: dict, max_retries: int = 5) -> dict:
    """balldontlie's free tier is rate-limited (5 req/min as of writing) --
    a 429 here is expected mid-pull, not a failure. Back off and retry
    rather than aborting the whole ingestion over it.
    """
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()

        retry_after = float(resp.headers.get("Retry-After", RATE_LIMIT_BACKOFF_SECONDS))
        print(f"  rate limited, waiting {retry_after:.0f}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(retry_after)

    raise RuntimeError(f"Still rate limited after {max_retries} retries -- try again later")


def transform(games: list[dict], season: int) -> pd.DataFrame:
    """One balldontlie game -> two team_game_logs rows (home + away)."""
    rows = []
    for game in games:
        if game.get("status") != "Final":
            continue  # skip postponed/future games

        home = game["home_team"]
        away = game["visitor_team"]
        home_score = game["home_team_score"]
        away_score = game["visitor_team_score"]
        if home_score == away_score:
            continue  # NBA games don't end in ties; equal scores usually mean incomplete data

        game_date = game["date"][:10]  # "YYYY-MM-DDT00:00:00.000Z" -> "YYYY-MM-DD"

        rows.append(
            dict(
                game_id=str(game["id"]),
                team_id=home["id"],
                sport="nba",
                season=str(season),
                game_date=game_date,
                team_abbr=home["abbreviation"],
                opponent_abbr=away["abbreviation"],
                is_home=True,
                wl="W" if home_score > away_score else "L",
                pts=home_score,
                plus_minus=float(home_score - away_score),
            )
        )
        rows.append(
            dict(
                game_id=str(game["id"]),
                team_id=away["id"],
                sport="nba",
                season=str(season),
                game_date=game_date,
                team_abbr=away["abbreviation"],
                opponent_abbr=home["abbreviation"],
                is_home=False,
                wl="W" if away_score > home_score else "L",
                pts=away_score,
                plus_minus=float(away_score - home_score),
            )
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seasons", nargs="+", type=int, required=True, help="Season-start years, e.g. 2021 2022 2023"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BALLDONTLIE_API_KEY"),
        help="or set BALLDONTLIE_API_KEY (get one free at https://app.balldontlie.io)",
    )
    parser.add_argument("--db", default=None, help="Override sqlite db path")
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Need an API key: pass --api-key or set BALLDONTLIE_API_KEY "
            "(get one free at https://app.balldontlie.io)"
        )

    engine = get_engine(args.db)
    session = get_session(engine)
    store = PointInTimeStore(session)

    for season in args.seasons:
        print(f"Fetching season {season} ({season}-{str(season + 1)[-2:]})...")
        games = fetch_season_games(season, args.api_key)
        clean = transform(games, season)
        n = store.ingest(clean)
        print(f"  ingested {n} team-game rows from {len(games)} games")


if __name__ == "__main__":
    main()
