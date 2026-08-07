"""Reshape the Kaggle "nba-betting-data-october-2007-to-june-2024" CSV
(cviaxmiwnptr) into the CSV shape ingest_closing_odds.py expects.

Source columns (relevant subset): date, home, away, moneyline_home,
moneyline_away -- already American odds, already ISO dates. The only real
work is team codes: this dataset uses its own lowercase short codes
("gs", "sa", "utah", "no", "ny", "wsh", ...) that don't match
balldontlie's abbreviations ("GSW", "SAS", "UTA", "NOP", "NYK", "WAS",
...). TEAM_ABBR_MAP below was built by diffing the actual distinct codes
in both this dataset and this repo's already-ingested team_game_logs --
not guessed. The dataset relabels historical relocated franchises (old
Seattle games as "okc", old New Jersey games as "bkn") under their
current code, so this is a clean 30-to-30 mapping with no historical
edge cases.

Usage:
    python scripts/reshape_kaggle_nba_odds.py path/to/nba_2008-2026.csv reshaped_odds.csv
    python scripts/ingest_closing_odds.py reshaped_odds.csv --sport nba --source kaggle-cviaxmiwnptr
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

TEAM_ABBR_MAP = {
    "atl": "ATL",
    "bkn": "BKN",
    "bos": "BOS",
    "cha": "CHA",
    "chi": "CHI",
    "cle": "CLE",
    "dal": "DAL",
    "den": "DEN",
    "det": "DET",
    "gs": "GSW",
    "hou": "HOU",
    "ind": "IND",
    "lac": "LAC",
    "lal": "LAL",
    "mem": "MEM",
    "mia": "MIA",
    "mil": "MIL",
    "min": "MIN",
    "no": "NOP",
    "ny": "NYK",
    "okc": "OKC",
    "orl": "ORL",
    "phi": "PHI",
    "phx": "PHX",
    "por": "POR",
    "sa": "SAS",
    "sac": "SAC",
    "tor": "TOR",
    "utah": "UTA",
    "wsh": "WAS",
}


def reshape(src_path: Path, dest_path: Path) -> tuple[int, int]:
    """Returns (rows_written, rows_skipped_for_missing_moneyline)."""
    written = 0
    skipped = 0

    with src_path.open(newline="") as src, dest_path.open("w", newline="") as dest:
        reader = csv.DictReader(src)
        writer = csv.writer(dest)
        writer.writerow(
            ["game_date", "home_team_abbr", "away_team_abbr", "home_moneyline", "away_moneyline"]
        )

        for row in reader:
            home_ml = row.get("moneyline_home", "").strip()
            away_ml = row.get("moneyline_away", "").strip()
            if not home_ml or not away_ml:
                skipped += 1
                continue

            home = TEAM_ABBR_MAP.get(row["home"])
            away = TEAM_ABBR_MAP.get(row["away"])
            if home is None or away is None:
                raise ValueError(
                    f"Unmapped team code: home={row['home']!r} away={row['away']!r} "
                    f"on {row.get('date')} -- TEAM_ABBR_MAP needs updating"
                )

            writer.writerow([row["date"], home, away, int(home_ml), int(away_ml)])
            written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", type=Path, help="Raw Kaggle CSV (nba_2008-2026.csv)")
    parser.add_argument("dest", type=Path, help="Where to write the reshaped CSV")
    args = parser.parse_args()

    written, skipped = reshape(args.src, args.dest)
    print(f"reshaped {written} rows -> {args.dest} ({skipped} skipped for missing moneyline)")


if __name__ == "__main__":
    main()
