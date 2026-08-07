from __future__ import annotations

import csv

import pytest

from scripts.reshape_kaggle_nba_odds import TEAM_ABBR_MAP, reshape

HEADER = (
    "season,date,regular,playoffs,away,home,score_away,score_home,"
    "q1_away,q2_away,q3_away,q4_away,ot_away,q1_home,q2_home,q3_home,q4_home,ot_home,"
    "whos_favored,spread,total,moneyline_away,moneyline_home,h2_spread,h2_total,id_spread,id_total"
)


def _write_source(tmp_path, rows: list[str]):
    path = tmp_path / "source.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_team_abbr_map_covers_all_30_teams():
    assert len(TEAM_ABBR_MAP) == 30
    assert len(set(TEAM_ABBR_MAP.values())) == 30  # every balldontlie code is unique too


def test_reshape_maps_teams_and_reorders_to_home_first(tmp_path):
    src = _write_source(
        tmp_path,
        [
            "2008,2007-10-30,True,False,por,sa,97,106,,,,,,,,,,,home,13,189.5,900,-1400,5,95,0,1",
        ],
    )
    dest = tmp_path / "reshaped.csv"

    written, skipped = reshape(src, dest)

    assert written == 1
    assert skipped == 0

    with dest.open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "game_date": "2007-10-30",
            "home_team_abbr": "SAS",
            "away_team_abbr": "POR",
            "home_moneyline": "-1400",
            "away_moneyline": "900",
        }
    ]


def test_reshape_skips_rows_with_missing_moneyline(tmp_path):
    src = _write_source(
        tmp_path,
        [
            "2008,2007-11-01,True,False,no,ny,,,,,,,,,,,,,,,,,,,,,",
        ],
    )
    dest = tmp_path / "reshaped.csv"

    written, skipped = reshape(src, dest)

    assert written == 0
    assert skipped == 1


def test_reshape_raises_on_unmapped_team_code(tmp_path):
    src = _write_source(
        tmp_path,
        [
            "2008,2007-10-30,True,False,xyz,sa,97,106,,,,,,,,,,,home,13,189.5,900,-1400,5,95,0,1",
        ],
    )
    dest = tmp_path / "reshaped.csv"

    with pytest.raises(ValueError, match="Unmapped team code"):
        reshape(src, dest)
