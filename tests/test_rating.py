from __future__ import annotations

from datetime import date

import pytest

from apila.rating import composite_rating, rating_diff, team_rating
from apila.store import TeamRating

TEAM = 1
OPPONENT = 2


def test_composite_rating_blends_season_and_recent_form():
    rating = TeamRating(
        team_id=1,
        as_of=date(2024, 1, 1),
        games_played=20,
        wins=12,
        losses=8,
        win_pct=0.6,
        ppg=105.0,
        opp_ppg=100.0,
        point_diff=5.0,  # season-long
        last10_win_pct=0.8,
        last10_point_diff=10.0,  # hotter recently than the season average
    )
    assert composite_rating(rating) == pytest.approx(0.7 * 5.0 + 0.3 * 10.0)


def test_composite_rating_falls_back_to_season_when_no_last10():
    rating = TeamRating(
        team_id=1,
        as_of=date(2024, 1, 1),
        games_played=1,
        wins=1,
        losses=0,
        win_pct=1.0,
        ppg=100.0,
        opp_ppg=90.0,
        point_diff=10.0,
        last10_win_pct=None,
        last10_point_diff=None,
    )
    assert composite_rating(rating) == pytest.approx(10.0)


def test_team_rating_none_without_history(store):
    assert team_rating(store, TEAM, as_of=date(2024, 1, 1)) is None


def test_rating_diff_is_antisymmetric(store):
    as_of = date(2024, 1, 9)
    forward = rating_diff(store, TEAM, OPPONENT, as_of)
    backward = rating_diff(store, OPPONENT, TEAM, as_of)
    assert forward == pytest.approx(-backward)


def test_rating_diff_none_when_either_team_missing_history(store):
    # Neither team has games before the very first fixture date.
    assert rating_diff(store, TEAM, OPPONENT, as_of=date(2024, 1, 1)) is None
