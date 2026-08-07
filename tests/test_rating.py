from __future__ import annotations

from datetime import date

import pytest

from apila.rating import composite_rating, rating_diff, team_rating
from apila.store import TeamRating

TEAM = 1
OPPONENT = 2
NBA = "nba"

SOCCER_TEAM = 101
SOCCER_OPPONENT = 102
SOCCER = "soccer"


def test_composite_rating_blends_season_and_recent_form():
    rating = TeamRating(
        team_id=1,
        sport=NBA,
        as_of=date(2024, 1, 1),
        games_played=20,
        wins=12,
        draws=0,
        losses=8,
        win_pct=0.6,
        draw_pct=0.0,
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
        sport=NBA,
        as_of=date(2024, 1, 1),
        games_played=1,
        wins=1,
        draws=0,
        losses=0,
        win_pct=1.0,
        draw_pct=0.0,
        ppg=100.0,
        opp_ppg=90.0,
        point_diff=10.0,
        last10_win_pct=None,
        last10_point_diff=None,
    )
    assert composite_rating(rating) == pytest.approx(10.0)


def test_team_rating_none_without_history(store):
    assert team_rating(store, TEAM, NBA, as_of=date(2024, 1, 1)) is None


def test_rating_diff_is_antisymmetric(store):
    as_of = date(2024, 1, 9)
    forward = rating_diff(store, TEAM, OPPONENT, NBA, as_of)
    backward = rating_diff(store, OPPONENT, TEAM, NBA, as_of)
    assert forward == pytest.approx(-backward)


def test_rating_diff_none_when_either_team_missing_history(store):
    # Neither team has games before the very first fixture date.
    assert rating_diff(store, TEAM, OPPONENT, NBA, as_of=date(2024, 1, 1)) is None


def test_soccer_rating_counts_draws_separately_from_losses(store):
    # CCC before 2024-02-15: W (m1), D (m2), D (m3), L (m4).
    rating = store.team_rating_asof(SOCCER_TEAM, SOCCER, as_of=date(2024, 2, 15))

    assert rating.games_played == 4
    assert rating.wins == 1
    assert rating.draws == 2
    assert rating.losses == 1
    assert rating.win_pct == pytest.approx(0.25)
    assert rating.draw_pct == pytest.approx(0.5)


def test_soccer_and_nba_ratings_do_not_cross_contaminate(store):
    # Team id 1 exists only in the nba fixture; asking for it under
    # "soccer" must find nothing, not silently reuse the nba rows.
    assert store.team_rating_asof(TEAM, SOCCER, as_of=date(2024, 1, 9)) is None


def test_soccer_rating_diff_is_antisymmetric(store):
    as_of = date(2024, 2, 15)
    forward = rating_diff(store, SOCCER_TEAM, SOCCER_OPPONENT, SOCCER, as_of)
    backward = rating_diff(store, SOCCER_OPPONENT, SOCCER_TEAM, SOCCER, as_of)
    assert forward == pytest.approx(-backward)
