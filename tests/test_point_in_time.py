from __future__ import annotations

from datetime import date

import pytest

TEAM = 1
SPORT = "nba"


def test_as_of_excludes_same_day_and_future_games(store):
    games = store.team_games_before(TEAM, SPORT, as_of=date(2024, 1, 5))
    assert set(games["game_id"]) == {"g1", "g2"}


def test_no_games_before_first_game_returns_none(store):
    assert store.team_rating_asof(TEAM, SPORT, as_of=date(2024, 1, 1)) is None


def test_rating_matches_hand_computed_values(store):
    rating = store.team_rating_asof(TEAM, SPORT, as_of=date(2024, 1, 9))

    # g1 W 100/+10, g2 L 95/-5, g3 W 110/+15, g4 W 105/+5 -- g5 excluded (same day)
    assert rating.games_played == 4
    assert rating.wins == 3
    assert rating.draws == 0
    assert rating.losses == 1
    assert rating.win_pct == pytest.approx(0.75)
    assert rating.ppg == pytest.approx((100 + 95 + 110 + 105) / 4)
    assert rating.opp_ppg == pytest.approx((90 + 100 + 95 + 100) / 4)
    assert rating.point_diff == pytest.approx((10 - 5 + 15 + 5) / 4)


def test_future_game_never_leaks_into_rating(store):
    """Adversarial check: the rating as-of a mid-season date must match a
    manual computation over only the games strictly before it. If the store
    leaked the game happening ON that date (or later), these numbers would
    silently drift -- which is exactly the failure mode Phase 0 depends on
    catching before it reaches a backtest.
    """
    as_of = date(2024, 1, 5)
    rating = store.team_rating_asof(TEAM, SPORT, as_of=as_of)

    # Correct: only g1 (W, 100) and g2 (L, 95) precede 2024-01-05.
    assert rating.games_played == 2
    assert rating.win_pct == pytest.approx(0.5)
    assert rating.ppg == pytest.approx((100 + 95) / 2)

    # If g3 (the game played ON as_of, W/110) had leaked in, ppg would be
    # 305/3 and win_pct would be 2/3. Assert the real result is not that.
    leaked_ppg = (100 + 95 + 110) / 3
    assert rating.ppg != pytest.approx(leaked_ppg)


def test_limit_returns_most_recent_games_first(store):
    games = store.team_games_before(TEAM, SPORT, as_of=date(2024, 1, 9), limit=2)
    assert list(games["game_id"]) == ["g4", "g3"]
