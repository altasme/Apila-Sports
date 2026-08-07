from __future__ import annotations

from datetime import date

import pytest

from apila.baselines import (
    fit_better_record_baseline,
    fit_better_record_baseline_three_way,
    home_win_rate,
    outcome_frequency_baseline,
    record_strength,
)
from apila.store import TeamRating


def _rating(**overrides) -> TeamRating:
    defaults = dict(
        team_id=1,
        sport="nba",
        as_of=date(2024, 1, 1),
        games_played=4,
        wins=3,
        draws=0,
        losses=1,
        win_pct=0.75,
        draw_pct=0.0,
        ppg=100.0,
        opp_ppg=95.0,
        point_diff=5.0,
        last10_win_pct=0.75,
        last10_point_diff=5.0,
    )
    defaults.update(overrides)
    return TeamRating(**defaults)


def test_record_strength_with_no_draws_is_three_times_win_pct():
    rating = _rating(wins=3, draws=0, losses=1, games_played=4)
    assert record_strength(rating) == pytest.approx(3 * 0.75)


def test_record_strength_accounts_for_draws():
    rating = _rating(wins=1, draws=2, losses=1, games_played=4)
    assert record_strength(rating) == pytest.approx((3 * 1 + 2) / 4)


def test_home_win_rate_basic():
    assert home_win_rate(["H", "A", "H", "D"]) == pytest.approx(0.5)


def test_home_win_rate_rejects_empty():
    with pytest.raises(ValueError):
        home_win_rate([])


def test_outcome_frequency_baseline_matches_counts_and_sums_to_one():
    home, draw, away = outcome_frequency_baseline(["H", "H", "D", "A"])
    assert (home, draw, away) == pytest.approx((0.5, 0.25, 0.25))
    assert home + draw + away == pytest.approx(1.0)


def test_fit_better_record_baseline_produces_positive_slope():
    diffs = [-3, -2, -1, 1, 2, 3]
    wins = [False, False, False, True, True, True]
    mapping = fit_better_record_baseline(diffs, wins, trained_on="synthetic")
    assert mapping.coef > 0
    assert mapping.engine_version == "baseline-better-record"


def test_fit_better_record_baseline_three_way_produces_valid_distribution():
    diffs = [-3, -2, -1, 0, 1, 2, 3]
    outcomes = ["A", "A", "A", "D", "H", "H", "H"]
    mapping = fit_better_record_baseline_three_way(diffs, outcomes, trained_on="synthetic")

    home, draw, away = mapping.predict_proba(0.0)
    assert home + draw + away == pytest.approx(1.0)
    assert mapping.engine_version == "baseline-better-record-3way"
