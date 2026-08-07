"""Naive baselines the engine has to actually beat (proposal section 4.4).

Every baseline here produces a probability, not just a pick, so it can be
scored with Brier/log loss alongside the real engine -- picking favorites
gets 60%+ accuracy for free, and that alone proves nothing about whether
it makes money (proposal section 3). "Market" is the third baseline; it
doesn't need a class here, since it's just PointInTimeStore.
market_probability() already devigged.
"""
from __future__ import annotations

from .calibration import (
    ProbabilityMapping,
    ThreeWayProbabilityMapping,
    fit_probability_mapping,
    fit_three_way_probability_mapping,
)
from .store import TeamRating


def record_strength(rating: TeamRating) -> float:
    """Standard 3-1-0 points-per-game. For sports with no draws this is
    just 3x win_pct (identical ranking to plain win_pct), but it's the
    historically correct definition once draws exist.
    """
    return (3 * rating.wins + rating.draws) / rating.games_played


def home_win_rate(outcomes: list[str]) -> float:
    """Constant "pick the home team" baseline probability: the fraction of
    games in the training set where the home side won outright. Using the
    actual base rate rather than a hardcoded 1.0 avoids an infinite log
    loss the first time the home team loses.
    """
    if not outcomes:
        raise ValueError("Need at least one outcome to compute a home-win rate")
    return sum(o == "H" for o in outcomes) / len(outcomes)


def outcome_frequency_baseline(outcomes: list[str]) -> tuple[float, float, float]:
    """Three-way analogue of home_win_rate: the empirical (home, draw,
    away) frequency in the training set, applied as a constant prediction
    to every game.
    """
    if not outcomes:
        raise ValueError("Need at least one outcome to compute frequencies")
    n = len(outcomes)
    return (
        sum(o == "H" for o in outcomes) / n,
        sum(o == "D" for o in outcomes) / n,
        sum(o == "A" for o in outcomes) / n,
    )


def fit_better_record_baseline(
    record_diffs: list[float],
    home_wins: list[bool],
    *,
    trained_on: str,
) -> ProbabilityMapping:
    """"Better record" baseline: the same logistic-fit machinery as the
    real engine, fed record_strength difference instead of the composite
    rating difference. Reusing fit_probability_mapping keeps this an
    honest, calibrated probability rather than a hand-drawn one.
    """
    return fit_probability_mapping(
        record_diffs,
        home_wins,
        engine_version="baseline-better-record",
        trained_on=trained_on,
    )


def fit_better_record_baseline_three_way(
    record_diffs: list[float],
    outcomes: list[str],
    *,
    trained_on: str,
) -> ThreeWayProbabilityMapping:
    return fit_three_way_probability_mapping(
        record_diffs,
        outcomes,
        engine_version="baseline-better-record-3way",
        trained_on=trained_on,
    )
