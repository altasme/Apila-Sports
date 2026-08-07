"""American-odds -> devigged implied probability.

Every function here is meant to run on *closing* odds — the price right
before tip-off — because that's the efficient number the prediction engine
has to beat (proposal sections 4.2/4.3).
"""
from __future__ import annotations


def american_to_decimal(odds: int) -> float:
    """American odds -> decimal odds (the multiplier on a winning stake)."""
    if odds > 0:
        return 1 + odds / 100
    if odds < 0:
        return 1 + 100 / abs(odds)
    raise ValueError("American odds cannot be 0")


def american_to_implied_prob(odds: int) -> float:
    """Raw implied probability from one side's American odds, vig included."""
    return 1 / american_to_decimal(odds)


def devig_multiplicative(prob_home: float, prob_away: float) -> tuple[float, float]:
    """Remove the vig by normalizing both raw implied probabilities to sum
    to 1. This spreads the overround proportionally across both sides.

    It's a simplification — it doesn't correct for favorite-longshot bias
    the way Shin's method does — but it's the standard baseline devig and
    good enough to establish whether the engine beats the market at all.
    """
    total = prob_home + prob_away
    if total <= 0:
        raise ValueError("Implied probabilities must be positive")
    return prob_home / total, prob_away / total


def devig_moneyline(home_moneyline: int, away_moneyline: int) -> tuple[float, float]:
    """American odds for both sides of a game -> devigged (home, away) probs."""
    return devig_multiplicative(
        american_to_implied_prob(home_moneyline),
        american_to_implied_prob(away_moneyline),
    )
