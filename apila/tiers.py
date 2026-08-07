"""Confidence tiers for reporting ROI/accuracy broken out by prediction strength.

The thresholds here are placeholders, same as v1's were — proposal section
7 is explicit that tier boundaries are guesses until recalibrated against
real distributions with confidence intervals. Keep reporting these tiers
*with* sample sizes; a tier's hit rate off a handful of games isn't
trustworthy no matter which threshold produced it (proposal section 4.8).
"""
from __future__ import annotations

HIGH_EDGE = 0.20
MODERATE_EDGE = 0.10


def confidence_tier(prob: float, baseline: float = 0.5) -> str:
    """`prob` is the predicted probability of the class being evaluated;
    `baseline` is the no-information probability for that market (0.5 for
    a two-outcome market, 1/3 for a naive three-outcome one). Tiered by
    distance from that baseline.
    """
    edge = abs(prob - baseline)
    if edge >= HIGH_EDGE:
        return "high"
    if edge >= MODERATE_EDGE:
        return "moderate"
    return "low"
