from __future__ import annotations

from apila.tiers import confidence_tier


def test_high_tier_for_large_edge():
    assert confidence_tier(0.75) == "high"
    assert confidence_tier(0.25) == "high"


def test_moderate_tier_for_medium_edge():
    assert confidence_tier(0.65) == "moderate"


def test_low_tier_for_small_edge():
    assert confidence_tier(0.52) == "low"
    assert confidence_tier(0.5) == "low"


def test_tier_respects_custom_baseline():
    # For a three-way market, "no edge" is 1/3, not 0.5.
    assert confidence_tier(1 / 3, baseline=1 / 3) == "low"
    assert confidence_tier(1 / 3 + 0.25, baseline=1 / 3) == "high"
