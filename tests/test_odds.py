from __future__ import annotations

import pytest

from apila.odds import (
    american_to_decimal,
    american_to_implied_prob,
    devig_1x2,
    devig_moneyline,
    devig_multiplicative,
)


def test_american_to_decimal_favorite_and_underdog():
    assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-4)
    assert american_to_decimal(130) == pytest.approx(2.3)


def test_american_to_decimal_rejects_zero():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_standard_vig_implied_prob():
    # -110 is the standard "vig" price: implied prob just over 52.4%.
    assert american_to_implied_prob(-110) == pytest.approx(0.5238, abs=1e-4)


def test_devig_multiplicative_removes_vig_on_pick_em():
    # Two -110 sides: raw implied probs sum to > 1 (the vig). Devigged,
    # a true pick-em should land at exactly 50/50.
    raw_home = american_to_implied_prob(-110)
    raw_away = american_to_implied_prob(-110)
    home, away = devig_multiplicative(raw_home, raw_away)
    assert home == pytest.approx(0.5)
    assert away == pytest.approx(0.5)
    assert home + away == pytest.approx(1.0)


def test_devig_moneyline_favorite_underdog_sums_to_one():
    home, away = devig_moneyline(-150, 130)
    assert home + away == pytest.approx(1.0)
    assert home > away  # -150 is the favorite


def test_devig_multiplicative_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        devig_multiplicative(0, 0)


def test_devig_multiplicative_handles_three_outcomes():
    # Symmetric three-way pick-em should land at exactly 1/3 each.
    p = american_to_implied_prob(200)  # same price on all three sides
    home, draw, away = devig_multiplicative(p, p, p)
    assert home == pytest.approx(1 / 3)
    assert draw == pytest.approx(1 / 3)
    assert away == pytest.approx(1 / 3)


def test_devig_1x2_sums_to_one_and_orders_by_favorite():
    home, draw, away = devig_1x2(-120, 220, 280)
    assert home + draw + away == pytest.approx(1.0)
    assert home > draw > away  # -120 favorite, then the draw, then the +280 underdog
