from __future__ import annotations

from datetime import date

import pytest

NBA = "nba"
SOCCER = "soccer"


def test_market_probability_matches_ingested_odds(store):
    # g1: 2024-01-01, AAA home at -150, BBB away at +130.
    market = store.market_probability(date(2024, 1, 1), "AAA", "BBB", NBA)

    assert market is not None
    assert market.home_moneyline == -150
    assert market.away_moneyline == 130
    assert not market.is_three_way
    assert market.draw_prob is None
    assert market.home_prob + market.away_prob == pytest.approx(1.0)
    assert market.home_prob > market.away_prob
    assert market.vig > 0  # both sides always overround before devig


def test_market_probability_none_when_no_odds_ingested(store):
    assert store.market_probability(date(2024, 1, 1), "ZZZ", "YYY", NBA) is None


def test_market_probability_respects_home_away_direction(store):
    # g2: 2024-01-03, BBB is home (-110), AAA is away (-110) -- reversing
    # the home/away args should not silently match the same row.
    market = store.market_probability(date(2024, 1, 3), "BBB", "AAA", NBA)
    assert market is not None

    reversed_lookup = store.market_probability(date(2024, 1, 3), "AAA", "BBB", NBA)
    assert reversed_lookup is None


def test_market_probability_respects_sport(store):
    # Same date/team-abbr shape could in principle collide across sports;
    # asking for a sport with no matching odds must not find the other
    # sport's row.
    assert store.market_probability(date(2024, 1, 1), "AAA", "BBB", SOCCER) is None


def test_market_probability_three_way_for_soccer(store):
    # m1: 2024-02-01, CCC home -120 / draw +220 / DDD away +280.
    market = store.market_probability(date(2024, 2, 1), "CCC", "DDD", SOCCER)

    assert market is not None
    assert market.is_three_way
    assert market.draw_moneyline == 220
    assert market.home_prob + market.draw_prob + market.away_prob == pytest.approx(1.0)
    assert market.home_prob > market.draw_prob > market.away_prob
    assert market.vig > 0
