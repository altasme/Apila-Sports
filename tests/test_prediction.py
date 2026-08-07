from __future__ import annotations

from datetime import date

import pytest

from apila.calibration import ProbabilityMapping, ThreeWayProbabilityMapping
from apila.prediction import PredictionEngine, ThreeWayPredictionEngine
from apila.rating import rating_diff

TEAM = 1
OPPONENT = 2
NBA = "nba"

SOCCER_TEAM = 101
SOCCER_OPPONENT = 102
SOCCER = "soccer"


@pytest.fixture
def mapping() -> ProbabilityMapping:
    return ProbabilityMapping(
        coef=0.05,
        intercept=0.0,
        engine_version="test-v0",
        trained_on="fixture",
        n_games=10,
    )


@pytest.fixture
def three_way_mapping() -> ThreeWayProbabilityMapping:
    return ThreeWayProbabilityMapping(
        home_coef=0.05,
        home_intercept=0.2,
        draw_coef=0.0,
        draw_intercept=0.0,
        away_coef=-0.05,
        away_intercept=-0.1,
        engine_version="test-soccer-v0",
        trained_on="fixture",
        n_games=10,
    )


def test_predict_matches_store_rating_diff_and_mapping(store, mapping):
    as_of = date(2024, 1, 9)
    engine = PredictionEngine(store, mapping, sport=NBA)

    prediction = engine.predict(TEAM, OPPONENT, as_of)

    assert prediction is not None
    expected_diff = rating_diff(store, TEAM, OPPONENT, NBA, as_of)
    assert prediction.rating_diff == pytest.approx(expected_diff)
    assert prediction.home_win_prob == pytest.approx(mapping.predict_proba(expected_diff))
    assert prediction.engine_version == "test-v0"


def test_predict_none_when_insufficient_history(store, mapping):
    engine = PredictionEngine(store, mapping, sport=NBA)
    assert engine.predict(TEAM, OPPONENT, as_of=date(2024, 1, 1)) is None


def test_three_way_predict_matches_store_rating_diff_and_mapping(store, three_way_mapping):
    as_of = date(2024, 2, 15)
    engine = ThreeWayPredictionEngine(store, three_way_mapping, sport=SOCCER)

    prediction = engine.predict(SOCCER_TEAM, SOCCER_OPPONENT, as_of)

    assert prediction is not None
    expected_diff = rating_diff(store, SOCCER_TEAM, SOCCER_OPPONENT, SOCCER, as_of)
    assert prediction.rating_diff == pytest.approx(expected_diff)

    expected_home, expected_draw, expected_away = three_way_mapping.predict_proba(expected_diff)
    assert prediction.home_prob == pytest.approx(expected_home)
    assert prediction.draw_prob == pytest.approx(expected_draw)
    assert prediction.away_prob == pytest.approx(expected_away)
    assert prediction.home_prob + prediction.draw_prob + prediction.away_prob == pytest.approx(1.0)


def test_three_way_predict_none_when_insufficient_history(store, three_way_mapping):
    engine = ThreeWayPredictionEngine(store, three_way_mapping, sport=SOCCER)
    assert engine.predict(SOCCER_TEAM, SOCCER_OPPONENT, as_of=date(2024, 2, 1)) is None
