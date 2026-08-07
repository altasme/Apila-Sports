from __future__ import annotations

from datetime import date

import pytest

from apila.calibration import ProbabilityMapping
from apila.prediction import PredictionEngine
from apila.rating import rating_diff

TEAM = 1
OPPONENT = 2


@pytest.fixture
def mapping() -> ProbabilityMapping:
    return ProbabilityMapping(
        coef=0.05,
        intercept=0.0,
        engine_version="test-v0",
        trained_on="fixture",
        n_games=10,
    )


def test_predict_matches_store_rating_diff_and_mapping(store, mapping):
    as_of = date(2024, 1, 9)
    engine = PredictionEngine(store, mapping)

    prediction = engine.predict(TEAM, OPPONENT, as_of)

    assert prediction is not None
    expected_diff = rating_diff(store, TEAM, OPPONENT, as_of)
    assert prediction.rating_diff == pytest.approx(expected_diff)
    assert prediction.home_win_prob == pytest.approx(mapping.predict_proba(expected_diff))
    assert prediction.engine_version == "test-v0"


def test_predict_none_when_insufficient_history(store, mapping):
    engine = PredictionEngine(store, mapping)
    assert engine.predict(TEAM, OPPONENT, as_of=date(2024, 1, 1)) is None
