from __future__ import annotations

import pytest

from apila.calibration import ProbabilityMapping, fit_probability_mapping


def _synthetic_training_set() -> tuple[list[float], list[bool]]:
    # Large negative diffs always lose, large positive diffs always win, a
    # couple of near-even games go either way -- enough signal to fit a
    # clean positive slope without being a degenerate single-class set.
    diffs = [-12, -10, -8, -6, -1, 1, 6, 8, 10, 12]
    wins = [False, False, False, False, False, True, True, True, True, True]
    return diffs, wins


def test_fit_produces_positive_slope_for_positive_relationship():
    diffs, wins = _synthetic_training_set()
    mapping = fit_probability_mapping(diffs, wins, engine_version="test", trained_on="synthetic")
    assert mapping.coef > 0


def test_predict_proba_monotonic_in_rating_diff():
    diffs, wins = _synthetic_training_set()
    mapping = fit_probability_mapping(diffs, wins, engine_version="test", trained_on="synthetic")

    assert mapping.predict_proba(-10) < mapping.predict_proba(0) < mapping.predict_proba(10)


def test_predict_proba_near_half_at_zero_diff_for_symmetric_data():
    diffs, wins = _synthetic_training_set()
    mapping = fit_probability_mapping(diffs, wins, engine_version="test", trained_on="synthetic")
    assert mapping.predict_proba(0.0) == pytest.approx(0.5, abs=0.15)


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_probability_mapping([1.0, 2.0], [True], engine_version="test", trained_on="x")


def test_fit_rejects_single_class():
    with pytest.raises(ValueError):
        fit_probability_mapping([1.0, 2.0, 3.0], [True, True, True], engine_version="test", trained_on="x")


def test_fit_rejects_too_few_games():
    with pytest.raises(ValueError):
        fit_probability_mapping([1.0], [True], engine_version="test", trained_on="x")


def test_mapping_json_roundtrip(tmp_path):
    diffs, wins = _synthetic_training_set()
    mapping = fit_probability_mapping(diffs, wins, engine_version="v1.0", trained_on="synthetic")

    path = tmp_path / "mapping.json"
    mapping.to_json(path)
    loaded = ProbabilityMapping.from_json(path)

    assert loaded == mapping
