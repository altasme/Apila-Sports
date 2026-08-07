from __future__ import annotations

import pytest

from apila.metrics import (
    accuracy,
    brier_score,
    log_loss,
    multiclass_accuracy,
    multiclass_brier_score,
    multiclass_log_loss,
)


def test_accuracy_counts_side_picked_not_calibration():
    # A wildly overconfident prob still counts as "correct" for accuracy
    # as long as it's on the right side of 0.5 -- that's the whole point
    # of needing Brier/log loss alongside it (proposal section 2).
    assert accuracy([0.99, 0.01], [True, False]) == pytest.approx(1.0)
    assert accuracy([0.51, 0.49], [False, True]) == pytest.approx(0.0)


def test_brier_score_zero_for_perfect_predictions():
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)


def test_brier_score_penalizes_overconfidence_more_than_log_loss_direction():
    # A confident wrong call (0.9 predicted, actually False) should score
    # worse than a coin-flip call that's also wrong.
    confident_wrong = brier_score([0.9], [False])
    coinflip_wrong = brier_score([0.5], [False])
    assert confident_wrong > coinflip_wrong


def test_log_loss_zero_for_perfect_confident_predictions():
    assert log_loss([1.0, 0.0], [True, False]) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_penalizes_confident_wrong_predictions_heavily():
    # A model that says 99% and is wrong should be punished far more than
    # accuracy alone would ever show.
    assert log_loss([0.99], [False]) > log_loss([0.5], [False])


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy([0.5], [True, False])
    with pytest.raises(ValueError):
        brier_score([0.5], [True, False])
    with pytest.raises(ValueError):
        log_loss([0.5], [True, False])


def test_multiclass_accuracy_uses_argmax():
    probs = [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]]
    outcomes = [0, 2]  # home win, away win
    assert multiclass_accuracy(probs, outcomes) == pytest.approx(1.0)


def test_multiclass_brier_score_zero_for_perfect_one_hot_predictions():
    probs = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    outcomes = [0, 2]
    assert multiclass_brier_score(probs, outcomes) == pytest.approx(0.0)


def test_multiclass_log_loss_zero_for_perfect_confident_predictions():
    probs = [[1.0, 0.0, 0.0]]
    outcomes = [0]
    assert multiclass_log_loss(probs, outcomes) == pytest.approx(0.0, abs=1e-9)


def test_multiclass_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        multiclass_accuracy([[0.3, 0.3, 0.4]], [0, 1])
    with pytest.raises(ValueError):
        multiclass_brier_score([[0.3, 0.3, 0.4]], [0, 1])
    with pytest.raises(ValueError):
        multiclass_log_loss([[0.3, 0.3, 0.4]], [0, 1])
