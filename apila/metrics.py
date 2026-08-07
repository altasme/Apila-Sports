"""Proper scoring rules for evaluating predicted probabilities.

Accuracy alone can't tell a calibrated model from a lucky one -- it's
trivial to hit 60%+ picking favorites and still lose money after vig
(proposal section 3). Brier score and log loss reward *calibrated*
probabilities, not just which side you picked, which is why the proposal
requires the engine beat baselines on these, not just on accuracy.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

_LOG_LOSS_EPS = 1e-15


def accuracy(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    if not probs:
        raise ValueError("Need at least one prediction")

    correct = sum((p >= 0.5) == outcome for p, outcome in zip(probs, outcomes))
    return correct / len(probs)


def brier_score(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    if not probs:
        raise ValueError("Need at least one prediction")

    squared_errors = [(p - float(o)) ** 2 for p, o in zip(probs, outcomes)]
    return sum(squared_errors) / len(squared_errors)


def log_loss(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    if not probs:
        raise ValueError("Need at least one prediction")

    total = 0.0
    for p, outcome in zip(probs, outcomes):
        p = min(max(p, _LOG_LOSS_EPS), 1 - _LOG_LOSS_EPS)
        total += -(math.log(p) if outcome else math.log(1 - p))
    return total / len(probs)


def multiclass_accuracy(prob_vectors: Sequence[Sequence[float]], outcome_indices: Sequence[int]) -> float:
    """`prob_vectors[i]` is a probability per class for game i; `outcome_indices[i]`
    is the index of the class that actually happened (e.g. 0=home, 1=draw,
    2=away). "Correct" means the highest-probability class matched.
    """
    if len(prob_vectors) != len(outcome_indices):
        raise ValueError("prob_vectors and outcome_indices must be the same length")
    if not prob_vectors:
        raise ValueError("Need at least one prediction")

    correct = sum(
        max(range(len(probs)), key=lambda i: probs[i]) == outcome
        for probs, outcome in zip(prob_vectors, outcome_indices)
    )
    return correct / len(prob_vectors)


def multiclass_brier_score(
    prob_vectors: Sequence[Sequence[float]], outcome_indices: Sequence[int]
) -> float:
    """Multi-class Brier score: mean squared error between each game's
    predicted probability vector and the one-hot actual outcome, summed
    across classes.
    """
    if len(prob_vectors) != len(outcome_indices):
        raise ValueError("prob_vectors and outcome_indices must be the same length")
    if not prob_vectors:
        raise ValueError("Need at least one prediction")

    total = 0.0
    for probs, outcome in zip(prob_vectors, outcome_indices):
        total += sum((p - (1.0 if i == outcome else 0.0)) ** 2 for i, p in enumerate(probs))
    return total / len(prob_vectors)


def multiclass_log_loss(
    prob_vectors: Sequence[Sequence[float]], outcome_indices: Sequence[int]
) -> float:
    if len(prob_vectors) != len(outcome_indices):
        raise ValueError("prob_vectors and outcome_indices must be the same length")
    if not prob_vectors:
        raise ValueError("Need at least one prediction")

    total = 0.0
    for probs, outcome in zip(prob_vectors, outcome_indices):
        p = min(max(probs[outcome], _LOG_LOSS_EPS), 1 - _LOG_LOSS_EPS)
        total += -math.log(p)
    return total / len(prob_vectors)
