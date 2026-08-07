"""Fitted rating-difference -> win-probability mapping.

v1 guessed this curve and guessed the confidence-tier thresholds built on
top of it. Guessed thresholds miscalibrate probabilities, which quietly
wrecks any EV calculation downstream (proposal section 4.6). This module
only ever produces a mapping by fitting logistic regression on labeled
(rating_diff, home_win) pairs -- never by hand.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ProbabilityMapping:
    """A frozen logistic mapping: P(home win) = sigmoid(coef * rating_diff + intercept).

    "Frozen" means exactly that: fit it once on a training set, then use
    it as-is against a test set and never touch it again for that engine
    version. Retuning on the data you evaluate on is how v1 manufactured
    fake improvement (proposal section 4.7).
    """

    coef: float
    intercept: float
    engine_version: str
    trained_on: str
    n_games: int

    def predict_proba(self, rating_diff: float) -> float:
        z = self.coef * rating_diff + self.intercept
        return 1.0 / (1.0 + math.exp(-z))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "ProbabilityMapping":
        return cls(**json.loads(Path(path).read_text()))


def fit_probability_mapping(
    rating_diffs: list[float],
    home_wins: list[bool],
    *,
    engine_version: str,
    trained_on: str,
) -> ProbabilityMapping:
    """Fit P(home win) ~ sigmoid(a * rating_diff + b) via logistic regression.

    Call this on a training set only -- it has no opinion about how the
    data was split (that's the caller's job, per proposal section 4.7),
    only that whatever it's given gets fit exactly once.
    """
    if len(rating_diffs) != len(home_wins):
        raise ValueError("rating_diffs and home_wins must be the same length")
    if len(rating_diffs) < 2:
        raise ValueError("Need at least 2 games to fit a mapping")

    x = np.array(rating_diffs, dtype=float).reshape(-1, 1)
    y = np.array(home_wins).astype(int)

    if len(set(y.tolist())) < 2:
        raise ValueError("Training set must contain both wins and losses")

    model = LogisticRegression()
    model.fit(x, y)

    return ProbabilityMapping(
        coef=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        engine_version=engine_version,
        trained_on=trained_on,
        n_games=len(rating_diffs),
    )
