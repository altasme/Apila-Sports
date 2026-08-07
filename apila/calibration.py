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


@dataclass(frozen=True)
class ThreeWayProbabilityMapping:
    """Frozen multinomial mapping for three-outcome markets (soccer 1X2):
    P(outcome) = softmax over {home, draw, away} of
    (coef_outcome * rating_diff + intercept_outcome).

    Same freeze discipline as ProbabilityMapping: fit once on a training
    set, then use as-is and never retune while evaluating.
    """

    home_coef: float
    home_intercept: float
    draw_coef: float
    draw_intercept: float
    away_coef: float
    away_intercept: float
    engine_version: str
    trained_on: str
    n_games: int

    def predict_proba(self, rating_diff: float) -> tuple[float, float, float]:
        """Returns (home_prob, draw_prob, away_prob), summing to 1."""
        z_home = self.home_coef * rating_diff + self.home_intercept
        z_draw = self.draw_coef * rating_diff + self.draw_intercept
        z_away = self.away_coef * rating_diff + self.away_intercept

        m = max(z_home, z_draw, z_away)
        e_home, e_draw, e_away = math.exp(z_home - m), math.exp(z_draw - m), math.exp(z_away - m)
        total = e_home + e_draw + e_away
        return e_home / total, e_draw / total, e_away / total

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "ThreeWayProbabilityMapping":
        return cls(**json.loads(Path(path).read_text()))


def fit_three_way_probability_mapping(
    rating_diffs: list[float],
    outcomes: list[str],
    *,
    engine_version: str,
    trained_on: str,
) -> ThreeWayProbabilityMapping:
    """Fit P(outcome) via multinomial logistic regression on rating_diff.

    `outcomes` entries must each be "H" (home win), "D" (draw), or "A"
    (away win). Requires all three to actually appear in the training set
    -- with only two, scikit-learn silently collapses to a binary fit and
    there'd be no real coefficient for the missing side, which would be
    worse than refusing to fit at all.
    """
    if len(rating_diffs) != len(outcomes):
        raise ValueError("rating_diffs and outcomes must be the same length")

    valid = {"H", "D", "A"}
    invalid = set(outcomes) - valid
    if invalid:
        raise ValueError(f"outcomes must be one of {sorted(valid)}, got {sorted(invalid)}")

    missing = valid - set(outcomes)
    if missing:
        raise ValueError(f"Training set must include all three outcomes; missing {sorted(missing)}")

    x = np.array(rating_diffs, dtype=float).reshape(-1, 1)
    y = np.array(outcomes)

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    coefs = {
        cls: (float(c[0]), float(i))
        for cls, c, i in zip(model.classes_, model.coef_, model.intercept_)
    }
    home_coef, home_intercept = coefs["H"]
    draw_coef, draw_intercept = coefs["D"]
    away_coef, away_intercept = coefs["A"]

    return ThreeWayProbabilityMapping(
        home_coef=home_coef,
        home_intercept=home_intercept,
        draw_coef=draw_coef,
        draw_intercept=draw_intercept,
        away_coef=away_coef,
        away_intercept=away_intercept,
        engine_version=engine_version,
        trained_on=trained_on,
        n_games=len(rating_diffs),
    )
