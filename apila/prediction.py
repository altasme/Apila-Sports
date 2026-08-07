"""Prediction engine: point-in-time ratings + a frozen probability mapping
-> a game prediction. This is what every later phase (betting sim, live
product) actually calls.

Two engines, one per market shape, mirroring the proposal's "separate MLB
engine" precedent (section 6): a binary engine for two-outcome sports
(basketball, MLB moneyline) and a three-way engine for soccer, where a
draw is a real third outcome rather than something to squeeze out of a
win/loss model. `sport` is bound at construction, since one engine
instance is always one sport + one engine version.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .calibration import ProbabilityMapping, ThreeWayProbabilityMapping
from .rating import rating_diff
from .store import PointInTimeStore


@dataclass
class Prediction:
    game_date: date
    home_team_id: int
    away_team_id: int
    rating_diff: float
    home_win_prob: float
    engine_version: str


class PredictionEngine:
    """Binary (two-outcome) prediction engine."""

    def __init__(self, store: PointInTimeStore, mapping: ProbabilityMapping, sport: str):
        self.store = store
        self.mapping = mapping
        self.sport = sport

    def predict(self, home_team_id: int, away_team_id: int, as_of: date) -> Prediction | None:
        diff = rating_diff(self.store, home_team_id, away_team_id, self.sport, as_of)
        if diff is None:
            return None

        return Prediction(
            game_date=as_of,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            rating_diff=diff,
            home_win_prob=self.mapping.predict_proba(diff),
            engine_version=self.mapping.engine_version,
        )


@dataclass
class ThreeWayPrediction:
    game_date: date
    home_team_id: int
    away_team_id: int
    rating_diff: float
    home_prob: float
    draw_prob: float
    away_prob: float
    engine_version: str


class ThreeWayPredictionEngine:
    """Three-outcome (home win / draw / away win) prediction engine, for soccer."""

    def __init__(self, store: PointInTimeStore, mapping: ThreeWayProbabilityMapping, sport: str):
        self.store = store
        self.mapping = mapping
        self.sport = sport

    def predict(
        self, home_team_id: int, away_team_id: int, as_of: date
    ) -> ThreeWayPrediction | None:
        diff = rating_diff(self.store, home_team_id, away_team_id, self.sport, as_of)
        if diff is None:
            return None

        home_prob, draw_prob, away_prob = self.mapping.predict_proba(diff)
        return ThreeWayPrediction(
            game_date=as_of,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            rating_diff=diff,
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            engine_version=self.mapping.engine_version,
        )
