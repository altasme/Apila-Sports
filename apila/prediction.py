"""Prediction engine: point-in-time ratings + a frozen probability mapping
-> a game prediction. This is what every later phase (betting sim, live
product) actually calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .calibration import ProbabilityMapping
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
    def __init__(self, store: PointInTimeStore, mapping: ProbabilityMapping):
        self.store = store
        self.mapping = mapping

    def predict(self, home_team_id: int, away_team_id: int, as_of: date) -> Prediction | None:
        diff = rating_diff(self.store, home_team_id, away_team_id, as_of)
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
