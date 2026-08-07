"""Deterministic composite team rating from point-in-time features.

Blends season-to-date point differential with recent-form point
differential. The weights are priors, not truth (proposal section 6) --
a reasonable starting point, not something to defend. They get superseded
once there's enough real data to actually validate weight choices.
"""
from __future__ import annotations

from datetime import date

from .store import PointInTimeStore, TeamRating

SEASON_WEIGHT = 0.7
RECENT_FORM_WEIGHT = 0.3


def composite_rating(rating: TeamRating) -> float:
    recent = rating.last10_point_diff if rating.last10_point_diff is not None else rating.point_diff
    return SEASON_WEIGHT * rating.point_diff + RECENT_FORM_WEIGHT * recent


def team_rating(store: PointInTimeStore, team_id: int, as_of: date) -> float | None:
    rating = store.team_rating_asof(team_id, as_of)
    if rating is None:
        return None
    return composite_rating(rating)


def rating_diff(
    store: PointInTimeStore, home_team_id: int, away_team_id: int, as_of: date
) -> float | None:
    home = team_rating(store, home_team_id, as_of)
    away = team_rating(store, away_team_id, as_of)
    if home is None or away is None:
        return None
    return home - away
