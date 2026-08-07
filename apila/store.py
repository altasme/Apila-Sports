from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import TeamGameLog


@dataclass
class TeamRating:
    team_id: int
    as_of: date
    games_played: int
    wins: int
    losses: int
    win_pct: float
    ppg: float
    opp_ppg: float
    point_diff: float
    last10_win_pct: float | None
    last10_point_diff: float | None


class PointInTimeStore:
    """Feature store where every read is parameterized by an as-of date.

    Every method here filters on `game_date < as_of` before computing
    anything. That's the entire lookahead-bias defense: there is no
    precomputed "current rating" a caller could accidentally read past its
    valid date, because ratings don't exist independent of the date they
    were asked for.
    """

    def __init__(self, session: Session):
        self.session = session

    def ingest(self, rows: pd.DataFrame) -> int:
        records = rows.to_dict(orient="records")
        for record in records:
            self.session.merge(TeamGameLog(**record))
        self.session.commit()
        return len(records)

    def team_games_before(
        self,
        team_id: int,
        as_of: date,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Games strictly before `as_of`, most recent first."""
        stmt = (
            select(TeamGameLog)
            .where(TeamGameLog.team_id == team_id)
            .where(TeamGameLog.game_date < as_of)
            .order_by(TeamGameLog.game_date.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = self.session.execute(stmt).scalars().all()
        return pd.DataFrame([_row_to_dict(r) for r in rows])

    def team_rating_asof(self, team_id: int, as_of: date) -> TeamRating | None:
        games = self.team_games_before(team_id, as_of)
        if games.empty:
            return None

        opp_pts = games["pts"] - games["plus_minus"]
        wins = int((games["wl"] == "W").sum())
        losses = len(games) - wins

        last10 = games.head(10)
        last10_win_pct = float((last10["wl"] == "W").mean()) if not last10.empty else None
        last10_point_diff = float(last10["plus_minus"].mean()) if not last10.empty else None

        return TeamRating(
            team_id=team_id,
            as_of=as_of,
            games_played=len(games),
            wins=wins,
            losses=losses,
            win_pct=wins / len(games),
            ppg=float(games["pts"].mean()),
            opp_ppg=float(opp_pts.mean()),
            point_diff=float(games["plus_minus"].mean()),
            last10_win_pct=last10_win_pct,
            last10_point_diff=last10_point_diff,
        )


def _row_to_dict(row: TeamGameLog) -> dict:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}
