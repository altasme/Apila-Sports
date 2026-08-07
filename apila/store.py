from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ClosingOdds, TeamGameLog
from .odds import american_to_implied_prob, devig_1x2, devig_moneyline


@dataclass
class TeamRating:
    team_id: int
    sport: str
    as_of: date
    games_played: int
    wins: int
    draws: int
    losses: int
    win_pct: float
    draw_pct: float
    ppg: float
    opp_ppg: float
    point_diff: float
    last10_win_pct: float | None
    last10_point_diff: float | None


@dataclass
class MarketProbability:
    game_date: date
    home_team_abbr: str
    away_team_abbr: str
    sport: str
    home_moneyline: int
    away_moneyline: int
    home_prob: float
    away_prob: float
    vig: float
    draw_moneyline: int | None = None
    draw_prob: float | None = None

    @property
    def is_three_way(self) -> bool:
        return self.draw_moneyline is not None


class PointInTimeStore:
    """Feature store where every read is parameterized by an as-of date.

    Every method here filters on `game_date < as_of` before computing
    anything. That's the entire lookahead-bias defense: there is no
    precomputed "current rating" a caller could accidentally read past its
    valid date, because ratings don't exist independent of the date they
    were asked for.

    `sport` is required on every read because `team_id` is only unique
    within a sport -- two different providers' ids can collide, and
    silently mixing e.g. an NBA team's games into a soccer team's rating
    would be a much worse bug than a required parameter.
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
        sport: str,
        as_of: date,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Games strictly before `as_of`, most recent first."""
        stmt = (
            select(TeamGameLog)
            .where(TeamGameLog.team_id == team_id)
            .where(TeamGameLog.sport == sport)
            .where(TeamGameLog.game_date < as_of)
            .order_by(TeamGameLog.game_date.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = self.session.execute(stmt).scalars().all()
        return pd.DataFrame([_row_to_dict(r) for r in rows])

    def team_rating_asof(self, team_id: int, sport: str, as_of: date) -> TeamRating | None:
        games = self.team_games_before(team_id, sport, as_of)
        if games.empty:
            return None

        opp_pts = games["pts"] - games["plus_minus"]
        wins = int((games["wl"] == "W").sum())
        draws = int((games["wl"] == "D").sum())
        losses = len(games) - wins - draws

        last10 = games.head(10)
        last10_win_pct = float((last10["wl"] == "W").mean()) if not last10.empty else None
        last10_point_diff = float(last10["plus_minus"].mean()) if not last10.empty else None

        return TeamRating(
            team_id=team_id,
            sport=sport,
            as_of=as_of,
            games_played=len(games),
            wins=wins,
            draws=draws,
            losses=losses,
            win_pct=wins / len(games),
            draw_pct=draws / len(games),
            ppg=float(games["pts"].mean()),
            opp_ppg=float(opp_pts.mean()),
            point_diff=float(games["plus_minus"].mean()),
            last10_win_pct=last10_win_pct,
            last10_point_diff=last10_point_diff,
        )

    def ingest_odds(self, rows: pd.DataFrame) -> int:
        records = rows.to_dict(orient="records")
        for record in records:
            self.session.merge(ClosingOdds(**record))
        self.session.commit()
        return len(records)

    def closing_odds(
        self, game_date: date, home_team_abbr: str, away_team_abbr: str, sport: str
    ) -> ClosingOdds | None:
        stmt = select(ClosingOdds).where(
            ClosingOdds.game_date == game_date,
            ClosingOdds.home_team_abbr == home_team_abbr,
            ClosingOdds.away_team_abbr == away_team_abbr,
            ClosingOdds.sport == sport,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def market_probability(
        self, game_date: date, home_team_abbr: str, away_team_abbr: str, sport: str
    ) -> MarketProbability | None:
        """Devigged closing-line probability for a historical game — the
        number a prediction has to beat. None if no odds were ingested for
        this game.

        Branches on whether a draw price was ingested: two-outcome markets
        (basketball, MLB moneyline) devig home/away only; three-outcome
        markets (soccer 1X2) devig all three sides together.
        """
        odds = self.closing_odds(game_date, home_team_abbr, away_team_abbr, sport)
        if odds is None:
            return None

        if odds.draw_moneyline is not None:
            home_prob, draw_prob, away_prob = devig_1x2(
                odds.home_moneyline, odds.draw_moneyline, odds.away_moneyline
            )
            raw_total = (
                american_to_implied_prob(odds.home_moneyline)
                + american_to_implied_prob(odds.draw_moneyline)
                + american_to_implied_prob(odds.away_moneyline)
            )
        else:
            home_prob, away_prob = devig_moneyline(odds.home_moneyline, odds.away_moneyline)
            draw_prob = None
            raw_total = american_to_implied_prob(odds.home_moneyline) + american_to_implied_prob(
                odds.away_moneyline
            )

        return MarketProbability(
            game_date=game_date,
            home_team_abbr=home_team_abbr,
            away_team_abbr=away_team_abbr,
            sport=sport,
            home_moneyline=odds.home_moneyline,
            away_moneyline=odds.away_moneyline,
            draw_moneyline=odds.draw_moneyline,
            home_prob=home_prob,
            away_prob=away_prob,
            draw_prob=draw_prob,
            vig=raw_total - 1.0,
        )


def _row_to_dict(row: TeamGameLog) -> dict:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}
