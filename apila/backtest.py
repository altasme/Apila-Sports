"""Assembles per-game backtest records: point-in-time rating diff, actual
outcome, and closing market odds, all joined for one game. Shared plumbing
between fitting (scripts/fit_probability_mapping.py builds a lighter
version of the same walk) and evaluation (scripts/run_backtest.py), kept
here so both walk the same query pattern and can't drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select

from .baselines import record_strength
from .models import TeamGameLog
from .rating import rating_diff
from .store import PointInTimeStore

_OUTCOME_MAP = {"W": "H", "D": "D", "L": "A"}


@dataclass
class GameRecord:
    game_id: str
    game_date: date
    sport: str
    home_team_id: int
    away_team_id: int
    home_abbr: str
    away_abbr: str
    rating_diff: float
    record_diff: float
    outcome: str  # "H" / "D" / "A"
    home_moneyline: int | None
    away_moneyline: int | None
    draw_moneyline: int | None

    @property
    def has_odds(self) -> bool:
        return self.home_moneyline is not None


def build_game_records(
    store: PointInTimeStore,
    sport: str,
    *,
    since: date | None = None,
    until: date | None = None,
    require_odds: bool = True,
) -> list[GameRecord]:
    """One record per game with a rating_diff, and closing odds when available.

    A game missing rating history for either team is always skipped -- there's
    no prediction to evaluate without one. A game missing closing odds is
    skipped too when `require_odds` is True (the default, and what betting
    simulation needs). Pass `require_odds=False` to keep those games anyway,
    with moneyline fields left None (`GameRecord.has_odds` is False) -- useful
    for evaluating raw prediction quality (accuracy/Brier/log loss against
    baselines) over a dataset wider than whatever odds happen to be ingested.
    """
    stmt = select(TeamGameLog).where(TeamGameLog.sport == sport).where(TeamGameLog.is_home.is_(True))
    if since is not None:
        stmt = stmt.where(TeamGameLog.game_date >= since)
    if until is not None:
        stmt = stmt.where(TeamGameLog.game_date < until)
    stmt = stmt.order_by(TeamGameLog.game_date)

    home_rows = store.session.execute(stmt).scalars().all()

    records: list[GameRecord] = []
    for row in home_rows:
        away = store.session.execute(
            select(TeamGameLog).where(
                TeamGameLog.game_id == row.game_id,
                TeamGameLog.sport == sport,
                TeamGameLog.is_home.is_(False),
            )
        ).scalar_one_or_none()
        if away is None:
            continue

        diff = rating_diff(store, row.team_id, away.team_id, sport, row.game_date)
        if diff is None:
            continue  # not enough prior history yet for one of the teams

        home_rating = store.team_rating_asof(row.team_id, sport, row.game_date)
        away_rating = store.team_rating_asof(away.team_id, sport, row.game_date)
        record_diff = record_strength(home_rating) - record_strength(away_rating)

        market = store.market_probability(row.game_date, row.team_abbr, away.team_abbr, sport)
        if market is None:
            if require_odds:
                continue  # no odds ingested for this game
            home_moneyline = away_moneyline = draw_moneyline = None
        else:
            home_moneyline = market.home_moneyline
            away_moneyline = market.away_moneyline
            draw_moneyline = market.draw_moneyline

        records.append(
            GameRecord(
                game_id=row.game_id,
                game_date=row.game_date,
                sport=sport,
                home_team_id=row.team_id,
                away_team_id=away.team_id,
                home_abbr=row.team_abbr,
                away_abbr=away.team_abbr,
                rating_diff=diff,
                record_diff=record_diff,
                outcome=_OUTCOME_MAP[row.wl],
                home_moneyline=home_moneyline,
                away_moneyline=away_moneyline,
                draw_moneyline=draw_moneyline,
            )
        )
    return records
