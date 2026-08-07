from __future__ import annotations

from sqlalchemy import Boolean, Column, Date, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TeamGameLog(Base):
    """One row per team per game. This is the only table in the store.

    Point-in-time correctness falls out of the data itself: a row's stats
    are only knowable once `game_date` has happened, so filtering on
    `game_date < as_of` is sufficient to prevent lookahead bias. There is
    deliberately no separate "team rating" table — ratings are always
    derived on read from this raw log, parameterized by an as-of date.
    """

    __tablename__ = "team_game_logs"

    game_id = Column(String, primary_key=True)
    team_id = Column(Integer, primary_key=True)
    season = Column(String, nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    team_abbr = Column(String, nullable=False)
    opponent_abbr = Column(String, nullable=False)
    is_home = Column(Boolean, nullable=False)
    wl = Column(String, nullable=False)
    pts = Column(Integer, nullable=False)
    plus_minus = Column(Float, nullable=False)
    fgm = Column(Float)
    fga = Column(Float)
    fg_pct = Column(Float)
    fg3m = Column(Float)
    fg3a = Column(Float)
    fg3_pct = Column(Float)
    ftm = Column(Float)
    fta = Column(Float)
    ft_pct = Column(Float)
    oreb = Column(Float)
    dreb = Column(Float)
    reb = Column(Float)
    ast = Column(Float)
    stl = Column(Float)
    blk = Column(Float)
    tov = Column(Float)
    pf = Column(Float)


class ClosingOdds(Base):
    """Closing moneyline for one historical game — the market benchmark.

    Keyed by (date, home, away) rather than a provider game id, since odds
    data and stats data almost never share an id scheme; this join key is
    the one thing every source agrees on.
    """

    __tablename__ = "closing_odds"

    game_date = Column(Date, primary_key=True)
    home_team_abbr = Column(String, primary_key=True)
    away_team_abbr = Column(String, primary_key=True)
    home_moneyline = Column(Integer, nullable=False)
    away_moneyline = Column(Integer, nullable=False)
    source = Column(String, nullable=False)
