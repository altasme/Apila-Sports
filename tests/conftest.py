from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apila.db import get_engine, get_session
from apila.store import PointInTimeStore

GAME_LOGS_FIXTURE = Path(__file__).parent / "fixtures" / "sample_game_logs.csv"
CLOSING_ODDS_FIXTURE = Path(__file__).parent / "fixtures" / "sample_closing_odds.csv"


@pytest.fixture
def store() -> PointInTimeStore:
    engine = get_engine(":memory:")
    session = get_session(engine)
    store = PointInTimeStore(session)

    games = pd.read_csv(GAME_LOGS_FIXTURE, parse_dates=["game_date"])
    games["game_date"] = games["game_date"].dt.date
    store.ingest(games)

    odds = pd.read_csv(CLOSING_ODDS_FIXTURE, parse_dates=["game_date"])
    odds["game_date"] = odds["game_date"].dt.date
    store.ingest_odds(odds)

    return store
