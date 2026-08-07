from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apila.db import get_engine, get_session
from apila.store import PointInTimeStore

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_game_logs.csv"


@pytest.fixture
def store() -> PointInTimeStore:
    engine = get_engine(":memory:")
    session = get_session(engine)
    store = PointInTimeStore(session)

    df = pd.read_csv(FIXTURE_PATH, parse_dates=["game_date"])
    df["game_date"] = df["game_date"].dt.date
    store.ingest(df)

    return store
