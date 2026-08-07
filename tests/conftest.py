from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apila.db import get_engine, get_session
from apila.store import PointInTimeStore

FIXTURES = Path(__file__).parent / "fixtures"


def _load_dated_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    df["game_date"] = df["game_date"].dt.date
    return df


@pytest.fixture
def store() -> PointInTimeStore:
    """A store loaded with both fixture sports: basketball (sport="nba",
    teams AAA/BBB, no draws) and soccer (sport="soccer", teams CCC/DDD,
    includes draws). Sport filtering keeps them from contaminating each
    other's ratings -- most tests only touch one sport at a time.
    """
    engine = get_engine(":memory:")
    session = get_session(engine)
    store = PointInTimeStore(session)

    store.ingest(_load_dated_csv(FIXTURES / "sample_game_logs.csv"))
    store.ingest_odds(_load_dated_csv(FIXTURES / "sample_closing_odds.csv"))
    store.ingest(_load_dated_csv(FIXTURES / "sample_soccer_matches.csv"))
    store.ingest_odds(_load_dated_csv(FIXTURES / "sample_soccer_odds.csv"))

    return store
