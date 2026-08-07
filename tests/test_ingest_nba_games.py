from __future__ import annotations

from datetime import date

from scripts.ingest_nba_games import transform


def _game(**overrides) -> dict:
    base = dict(
        id=1,
        date="2024-01-01T00:00:00.000Z",
        status="Final",
        home_team={"id": 14, "abbreviation": "LAL"},
        visitor_team={"id": 2, "abbreviation": "BOS"},
        home_team_score=110,
        visitor_team_score=105,
    )
    base.update(overrides)
    return base


def test_transform_produces_real_date_objects_not_strings():
    # Regression: transform() used to leave game_date as an ISO string,
    # which SQLAlchemy's Date column silently rejects at insert time --
    # this only ever showed up against a real ingestion run, not in any
    # existing test, since nothing here previously touched this script.
    df = transform([_game()], season=2023)
    assert isinstance(df.iloc[0]["game_date"], date)
    assert df.iloc[0]["game_date"] == date(2024, 1, 1)


def test_transform_produces_home_and_away_rows_with_correct_result():
    df = transform([_game(home_team_score=110, visitor_team_score=105)], season=2023)
    assert len(df) == 2

    home_row = df[df["team_abbr"] == "LAL"].iloc[0]
    away_row = df[df["team_abbr"] == "BOS"].iloc[0]

    assert bool(home_row["is_home"]) is True
    assert home_row["wl"] == "W"
    assert home_row["pts"] == 110
    assert home_row["plus_minus"] == 5.0

    assert bool(away_row["is_home"]) is False
    assert away_row["wl"] == "L"
    assert away_row["pts"] == 105
    assert away_row["plus_minus"] == -5.0


def test_transform_skips_games_not_final():
    df = transform([_game(status="Scheduled", home_team_score=0, visitor_team_score=0)], season=2023)
    assert df.empty


def test_transform_skips_tied_scores():
    df = transform([_game(home_team_score=100, visitor_team_score=100)], season=2023)
    assert df.empty


def test_transform_output_ingests_cleanly_into_the_real_store():
    # The actual regression check: does this data survive a real
    # PointInTimeStore.ingest() call, not just look right in a DataFrame.
    from apila.db import get_engine, get_session
    from apila.store import PointInTimeStore

    df = transform([_game()], season=2023)

    engine = get_engine(":memory:")
    session = get_session(engine)
    store = PointInTimeStore(session)
    n = store.ingest(df)

    assert n == 2
    rating = store.team_rating_asof(14, "nba", as_of=date(2024, 1, 2))
    assert rating is not None
    assert rating.wins == 1
