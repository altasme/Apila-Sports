from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from apila.backtest import build_game_records
from apila.db import get_engine, get_session
from apila.store import PointInTimeStore


def test_build_game_records_skips_first_game_for_lack_of_history(store):
    records = build_game_records(store, "nba")
    # g1 is the very first game for both AAA and BBB -- neither has prior
    # history yet, so it can never produce a rating_diff.
    assert {r.game_id for r in records} == {"g2", "g3", "g4", "g5"}


def test_build_game_records_outcomes_match_home_team_result(store):
    records = {r.game_id: r for r in build_game_records(store, "nba")}
    # g2: BBB home, won -> "H". g4: BBB home, lost -> "A".
    assert records["g2"].outcome == "H"
    assert records["g4"].outcome == "A"


def test_build_game_records_respects_since_and_until(store):
    records = build_game_records(store, "nba", since=date(2024, 1, 7))
    assert {r.game_id for r in records} == {"g4", "g5"}

    records = build_game_records(store, "nba", until=date(2024, 1, 7))
    assert {r.game_id for r in records} == {"g2", "g3"}


def test_build_game_records_includes_draw_moneyline_for_soccer(store):
    records = {r.game_id: r for r in build_game_records(store, "soccer")}
    # m2 is CCC/DDD's second meeting -- both teams have one prior game by then.
    assert "m2" in records
    assert records["m2"].draw_moneyline == 210
    assert records["m2"].outcome == "D"


def test_build_game_records_soccer_and_nba_are_isolated(store):
    nba_ids = {r.game_id for r in build_game_records(store, "nba")}
    soccer_ids = {r.game_id for r in build_game_records(store, "soccer")}
    assert nba_ids.isdisjoint(soccer_ids)


def test_build_game_records_skips_games_with_no_ingested_odds():
    engine = get_engine(":memory:")
    session = get_session(engine)
    store = PointInTimeStore(session)

    games = pd.DataFrame(
        [
            dict(
                game_id="x1", team_id=1, sport="nba", season="2023-24",
                game_date=date(2024, 1, 1), team_abbr="XXX", opponent_abbr="YYY",
                is_home=True, wl="W", pts=100, plus_minus=10,
            ),
            dict(
                game_id="x1", team_id=2, sport="nba", season="2023-24",
                game_date=date(2024, 1, 1), team_abbr="YYY", opponent_abbr="XXX",
                is_home=False, wl="L", pts=90, plus_minus=-10,
            ),
            dict(
                game_id="x2", team_id=1, sport="nba", season="2023-24",
                game_date=date(2024, 1, 3), team_abbr="XXX", opponent_abbr="YYY",
                is_home=True, wl="W", pts=100, plus_minus=5,
            ),
            dict(
                game_id="x2", team_id=2, sport="nba", season="2023-24",
                game_date=date(2024, 1, 3), team_abbr="YYY", opponent_abbr="XXX",
                is_home=False, wl="L", pts=95, plus_minus=-5,
            ),
        ]
    )
    store.ingest(games)
    # No ingest_odds call at all -- x2 has a computable rating_diff
    # (x1 is prior history) but no market to join against.
    assert build_game_records(store, "nba") == []
