from __future__ import annotations

from datetime import date

import pytest

from apila.odds import american_to_decimal
from apila.simulate import GameBet, SideBet, simulate_flat_stake, summarize, summarize_by_tier
from apila.tiers import confidence_tier

GAME_DATE = date(2024, 3, 1)


def test_bets_placed_only_when_model_beats_break_even():
    game = GameBet(
        game_id="gA",
        game_date=GAME_DATE,
        sides=[
            # break-even = 150/250 = 0.6; model 0.65 clears it.
            SideBet(side="home", model_prob=0.65, moneyline=-150, won=True),
            # break-even = 100/230 = 0.4348; model's implied 0.35 doesn't clear it.
            SideBet(side="away", model_prob=0.35, moneyline=130, won=False),
        ],
    )

    outcomes = simulate_flat_stake([game], stake=10.0)

    assert len(outcomes) == 1
    assert outcomes[0].side == "home"


def test_no_bets_when_no_side_has_positive_ev():
    game = GameBet(
        game_id="gB",
        game_date=GAME_DATE,
        sides=[
            SideBet(side="home", model_prob=0.55, moneyline=-150, won=True),  # break-even 0.6
            SideBet(side="away", model_prob=0.30, moneyline=130, won=False),  # break-even 0.435
        ],
    )
    assert simulate_flat_stake([game], stake=10.0) == []


def test_profit_on_win_matches_decimal_odds_payout():
    game = GameBet(
        game_id="gC",
        game_date=GAME_DATE,
        sides=[SideBet(side="home", model_prob=0.65, moneyline=-150, won=True)],
    )
    [outcome] = simulate_flat_stake([game], stake=10.0)

    expected_decimal_odds = american_to_decimal(-150)
    assert outcome.decimal_odds == pytest.approx(expected_decimal_odds)
    assert outcome.profit == pytest.approx(10.0 * (expected_decimal_odds - 1))


def test_profit_on_loss_is_negative_stake():
    game = GameBet(
        game_id="gD",
        game_date=GAME_DATE,
        sides=[SideBet(side="home", model_prob=0.65, moneyline=-150, won=False)],
    )
    [outcome] = simulate_flat_stake([game], stake=10.0)
    assert outcome.profit == pytest.approx(-10.0)


def test_three_way_game_can_place_more_than_one_bet():
    # Fabricated disagreement: model favors both home and away over what
    # their individual break-evens require. The simulator has no opinion
    # on whether that's realistic -- it just evaluates each side on its
    # own merits, same as it would for two sides.
    game = GameBet(
        game_id="gE",
        game_date=GAME_DATE,
        sides=[
            SideBet(side="home", model_prob=0.55, moneyline=-120, won=True),  # break-even ~0.545
            SideBet(side="draw", model_prob=0.20, moneyline=220, won=False),  # break-even ~0.3125
            SideBet(side="away", model_prob=0.35, moneyline=280, won=False),  # break-even ~0.263
        ],
    )
    outcomes = simulate_flat_stake([game], stake=10.0)
    sides_bet = {o.side for o in outcomes}
    assert sides_bet == {"home", "away"}


def test_summarize_computes_roi():
    game = GameBet(
        game_id="gF",
        game_date=GAME_DATE,
        sides=[SideBet(side="home", model_prob=0.65, moneyline=-150, won=True)],
    )
    outcomes = simulate_flat_stake([game], stake=10.0)
    summary = summarize(outcomes)

    assert summary.n_bets == 1
    assert summary.total_staked == pytest.approx(10.0)
    assert summary.roi == pytest.approx(summary.total_profit / summary.total_staked)


def test_summarize_empty_list_is_zeroed_not_a_division_error():
    summary = summarize([])
    assert summary.n_bets == 0
    assert summary.roi == 0.0


def test_summarize_by_tier_groups_bets_by_confidence():
    games = [
        GameBet(
            game_id="g1",
            game_date=GAME_DATE,
            sides=[SideBet(side="home", model_prob=0.85, moneyline=-150, won=True)],  # high tier
        ),
        GameBet(
            game_id="g2",
            game_date=GAME_DATE,
            sides=[SideBet(side="home", model_prob=0.62, moneyline=-105, won=False)],  # moderate tier
        ),
    ]
    outcomes = simulate_flat_stake(games, stake=10.0)
    by_tier = summarize_by_tier(outcomes, lambda o: confidence_tier(o.model_prob))

    assert set(by_tier.keys()) == {"high", "moderate"}
    assert by_tier["high"].n_bets == 1
    assert by_tier["moderate"].n_bets == 1
