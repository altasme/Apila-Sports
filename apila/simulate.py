"""Flat-stake betting simulation (proposal section 4.5) — the true metric.

Deliberately sport-agnostic: a game is just a list of sides, each with a
model probability and a moneyline. Basketball/MLB pass two sides (home,
away); soccer passes three (home, draw, away). The decision rule and
staking logic don't change either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .odds import american_to_decimal, american_to_implied_prob


@dataclass
class SideBet:
    side: str
    model_prob: float
    moneyline: int
    won: bool


@dataclass
class GameBet:
    game_id: str
    game_date: date
    sides: list[SideBet]


@dataclass
class BetOutcome:
    game_id: str
    game_date: date
    side: str
    stake: float
    decimal_odds: float
    model_prob: float
    break_even_prob: float
    won: bool
    profit: float


def simulate_flat_stake(games: list[GameBet], stake: float = 1.0) -> list[BetOutcome]:
    """Bet a side only when model_prob > break-even implied probability
    (positive EV) at that side's actual price -- not the devigged market
    probability, which is a separate benchmark (see PointInTimeStore.
    market_probability), not the price you'd actually get paid at.
    """
    outcomes: list[BetOutcome] = []
    for game in games:
        for side in game.sides:
            break_even = american_to_implied_prob(side.moneyline)
            if side.model_prob <= break_even:
                continue

            decimal_odds = american_to_decimal(side.moneyline)
            profit = stake * (decimal_odds - 1) if side.won else -stake

            outcomes.append(
                BetOutcome(
                    game_id=game.game_id,
                    game_date=game.game_date,
                    side=side.side,
                    stake=stake,
                    decimal_odds=decimal_odds,
                    model_prob=side.model_prob,
                    break_even_prob=break_even,
                    won=side.won,
                    profit=profit,
                )
            )
    return outcomes


@dataclass
class SimulationSummary:
    n_bets: int
    total_staked: float
    total_profit: float
    roi: float


def summarize(outcomes: list[BetOutcome]) -> SimulationSummary:
    if not outcomes:
        return SimulationSummary(n_bets=0, total_staked=0.0, total_profit=0.0, roi=0.0)

    total_staked = sum(o.stake for o in outcomes)
    total_profit = sum(o.profit for o in outcomes)
    return SimulationSummary(
        n_bets=len(outcomes),
        total_staked=total_staked,
        total_profit=total_profit,
        roi=total_profit / total_staked,
    )


def summarize_by_tier(
    outcomes: list[BetOutcome], tier_fn
) -> dict[str, SimulationSummary]:
    """`tier_fn(outcome: BetOutcome) -> str` assigns each bet to a tier
    (see apila.tiers.confidence_tier). Returns a summary per tier.
    """
    by_tier: dict[str, list[BetOutcome]] = {}
    for outcome in outcomes:
        by_tier.setdefault(tier_fn(outcome), []).append(outcome)
    return {tier: summarize(bets) for tier, bets in by_tier.items()}
