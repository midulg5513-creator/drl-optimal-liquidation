"""Almgren-Chriss-style liquidation environment.

The environment is intentionally lightweight and fully reproducible so the
research demo can run without external RL frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from config import ExperimentConfig
from execution_reference import reference_inventory_targets, reference_sell_fraction


@dataclass
class StepInfo:
    """Metadata returned after each environment step."""

    executed_quantity: float
    executed_fraction: float
    reference_fraction: float
    reference_inventory_after: float
    inventory_gap_after: float
    execution_price: float
    inventory_after: float
    mid_price_after: float
    step_cost: float
    normalized_value_drop: float
    backlog_penalty: float
    deviation_penalty: float
    cumulative_reward: float
    implementation_shortfall: float
    done: bool


class OptimalExecutionEnv:
    """Continuous-action optimal liquidation environment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.reference_inventory_path = reference_inventory_targets(config)
        self.rng = np.random.default_rng(config.random_seed)
        self.price = config.initial_price
        self.inventory = config.initial_inventory
        self.step_index = 0
        self.cash = 0.0
        self.cumulative_reward = 0.0
        self.return_history = [0.0 for _ in range(config.lookback)]

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.price = self.config.initial_price
        self.inventory = self.config.initial_inventory
        self.step_index = 0
        self.cash = 0.0
        self.cumulative_reward = 0.0
        self.return_history = [0.0 for _ in range(self.config.lookback)]
        return self._build_state()

    def _remaining_steps(self) -> int:
        return self.config.horizon - self.step_index

    def _reference_fraction(self) -> float:
        return reference_sell_fraction(
            self.config,
            self.inventory,
            self.step_index,
            self.reference_inventory_path,
        )

    def _risk_feature(self) -> float:
        return float(np.clip(self.config.risk_aversion * 1000.0, 0.0, 2.0))

    def _inventory_gap(self) -> float:
        reference_inventory_now = self.reference_inventory_path[min(self.step_index, self.config.horizon)]
        return (self.inventory - reference_inventory_now) / self.config.initial_inventory

    def _build_state(self) -> np.ndarray:
        remaining_time = self._remaining_steps() / self.config.horizon
        inventory_ratio = self.inventory / self.config.initial_inventory
        price_deviation = self.price / self.config.initial_price - 1.0
        reference_fraction = self._reference_fraction()
        inventory_gap = self._inventory_gap()
        risk_feature = self._risk_feature()
        state = np.array(
            self.return_history
            + [remaining_time, inventory_ratio, price_deviation, reference_fraction, inventory_gap, risk_feature],
            dtype=np.float64,
        )
        return state

    def _market_noise(self) -> float:
        variance = self.config.volatility * np.sqrt(self.config.dt)
        diffusion = self.rng.normal(loc=self.config.drift * self.config.dt, scale=variance)
        return diffusion

    def _map_action_to_fraction(self, policy_action: float, reference_fraction: float) -> float:
        lower = max(0.0, reference_fraction - self.config.action_deviation_limit)
        upper = min(1.0, reference_fraction + self.config.action_deviation_limit)
        if policy_action <= 0.5:
            scaled = 0.0 if reference_fraction <= lower else policy_action / 0.5
            return float(lower + scaled * (reference_fraction - lower))
        scaled = (policy_action - 0.5) / 0.5
        return float(reference_fraction + scaled * (upper - reference_fraction))

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Execute one liquidation step."""
        if self.step_index >= self.config.horizon:
            raise RuntimeError("Environment already terminated.")

        policy_action = float(np.clip(action, 0.0, 1.0))
        remaining_steps = self._remaining_steps()
        reference_fraction = self._reference_fraction()
        if remaining_steps <= 1:
            executed_fraction = 1.0
        else:
            executed_fraction = self._map_action_to_fraction(policy_action, reference_fraction)
        executed_quantity = float(np.clip(executed_fraction * self.inventory, 0.0, self.inventory))

        pre_trade_price = self.price
        cash_before = self.cash
        inventory_before = self.inventory
        portfolio_before = cash_before + inventory_before * pre_trade_price

        participation = executed_quantity / self.config.initial_inventory
        temp_impact = self.config.temporary_impact * participation + self.config.fixed_fee
        execution_price = max(0.01, pre_trade_price - temp_impact)
        proceeds = executed_quantity * execution_price

        self.cash += proceeds
        inventory_after = self.inventory - executed_quantity

        permanent_impact = self.config.permanent_impact * participation
        diffusion = self._market_noise()
        next_price = max(0.01, pre_trade_price * np.exp(diffusion) - permanent_impact)

        raw_step_cost = executed_quantity * (pre_trade_price - execution_price)
        portfolio_after = self.cash + inventory_after * next_price
        normalized_value_drop = (portfolio_before - portfolio_after) / (
            self.config.initial_price * self.config.initial_inventory
        )
        inventory_penalty = self.config.risk_aversion * (inventory_after / self.config.initial_inventory) ** 2
        reference_inventory_after = self.reference_inventory_path[min(self.step_index + 1, self.config.horizon)]
        inventory_gap_after = max(
            0.0,
            (inventory_after - reference_inventory_after) / self.config.initial_inventory,
        )
        backlog_penalty = self.config.reference_backlog_penalty * (inventory_gap_after ** 2)
        deviation_penalty = 0.0
        reward = -(normalized_value_drop + inventory_penalty + backlog_penalty + deviation_penalty)

        price_return = np.log(next_price / pre_trade_price)
        self.return_history = self.return_history[1:] + [price_return]

        self.price = next_price
        self.inventory = inventory_after
        self.step_index += 1
        self.cumulative_reward += reward
        done = self.step_index >= self.config.horizon or self.inventory <= 1.0e-8

        implementation_shortfall = (
            self.config.initial_inventory * self.config.initial_price - self.cash
        )

        info = StepInfo(
            executed_quantity=executed_quantity,
            executed_fraction=executed_fraction,
            reference_fraction=reference_fraction,
            reference_inventory_after=reference_inventory_after,
            inventory_gap_after=inventory_gap_after,
            execution_price=execution_price,
            inventory_after=inventory_after,
            mid_price_after=next_price,
            step_cost=raw_step_cost,
            normalized_value_drop=normalized_value_drop,
            backlog_penalty=backlog_penalty,
            deviation_penalty=deviation_penalty,
            cumulative_reward=self.cumulative_reward,
            implementation_shortfall=implementation_shortfall,
            done=done,
        )
        return self._build_state(), reward, done, info.__dict__
