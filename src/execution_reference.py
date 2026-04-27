"""Reference liquidation trajectory helpers."""

from __future__ import annotations

import math

import numpy as np

from config import ExperimentConfig


def ac_inventory_path(config: ExperimentConfig) -> np.ndarray:
    """Compute the discrete Almgren-Chriss inventory path."""
    horizon = config.horizon
    dt = config.dt
    eta = max(config.temporary_impact, 1.0e-6)
    lam = max(config.risk_aversion, 1.0e-9)
    sigma = config.volatility * config.initial_price

    kappa_hat_sq = lam * sigma * sigma / eta
    if kappa_hat_sq < 1.0e-10:
        inventory = np.linspace(config.initial_inventory, 0.0, horizon + 1)
        inventory[-1] = 0.0
        return inventory

    kappa = math.acosh(1.0 + 0.5 * kappa_hat_sq * dt * dt) / dt
    if abs(kappa) < 1.0e-9:
        inventory = np.linspace(config.initial_inventory, 0.0, horizon + 1)
        inventory[-1] = 0.0
        return inventory

    inventory = []
    for step in range(horizon + 1):
        remaining_time = config.total_time - step * dt
        position = (
            math.sinh(kappa * remaining_time) / math.sinh(kappa * config.total_time)
        ) * config.initial_inventory
        inventory.append(position)
    inventory_array = np.maximum(np.array(inventory, dtype=np.float64), 0.0)
    inventory_array[-1] = 0.0
    return inventory_array


def reference_inventory_targets(config: ExperimentConfig) -> np.ndarray:
    """Return the reference inventory trajectory for the current config."""
    return ac_inventory_path(config)


def reference_sell_fraction(
    config: ExperimentConfig,
    inventory_now: float,
    step_index: int,
    reference_inventory_path: np.ndarray,
) -> float:
    """Compute the sell fraction needed to move toward the next reference inventory."""
    remaining_steps = config.horizon - step_index
    if remaining_steps <= 1 or inventory_now <= 1.0e-12:
        return 1.0 if remaining_steps <= 1 else 0.0
    next_target_inventory = reference_inventory_path[min(step_index + 1, config.horizon)]
    fraction = (inventory_now - next_target_inventory) / max(inventory_now, 1.0e-12)
    return float(np.clip(fraction, 0.0, 1.0))
