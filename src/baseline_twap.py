"""Time-weighted average price benchmark strategy."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import ExperimentConfig
from env_execution import OptimalExecutionEnv


def _twap_target_fraction(remaining_steps: int) -> float:
    """Sell an equal fraction of the remaining inventory each step."""
    if remaining_steps <= 1:
        return 1.0
    return 1.0 / remaining_steps


def _fraction_to_policy_action(
    config: ExperimentConfig,
    reference_fraction: float,
    target_fraction: float,
) -> float:
    """Invert the environment's piecewise action mapping."""
    lower = max(0.0, reference_fraction - config.action_deviation_limit)
    upper = min(1.0, reference_fraction + config.action_deviation_limit)
    clipped_target = float(np.clip(target_fraction, lower, upper))

    if clipped_target <= reference_fraction:
        denom = max(reference_fraction - lower, 1.0e-12)
        return float(np.clip(0.5 * (clipped_target - lower) / denom, 0.0, 0.5))

    denom = max(upper - reference_fraction, 1.0e-12)
    return float(np.clip(0.5 + 0.5 * (clipped_target - reference_fraction) / denom, 0.5, 1.0))


def evaluate_twap(config: ExperimentConfig, episodes: int, base_seed: int) -> Dict[str, object]:
    """Evaluate the TWAP strategy over multiple episodes."""
    env = OptimalExecutionEnv(config)
    records = []
    for episode in range(episodes):
        env.reset(seed=base_seed + episode)
        done = False
        last_info = {}
        cumulative_reward = 0.0
        while not done:
            remaining_steps = config.horizon - env.step_index
            reference_fraction = env._reference_fraction()
            target_fraction = _twap_target_fraction(remaining_steps)
            action = _fraction_to_policy_action(config, reference_fraction, target_fraction)
            _, reward, done, info = env.step(action)
            cumulative_reward += reward
            last_info = info
        records.append(
            {
                'episode': episode,
                'shortfall': last_info['implementation_shortfall'],
                'total_reward': cumulative_reward,
                'completion': 1.0 if last_info['inventory_after'] <= 1.0e-8 else 0.0,
            }
        )
    frame = pd.DataFrame(records)
    metrics = {
        'avg_shortfall': float(frame['shortfall'].mean()),
        'shortfall_std': float(frame['shortfall'].std(ddof=0)),
        'avg_reward': float(frame['total_reward'].mean()),
        'completion_rate': float(frame['completion'].mean()),
    }
    return {'metrics': metrics, 'episode_frame': frame}
