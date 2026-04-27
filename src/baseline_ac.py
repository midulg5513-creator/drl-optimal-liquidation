"""Almgren-Chriss benchmark strategy."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import ExperimentConfig
from env_execution import OptimalExecutionEnv
from execution_reference import ac_inventory_path


def evaluate_ac(config: ExperimentConfig, episodes: int, base_seed: int) -> Dict[str, object]:
    """Evaluate the AC strategy over multiple episodes."""
    ac_inventory_path(config)
    env = OptimalExecutionEnv(config)
    records = []
    for episode in range(episodes):
        env.reset(seed=base_seed + episode)
        done = False
        last_info = {}
        cumulative_reward = 0.0
        while not done:
            _, reward, done, info = env.step(0.5)
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
