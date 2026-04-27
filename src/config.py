"""Project configuration for optimal-liquidation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Centralized experiment parameters."""

    project_title: str = "Optimal Liquidation with Deep Reinforcement Learning"

    random_seed: int = 7
    horizon: int = 20
    total_time: float = 1.0
    lookback: int = 5

    initial_price: float = 100.0
    initial_inventory: float = 1.0
    drift: float = 0.0
    volatility: float = 0.16
    permanent_impact: float = 0.90
    temporary_impact: float = 1.20
    fixed_fee: float = 0.02
    risk_aversion: float = 5.0e-4
    action_deviation_limit: float = 0.15
    reference_backlog_penalty: float = 0.0
    reference_deviation_penalty: float = 0.0

    hidden_size: int = 32
    hidden_size_large: int = 32

    ddpg_episodes: int = 180
    ddpg_batch_size: int = 64
    ddpg_buffer_size: int = 20000
    ddpg_gamma: float = 0.98
    ddpg_tau: float = 0.010
    ddpg_actor_lr: float = 0.0015
    ddpg_critic_lr: float = 0.0025
    ddpg_warmup_steps: int = 150
    ddpg_exploration_noise: float = 0.03
    ddpg_exploration_noise_min: float = 0.005
    ddpg_updates_per_step: int = 2

    ppo_episodes: int = 180
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.12
    ppo_policy_lr: float = 0.0010
    ppo_value_lr: float = 0.0020
    ppo_epochs: int = 6
    ppo_minibatch_size: int = 64
    ppo_action_std_start: float = 0.03
    ppo_action_std_end: float = 0.01
    ppo_rollout_episodes: int = 8
    ppo_advantage_clip: float = 2.5
    ppo_policy_blend: float = 0.60
    ppo_teacher_guidance: float = 8.0

    eval_episodes: int = 100
    sensitivity_eval_episodes: int = 60
    sensitivity_train_episodes_ddpg: int = 100
    sensitivity_train_episodes_ppo: int = 90

    lambda_grid: List[float] = field(
        default_factory=lambda: [1.0e-4, 3.0e-4, 5.0e-4, 8.0e-4]
    )
    fee_grid: List[float] = field(default_factory=lambda: [0.00, 0.01, 0.02, 0.05])

    experiment_version: str = "v1.6.0"
    checkpoint_dirname: str = "checkpoints"
    reuse_checkpoints: bool = True
    warm_start_compatible: bool = True
    force_retrain: bool = False
    ddpg_resume_extra_episodes: int = 0
    ppo_resume_extra_episodes: int = 0
    statistical_train_seeds: List[int] = field(default_factory=lambda: [7, 17, 27])
    statistical_eval_episodes: int = 80
    generate_multi_seed_summary: bool = True
    include_twap_baseline: bool = True

    @property
    def state_dim(self) -> int:
        return self.lookback + 6

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def dt(self) -> float:
        return self.total_time / self.horizon

    def with_overrides(self, **kwargs: float) -> "ExperimentConfig":
        """Return a shallow copy with updated values."""
        data = self.__dict__.copy()
        data.update(kwargs)
        return ExperimentConfig(**data)


DEFAULT_CONFIG = ExperimentConfig()
