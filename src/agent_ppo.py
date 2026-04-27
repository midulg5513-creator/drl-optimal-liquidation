"""Pure-numpy PPO implementation for continuous liquidation control."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import ExperimentConfig
from env_execution import OptimalExecutionEnv
from utils import AdamOptimizer, NumpyMLP, sigmoid


@dataclass
class RolloutBatch:
    """Single PPO rollout storage."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class PPOAgent:
    """Small PPO agent with Gaussian policy and manual gradients."""

    def __init__(self, config: ExperimentConfig, seed: int, teacher_agent: object | None = None) -> None:
        self.config = config
        self.teacher_agent = teacher_agent
        self.rng = np.random.default_rng(seed)
        self.current_action_std = config.ppo_action_std_start
        policy_layers = [
            config.state_dim,
            config.hidden_size,
            config.hidden_size_large,
            config.action_dim,
        ]
        value_layers = [config.state_dim, config.hidden_size_large, config.hidden_size, 1]
        activations = ["tanh", "tanh", "linear"]
        self.policy = NumpyMLP(policy_layers, activations, seed)
        self.value = NumpyMLP(value_layers, activations, seed + 1)
        self.policy_optimizer = AdamOptimizer(
            self.policy.parameters(), learning_rate=config.ppo_policy_lr
        )
        self.value_optimizer = AdamOptimizer(
            self.value.parameters(), learning_rate=config.ppo_value_lr
        )
        self.completed_episodes = 0
        self.training_history: List[Dict[str, float]] = []

    def policy_mean(self, states: np.ndarray, store_cache: bool = False) -> np.ndarray:
        raw = self.policy.forward(states, store_cache=store_cache)
        base_mean = sigmoid(raw)
        return 0.5 + self.config.ppo_policy_blend * (base_mean - 0.5)

    def value_predict(self, states: np.ndarray, store_cache: bool = False) -> np.ndarray:
        return self.value.forward(states, store_cache=store_cache)

    def log_prob(self, actions: np.ndarray, mean: np.ndarray, action_std: float | None = None) -> np.ndarray:
        std = action_std if action_std is not None else self.current_action_std
        variance = std ** 2
        return -0.5 * (
            np.square((actions - mean) / std)
            + np.log(2.0 * np.pi * variance)
        )

    def select_action(self, state: np.ndarray, deterministic: bool) -> tuple[float, float, float]:
        mean = self.policy_mean(state.reshape(1, -1), store_cache=False)[0, 0]
        if deterministic:
            action = mean
        else:
            action = mean + self.rng.normal(0.0, self.current_action_std)
        action = float(np.clip(action, 0.0, 1.0))
        log_prob = float(self.log_prob(np.array([[action]]), np.array([[mean]]))[0, 0])
        value = float(self.value_predict(state.reshape(1, -1), store_cache=False)[0, 0])
        return action, log_prob, value

    def compute_returns_advantages(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE returns and advantages."""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.config.ppo_gamma * next_value * mask - values[step]
            gae = delta + self.config.ppo_gamma * self.config.ppo_gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
            next_value = values[step]
        return returns, advantages

    def collect_episode(self, env: OptimalExecutionEnv, seed: int) -> tuple[RolloutBatch, Dict[str, float]]:
        """Collect one full episode."""
        states: List[np.ndarray] = []
        actions: List[float] = []
        rewards: List[float] = []
        dones: List[float] = []
        log_probs: List[float] = []
        values: List[float] = []

        state = env.reset(seed=seed)
        done = False
        info: Dict[str, float] = {}
        while not done:
            action, log_prob, value = self.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            states.append(state.copy())
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob)
            values.append(value)
            state = next_state

        rewards_arr = np.array(rewards, dtype=np.float64).reshape(-1, 1)
        dones_arr = np.array(dones, dtype=np.float64).reshape(-1, 1)
        values_arr = np.array(values, dtype=np.float64).reshape(-1, 1)
        returns, advantages = self.compute_returns_advantages(rewards_arr, dones_arr, values_arr)
        batch = RolloutBatch(
            states=np.vstack(states),
            actions=np.array(actions, dtype=np.float64).reshape(-1, 1),
            rewards=rewards_arr,
            dones=dones_arr,
            log_probs=np.array(log_probs, dtype=np.float64).reshape(-1, 1),
            values=values_arr,
            returns=returns,
            advantages=advantages,
        )
        return batch, info

    def merge_batches(self, batches: List[RolloutBatch]) -> RolloutBatch:
        """Merge several rollout batches and normalize advantages globally."""
        merged = RolloutBatch(
            states=np.vstack([batch.states for batch in batches]),
            actions=np.vstack([batch.actions for batch in batches]),
            rewards=np.vstack([batch.rewards for batch in batches]),
            dones=np.vstack([batch.dones for batch in batches]),
            log_probs=np.vstack([batch.log_probs for batch in batches]),
            values=np.vstack([batch.values for batch in batches]),
            returns=np.vstack([batch.returns for batch in batches]),
            advantages=np.vstack([batch.advantages for batch in batches]),
        )
        merged.advantages = (merged.advantages - merged.advantages.mean()) / (
            merged.advantages.std() + 1.0e-8
        )
        merged.advantages = np.clip(
            merged.advantages,
            -self.config.ppo_advantage_clip,
            self.config.ppo_advantage_clip,
        )
        return merged

    def update_policy(self, batch: RolloutBatch) -> tuple[float, float]:
        """Run PPO optimization on one batch."""
        num_samples = batch.states.shape[0]
        policy_losses = []
        value_losses = []
        indices = np.arange(num_samples)

        for _ in range(self.config.ppo_epochs):
            self.rng.shuffle(indices)
            for start in range(0, num_samples, self.config.ppo_minibatch_size):
                batch_indices = indices[start : start + self.config.ppo_minibatch_size]
                states = batch.states[batch_indices]
                actions = batch.actions[batch_indices]
                old_log_probs = batch.log_probs[batch_indices]
                advantages = batch.advantages[batch_indices]
                returns = batch.returns[batch_indices]

                mean = self.policy_mean(states, store_cache=True)
                new_log_probs = self.log_prob(actions, mean)
                ratios = np.exp(new_log_probs - old_log_probs)

                active_mask = np.ones_like(ratios, dtype=np.float64)
                clipped_high = (advantages >= 0.0) & (ratios > 1.0 + self.config.ppo_clip_ratio)
                clipped_low = (advantages < 0.0) & (ratios < 1.0 - self.config.ppo_clip_ratio)
                active_mask[clipped_high | clipped_low] = 0.0

                surrogate = np.minimum(
                    ratios * advantages,
                    np.clip(ratios, 1.0 - self.config.ppo_clip_ratio, 1.0 + self.config.ppo_clip_ratio)
                    * advantages,
                )
                policy_loss = -float(np.mean(surrogate))
                dloss_dlogp = -(active_mask * ratios * advantages) / len(batch_indices)
                dlogp_dmean = (actions - mean) / (self.current_action_std ** 2)
                grad_mean = dloss_dlogp * dlogp_dmean
                imitation_loss = 0.0
                if self.teacher_agent is not None and self.config.ppo_teacher_guidance > 0.0:
                    teacher_actions = self.teacher_agent.actor_forward(states, store_cache=False)
                    imitation_loss = self.config.ppo_teacher_guidance * float(
                        np.mean(np.square(mean - teacher_actions))
                    )
                    grad_mean += (
                        (2.0 * self.config.ppo_teacher_guidance / len(batch_indices))
                        * (mean - teacher_actions)
                    )
                raw_policy_output = self.policy.cache_outputs[-1]
                base_mean = sigmoid(raw_policy_output)
                grad_raw = (
                    grad_mean
                    * self.config.ppo_policy_blend
                    * base_mean
                    * (1.0 - base_mean)
                )
                policy_grads, _ = self.policy.backward(grad_raw)
                self.policy_optimizer.step(policy_grads)
                policy_losses.append(policy_loss + imitation_loss)

                value_pred = self.value_predict(states, store_cache=True)
                value_loss_grad = 2.0 * (value_pred - returns) / len(batch_indices)
                value_grads, _ = self.value.backward(value_loss_grad)
                self.value_optimizer.step(value_grads)
                value_losses.append(float(np.mean(np.square(value_pred - returns))))

        return float(np.mean(policy_losses)), float(np.mean(value_losses))

    def train(self, episodes: int | None = None, base_seed: int = 2000) -> pd.DataFrame:
        """Train PPO and return cumulative episode logs."""
        additional_episodes = episodes if episodes is not None else self.config.ppo_episodes
        if additional_episodes <= 0:
            return self.get_training_frame()

        env = OptimalExecutionEnv(self.config)
        episodes_remaining = additional_episodes
        decay_span = max(self.config.ppo_episodes - 1, 1)

        while episodes_remaining > 0:
            progress = min(self.completed_episodes / decay_span, 1.0)
            self.current_action_std = self.config.ppo_action_std_start + progress * (
                self.config.ppo_action_std_end - self.config.ppo_action_std_start
            )
            rollout_batches: List[RolloutBatch] = []
            rollout_logs: List[Dict[str, float]] = []
            current_rollouts = min(self.config.ppo_rollout_episodes, episodes_remaining)
            for _ in range(current_rollouts):
                episode_number = self.completed_episodes + 1
                batch, info = self.collect_episode(env, base_seed + self.completed_episodes)
                rollout_batches.append(batch)
                rollout_logs.append(
                    {
                        'episode': episode_number,
                        'episode_reward': float(batch.rewards.sum()),
                        'implementation_shortfall': float(info['implementation_shortfall']),
                        'action_std': self.current_action_std,
                    }
                )
                self.completed_episodes += 1
                episodes_remaining -= 1

            merged_batch = self.merge_batches(rollout_batches)
            policy_loss, value_loss = self.update_policy(merged_batch)
            for record in rollout_logs:
                record['policy_loss'] = policy_loss
                record['value_loss'] = value_loss
                self.training_history.append(record)
        return self.get_training_frame()

    def evaluate(self, episodes: int, base_seed: int) -> Dict[str, object]:
        """Evaluate a trained PPO agent."""
        env = OptimalExecutionEnv(self.config)
        records: List[Dict[str, float]] = []
        for episode in range(episodes):
            state = env.reset(seed=base_seed + episode)
            done = False
            total_reward = 0.0
            info: Dict[str, float] = {}
            while not done:
                action, _, _ = self.select_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                total_reward += reward
            records.append(
                {
                    'episode': episode,
                    'shortfall': info['implementation_shortfall'],
                    'total_reward': total_reward,
                    'completion': 1.0 if info['inventory_after'] <= 1.0e-8 else 0.0,
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

    def get_training_frame(self) -> pd.DataFrame:
        """Return cumulative training logs as a DataFrame."""
        return pd.DataFrame(self.training_history)

    def get_checkpoint_state(self) -> Dict[str, object]:
        """Serialize the full PPO state for warm starts."""
        return {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'rng_state': deepcopy(self.rng.bit_generator.state),
            'current_action_std': self.current_action_std,
            'completed_episodes': self.completed_episodes,
            'training_history': [dict(record) for record in self.training_history],
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        """Restore a saved PPO state."""
        self.policy.load_state_dict(state['policy'])
        self.value.load_state_dict(state['value'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        self.value_optimizer.load_state_dict(state['value_optimizer'])
        self.rng.bit_generator.state = deepcopy(state['rng_state'])
        self.current_action_std = float(state.get('current_action_std', self.config.ppo_action_std_start))
        self.completed_episodes = int(state.get('completed_episodes', 0))
        self.training_history = [dict(record) for record in state.get('training_history', [])]
