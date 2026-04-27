"""Pure-numpy DDPG implementation for continuous liquidation control."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ExperimentConfig
from env_execution import OptimalExecutionEnv
from utils import AdamOptimizer, NumpyMLP, sigmoid


@dataclass
class Transition:
    """Replay buffer transition."""

    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    """A fixed-size replay buffer."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.storage: List[Transition] = []
        self.position = 0

    def add(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        states = np.vstack([self.storage[idx].state for idx in indices])
        actions = np.array([[self.storage[idx].action] for idx in indices], dtype=np.float64)
        rewards = np.array([[self.storage[idx].reward] for idx in indices], dtype=np.float64)
        next_states = np.vstack([self.storage[idx].next_state for idx in indices])
        dones = np.array([[self.storage[idx].done] for idx in indices], dtype=np.float64)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.storage)

    def state_dict(self) -> Dict[str, object]:
        """Serialize replay buffer state."""
        if not self.storage:
            return {
                'capacity': self.capacity,
                'position': self.position,
                'rng_state': deepcopy(self.rng.bit_generator.state),
                'states': np.empty((0, 0), dtype=np.float64),
                'actions': np.empty((0,), dtype=np.float64),
                'rewards': np.empty((0,), dtype=np.float64),
                'next_states': np.empty((0, 0), dtype=np.float64),
                'dones': np.empty((0,), dtype=np.float64),
            }
        return {
            'capacity': self.capacity,
            'position': self.position,
            'rng_state': deepcopy(self.rng.bit_generator.state),
            'states': np.vstack([transition.state for transition in self.storage]),
            'actions': np.array([transition.action for transition in self.storage], dtype=np.float64),
            'rewards': np.array([transition.reward for transition in self.storage], dtype=np.float64),
            'next_states': np.vstack([transition.next_state for transition in self.storage]),
            'dones': np.array([transition.done for transition in self.storage], dtype=np.float64),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore replay buffer state."""
        self.capacity = int(state['capacity'])
        self.position = int(state['position'])
        self.rng.bit_generator.state = deepcopy(state['rng_state'])
        self.storage = []
        states = np.asarray(state['states'], dtype=np.float64)
        if states.size == 0:
            return
        actions = np.asarray(state['actions'], dtype=np.float64)
        rewards = np.asarray(state['rewards'], dtype=np.float64)
        next_states = np.asarray(state['next_states'], dtype=np.float64)
        dones = np.asarray(state['dones'], dtype=np.float64)
        for idx in range(states.shape[0]):
            self.storage.append(
                Transition(
                    state=states[idx].copy(),
                    action=float(actions[idx]),
                    reward=float(rewards[idx]),
                    next_state=next_states[idx].copy(),
                    done=float(dones[idx]),
                )
            )


class DDPGAgent:
    """Minimal DDPG agent using manual backpropagation."""

    def __init__(self, config: ExperimentConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        actor_layers = [
            config.state_dim,
            config.hidden_size,
            config.hidden_size_large,
            config.action_dim,
        ]
        critic_layers = [
            config.state_dim + config.action_dim,
            config.hidden_size_large,
            config.hidden_size,
            1,
        ]
        activations = ["tanh", "tanh", "linear"]
        self.actor = NumpyMLP(actor_layers, activations, seed)
        self.actor_target = NumpyMLP(actor_layers, activations, seed + 1)
        self.actor_target.copy_from(self.actor)
        self.critic = NumpyMLP(critic_layers, activations, seed + 2)
        self.critic_target = NumpyMLP(critic_layers, activations, seed + 3)
        self.critic_target.copy_from(self.critic)

        self.actor_optimizer = AdamOptimizer(
            self.actor.parameters(), learning_rate=config.ddpg_actor_lr
        )
        self.critic_optimizer = AdamOptimizer(
            self.critic.parameters(), learning_rate=config.ddpg_critic_lr
        )
        self.buffer = ReplayBuffer(config.ddpg_buffer_size, seed + 4)
        self.total_steps = 0
        self.completed_episodes = 0
        self.training_history: List[Dict[str, float]] = []

    def actor_forward(self, states: np.ndarray, store_cache: bool = True) -> np.ndarray:
        raw = self.actor.forward(states, store_cache=store_cache)
        return sigmoid(raw)

    def actor_target_forward(self, states: np.ndarray) -> np.ndarray:
        raw = self.actor_target.forward(states, store_cache=False)
        return sigmoid(raw)

    def critic_forward(
        self, states: np.ndarray, actions: np.ndarray, store_cache: bool = True
    ) -> np.ndarray:
        inputs = np.hstack([states, actions])
        return self.critic.forward(inputs, store_cache=store_cache)

    def critic_target_forward(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        inputs = np.hstack([states, actions])
        return self.critic_target.forward(inputs, store_cache=False)

    def select_action(self, state: np.ndarray, noise_scale: float = 0.0) -> float:
        action = float(self.actor_forward(state.reshape(1, -1), store_cache=False)[0, 0])
        if noise_scale > 0.0:
            action += self.rng.normal(0.0, noise_scale)
        return float(np.clip(action, 0.0, 1.0))

    def update(self, batch: Transition) -> Tuple[float, float]:
        batch_size = batch.state.shape[0]

        next_actions = self.actor_target_forward(batch.next_state)
        target_q = self.critic_target_forward(batch.next_state, next_actions)
        td_target = batch.reward + (1.0 - batch.done) * self.config.ddpg_gamma * target_q

        predicted_q = self.critic_forward(batch.state, batch.action, store_cache=True)
        critic_grad_output = 2.0 * (predicted_q - td_target) / batch_size
        critic_grads, _ = self.critic.backward(critic_grad_output)
        self.critic_optimizer.step(critic_grads)
        critic_loss = float(np.mean(np.square(predicted_q - td_target)))

        actions = self.actor_forward(batch.state, store_cache=True)
        q_values = self.critic_forward(batch.state, actions, store_cache=True)
        actor_loss = -float(np.mean(q_values))
        critic_input_grad_output = -np.ones_like(q_values) / batch_size
        _, critic_input_grad = self.critic.backward(critic_input_grad_output)
        action_grad = critic_input_grad[:, -1:]
        raw_actions = self.actor.cache_outputs[-1]
        action_sigmoid = sigmoid(raw_actions)
        actor_grad_output = action_grad * action_sigmoid * (1.0 - action_sigmoid)
        actor_grads, _ = self.actor.backward(actor_grad_output)
        self.actor_optimizer.step(actor_grads)

        self.actor_target.soft_update(self.actor, self.config.ddpg_tau)
        self.critic_target.soft_update(self.critic, self.config.ddpg_tau)
        return critic_loss, actor_loss

    def train(self, episodes: int | None = None, base_seed: int = 1000) -> pd.DataFrame:
        """Train the DDPG agent and return cumulative episode logs."""
        additional_episodes = episodes if episodes is not None else self.config.ddpg_episodes
        if additional_episodes <= 0:
            return self.get_training_frame()

        env = OptimalExecutionEnv(self.config)
        schedule_span = max(self.config.ddpg_episodes - 1, 1)
        for _ in range(additional_episodes):
            episode_number = self.completed_episodes + 1
            state = env.reset(seed=base_seed + self.completed_episodes)
            done = False
            episode_reward = 0.0
            last_info: Dict[str, float] = {}
            critic_loss = np.nan
            actor_loss = np.nan
            progress = min(self.completed_episodes / schedule_span, 1.0)
            noise_scale = self.config.ddpg_exploration_noise + progress * (
                self.config.ddpg_exploration_noise_min - self.config.ddpg_exploration_noise
            )
            while not done:
                action = self.select_action(state, noise_scale=noise_scale)
                next_state, reward, done, info = env.step(action)
                self.buffer.add(
                    Transition(
                        state=state.copy(),
                        action=action,
                        reward=reward,
                        next_state=next_state.copy(),
                        done=float(done),
                    )
                )
                state = next_state
                episode_reward += reward
                last_info = info
                self.total_steps += 1

                if len(self.buffer) >= self.config.ddpg_batch_size and self.total_steps >= self.config.ddpg_warmup_steps:
                    for _ in range(self.config.ddpg_updates_per_step):
                        batch = self.buffer.sample(self.config.ddpg_batch_size)
                        critic_loss, actor_loss = self.update(batch)

            self.training_history.append(
                {
                    'episode': episode_number,
                    'episode_reward': episode_reward,
                    'implementation_shortfall': last_info.get('implementation_shortfall', np.nan),
                    'critic_loss': critic_loss,
                    'actor_loss': actor_loss,
                    'exploration_noise': noise_scale,
                }
            )
            self.completed_episodes += 1
        return self.get_training_frame()

    def evaluate(self, episodes: int, base_seed: int) -> Dict[str, object]:
        """Evaluate a trained DDPG agent."""
        env = OptimalExecutionEnv(self.config)
        records: List[Dict[str, float]] = []
        for episode in range(episodes):
            state = env.reset(seed=base_seed + episode)
            done = False
            total_reward = 0.0
            info: Dict[str, float] = {}
            while not done:
                action = self.select_action(state, noise_scale=0.0)
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
        """Serialize the full agent state for warm starts."""
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'buffer': self.buffer.state_dict(),
            'rng_state': deepcopy(self.rng.bit_generator.state),
            'total_steps': self.total_steps,
            'completed_episodes': self.completed_episodes,
            'training_history': [dict(record) for record in self.training_history],
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        """Restore a saved agent state."""
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.buffer.load_state_dict(state['buffer'])
        self.rng.bit_generator.state = deepcopy(state['rng_state'])
        self.total_steps = int(state.get('total_steps', 0))
        self.completed_episodes = int(state.get('completed_episodes', 0))
        self.training_history = [dict(record) for record in state.get('training_history', [])]
