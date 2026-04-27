"""Utility helpers for reproducible optimal-liquidation experiments.

"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.family"] = ["Microsoft YaHei"]
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create a directory if needed and return the resolved path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid function."""
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def tanh_derivative(output: np.ndarray) -> np.ndarray:
    """Derivative of tanh activation expressed via its output."""
    return 1.0 - np.square(output)


def relu_derivative(output: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation expressed via its output."""
    return (output > 0.0).astype(np.float64)


class AdamOptimizer:
    """Minimal Adam optimizer for numpy parameters."""

    def __init__(
        self,
        params: Sequence[np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-8,
    ) -> None:
        self.params = list(params)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one Adam update."""
        self.step_count += 1
        for index, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[index] = self.beta1 * self.m[index] + (1.0 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1.0 - self.beta2) * np.square(grad)
            m_hat = self.m[index] / (1.0 - self.beta1 ** self.step_count)
            v_hat = self.v[index] / (1.0 - self.beta2 ** self.step_count)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> Dict[str, object]:
        """Serialize optimizer state for checkpoints."""
        return {
            'learning_rate': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'step_count': self.step_count,
            'm': [value.copy() for value in self.m],
            'v': [value.copy() for value in self.v],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore optimizer state from a checkpoint."""
        self.lr = float(state['learning_rate'])
        self.beta1 = float(state['beta1'])
        self.beta2 = float(state['beta2'])
        self.eps = float(state['eps'])
        self.step_count = int(state['step_count'])
        self.m = [np.array(value, dtype=np.float64, copy=True) for value in state['m']]
        self.v = [np.array(value, dtype=np.float64, copy=True) for value in state['v']]


class NumpyMLP:
    """A small fully connected network with manual backpropagation."""

    def __init__(self, layer_sizes: Sequence[int], activations: Sequence[str], seed: int) -> None:
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("layer_sizes and activations length mismatch")
        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            self.weights.append(rng.uniform(-limit, limit, size=(in_dim, out_dim)))
            self.biases.append(np.zeros((1, out_dim), dtype=np.float64))
        self.cache_inputs: List[np.ndarray] = []
        self.cache_outputs: List[np.ndarray] = []

    def forward(self, x: np.ndarray, store_cache: bool = True) -> np.ndarray:
        """Forward pass through the network."""
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if store_cache:
            self.cache_inputs = []
            self.cache_outputs = []
        current = a
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            z = current @ weight + bias
            if activation == "tanh":
                current = np.tanh(z)
            elif activation == "relu":
                current = np.maximum(z, 0.0)
            elif activation == "linear":
                current = z
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            if store_cache:
                self.cache_inputs.append(a if len(self.cache_inputs) == 0 else self.cache_outputs[-1])
                self.cache_outputs.append(current.copy())
            a = current
        return current

    def backward(self, grad_output: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Backpropagate and return parameter gradients and input gradient."""
        grad = np.asarray(grad_output, dtype=np.float64)
        if grad.ndim == 1:
            grad = grad.reshape(-1, 1)

        grad_weights: List[np.ndarray] = []
        grad_biases: List[np.ndarray] = []
        grad_current = grad

        for layer_idx in reversed(range(len(self.weights))):
            output = self.cache_outputs[layer_idx]
            activation = self.activations[layer_idx]
            if activation == "tanh":
                grad_current = grad_current * tanh_derivative(output)
            elif activation == "relu":
                grad_current = grad_current * relu_derivative(output)
            elif activation == "linear":
                grad_current = grad_current
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            input_to_layer = self.cache_inputs[layer_idx]
            grad_w = input_to_layer.T @ grad_current
            grad_b = grad_current.sum(axis=0, keepdims=True)
            grad_input = grad_current @ self.weights[layer_idx].T
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            grad_current = grad_input

        grads = []
        for grad_w, grad_b in zip(grad_weights, grad_biases):
            grads.extend([grad_w, grad_b])
        return grads, grad_current

    def parameters(self) -> List[np.ndarray]:
        """Return parameters in optimizer order."""
        params = []
        for weight, bias in zip(self.weights, self.biases):
            params.extend([weight, bias])
        return params

    def copy_from(self, other: "NumpyMLP") -> None:
        """Copy parameters from another network."""
        for current, source in zip(self.parameters(), other.parameters()):
            np.copyto(current, source)

    def soft_update(self, source: "NumpyMLP", tau: float) -> None:
        """Soft-update parameters from another network."""
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param *= 1.0 - tau
            target_param += tau * source_param

    def state_dict(self) -> Dict[str, object]:
        """Serialize network parameters."""
        return {
            'layer_sizes': list(self.layer_sizes),
            'activations': list(self.activations),
            'weights': [weight.copy() for weight in self.weights],
            'biases': [bias.copy() for bias in self.biases],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore network parameters."""
        if list(state['layer_sizes']) != self.layer_sizes:
            raise ValueError('Checkpoint layer sizes do not match the current network.')
        if list(state['activations']) != self.activations:
            raise ValueError('Checkpoint activations do not match the current network.')
        for current, source in zip(self.weights, state['weights']):
            np.copyto(current, source)
        for current, source in zip(self.biases, state['biases']):
            np.copyto(current, source)


def moving_average(values: Sequence[float], window: int = 15) -> np.ndarray:
    """Smoothed moving average for plotting."""
    series = pd.Series(list(values), dtype=np.float64)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as UTF-8 CSV."""
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text file."""
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, content: Dict[str, object]) -> None:
    """Write a JSON file with UTF-8 encoding."""
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')


def save_pickle(path: Path, content: Dict[str, object]) -> None:
    """Write a pickle checkpoint."""
    with path.open('wb') as handle:
        pickle.dump(content, handle)


def load_pickle(path: Path) -> Dict[str, object]:
    """Load a pickle checkpoint."""
    with path.open('rb') as handle:
        return pickle.load(handle)


def _normalize_for_hash(value: object) -> object:
    """Normalize nested values into stable JSON-friendly objects."""
    if isinstance(value, dict):
        return {key: _normalize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def compute_config_hash(config_dict: Dict[str, object], exclude_keys: Sequence[str] = ()) -> str:
    """Compute a stable short hash for a configuration subset."""
    filtered = {
        key: value
        for key, value in config_dict.items()
        if key not in set(exclude_keys)
    }
    normalized = _normalize_for_hash(filtered)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


def sync_directory_contents(source_dir: Path, target_dir: Path, patterns: Sequence[str]) -> None:
    """Copy selected files into the latest-output directory."""
    ensure_dir(target_dir)
    for pattern in patterns:
        for stale in target_dir.glob(pattern):
            if stale.is_file():
                stale.unlink()

    copied = set()
    for pattern in patterns:
        for source_path in source_dir.glob(pattern):
            if source_path.is_file() and source_path.name not in copied:
                shutil.copy2(source_path, target_dir / source_path.name)
                copied.add(source_path.name)


def build_metric_row(method: str, metrics: Dict[str, float]) -> Dict[str, float]:
    """Assemble a flat metric row for CSV export."""
    row: Dict[str, float] = {"method": method}
    row.update(metrics)
    return row


def plot_training_curve(frame: pd.DataFrame, output: Path, title: str, value_column: str) -> None:
    """Plot a training curve with smoothing."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["episode"], frame[value_column], alpha=0.35, label="原始值")
    ax.plot(frame["episode"], moving_average(frame[value_column], 15), linewidth=2.0, label="滑动平均")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(value_column)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_compare_bar(frame: pd.DataFrame, output: Path) -> None:
    """Plot mean implementation shortfall by method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=frame, x="method", y="avg_shortfall", hue="method", legend=False, ax=ax, palette="crest")
    ax.set_title("三种方法平均实施缺口对比")
    ax.set_xlabel("方法")
    ax.set_ylabel("平均实施缺口")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_shortfall_box(frame: pd.DataFrame, output: Path) -> None:
    """Plot shortfall distribution by method."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=frame, x="method", y="shortfall", hue="method", legend=False, ax=ax, palette="Set2")
    ax.set_title("实施缺口分布箱线图")
    ax.set_xlabel("方法")
    ax.set_ylabel("实施缺口")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_sensitivity(frame: pd.DataFrame, x_column: str, output: Path, title: str) -> None:
    """Plot sensitivity curves for all methods."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=frame,
        x=x_column,
        y="avg_shortfall",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("平均实施缺口")
    ax.set_xlabel(x_column)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def config_to_markdown(config_dict: Dict[str, object]) -> str:
    """Render config parameters as a Markdown table."""
    lines = ["| 参数 | 值 |", "| --- | --- |"]
    for key, value in config_dict.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def dataclass_to_dict(obj: object) -> Dict[str, object]:
    """Serialize a dataclass to a plain dictionary."""
    return asdict(obj)
