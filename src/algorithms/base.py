from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from tensordict import TensorDict


@dataclass
class TrainingState:
    """Full snapshot of algorithm state for checkpointing and resuming."""
    step: int
    policy_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    replay_buffer_state: dict[str, Any] | None = field(default=None)


class BaseAlgorithm(ABC):
    """Abstract base class for all RL algorithms.

    Each algorithm owns its environment, network(s), loss module, optimizer,
    and data collector. The trainer calls setup() once, then train().

    Args:
        cfg: full Hydra config (access algorithm params via cfg.algorithm,
             environment params via cfg.environment, etc.)
        device: resolved torch.device
    """

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self._step: int = 0

    @abstractmethod
    def setup(self) -> None:
        """Initialize networks, loss modules, optimizers, and data collector."""

    @abstractmethod
    def train(
        self,
        trainer_cfg: DictConfig,
        callbacks: list,
    ) -> dict[str, float]:
        """Run the full training loop.

        Args:
            trainer_cfg: trainer sub-config (max_steps, log_every_n_steps)
            callbacks: list of callback objects

        Returns:
            dict of final training metrics
        """

    @abstractmethod
    def eval(self, num_episodes: int) -> dict[str, float]:
        """Run deterministic evaluation episodes.

        Args:
            num_episodes: number of episodes to evaluate

        Returns:
            dict with at least "eval/return_mean" and "eval/return_std"
        """

    @abstractmethod
    def _update(self, batch: TensorDict) -> dict[str, float]:
        """Perform a single gradient update step.

        Args:
            batch: TensorDict sampled from collector or replay buffer

        Returns:
            dict of scalar metrics (losses, Q-values, etc.)
        """

    def save_checkpoint(self, path: Path) -> None:
        """Serialize the current TrainingState to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._get_training_state()
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        """Restore TrainingState from a checkpoint file."""
        state: TrainingState = torch.load(path, map_location=self.device, weights_only=False)
        self._step = state.step
        self._load_training_state(state)

    @abstractmethod
    def _get_training_state(self) -> TrainingState:
        """Collect current state dicts into a TrainingState for serialization."""

    @abstractmethod
    def _load_training_state(self, state: TrainingState) -> None:
        """Restore network/optimizer/buffer state from a loaded TrainingState."""
