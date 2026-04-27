"""Environment: config wrapper + factory for TorchRL environments.

The Environment holds configuration and produces TransformedEnv instances
on demand.  It never holds a live env itself — the Trainer controls env
lifecycle by calling ``make_env()`` when needed.
"""
from __future__ import annotations

from typing import Sequence

from torchrl.envs import EnvBase

from src.environments.factory import make_env


class Environment:
    """Wraps environment parameters and produces TorchRL env instances.

    Args:
        name: Environment name (e.g. ``"CartPole-v1"``, ``"ALE/Pong-v5"``).
        backend: Backend to use (``"gymnasium"``, ``"dm_control"``, ``"envpool"``).
        obs_shape: Observation shape after preprocessing (e.g. ``[4]`` or ``[4, 84, 84]``).
        num_actions: Number of actions (discrete count or continuous dim).
        transforms: List of transform dicts (each with ``_target_`` key and kwargs),
            instantiated fresh per ``make_env()`` call via ``hydra.utils.instantiate``.
            Controls the full transform pipeline; include ``StepCounter`` explicitly.
            Gymnasium only — dm_control and envpool use built-in pipelines.
        from_pixels: Pass ``from_pixels=True`` to ``GymEnv`` for pixel observations.
            Set to ``True`` for any Atari / pixel-based gymnasium environment.
        task: dm_control task string (e.g. ``"walk"`` for ``humanoid-walk``).
        max_episode_steps: Maximum steps per episode (envpool only).
        **kwargs: Extra keyword arguments forwarded to backend-specific helpers
            (e.g. ``normalize_obs`` for dm_control, ``clip_rewards`` for envpool).
    """

    def __init__(
        self,
        name: str,
        backend: str,
        obs_shape: Sequence[int],
        num_actions: int,
        transforms: list | None = None,
        from_pixels: bool = False,
        task: str | None = None,
        max_episode_steps: int | None = None,
        **kwargs,
    ) -> None:
        self.obs_shape: tuple[int, ...] = tuple(obs_shape)
        self.num_actions = int(num_actions)
        self._factory_kwargs: dict = {
            "name": name,
            "backend": backend,
            "obs_shape": obs_shape,
            "num_actions": num_actions,
            "transforms": transforms,
            "from_pixels": from_pixels,
            "task": task,
            "max_episode_steps": max_episode_steps,
            **kwargs,
        }

    def make_env(
        self,
        num_envs: int = 1,
        device: str = "cpu",
    ) -> EnvBase:
        """Create a (possibly vectorised) TorchRL env from stored parameters.

        Args:
            num_envs: Number of parallel envs (>1 → ParallelEnv).
            device: Target device string.

        Returns:
            TransformedEnv (or ParallelEnv wrapping TransformedEnvs).
        """
        return make_env(**self._factory_kwargs, num_envs=num_envs, device=device)
