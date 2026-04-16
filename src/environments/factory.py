"""Environment factory: builds a TorchRL TransformedEnv from Hydra config params.

Usage (from algorithm setup):
    from hydra.utils import instantiate
    env = instantiate(cfg.environment, device=str(self.device))

Or directly:
    from src.environments.factory import make_env
    env = make_env(**OmegaConf.to_container(cfg.environment, resolve=True), device="cpu")
"""
from __future__ import annotations

from typing import Sequence


def make_env(
    name: str,
    backend: str,
    obs_shape: Sequence[int],
    num_actions: int,
    num_envs: int = 1,
    frame_stack: int = 1,
    grayscale: bool = False,
    resize: Sequence[int] | None = None,
    clip_rewards: bool = False,
    normalize_obs: bool = False,
    device: str = "cpu",
    task: str | None = None,
    **kwargs,
):
    """Build a (possibly vectorised) TransformedEnv.

    Args:
        name: environment name (e.g. "CartPole-v1", "ALE/Breakout-v5", "humanoid")
        backend: "gymnasium" or "dm_control"
        obs_shape: expected observation shape after preprocessing (for validation)
        num_actions: number of actions (discrete or continuous dim)
        num_envs: number of parallel envs (>1 → ParallelEnv)
        frame_stack: number of frames to stack (CatFrames)
        grayscale: convert RGB to grayscale
        resize: [H, W] to resize pixel observations
        clip_rewards: clip rewards to {-1, 0, +1} (standard Atari)
        normalize_obs: apply running mean/std normalisation to observations
        device: target device string
        task: dm_control task string (e.g. "walk")
        **kwargs: extra env kwargs forwarded to the base env constructor

    Returns:
        TransformedEnv (single env or wrapped in ParallelEnv)
    """
    if backend == "gymnasium":
        env_fn = lambda: _make_gymnasium_env(  # noqa: E731
            name=name,
            grayscale=grayscale,
            resize=resize,
            frame_stack=frame_stack,
            clip_rewards=clip_rewards,
            normalize_obs=normalize_obs,
            device=device,
        )
    elif backend == "dm_control":
        env_fn = lambda: _make_dmcontrol_env(  # noqa: E731
            name=name,
            task=task or "walk",
            normalize_obs=normalize_obs,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: 'gymnasium', 'dm_control'."
        )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv
        return ParallelEnv(num_envs, env_fn)
    else:
        return env_fn()


def _make_gymnasium_env(
    name: str,
    grayscale: bool,
    resize: Sequence[int] | None,
    frame_stack: int,
    clip_rewards: bool,
    normalize_obs: bool,
    device: str,
):
    from torchrl.envs import GymEnv, TransformedEnv
    from torchrl.envs.transforms import (
        CatFrames,
        Compose,
        GrayScale,
        RewardClipping,
        ToTensorImage,
    )

    # Determine if this is a pixel-based env
    pixel_obs = grayscale or resize is not None

    base_env = GymEnv(name, device=device, from_pixels=pixel_obs)

    transforms = []

    if pixel_obs:
        transforms.append(ToTensorImage(in_keys=["pixels"], out_keys=["pixels"]))

    if grayscale:
        transforms.append(GrayScale(in_keys=["pixels"], out_keys=["pixels"]))

    if resize is not None:
        from torchrl.envs.transforms import Resize
        h, w = resize
        transforms.append(Resize(h, w, in_keys=["pixels"], out_keys=["pixels"]))

    if frame_stack > 1:
        transforms.append(
            CatFrames(N=frame_stack, dim=-3, in_keys=["pixels"], out_keys=["observation"])
        )
    elif pixel_obs:
        # Rename pixels → observation for a uniform key across all envs
        from torchrl.envs.transforms import RenameTransform
        transforms.append(RenameTransform(["pixels"], ["observation"]))

    if clip_rewards:
        transforms.append(RewardClipping(-1.0, 1.0))

    if normalize_obs:
        from torchrl.envs.transforms import ObservationNorm
        transforms.append(ObservationNorm(in_keys=["observation"]))

    from torchrl.envs.transforms import StepCounter
    transforms.append(StepCounter())

    if transforms:
        from torchrl.envs.transforms import Compose
        return TransformedEnv(base_env, Compose(*transforms))
    return base_env


def _make_dmcontrol_env(
    name: str,
    task: str,
    normalize_obs: bool,
    device: str,
):
    from torchrl.envs import DMControlEnv, TransformedEnv
    from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter

    base_env = DMControlEnv(name, task, device=device)

    obs_keys = list(base_env.observation_spec.keys())
    from torchrl.envs.transforms import CatTensors
    transforms: list = [
        DoubleToFloat(in_keys=obs_keys),
        CatTensors(in_keys=obs_keys, out_key="observation", del_keys=True),
    ]

    if normalize_obs:
        from torchrl.envs.transforms import ObservationNorm
        transforms.append(ObservationNorm(in_keys=["observation"]))

    transforms.append(StepCounter())

    return TransformedEnv(base_env, Compose(*transforms))
