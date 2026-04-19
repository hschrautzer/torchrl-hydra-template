"""Environment factory: builds a TorchRL TransformedEnv from Hydra config params.

Usage (from algorithm setup):
    from hydra.utils import instantiate
    env = instantiate(cfg.environment, device=str(self.device))

Or directly:
    from src.environments.factory import make_env
    env = make_env(**OmegaConf.to_container(cfg.environment, resolve=True), device="cpu")
"""
from __future__ import annotations

from functools import partial
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
    max_episode_steps: int | None = None,
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
        env_fn = partial(
            _make_gymnasium_env,
            name=name,
            grayscale=grayscale,
            resize=resize,
            frame_stack=frame_stack,
            clip_rewards=clip_rewards,
            normalize_obs=normalize_obs,
            device=device,
        )
    elif backend == "dm_control":
        env_fn = partial(
            _make_dmcontrol_env,
            name=name,
            task=task or "walk",
            normalize_obs=normalize_obs,
            device=device,
        )
    elif backend == "envpool":
        return _make_envpool_env(
            name=name,
            num_envs=num_envs,
            clip_rewards=clip_rewards,
            normalize_obs=normalize_obs,
            device=device,
            max_episode_steps=max_episode_steps,
        )
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: 'gymnasium', 'dm_control', 'envpool'."
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


def _unsqueeze_signals(td):
    """Ensure reward/done signals have a trailing singleton dim.

    EnvPool produces shape ``[N]`` but torchrl losses expect ``[N, 1]``.
    """
    for key in ["reward", "done", "terminated", "truncated"]:
        t = td.get(key, None)
        if t is not None and t.dim() >= 1 and t.shape[-1] != 1:
            td.set(key, t.unsqueeze(-1))
    next_td = td.get("next", None)
    if next_td is not None:
        _unsqueeze_signals(next_td)
    return td


def _patch_envpool_shapes(env, num_envs: int, max_episode_steps: int | None = None):
    """Patch an envpool env: fix signal shapes and split done → terminated/truncated.

    envpool auto-resets on done and collapses terminated with truncated (both
    equal to done). DQN needs them distinct to correctly bootstrap from
    time-limit truncations; otherwise it learns Q-values as if the episode
    truly ended at the time limit. When ``max_episode_steps`` is provided, we
    track per-env step counts and derive ``terminated = done AND step_count <
    max_episode_steps`` (a done at the step limit is a truncation, not a real
    termination).
    """
    import torch as _torch

    step_counters = (
        _torch.zeros(num_envs, dtype=_torch.long)
        if max_episode_steps is not None
        else None
    )

    def _split_done(next_td):
        if step_counters is None:
            return
        done = next_td.get("done", None)
        if done is None:
            return
        step_counters.add_(1)
        done_cpu = done.view(-1).bool().cpu()
        trunc = done_cpu & (step_counters >= max_episode_steps)
        term = done_cpu & ~trunc
        next_td.set(
            "terminated",
            term.view_as(done.cpu()).to(device=done.device, dtype=done.dtype),
        )
        next_td.set(
            "truncated",
            trunc.view_as(done.cpu()).to(device=done.device, dtype=done.dtype),
        )
        step_counters[done_cpu] = 0

    orig_step = env.step

    def patched_step(td, **kwargs):
        result = orig_step(td, **kwargs)
        _unsqueeze_signals(result)
        next_td = result.get("next", None)
        if next_td is not None:
            _split_done(next_td)
        return result

    env.step = patched_step

    orig_reset = env.reset

    def patched_reset(td=None, **kwargs):
        result = orig_reset(td, **kwargs)
        _unsqueeze_signals(result)
        if step_counters is not None:
            step_counters.zero_()
        return result

    env.reset = patched_reset
    return env


def _make_envpool_env(
    name: str,
    num_envs: int,
    clip_rewards: bool,
    normalize_obs: bool,
    device: str,
    max_episode_steps: int | None = None,
):
    from torchrl.envs import MultiThreadedEnv, TransformedEnv
    from torchrl.envs.transforms import Compose, RewardClipping

    env = _patch_envpool_shapes(
        MultiThreadedEnv(num_workers=num_envs, env_name=name),
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
    )

    transforms = []
    if clip_rewards:
        transforms.append(RewardClipping(-1.0, 1.0))
    if normalize_obs:
        from torchrl.envs.transforms import ObservationNorm
        transforms.append(ObservationNorm(in_keys=["observation"]))

    if transforms:
        return TransformedEnv(env, Compose(*transforms))
    return env


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
