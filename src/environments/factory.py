"""Environment factory: builds a TorchRL TransformedEnv from Hydra config params.

Or directly:
    from src.environments.factory import make_env
    env = make_env(**OmegaConf.to_container(cfg.environment, resolve=True), device="cpu")
"""
from __future__ import annotations

import importlib
from functools import partial
from typing import Sequence


def make_env(
    name: str,
    backend: str,
    obs_shape: Sequence[int],
    num_actions: int,
    num_envs: int = 1,
    device: str = "cpu",
    transforms: list | None = None,
    from_pixels: bool = False,
    task: str | None = None,
    max_episode_steps: int | None = None,
    **kwargs,
):
    """Build a (possibly vectorised) TransformedEnv.

    Args:
        name: environment name (e.g. "CartPole-v1", "ALE/Pong-v5", "humanoid")
        backend: "gymnasium", "dm_control", or "envpool"
        obs_shape: expected observation shape after preprocessing (for validation)
        num_actions: number of actions (discrete count or continuous dim)
        num_envs: number of parallel envs (>1 → ParallelEnv for gymnasium/dm_control)
        device: target device string; ParallelEnv workers always run on CPU
            (CUDA contexts cannot survive fork) — the collector moves data to GPU
        transforms: list of transform dicts (each with ``_target_`` key), instantiated
            fresh per call.  Gymnasium only.
        from_pixels: pass ``from_pixels=True`` to ``GymEnv``.  Gymnasium only.
        task: dm_control task string (e.g. "walk")
        max_episode_steps: maximum steps per episode (envpool only)
        **kwargs: extra kwargs forwarded to backend-specific helpers
            (e.g. ``normalize_obs`` for dm_control, ``clip_rewards`` for envpool)

    Returns:
        TransformedEnv (single env or wrapped in ParallelEnv)
    """
    # ParallelEnv spawns/forks worker processes. CUDA contexts cannot be used
    # in forked children — passing a GPU device to GymEnv inside a worker causes
    # a bus error during the first CUDA call. Workers always use CPU; the
    # SyncDataCollector moves batched tensors to the target device after collection.
    worker_device = "cpu" if num_envs > 1 else device

    if backend == "gymnasium":
        env_fn = partial(
            _make_gymnasium_env,
            name=name,
            transforms=transforms,
            from_pixels=from_pixels,
            device=worker_device,
        )
    elif backend == "dm_control":
        env_fn = partial(
            _make_dmcontrol_env,
            name=name,
            task=task or "walk",
            device=worker_device,
            **kwargs,
        )
    elif backend == "envpool":
        return _make_envpool_env(
            name=name,
            num_envs=num_envs,
            device=device,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: 'gymnasium', 'dm_control', 'envpool'."
        )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv
        # "fork" (Linux default) creates workers as copies of the parent process.
        # ALE (Atari) and other native libs with global C-level state can corrupt
        # after forking, producing bus errors tens of thousands of steps in.
        # "spawn" starts each worker from a clean Python interpreter so there is
        # no inherited C state — safe for any env at the cost of slower startup.
        return ParallelEnv(num_envs, env_fn, mp_start_method="spawn")
    else:
        return env_fn()


def _instantiate_transform(cfg: dict):
    """Instantiate a TorchRL transform from a ``_target_``-keyed dict.

    Uses importlib directly instead of ``hydra.utils.instantiate`` so this
    function is safe to call inside forked ``ParallelEnv`` worker processes
    where Hydra's global initialisation state is not guaranteed.
    """
    cfg = dict(cfg)  # copy — do not mutate the caller's list element
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg)


def _make_gymnasium_env(
    name: str,
    transforms: list | None,
    from_pixels: bool,
    device: str,
):
    """Build a gymnasium TransformedEnv from an explicit transforms list.

    Each element of ``transforms`` must be a dict with a ``_target_`` key
    (and any constructor kwargs) as produced by ``OmegaConf.to_container``.
    ``_instantiate_transform`` is called fresh per element so each call
    produces independent transform objects with independent state
    (important for stateful transforms like ``CatFrames``).

    Args:
        name: gymnasium env name (e.g. "CartPole-v1", "ALE/Pong-v5")
        transforms: list of dicts, each with ``_target_`` pointing to a
            ``torchrl.envs.transforms`` class and any constructor kwargs.
            If ``None`` or empty, the bare base env is returned.
        from_pixels: if ``True``, pass ``from_pixels=True`` to ``GymEnv``
            so the ``"pixels"`` observation key is available.
        device: target device string.

    Returns:
        TransformedEnv (or bare GymEnv when transforms is empty/None).
    """
    from torchrl.envs import GymEnv, TransformedEnv
    from torchrl.envs.transforms import Compose

    base_env = GymEnv(name, device=device, from_pixels=from_pixels)

    if not transforms:
        return base_env

    transform_objects = [_instantiate_transform(t) for t in transforms]
    return TransformedEnv(base_env, Compose(*transform_objects))


def _patch_envpool_reset_mask(env):
    """Squeeze a trailing singleton from the ``_reset`` mask before envpool sees it.

    Once we unsqueeze done to ``[num_envs, 1]`` (see ``_make_envpool_env``),
    torchrl's ``maybe_reset`` forwards ``_reset`` with the same trailing
    dim, but envpool's ``_reset`` does ``self.obs[reset_workers]`` where
    ``self.obs`` has shape ``[num_envs]`` — so a ``[N, 1]`` mask raises
    ``IndexError``. Squeezing it keeps both sides happy.
    """
    _orig = env._reset

    def _reset(td, **kwargs):
        if td is not None:
            r = td.get("_reset", None)
            if r is not None and r.ndim > 1 and r.shape[-1] == 1:
                td = td.clone(False)
                td.set("_reset", r.squeeze(-1))
        return _orig(td, **kwargs)

    env._reset = _reset
    return env


def _make_envpool_env(
    name: str,
    num_envs: int,
    device: str,
    max_episode_steps: int | None = None,
    **kwargs,
):
    """envpool-backed vectorised env via torchrl's ``MultiThreadedEnv``.

    Two non-obvious details:

    * ``MultiThreadedEnvWrapper._get_action_spec`` hardcodes
      ``categorical_action_encoding=True``, so actions are scalar ints as the
      rest of the pipeline expects — no config knob needed.
    * Reward/done/terminated/truncated are emitted with shape ``[num_envs]``
      rather than torchrl's standard ``[num_envs, 1]``. Without the trailing
      singleton, ``DQNLoss`` raises ``"All input tensors (value, reward and
      done states) must share a unique shape"``. We fix this with an
      ``UnsqueezeTransform`` on those keys, plus a tiny patch to the base
      env's ``_reset`` so the now-2D ``_reset`` mask gets squeezed back to 1D
      before indexing envpool's internal obs buffer.
    """
    clip_rewards = kwargs.get("clip_rewards", False)
    normalize_obs = kwargs.get("normalize_obs", False)

    from torchrl.envs import MultiThreadedEnv, TransformedEnv
    from torchrl.envs.transforms import (
        Compose,
        ObservationNorm,
        RewardClipping,
        UnsqueezeTransform,
    )

    env = _patch_envpool_reset_mask(
        MultiThreadedEnv(
            num_workers=num_envs,
            env_name=name,
            device=device,
        )
    )

    # StepCounter is omitted here: its internal step_count is 1D while the
    # unsqueeze makes the _reset mask 2D, and it can't expand across them.
    # Last-episode logging reads "next.done" directly, so nothing depends on
    # step_count for this env path.
    transforms: list = [
        UnsqueezeTransform(
            dim=-1,
            in_keys=["reward", "done", "terminated", "truncated"],
            in_keys_inv=[],
        ),
    ]
    if clip_rewards:
        transforms.append(RewardClipping(-1.0, 1.0))
    if normalize_obs:
        transforms.append(ObservationNorm(in_keys=["observation"]))

    return TransformedEnv(env, Compose(*transforms))


def _make_dmcontrol_env(
    name: str,
    task: str,
    device: str,
    **kwargs,
):
    normalize_obs = kwargs.get("normalize_obs", False)

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
