from __future__ import annotations

import torch
from tensordict import TensorDict


def last_episode_return(batch: TensorDict) -> float:
    """Return the cumulative reward of the most recently completed episode.

    Flattens the batch, locates the last ``done`` transition, then sums
    rewards backwards to the start of that episode.  Returns 0.0 when no
    episode completed in this batch.
    """
    flat = batch.reshape(-1)
    rewards = flat.get(("next", "reward"))
    dones = flat.get(("next", "done")).bool()

    done_idx = dones.nonzero(as_tuple=True)[0]
    if done_idx.numel() == 0:
        return 0.0

    end = done_idx[-1].item()
    prev = done_idx[done_idx < end]
    start = prev[-1].item() + 1 if prev.numel() > 0 else 0
    return rewards[start : end + 1].sum().item()
