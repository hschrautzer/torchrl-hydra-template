"""Training entry point.

Usage:
    python src/train.py experiment=reinforce/cartpole
    python src/train.py experiment=dqn/cartpole logger=[wandb,tensorboard]
    python src/train.py experiment=dqn/atari_breakout trainer.accelerator=gpu trainer.devices=[0]
    python src/train.py experiment=ppo/dmc_humanoid trainer.accelerator=gpu
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    _train(cfg)


def _train(cfg: DictConfig) -> dict[str, float]:
    """Separated from the Hydra decorator for testability.

    Args:
        cfg: fully composed Hydra config

    Returns:
        dict of final training metrics
    """
    from hydra.utils import get_class

    from src.utils.device import resolve_device
    from src.utils.instantiate import build_callbacks, build_loggers
    from src.utils.seeding import seed_everything

    seed_everything(int(cfg.trainer.seed))
    device = resolve_device(cfg.trainer.accelerator, list(cfg.trainer.devices))

    # Instantiate loggers (may be empty list = no logging)
    loggers = build_loggers(cfg.logger)

    # Instantiate algorithm (select class via _target_, pass full cfg + device)
    AlgClass = get_class(cfg.algorithm._target_)
    algorithm = AlgClass(cfg=cfg, device=device)
    algorithm.setup()

    # Optionally resume from a checkpoint
    if cfg.checkpoint.get("resume_from") is not None:
        algorithm.load_checkpoint(cfg.checkpoint.resume_from)

    callbacks = build_callbacks(cfg.trainer, cfg.checkpoint, algorithm, loggers)

    return algorithm.train(cfg.trainer, callbacks)


if __name__ == "__main__":
    train()
