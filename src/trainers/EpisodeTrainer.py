"""Episodic trainer for algorithms like REINFORCE."""
from __future__ import annotations

from src.algorithms.utils import last_episode_return
from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


class EpisodeTrainer(BaseTrainer):
    """Trainer for episodic algorithms that use ``env.rollout()``.

    Each iteration: roll out one full episode → ``algorithm.step(episode)``.
    No ``SyncDataCollector`` is used. Used by REINFORCE.
    """

    def _training_loop(self) -> dict[str, float]:
        total_frames = int(self.trainer_cfg.total_frames)
        log_every = int(self.trainer_cfg.log_every_n_steps)

        explore_policy = self.algorithm.get_explore_policy()
        metrics: dict[str, float] = {}

        while self._step < total_frames:
            remaining = total_frames - self._step

            rollout = self.train_env.rollout(
                max_steps=remaining,
                policy=explore_policy,
                auto_reset=True,
            )

            rollout = self.algorithm.on_batch_collected(rollout)

            batch_frames = rollout.batch_size[0]

            if not self.algorithm.should_skip_update(self._step + batch_frames):
                metrics = self.algorithm.step(rollout)

            self._step += batch_frames
            self.algorithm.on_step_complete(self._step)

            if self._should_log(log_every, batch_frames):
                metrics["reward/last"] = last_episode_return(rollout)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics
