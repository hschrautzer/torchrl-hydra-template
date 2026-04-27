"""Step-based trainer for algorithms like DQN and PPO."""
from __future__ import annotations

from src.algorithms.utils import last_episode_return
from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


class StepTrainer(BaseTrainer):
    """Trainer for step-based algorithms using ``SyncDataCollector``.

    Each iteration: collector yields a batch → ``algorithm.step(batch)``.
    Used by DQN, PPO.
    """

    def setup(self) -> None:
        """Create env, set up algorithm, then create the collector."""
        super().setup()
        self._create_collector()

    def _create_collector(self) -> None:
        from torchrl.collectors import SyncDataCollector

        collector_cfg = self.algorithm.get_collector_config()

        self.collector = SyncDataCollector(
            create_env_fn=self.train_env,
            policy=self.algorithm.get_explore_policy(),
            frames_per_batch=collector_cfg.frames_per_batch,
            total_frames=collector_cfg.total_frames,
            split_trajs=collector_cfg.split_trajs,
            device=self.device,
            storing_device=self.device,
        )

    def _training_loop(self) -> dict[str, float]:
        log_every = int(self.trainer_cfg.log_every_n_steps)

        metrics: dict[str, float] = {}
        for batch in self.collector:
            batch = self.algorithm.on_batch_collected(batch)

            batch_frames = batch.numel()
            self._step += batch_frames

            if not self.algorithm.should_skip_update(self._step):
                metrics = self.algorithm.step(batch)

            self.algorithm.on_step_complete(self._step)

            if self._should_log(log_every, batch_frames):
                metrics["reward/last"] = last_episode_return(batch)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics
