from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.algorithms.base import BaseAlgorithm


class CheckpointCallback:
    """Saves full training state at fixed step intervals and optionally at the end.

    Args:
        save_dir: directory where checkpoint files are written
        save_every_n_steps: save a checkpoint every this many environment steps
        save_last: if True, save "last.pt" when training finishes
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_every_n_steps: int,
        save_last: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_last = save_last
        self._algorithm: BaseAlgorithm | None = None
        self._last_saved_step: int = 0

    def set_algorithm(self, algorithm: BaseAlgorithm) -> None:
        """Inject the algorithm instance (called by build_callbacks)."""
        self._algorithm = algorithm

    def on_train_start(self, state: dict[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._algorithm is None:
            return
        # Check if we've crossed a save boundary since the last save
        if step // self.save_every_n_steps > self._last_saved_step // self.save_every_n_steps:
            path = self.save_dir / f"step_{step:010d}.pt"
            self._algorithm.save_checkpoint(path)
            self._last_saved_step = step

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._algorithm is None:
            return
        if self.save_last:
            self._algorithm.save_checkpoint(self.save_dir / "last.pt")
