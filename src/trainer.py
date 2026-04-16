from __future__ import annotations

from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


class TrainerEvent(Enum):
    ON_TRAIN_START = auto()
    ON_STEP_END = auto()
    ON_TRAIN_END = auto()
    ON_EVAL_START = auto()
    ON_EVAL_END = auto()


@runtime_checkable
class Callback(Protocol):
    def on_train_start(self, state: dict[str, Any]) -> None: ...
    def on_step_end(self, metrics: dict[str, float], step: int) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...


def fire_callbacks(
    event: TrainerEvent,
    callbacks: list,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch a training event to all callbacks that implement the matching method.

    Args:
        event: the TrainerEvent to fire
        callbacks: list of callback objects
        *args, **kwargs: forwarded to the matching method on each callback
    """
    method_name = event.name.lower()
    for cb in callbacks:
        method = getattr(cb, method_name, None)
        if callable(method):
            method(*args, **kwargs)
