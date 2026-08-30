# SPDX-License-Identifier: Apache-2.0
"""Low-overhead, opt-in diagnostics for engine startup phases."""

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_STARTUP_PHASE_LOGGING = os.getenv("VLLM_STARTUP_PHASE_LOGGING", "0") == "1"
_PROCESS_START = time.perf_counter()
_R = TypeVar("_R")


def log_startup_phase(phase: str, event: str, elapsed: float | None = None) -> None:
    """Emit one structured startup phase event when diagnostics are enabled."""
    if not _STARTUP_PHASE_LOGGING:
        return
    fields: list[str] = [
        f"phase={phase}",
        f"event={event}",
        f"pid={os.getpid()}",
        f"process_elapsed={time.perf_counter() - _PROCESS_START:.3f}",
    ]
    if elapsed is not None:
        fields.append(f"elapsed={elapsed:.3f}")
    logger.info("STARTUP_PHASE %s", " ".join(fields))


@contextmanager
def startup_phase(phase: str) -> Iterator[None]:
    """Log the start and completion of a startup phase."""
    if not _STARTUP_PHASE_LOGGING:
        yield
        return

    start = time.perf_counter()
    log_startup_phase(phase, "start")
    try:
        yield
    finally:
        log_startup_phase(phase, "end", time.perf_counter() - start)


def startup_phase_decorator(
    phase: str,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """Decorate a startup method without changing its exception behavior."""

    def decorate(func: Callable[..., _R]) -> Callable[..., _R]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> _R:
            with startup_phase(phase):
                return func(*args, **kwargs)

        return wrapped

    return decorate
