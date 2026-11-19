"""Keep the computer from idle-sleeping while a long experiment runs, and time work without
counting sleep.

On Windows keep_awake calls SetThreadExecutionState, the same request media players make. It
changes no setting and ends by itself when the process exits. A closed lid or a manual Sleep
still sleeps the computer. On other systems it does nothing.

awake_seconds is a clock that stops while the computer sleeps. time.perf_counter keeps running
during sleep on Windows, so a fit that spans a sleep would otherwise be recorded as too long.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _make_awake_clock() -> Callable[[], float]:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ticks = ctypes.c_ulonglong()

        def clock() -> float:
            # Interrupt time without the time spent in sleep or hibernation, in 100 ns units.
            kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks))
            return ticks.value / 1e7

        return clock
    if sys.platform.startswith("linux"):
        return time.monotonic  # CLOCK_MONOTONIC does not count suspend on Linux
    return time.perf_counter


awake_seconds: Callable[[], float] = _make_awake_clock()
"""Seconds on a clock that does not advance while the computer sleeps (resolution about
16 ms on Windows: use it for fits and batch predictions, not for single-row latency)."""


@contextmanager
def keep_awake(log=None) -> Iterator[bool]:
    """Inside the block the system will not idle-sleep (Windows only). Yields whether the
    request was accepted."""
    accepted = False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        accepted = bool(kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
        if log is not None:
            log.info("keep-awake request %s", "accepted" if accepted else "refused")
    try:
        yield accepted
    finally:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)  # type: ignore[attr-defined]
