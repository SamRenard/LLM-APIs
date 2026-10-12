"""Retry helpers with exponential backoff and full jitter."""
import random
import time
from typing import Callable, Iterator, Optional, TypeVar

T = TypeVar("T")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRYABLE_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
}


def is_retryable(exc: Exception) -> bool:
    """Retry only transient failures (rate limits, 5xx, network)."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return type(exc).__name__ in RETRYABLE_NAMES


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Full jitter: random value in [0, min(cap, base * 2^attempt)]."""
    return random.uniform(0, min(cap, base * (2 ** attempt)))


def retry_call(
    fn: Callable[[], T],
    max_retries: int = 5,
    base: float = 1.0,
    cap: float = 30.0,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries or not is_retryable(exc):
                raise
            delay = backoff_delay(attempt, base, cap)
            if on_retry:
                on_retry(attempt + 1, delay, exc)
            sleep(delay)
    raise RuntimeError("unreachable")


def retry_stream(
    factory: Callable[[], Iterator[T]],
    max_retries: int = 5,
    base: float = 1.0,
    cap: float = 30.0,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[T]:
    """Retry a streaming call, but only if it fails BEFORE the first chunk.

    Once output has been shown to the user, retrying would duplicate text,
    so the error is re-raised instead.
    """
    for attempt in range(max_retries + 1):
        started = False
        try:
            for item in factory():
                started = True
                yield item
            return
        except Exception as exc:
            if started or attempt == max_retries or not is_retryable(exc):
                raise
            delay = backoff_delay(attempt, base, cap)
            if on_retry:
                on_retry(attempt + 1, delay, exc)
            sleep(delay)
