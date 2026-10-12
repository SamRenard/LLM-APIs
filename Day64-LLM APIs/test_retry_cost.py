import pytest

from src.cost import calculate_cost
from src.retry import retry_call, retry_stream


class Transient(Exception):
    status_code = 429


class Fatal(Exception):
    status_code = 400


def test_retry_succeeds_after_transient_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient()
        return "ok"

    assert retry_call(flaky, sleep=lambda _: None) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_fatal():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise Fatal()

    with pytest.raises(Fatal):
        retry_call(bad, sleep=lambda _: None)
    assert calls["n"] == 1


def test_stream_retries_only_before_first_chunk():
    attempts = {"n": 0}

    def factory():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Transient()
        yield "a"
        yield "b"

    assert list(retry_stream(factory, sleep=lambda _: None)) == ["a", "b"]

    def broken_midway():
        yield "a"
        raise Transient()

    with pytest.raises(Transient):
        list(retry_stream(broken_midway, sleep=lambda _: None))


def test_cost():
    pricing = {"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}}
    assert calculate_cost("m", 1_000_000, 500_000, pricing) == pytest.approx(2.0)
    assert calculate_cost("unknown", 1, 1, pricing) is None
