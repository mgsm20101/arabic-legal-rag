"""`run_isolated` — a killable subprocess with a wall-clock timeout (T04).

Generic, no PDF knowledge: worker functions here are trivial and module-level
(spawn re-imports this file fresh in the child, so a worker must be an
importable name, not a lambda or closure — see isolate.py's docstring).
"""

import multiprocessing
import sys
import threading
import time
from pathlib import Path
from time import monotonic

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.isolate import IsolationCrash, IsolationTimeout, run_isolated  # noqa: E402

# Kept well under EXTRACTION_BUDGET_SECONDS-scale numbers so this file's own runtime stays
# small: a timeout test only has to prove "much less than the sleep", not exercise a realistic
# production-sized budget.
SHORT_TIMEOUT = 0.3
LONG_SLEEP = 5.0


def _hangs_forever(seconds):
    time.sleep(seconds)


def _hangs_forever_after_reporting_pid(pid_queue, seconds):
    import os
    pid_queue.put(os.getpid())
    time.sleep(seconds)


def _returns_this(value):
    return value


def _raises_this(message):
    raise ValueError(message)


def _returns_a_lot_of_data(size):
    return "x" * size


def _crashes_hard():
    import os
    os._exit(1)


def test_a_worker_that_hangs_is_killed_within_roughly_the_timeout_not_the_full_sleep():
    start = monotonic()

    with pytest.raises(IsolationTimeout):
        run_isolated(_hangs_forever, (LONG_SLEEP,), timeout_seconds=SHORT_TIMEOUT)

    elapsed = monotonic() - start
    assert elapsed < LONG_SLEEP / 2


def test_a_worker_that_hangs_leaves_no_process_still_alive_afterward():
    """Proves the OS process is actually gone, not merely abandoned: a thread-based timeout
    that lets the underlying work keep running is exactly the failure mode T04 rejects."""
    ctx = multiprocessing.get_context("spawn")
    pid_queue = ctx.Queue()
    # A more generous timeout than SHORT_TIMEOUT: this test only needs "killed well before the
    # full sleep", not timing precision, and the child must have enough headroom to actually
    # start up and report its pid before it gets killed.
    generous_timeout = 2.0

    with pytest.raises(IsolationTimeout):
        run_isolated(_hangs_forever_after_reporting_pid, (pid_queue, LONG_SLEEP), timeout_seconds=generous_timeout)

    child_pid = pid_queue.get(timeout=5)
    if sys.platform != "win32":
        import errno
        import os
        with pytest.raises(OSError) as no_such_process:
            os.kill(child_pid, 0)  # signal 0: probe liveness without actually signaling
        assert no_such_process.value.errno == errno.ESRCH
    # Portable check, works on every platform `run_isolated` supports: `run_isolated` only
    # returns after `process.join()`, so multiprocessing's own bookkeeping must already
    # consider that child finished.
    assert child_pid not in {p.pid for p in multiprocessing.active_children()}


def test_a_worker_that_returns_quickly_returns_its_value():
    assert run_isolated(_returns_this, ("hello",), timeout_seconds=5.0) == "hello"


def test_a_worker_that_raises_surfaces_as_a_crash_chaining_the_original_exception():
    with pytest.raises(IsolationCrash) as failed:
        run_isolated(_raises_this, ("boom",), timeout_seconds=5.0)

    assert "boom" in str(failed.value)
    assert isinstance(failed.value.__cause__, ValueError)
    assert str(failed.value.__cause__) == "boom"


def test_a_worker_that_exits_hard_is_a_crash_not_a_silent_hang_or_queue_empty():
    with pytest.raises(IsolationCrash) as failed:
        run_isolated(_crashes_hard, (), timeout_seconds=5.0)

    assert "without producing a result" in str(failed.value)


def test_a_moderately_large_result_round_trips_through_the_queue():
    size = 2_000_000
    result = run_isolated(_returns_a_lot_of_data, (size,), timeout_seconds=15.0)
    assert result == "x" * size


def test_two_concurrent_isolated_calls_do_not_interfere_with_each_others_results():
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def _run(key, value):
        try:
            results[key] = run_isolated(_returns_this, (value,), timeout_seconds=10.0)
        except BaseException as e:  # noqa: BLE001 - surfaced to the main thread via `errors`
            errors.append(e)

    threads = [
        threading.Thread(target=_run, args=("a", "value-a")),
        threading.Thread(target=_run, args=("b", "value-b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)

    assert not errors, errors
    assert results == {"a": "value-a", "b": "value-b"}
