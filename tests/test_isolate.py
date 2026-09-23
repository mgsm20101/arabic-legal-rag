"""`run_isolated` — a killable subprocess with a wall-clock timeout (T04).

Generic, no PDF knowledge: worker functions here are trivial and module-level
(spawn re-imports this file fresh in the child, so a worker must be an
importable name, not a lambda or closure — see isolate.py's docstring).
"""

import multiprocessing
import queue
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


# The timeout `run_isolated` is given starts at spawn, not at the point the child is
# running, so it has to cover interpreter start-up as well as the work. A cold `spawn`
# on Windows re-imports this module in a fresh interpreter, and under load that took
# longer than the 2.0s this test used to allow: the child was killed before it reached
# its `pid_queue.put`, and the test failed on an empty queue — a flake, roughly 1 run
# in 8 while the machine was busy, and never once when the file ran on its own.
#
# The fix is headroom, not precision. This test asserts that the process is *gone*,
# not that it died on schedule (the test above owns that), so widening the gap from
# 2-against-5 to 8-against-60 strengthens "well before the full sleep" while giving
# start-up room it demonstrably needed. It costs ~6s of suite time, once.
#
# Later: the same flake reappeared in three more tests during a heavy run, including
# one whose worker only returns a string — a trivial worker timed out at 5.0s, which
# can only be start-up. Those tests were making the same mistake in a quieter way:
# each picked its own comfortable-looking timeout, and every one of them was really
# a bet on how fast a fresh Windows interpreter starts on a loaded machine. None of
# them is a timing test. `SPAWN_HEADROOM_SECONDS` is now the single place that bet
# is made, so raising it once fixes all of them, and no test asserts speed unless
# that is the behaviour it exists to check.
SPAWN_HEADROOM_SECONDS = 20.0
UNREACHABLE_SLEEP = 60.0


def test_a_worker_that_hangs_leaves_no_process_still_alive_afterward():
    """Proves the OS process is actually gone, not merely abandoned: a thread-based timeout
    that lets the underlying work keep running is exactly the failure mode T04 rejects."""
    ctx = multiprocessing.get_context("spawn")
    pid_queue = ctx.Queue()

    with pytest.raises(IsolationTimeout):
        run_isolated(
            _hangs_forever_after_reporting_pid,
            (pid_queue, UNREACHABLE_SLEEP),
            timeout_seconds=SPAWN_HEADROOM_SECONDS,
        )

    try:
        child_pid = pid_queue.get(timeout=5)
    except queue.Empty:  # pragma: no cover - only on a regression or a very slow box
        pytest.fail(
            f"the child never reported its pid within {SPAWN_HEADROOM_SECONDS}s. "
            "Either spawn start-up is slower than that here (raise "
            "SPAWN_HEADROOM_SECONDS) or run_isolated is killing the child before it "
            "runs at all, which would be the real bug this test exists to catch."
        )
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
    assert run_isolated(_returns_this, ("hello",), timeout_seconds=SPAWN_HEADROOM_SECONDS) == "hello"


def test_a_worker_that_raises_surfaces_as_a_crash_chaining_the_original_exception():
    with pytest.raises(IsolationCrash) as failed:
        run_isolated(_raises_this, ("boom",), timeout_seconds=SPAWN_HEADROOM_SECONDS)

    assert "boom" in str(failed.value)
    assert isinstance(failed.value.__cause__, ValueError)
    assert str(failed.value.__cause__) == "boom"


def test_a_worker_that_exits_hard_is_a_crash_not_a_silent_hang_or_queue_empty():
    with pytest.raises(IsolationCrash) as failed:
        run_isolated(_crashes_hard, (), timeout_seconds=SPAWN_HEADROOM_SECONDS)

    assert "without producing a result" in str(failed.value)


def test_a_moderately_large_result_round_trips_through_the_queue():
    size = 2_000_000
    result = run_isolated(_returns_a_lot_of_data, (size,), timeout_seconds=SPAWN_HEADROOM_SECONDS + 10.0)
    assert result == "x" * size


def test_two_concurrent_isolated_calls_do_not_interfere_with_each_others_results():
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def _run(key, value):
        try:
            results[key] = run_isolated(_returns_this, (value,), timeout_seconds=SPAWN_HEADROOM_SECONDS)
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
