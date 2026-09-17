"""Run a picklable callable in its own process, with a wall-clock timeout that
actually kills it. Nothing here is PDF-specific — `docextract.py` is the first
caller, wiring `pdf_text.extract_pages` through it, but this module knows
nothing about PDFs.

`multiprocessing.Process` + `Queue` directly, not `ProcessPoolExecutor`:
`Future.result(timeout=)` raises `TimeoutError` on a pool without stopping the
worker underneath it, which is exactly the "the caller gives up waiting but
the work keeps running" failure this module exists to avoid. `run_isolated`
instead joins the process itself and, on a timeout, terminates (then kills)
the OS process before it ever raises — the process is provably gone
(`is_alive()` is `False`) by the time the caller sees the exception, not just
abandoned.

Always `multiprocessing.get_context("spawn")`, on every platform: fork is not
available on Windows at all, and relying on the platform default would make
Windows and Linux/Mac behave (and impose picklability constraints) differently
for no benefit. Under spawn the child re-imports every module from scratch and
only unpickles `target`/`args`/the result by reference, so:

- `target` MUST be a module-level function — importable as `module.qualname`
  from a fresh interpreter. Not a lambda, not a closure, not a bound method of
  a local object: none of those exist yet when the child re-imports the
  module that defines them.
- `args`, and whatever `target` returns, must themselves be picklable: plain
  data (paths, strings, numbers, lists/dicts of the same) — no open files,
  sockets, locks, or local closures.
"""

from __future__ import annotations

import multiprocessing
import pickle
import queue as _queue_module
from time import monotonic
from typing import Any, Callable

# One context object, reused for every call: constructing it is cheap and
# this keeps every `run_isolated` call unambiguously on "spawn" regardless of
# what `multiprocessing.set_start_method` some other part of the process may
# have configured (or not) as the platform default.
_MP_CONTEXT = multiprocessing.get_context("spawn")

# After terminate() a process still needs a moment to actually unwind and
# exit (signal delivery and interpreter teardown are not instantaneous even
# on POSIX; on Windows terminate() is already the hard TerminateProcess call).
# If it is still alive after this long, kill() is used instead, which — on
# POSIX — sends SIGKILL, a signal the process cannot catch or ignore.
_TERMINATE_GRACE_SECONDS = 1.0
_KILL_REAP_SECONDS = 1.0

# How often `run_isolated` re-checks the queue/process while waiting: frequent enough that a
# crash (the process exits without a result) is noticed promptly rather than only at the full
# timeout, coarse enough not to busy-loop. This also bounds how quickly a genuine timeout is
# detected past `timeout_seconds` itself.
_POLL_INTERVAL_SECONDS = 0.05

# How long a process gets to actually exit after `run_isolated` has already read its result off
# the queue. Should be near-instant (the worker already returned and the message the parent just
# read is, by construction, everything the feeder thread needed to send) — this only guards
# against something unexpected keeping the process alive a moment longer.
_EXIT_AFTER_RESULT_GRACE_SECONDS = 5.0


class IsolationError(Exception):
    """Base class for everything `run_isolated` raises itself, as opposed to
    a worker's own exception surfacing through it unchanged."""


class IsolationTimeout(IsolationError):
    """`target` did not finish within `timeout_seconds`. By the time this is
    raised, the worker process has already been terminated (killed if it
    ignored that) and reaped: it is not still running somewhere."""


class IsolationCrash(IsolationError):
    """The worker did not return a result: it raised an exception (available
    as `__cause__` when that exception pickled successfully — see
    `_bootstrap`), or the process exited on its own (a hard crash: a
    segfault, `os._exit`, a fatal interpreter error) without the `_bootstrap`
    wrapper getting a chance to report anything."""


def run_isolated(target: Callable[..., Any], args: tuple, timeout_seconds: float) -> Any:
    """Run `target(*args)` in a fresh subprocess and return what it returns.

    Raises `IsolationTimeout` if `target` is still running after
    `timeout_seconds` (the process is terminated/killed and reaped before
    this is raised — never left running in the background). Raises
    `IsolationCrash` if the worker raised an exception (chained as
    `__cause__` when picklable) or the process exited without a result.
    """
    result_queue: multiprocessing.Queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(target=_bootstrap, args=(target, args, result_queue))
    process.start()
    try:
        got_result, status, payload = _await_result(process, result_queue, timeout_seconds)
        if not got_result:
            if process.is_alive():
                _kill(process)
                raise IsolationTimeout(
                    f"{_name(target)} did not finish within {timeout_seconds:.3f}s and was terminated"
                )
            raise IsolationCrash(
                f"{_name(target)} exited with code {process.exitcode} without producing a result"
            )
        # A result arrived; let the process finish exiting on its own (fast — see
        # _EXIT_AFTER_RESULT_GRACE_SECONDS) before reporting it, so cleanup on the way out
        # doesn't have to kill a process that was already finishing up normally.
        process.join(_EXIT_AFTER_RESULT_GRACE_SECONDS)
        if status == "error":
            raise IsolationCrash(f"{_name(target)} raised {type(payload).__name__}: {payload}") from payload
        return payload
    finally:
        if process.is_alive():
            _kill(process)
        process.join()


def _await_result(
    process: "multiprocessing.Process", result_queue: "multiprocessing.Queue", timeout_seconds: float,
) -> tuple[bool, str | None, Any]:
    """Waits up to `timeout_seconds` for a result, polling both the queue and the process so
    neither a huge result nor a dead-with-nothing-to-say worker has to wait out the full
    timeout.

    Deliberately does NOT simply `process.join(timeout_seconds)` and then read the queue
    afterward: a worker's result larger than the OS pipe buffer (a `Queue.put()` is handed off
    to a background feeder thread, not written synchronously) makes that feeder thread block
    until someone reads — and `join()` blocks the only someone who could, until timeout_seconds
    silently ticks by as an actual deadlock, not real work. Reading the queue *while* waiting
    for the process, in a short poll loop, keeps the pipe draining as data arrives instead.
    """
    deadline = monotonic() + max(0.0, timeout_seconds)
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        try:
            status, payload = result_queue.get(timeout=min(remaining, _POLL_INTERVAL_SECONDS))
            return True, status, payload
        except _queue_module.Empty:
            if not process.is_alive():
                break  # give one last non-blocking read a chance below, then call it a crash
    try:
        status, payload = result_queue.get_nowait()
        return True, status, payload
    except _queue_module.Empty:
        return False, None, None


def _bootstrap(target: Callable[..., Any], args: tuple, result_queue: "multiprocessing.Queue") -> None:
    """The actual `Process` target. Calls `target(*args)` and puts exactly one
    `("ok", value)` or `("error", exc)` tuple on `result_queue`.

    On the error path, the worker's own exception object is put on the queue
    as-is, which preserves its exact type (and any `__reduce__`-restorable
    state) so a caller can `isinstance()`-check the `__cause__` `run_isolated`
    chains it as. Not every exception pickles cleanly (a custom `__init__`
    signature that does not match `self.args`, attached unpicklable state,
    ...), so `_picklable_or_fallback` checks that up front — `Queue.put()`
    itself would not catch it: its actual pickling happens later, in a
    background feeder thread, so a `try/except` around `put()` never sees a
    pickling failure and the item would just be silently dropped instead of
    reported. The fallback, a plain `RuntimeError` carrying the original type
    name and message, is always picklable, so the child never simply hangs
    or vanishes instead of reporting something.
    """
    try:
        result = target(*args)
    except Exception as exc:
        result_queue.put(("error", _picklable_or_fallback(exc)))
        return
    result_queue.put(("ok", result))


def _picklable_or_fallback(exc: Exception) -> Exception:
    try:
        pickle.dumps(exc)
    except Exception:
        return RuntimeError(f"{type(exc).__name__}: {exc}")
    return exc


def _kill(process: "multiprocessing.Process") -> None:
    process.terminate()
    process.join(_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(_KILL_REAP_SECONDS)


def _name(target: Callable[..., Any]) -> str:
    return getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))
