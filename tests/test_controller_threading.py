# SPDX-License-Identifier: Apache-2.0
"""The controller has two callers, and they overlap on purpose.

The lifecycle runner drives transitions; whoever owns the rollouts admits and
releases them. They touch the controller from different threads exactly where it
matters: a rollout must be released *as its request completes*, because that is
what makes the engine's "drain completed" and RolloutCore's "no active rollouts"
agree (`confirm_drained`). These tests pin that behaviour down.
"""

from __future__ import annotations

import sys
import threading

from support import bootstrapped, function_tests, good_drain

from rolloutcore import LifecycleState

#: Without the controller's lock, `self._seq += 1` is a load-add-store that the
#: GIL may switch in the middle of. A tiny switch interval makes that likely
#: rather than theoretical, so this file actually fails if the lock is removed.
TIGHT_SWITCH_INTERVAL = 1e-6


def test_a_rollout_released_while_draining_lets_the_confirm_through() -> None:
    """The demo's ordering: release first, confirm second, no taint.

    `confirm_drained` taints when the engine claims quiescence while RolloutCore
    still counts live work -- but a rollout released *during* DRAINING is the
    normal case, not a disagreement.
    """
    ctrl = bootstrapped()
    ctrl.admit_rollout("R1")
    ctrl.begin_drain()
    assert ctrl.state is LifecycleState.DRAINING
    assert ctrl.active_rollout_count == 1

    releaser = threading.Thread(target=ctrl.finish_rollout, args=("R1",))
    releaser.start()
    releaser.join(timeout=5)
    assert not releaser.is_alive()

    ctrl.confirm_drained(good_drain())
    assert ctrl.state is LifecycleState.QUIESCED
    assert not ctrl.is_tainted


def test_the_same_rollout_cannot_be_released_twice() -> None:
    """Two actors racing to release the same rollout is a bookkeeping bug.

    One of them must lose loudly (`UnknownRolloutError`), and the controller must
    not have counted the rollout out twice.
    """
    ctrl = bootstrapped()
    ctrl.admit_rollout("R1")
    ctrl.begin_drain()

    outcomes: list[object] = []
    barrier = threading.Barrier(2)

    def release() -> None:
        barrier.wait(timeout=5)
        try:
            outcomes.append(ctrl.finish_rollout("R1"))
        except Exception as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=release) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(outcomes) == 2
    assert sum(isinstance(o, Exception) for o in outcomes) == 1
    assert ctrl.active_rollout_count == 0


def test_concurrent_admit_and_release_never_duplicates_journal_seq() -> None:
    """Contention must not corrupt the journal or lose a rollout.

    Every transition takes `self._seq += 1` and then appends a record. If those
    two steps are not serialized, two threads can read the same counter and write
    duplicate sequence numbers -- a corrupted audit trail that no later check
    would notice.
    """
    ctrl = bootstrapped()
    workers, per_worker = 4, 250
    # `bootstrapped()` already journaled the initialization transition.
    before = len(ctrl.journal)
    failures: list[BaseException] = []
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(TIGHT_SWITCH_INTERVAL)
    try:

        def churn(worker: int) -> None:
            try:
                for i in range(per_worker):
                    rid = f"R{worker}-{i}"
                    ctrl.admit_rollout(rid)
                    ctrl.finish_rollout(rid)
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=churn, args=(w,)) for w in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads), "a worker deadlocked"
    finally:
        sys.setswitchinterval(old_interval)

    assert not failures, failures
    assert ctrl.active_rollout_count == 0
    seqs = [record.seq for record in ctrl.journal]
    assert seqs == sorted(seqs), "journal is out of order"
    assert len(seqs) == len(set(seqs)), "duplicate seq: the counter raced"
    assert len(seqs) - before == workers * per_worker * 2, "lost a transition record"


def test_a_reader_never_sees_a_half_updated_rollout_set() -> None:
    """`active_rollouts` snapshots; it must not raise mid-iteration.

    Iterating `dict.values()` while another thread pops raises
    `RuntimeError: dictionary changed size during iteration`. The lock is what
    makes the snapshot safe, so this fails loudly without it.
    """
    ctrl = bootstrapped()
    stop = threading.Event()
    seen: list[int] = []
    failures: list[BaseException] = []

    def churn() -> None:
        try:
            n = 0
            while not stop.is_set():
                rid = f"R{n}"
                ctrl.admit_rollout(rid)
                ctrl.finish_rollout(rid)
                n += 1
        except BaseException as exc:
            failures.append(exc)

    churner = threading.Thread(target=churn)
    churner.start()
    try:
        for _ in range(2000):
            seen.append(ctrl.active_rollout_count)
    finally:
        stop.set()
        churner.join(timeout=10)

    assert not failures, failures
    assert not churner.is_alive()
    assert all(isinstance(count, int) for count in seen)


# `unittest` collects only TestCase subclasses, so without this the four functions
# above are invisible to the dependency-free runner that CI runs first. pytest
# ignores this hook and collects the functions directly.
load_tests = function_tests(globals())
