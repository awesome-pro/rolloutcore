# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the tests.

The important helper is :func:`drive_to`, which produces a controller sitting in
any one of the nine states using only *valid* evidence. Every test that needs "a
controller in state X" goes through it, so the states are constructed exactly one
way and the illegal-transition matrix cannot accidentally test an
invalidly-constructed controller.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

from rolloutcore import (
    BootstrapEvidence,
    DrainEvidence,
    Event,
    InvalidateEvidence,
    LifecycleController,
    LifecycleState,
    ResumeEvidence,
    UpdateEvidence,
    UpdateTarget,
    ValidateEvidence,
    WeightIdentity,
    WeightVersion,
)
from rolloutcore.adapters import manifest_identity

#: Two distinguishable weight sets, modelling the plan's Model A / Model B.
IDENTITY_V0: WeightIdentity = manifest_identity("v0")
IDENTITY_V1: WeightIdentity = manifest_identity("v1")
IDENTITY_V2: WeightIdentity = manifest_identity("v2")

#: A bootstrap a correct adapter produces against a fresh ``vllm serve``: the
#: engine reports the unmanaged literal ``"default"``, the driver initializes the
#: transfer engine, the adapter seeds ``rc-0``, and the post-seed read confirms it.
GOOD_BOOTSTRAP = BootstrapEvidence(
    observed_engine_label="rc-0",
    weight_transfer_initialised=True,
    pre_seed_label="default",
    weight_identity=IDENTITY_V0,
    backend="nccl",
    world_size=2,
    weight_transfer_driver="test-driver",
)

#: The control-plane-only variant: no transfer engine, no driver.
GOOD_BOOTSTRAP_NO_DRIVER = BootstrapEvidence(
    observed_engine_label="rc-0",
    weight_transfer_initialised=False,
    pre_seed_label="default",
    weight_identity=IDENTITY_V0,
    backend="none",
    world_size=None,
    weight_transfer_driver=None,
)


def target(version: int, identity: WeightIdentity | None = None) -> UpdateTarget:
    """An update target for a generation, with a distinct identity by default."""
    return UpdateTarget(
        version=WeightVersion(version),
        identity=identity or manifest_identity(f"v{version}"),
    )


#: The identity ``drive_to`` installs when it advances to generation 1.
TARGET_V1 = target(1)


def good_drain(completed: bool = True, engine_active: int | None = None) -> DrainEvidence:
    return DrainEvidence(
        engine_drain_completed=completed,
        engine_active_requests=engine_active,
    )


def good_update(t: UpdateTarget, **overrides: object) -> UpdateEvidence:
    kwargs: dict = dict(
        target=t,
        weights_loaded=True,
        finish_acknowledged=True,
        chunks_transferred=12,
        data_plane_complete=True,
        observed_identity=t.identity,
    )
    kwargs.update(overrides)
    return UpdateEvidence(**kwargs)


def good_invalidate(**overrides: object) -> InvalidateEvidence:
    kwargs: dict = dict(prefix_cache_reset=True, encoder_cache_reset=True, mm_cache_reset=True)
    kwargs.update(overrides)
    return InvalidateEvidence(**kwargs)


def good_validate(t: UpdateTarget, **overrides: object) -> ValidateEvidence:
    """Pre-resume evidence. ``is_paused`` is **True** -- we have not resumed yet."""
    kwargs: dict = dict(
        target=t,
        observed_engine_label=t.label,
        is_paused=True,
        caches_still_clean=True,
    )
    kwargs.update(overrides)
    return ValidateEvidence(**kwargs)


def good_resume(t: UpdateTarget, **overrides: object) -> ResumeEvidence:
    kwargs: dict = dict(
        target=t,
        resume_acknowledged=True,
        is_paused=False,
        observed_engine_label=t.label,
    )
    kwargs.update(overrides)
    return ResumeEvidence(**kwargs)


def bootstrapped(**overrides: object) -> LifecycleController:
    """A controller in READY at generation 0."""
    ctrl = LifecycleController()
    ctrl.initialize(overrides.pop("evidence", GOOD_BOOTSTRAP))
    return ctrl


def drive_to(state: LifecycleState) -> LifecycleController:
    """Return a controller in ``state``.

    UNINITIALIZED -- no target
    READY         -- v0 committed
    DRAINING      -- v0 committed, 0 active rollouts
    QUIESCED      -- v0 committed
    UPDATING      -- v0 committed, v1 pending
    INVALIDATING  -- v0 committed, v1 pending
    VALIDATING    -- v0 committed, v1 pending, engine still paused
    RESUMING      -- v0 committed, v1 pending, resume issued
    TAINTED       -- v0 committed, tainted from READY
    """
    ctrl = LifecycleController()

    if state is LifecycleState.UNINITIALIZED:
        return ctrl

    ctrl.initialize(GOOD_BOOTSTRAP)
    if state is LifecycleState.READY:
        return ctrl
    if state is LifecycleState.TAINTED:
        ctrl.taint("operator abort")
        return ctrl

    ctrl.begin_drain()
    if state is LifecycleState.DRAINING:
        return ctrl

    ctrl.confirm_drained(good_drain())
    if state is LifecycleState.QUIESCED:
        return ctrl

    t = TARGET_V1
    ctrl.begin_update(t)
    if state is LifecycleState.UPDATING:
        return ctrl

    ctrl.confirm_updated(good_update(t))
    if state is LifecycleState.INVALIDATING:
        return ctrl

    ctrl.confirm_invalidated(good_invalidate())
    if state is LifecycleState.VALIDATING:
        return ctrl

    ctrl.confirm_validated(good_validate(t))
    if state is LifecycleState.RESUMING:
        return ctrl

    raise AssertionError(f"drive_to does not support {state}")  # pragma: no cover


def run_full_cycle(ctrl: LifecycleController, t: UpdateTarget) -> None:
    """Drive one complete READY -> ... -> READY cycle on ``ctrl``."""
    from rolloutcore import Event as _E  # noqa: F401

    ctrl.begin_drain()
    ctrl.confirm_drained(good_drain())
    ctrl.begin_update(t)
    ctrl.confirm_updated(good_update(t))
    ctrl.confirm_invalidated(good_invalidate())
    ctrl.confirm_validated(good_validate(t))
    ctrl.confirm_resumed(good_resume(t))


def advance_to_version(ctrl: LifecycleController, target_version: WeightVersion) -> None:
    """Run whole cycles until ``target_version`` is committed."""
    while ctrl.current_version is None or ctrl.current_version < target_version:
        assert ctrl.current_version is not None
        run_full_cycle(ctrl, target(ctrl.current_version.next().value))


def invoke(ctrl: LifecycleController, event: Event) -> None:
    """Invoke ``event`` with otherwise-valid arguments.

    Used by the illegal-transition matrix so that any exception raised is
    attributable to the *state*, not to malformed arguments.
    """
    if event is Event.INITIALIZE:
        ctrl.initialize(GOOD_BOOTSTRAP)
    elif event is Event.BEGIN_DRAIN:
        ctrl.begin_drain()
    elif event is Event.CONFIRM_DRAINED:
        ctrl.confirm_drained(good_drain())
    elif event is Event.BEGIN_UPDATE:
        current = ctrl.current_version or WeightVersion(0)
        ctrl.begin_update(target(current.next().value))
    elif event is Event.CONFIRM_UPDATED:
        ctrl.confirm_updated(good_update(ctrl.pending_target or TARGET_V1))
    elif event is Event.CONFIRM_INVALIDATED:
        ctrl.confirm_invalidated(good_invalidate())
    elif event is Event.CONFIRM_VALIDATED:
        ctrl.confirm_validated(good_validate(ctrl.pending_target or TARGET_V1))
    elif event is Event.CONFIRM_RESUMED:
        ctrl.confirm_resumed(good_resume(ctrl.pending_target or TARGET_V1))
    elif event is Event.TAINT:
        ctrl.taint("matrix probe")
    else:  # pragma: no cover
        raise AssertionError(f"unhandled event {event}")


def assert_state_unchanged(test, ctrl: LifecycleController, fn):
    """Assert ``fn`` raises, and that the controller is untouched by it.

    Checks the *observable* identity of the controller -- state, targets, active
    rollouts, orphan count and journal length -- rather than just the state, so a
    mutation performed before the legality check is caught too.
    """
    before = (
        ctrl.state,
        ctrl.current_target,
        ctrl.pending_target,
        ctrl.active_rollouts,
        len(ctrl.orphaned_rollouts),
        len(ctrl.journal),
    )
    with test.assertRaises(Exception) as ctx:
        fn()
    after = (
        ctrl.state,
        ctrl.current_target,
        ctrl.pending_target,
        ctrl.active_rollouts,
        len(ctrl.orphaned_rollouts),
        len(ctrl.journal),
    )
    test.assertEqual(before, after, "controller mutated by a rejected event")
    return ctx.exception


def function_tests(namespace: dict[str, Any]) -> Callable[..., unittest.TestSuite]:
    """A ``load_tests`` hook that adopts module-level ``test_*`` functions.

    ``unittest`` only collects ``TestCase`` subclasses; pytest collects plain
    module-level functions as well. Two modules here are written as functions
    (`test_controller_threading.py`, `test_script_hygiene.py`), and without this
    the dependency-free runner -- the step CI runs *before* installing pytest --
    silently executed 317 of 326 tests. Silently is the problem: the five hygiene
    tests were among the invisible ones.

    A module that defines plain test functions ends with::

        load_tests = function_tests(globals())

    which keeps the bodies as ordinary functions (no re-indentation, no change in
    what they assert) and hands ``unittest`` a suite containing both the
    functions and anything ``unittest`` found on its own, so a future
    ``TestCase`` in the same module is not dropped.
    """

    functions = {
        name: value
        for name, value in namespace.items()
        if name.startswith("test_") and callable(value)
    }

    class _Functions(unittest.TestCase):
        pass

    for name, function in functions.items():
        setattr(_Functions, name, staticmethod(function))

    def load_tests(
        loader: unittest.TestLoader,
        standard_tests: unittest.TestSuite,
        pattern: str | None,
    ) -> unittest.TestSuite:
        del pattern
        suite = unittest.TestSuite()
        suite.addTests(standard_tests)
        suite.addTests(loader.loadTestsFromTestCase(_Functions))
        return suite

    return load_tests
