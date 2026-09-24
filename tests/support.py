# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Phase 1 tests.

The important helper is :func:`drive_to`, which produces a controller sitting in
any one of the eight states using only *valid* evidence. Every test that needs
"a controller in state X" goes through it, so the states are constructed exactly
one way and the illegal-transition matrix cannot accidentally test an
invalidly-constructed controller.
"""

from __future__ import annotations

from rolloutcore import (
    BootstrapEvidence,
    DrainEvidence,
    Event,
    InvalidateEvidence,
    LifecycleController,
    LifecycleState,
    UpdateEvidence,
    ValidateEvidence,
    WeightVersion,
)

#: A bootstrap that a correct Phase 2 adapter would produce against a fresh
#: ``vllm serve``: the engine reports the unmanaged literal ``"default"``, the
#: adapter seeds ``rc-0``, and the post-seed read confirms it.
GOOD_BOOTSTRAP = BootstrapEvidence(
    observed_engine_label="rc-0",
    weight_transfer_initialised=True,
    seeded=True,
    pre_seed_label="default",
    world_size=2,
    backend="nccl",
)

GOOD_DRAIN = DrainEvidence(active_rollouts=0, engine_pause_confirmed=True)
GOOD_INVALIDATE = InvalidateEvidence(
    prefix_cache_reset=True, encoder_cache_reset=True, mm_cache_reset=True
)


def good_update(target: WeightVersion, **overrides) -> UpdateEvidence:
    kwargs = dict(
        target_version=target,
        weights_loaded=True,
        finish_acknowledged=True,
        chunks_transferred=12,
        data_plane_complete=True,
    )
    kwargs.update(overrides)
    return UpdateEvidence(**kwargs)


def good_validate(target: WeightVersion, **overrides) -> ValidateEvidence:
    kwargs = dict(
        target_version=target,
        observed_engine_label=target.label,
        resume_acknowledged=True,
        is_paused=False,
    )
    kwargs.update(overrides)
    return ValidateEvidence(**kwargs)


def bootstrapped(**overrides) -> LifecycleController:
    """A controller in READY at version 0."""
    ctrl = LifecycleController()
    ctrl.initialize(overrides.pop("evidence", GOOD_BOOTSTRAP))
    return ctrl


def drive_to(state: LifecycleState) -> LifecycleController:
    """Return a controller in ``state`` at the committed version noted below.

    UNINITIALIZED -- no version
    READY         -- v0 committed
    DRAINING      -- v0 committed, 0 active rollouts
    QUIESCED      -- v0 committed
    UPDATING      -- v0 committed, v1 pending
    INVALIDATING  -- v0 committed, v1 pending
    VALIDATING    -- v0 committed, v1 pending
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

    ctrl.confirm_drained(GOOD_DRAIN)
    if state is LifecycleState.QUIESCED:
        return ctrl

    target = WeightVersion(1)
    ctrl.begin_update(target)
    if state is LifecycleState.UPDATING:
        return ctrl

    ctrl.confirm_updated(good_update(target))
    if state is LifecycleState.INVALIDATING:
        return ctrl

    ctrl.confirm_invalidated(GOOD_INVALIDATE)
    if state is LifecycleState.VALIDATING:
        return ctrl

    raise AssertionError(f"drive_to does not support {state}")  # pragma: no cover


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
        ctrl.confirm_drained(GOOD_DRAIN)
    elif event is Event.BEGIN_UPDATE:
        current = ctrl.current_version or WeightVersion(0)
        ctrl.begin_update(current.next())
    elif event is Event.CONFIRM_UPDATED:
        target = ctrl.pending_version or WeightVersion(1)
        ctrl.confirm_updated(good_update(target))
    elif event is Event.CONFIRM_INVALIDATED:
        ctrl.confirm_invalidated(GOOD_INVALIDATE)
    elif event is Event.CONFIRM_VALIDATED:
        target = ctrl.pending_version or WeightVersion(1)
        ctrl.confirm_validated(good_validate(target))
    elif event is Event.TAINT:
        ctrl.taint("matrix probe")
    else:  # pragma: no cover
        raise AssertionError(f"unhandled event {event}")


def advance_to_version(ctrl: LifecycleController, target: WeightVersion) -> None:
    """Run whole cycles until ``target`` is committed. Controller ends in READY."""
    while ctrl.current_version is None or ctrl.current_version < target:
        assert ctrl.current_version is not None
        nxt = ctrl.current_version.next()
        ctrl.begin_drain()
        ctrl.confirm_drained(GOOD_DRAIN)
        ctrl.begin_update(nxt)
        ctrl.confirm_updated(good_update(nxt))
        ctrl.confirm_invalidated(GOOD_INVALIDATE)
        ctrl.confirm_validated(good_validate(nxt))


def assert_state_unchanged(test, ctrl: LifecycleController, fn) -> None:
    """Assert ``fn`` raises, and that the controller is untouched by it.

    Checks the *observable* identity of the controller -- state, versions,
    active rollouts and journal length -- rather than just the state, so a
    mutation performed before the legality check is caught too.
    """
    before = (
        ctrl.state,
        ctrl.current_version,
        ctrl.pending_version,
        ctrl.active_rollouts,
        len(ctrl.journal),
    )
    with test.assertRaises(Exception) as ctx:
        fn()
    after = (
        ctrl.state,
        ctrl.current_version,
        ctrl.pending_version,
        ctrl.active_rollouts,
        len(ctrl.journal),
    )
    test.assertEqual(before, after, "controller mutated by a rejected event")
    return ctx.exception
