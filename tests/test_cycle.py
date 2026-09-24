# SPDX-License-Identifier: Apache-2.0
"""Phase 2: the real ``READY -> ... -> READY`` cycle over the fake engine.

This is the no-GPU end-to-end test. It exercises ``LifecycleRunner`` against
``FakeVLLMAdapter``, which implements the same ``LifecycleAdapter`` port that
``HttpVLLMAdapter`` does -- so the controller, the ordering and the fail-closed
behaviour are all verified here, and the HTTP adapter only needs its own
transport-level tests.

The fake engine reproduces the vLLM behaviours the design depends on:

* ``weight_version`` starts as ``"default"`` and is never auto-incremented;
* ``pause(mode="wait")`` does not complete while requests are active;
* ``finish_weight_update`` writes the version but invalidates **no** cache;
* ``reset_prefix_cache`` can report failure while blocks are held.
"""

from __future__ import annotations

import unittest

from support import IDENTITY_V0

from rolloutcore import (
    AlreadyManagedEngineError,
    DrainFailedError,
    EngineTaintedError,
    EvidenceNotReady,
    LifecycleController,
    LifecycleRunner,
    LifecycleState,
    NotServingError,
    RolloutCoreError,
    VersionMismatchError,
    WeightTransferNotConfiguredError,
    WeightVersion,
)
from rolloutcore.adapters import (
    FakeEngineError,
    FakeVLLMAdapter,
    FakeVLLMEngine,
    manifest_identity,
    seeded_engine,
)


def make_runner(
    engine: FakeVLLMEngine | None = None,
    adapter: FakeVLLMAdapter | None = None,
    sleep=None,
    **kwargs,
):
    ctrl = LifecycleController()
    ad = adapter or FakeVLLMAdapter(engine if engine is not None else seeded_engine())
    runner = LifecycleRunner(ctrl, ad, sleep=sleep or (lambda _s: None), **kwargs)
    return runner, ctrl, ad


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_seeds_rc0_and_lands_in_ready(self):
        engine = seeded_engine(IDENTITY_V0)
        self.assertEqual(engine.weight_version, "default")
        runner, ctrl, _ = make_runner(engine)

        runner.bootstrap()

        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.current_identity, IDENTITY_V0)
        # The adapter seeded over an unmanaged label.
        self.assertEqual(engine.weight_version, "rc-0")
        self.assertTrue(engine.is_managed())

    def test_bootstrap_records_the_world_size(self):
        engine = seeded_engine(IDENTITY_V0, world_size=4)
        _runner, ctrl, adapter = make_runner(engine)
        evidence = adapter.bootstrap()
        self.assertEqual(evidence.world_size, 4)
        ctrl.initialize(evidence)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_bootstrap_refuses_to_adopt_a_managed_engine(self):
        """Amendment 4 + review item 1, end to end.

        The refusal must leave the *engine* untouched, not just our controller
        uninitialized: a late check would have written ``rc-0`` first and stolen
        the other controller's ownership on the way out.
        """
        engine = seeded_engine(IDENTITY_V0)
        engine.weight_version = "rc-7"  # a previous controller already owns it
        runner, ctrl, _ = make_runner(engine)
        engine.calls.clear()

        with self.assertRaises(AlreadyManagedEngineError) as ctx:
            runner.bootstrap()

        self.assertIn("lease/recovery", str(ctx.exception))
        # Review item 1: no mutation, so no taint. The engine is healthy and
        # simply not ours; a taint here would demand restarting a working engine.
        self.assertIs(ctrl.state, LifecycleState.UNINITIALIZED)
        self.assertEqual(engine.weight_version, "rc-7", "the label must not move")
        self.assertNotIn("update_weight_version", engine.calls)
        self.assertNotIn("init_weight_transfer_engine", engine.calls)
        self.assertEqual(engine.calls, ["get_weight_info"])

    def test_bootstrap_after_a_refusal_can_still_succeed(self):
        """A refusal is retryable: nothing was written and nothing was tainted."""
        engine = seeded_engine(IDENTITY_V0)
        engine.weight_version = "rc-7"
        runner, ctrl, _ = make_runner(engine)
        with self.assertRaises(AlreadyManagedEngineError):
            runner.bootstrap()

        engine.weight_version = "default"  # the other controller released it
        runner.bootstrap()
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(engine.weight_version, "rc-0")


class TestFullCycle(unittest.TestCase):
    def test_one_full_cycle_reaches_ready_at_the_next_generation(self):
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        result = runner.install_next(manifest_identity("v1"))

        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(1))
        self.assertEqual(ctrl.current_identity, manifest_identity("v1"))
        self.assertEqual(engine.weight_version, "rc-1")
        self.assertFalse(engine.paused)
        self.assertEqual(
            result.states_visited,
            (
                LifecycleState.READY,
                LifecycleState.DRAINING,
                LifecycleState.QUIESCED,
                LifecycleState.UPDATING,
                LifecycleState.INVALIDATING,
                LifecycleState.VALIDATING,
                LifecycleState.RESUMING,
                LifecycleState.READY,
            ),
        )

    def test_three_consecutive_cycles(self):
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        for n in (1, 2, 3):
            runner.install_next(manifest_identity(f"v{n}"))
            self.assertEqual(ctrl.current_version, WeightVersion(n))
            self.assertEqual(ctrl.current_identity, manifest_identity(f"v{n}"))
            self.assertEqual(engine.weight_version, f"rc-{n}")

    def test_engine_is_paused_from_draining_through_validating(self):
        """Amendment 1 ordering, observed on the engine rather than the state."""
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, adapter = make_runner(engine)
        runner.bootstrap()
        self.assertFalse(engine.paused)

        target = ctrl.next_target(manifest_identity("v1"))

        ctrl.begin_drain()
        adapter.begin_drain()
        self.assertTrue(engine.paused, "paused after drain")

        ctrl.confirm_drained(adapter.await_drain())
        ctrl.begin_update(target)
        adapter.start_weight_update(target)
        ctrl.confirm_updated(adapter.complete_weight_update(target))
        self.assertTrue(engine.paused, "still paused during invalidation")

        ctrl.confirm_invalidated(adapter.invalidate_caches(target))
        self.assertTrue(engine.paused, "still paused during validation")

        # Validation runs while the engine is still paused -- this is the point.
        evidence = adapter.validate_pre_resume(target)
        self.assertTrue(evidence.is_paused)
        ctrl.confirm_validated(evidence)
        self.assertIs(ctrl.state, LifecycleState.RESUMING)
        self.assertTrue(engine.paused, "still paused on entry to RESUMING")

        # Only now.
        ctrl.confirm_resumed(adapter.resume(target))
        self.assertFalse(engine.paused)
        self.assertIs(ctrl.state, LifecycleState.READY)

    def test_caches_are_clean_before_the_engine_is_resumed(self):
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, adapter = make_runner(engine)
        runner.bootstrap()

        target = ctrl.next_target(manifest_identity("v1"))
        ctrl.begin_drain()
        adapter.begin_drain()
        ctrl.confirm_drained(adapter.await_drain())
        ctrl.begin_update(target)
        adapter.start_weight_update(target)
        ctrl.confirm_updated(adapter.complete_weight_update(target))

        # pause(mode="wait", clear_cache=True) already cleaned all three as a
        # side effect (vllm/v1/engine/core.py:877-882 -> :861-875). Re-dirty them
        # to model a cache that pause does not know about, then show that
        # finish_weight_update leaves it dirty -- exactly like vLLM, which only
        # calls reset_lora_state() (vllm/v1/worker/gpu_worker.py:1504-1505).
        engine.encoder_cache_dirty = True
        engine.mm_cache_dirty = True
        engine.prefix_cache_dirty = True
        self.assertTrue(engine.encoder_cache_dirty)

        ctrl.confirm_invalidated(adapter.invalidate_caches(target))
        self.assertFalse(engine.encoder_cache_dirty)
        self.assertFalse(engine.mm_cache_dirty)
        self.assertFalse(engine.prefix_cache_dirty)

        ctrl.confirm_validated(adapter.validate_pre_resume(target))
        ctrl.confirm_resumed(adapter.resume(target))
        self.assertIs(ctrl.state, LifecycleState.READY)

    def test_rollouts_across_a_cycle_get_distinct_bindings(self):
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        before = ctrl.admit_rollout("R1")
        ctrl.finish_rollout("R1")
        runner.install_next(manifest_identity("v1"))
        after = ctrl.admit_rollout("R2")

        self.assertEqual(before.version, WeightVersion(0))
        self.assertEqual(after.version, WeightVersion(1))
        self.assertNotEqual(before.weight_identity, after.weight_identity)
        self.assertNotEqual(before.cache_salt, after.cache_salt)


class TestDrainWhileRequestsActive(unittest.TestCase):
    def test_runner_polls_until_the_engine_drains(self):
        """The engine refuses to finish pausing while work is active."""
        engine = seeded_engine(IDENTITY_V0)
        engine.active_requests = 3
        sleep_calls: list[float] = []

        runner, ctrl, adapter = make_runner(
            engine, drain_polls=5, sleep=lambda s: sleep_calls.append(s)
        )
        runner.bootstrap()

        target = ctrl.next_target(manifest_identity("v1"))
        ctrl.begin_drain()
        adapter.begin_drain()
        ctrl.begin_update  # noqa: B018 - documented below
        # Draining cannot complete yet.
        with self.assertRaises(EvidenceNotReady):
            ctrl.confirm_drained(adapter.await_drain())
        self.assertIs(ctrl.state, LifecycleState.DRAINING)

        # Work finishes; now the drain completes.
        engine.active_requests = 0
        ctrl.confirm_drained(adapter.await_drain())
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)
        # And the cycle still completes from here.
        ctrl.begin_update(target)
        adapter.start_weight_update(target)
        ctrl.confirm_updated(adapter.complete_weight_update(target))
        ctrl.confirm_invalidated(adapter.invalidate_caches(target))
        ctrl.confirm_validated(adapter.validate_pre_resume(target))
        ctrl.confirm_resumed(adapter.resume(target))
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(1))

    def test_runner_gives_up_after_the_poll_budget(self):
        engine = seeded_engine(IDENTITY_V0)
        engine.active_requests = 99  # never drains
        runner, ctrl, _ = make_runner(engine, drain_polls=3, sleep=lambda _s: None)
        runner.bootstrap()

        with self.assertRaises(RolloutCoreError) as ctx:
            runner.install_next(manifest_identity("v1"))
        self.assertIn("did not complete", str(ctx.exception))
        # Fail-closed: the engine stays paused and nothing was published.
        self.assertIs(ctrl.state, LifecycleState.DRAINING)
        self.assertTrue(engine.paused)
        self.assertEqual(ctrl.current_version, WeightVersion(0))


class TestFailurePaths(unittest.TestCase):
    def test_failed_finish_weight_update_leaves_the_engine_paused_and_unpublished(self):
        engine = seeded_engine(IDENTITY_V0)
        engine.faults["finish_weight_update"] = "worker died"
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        with self.assertRaises(EngineTaintedError):
            runner.install_next(manifest_identity("v1"))

        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(engine.weight_version, "rc-0", "version must not advance")
        self.assertTrue(engine.paused, "engine must not be resumed")

    def test_failed_prefix_cache_reset_taints_and_does_not_resume(self):
        engine = seeded_engine(IDENTITY_V0)
        engine.prefix_reset_succeeds = False
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        with self.assertRaises(EngineTaintedError) as ctx:
            runner.install_next(manifest_identity("v1"))
        self.assertIn("prefix cache", str(ctx.exception))

        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertTrue(engine.paused, "engine must not be resumed after a failed reset")
        # The engine's own label HAS advanced: vLLM writes it inside
        # finish_weight_update, before any cache is touched
        # (vllm/v1/engine/async_llm.py:1284-1288). This is exactly why the label
        # is not a commit marker and RolloutCore keeps its own committed target.
        self.assertEqual(engine.weight_version, "rc-1")
        self.assertEqual(ctrl.current_version, WeightVersion(0), "not committed by us")

    def test_failed_start_weight_update_taints(self):
        """Review item 3: an ambiguous mutating failure is not left as UPDATING.

        ``start_weight_update`` is exactly the ambiguous case: it may have opened
        a session on some ranks and not others, so "the call raised" does not
        mean "nothing happened".
        """
        engine = seeded_engine(IDENTITY_V0)
        engine.faults["start_weight_update"] = "session already open"
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        with self.assertRaises(EngineTaintedError) as ctx:
            runner.install_next(manifest_identity("v1"))
        self.assertIn("start_weight_update", str(ctx.exception))
        self.assertIn("unknown engine outcome", str(ctx.exception))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertTrue(engine.paused)
        self.assertEqual(ctrl.current_version, WeightVersion(0), "nothing was published")

    def test_failed_cache_reset_taints(self):
        engine = seeded_engine(IDENTITY_V0)
        engine.faults["reset_prefix_cache"] = "block pool died"
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()
        with self.assertRaises(EngineTaintedError):
            runner.install_next(manifest_identity("v1"))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_failed_resume_taints(self):
        """A failed resume may have left the engine serving: never guess."""
        engine = seeded_engine(IDENTITY_V0)
        engine.faults["resume"] = "scheduler refused"
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()
        with self.assertRaises(EngineTaintedError):
            runner.install_next(manifest_identity("v1"))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_failed_observation_does_not_taint(self):
        """Review item 3's nuance: a failed *read* is retryable, not terminal.

        Validation runs while the engine is paused, so a read failure leaves it
        unable to serve anything new. Tainting would demand a restart for what
        may be one dropped connection.
        """
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()

        engine.faults["get_weight_info"] = "connection reset"

        with self.assertRaises(FakeEngineError):
            runner.install_next(manifest_identity("v1"))

        self.assertIs(ctrl.state, LifecycleState.VALIDATING, "still fail-closed: paused")
        self.assertFalse(ctrl.is_tainted)
        self.assertTrue(engine.paused)
        self.assertEqual(ctrl.current_version, WeightVersion(0), "nothing was published")

        # And the cycle completes once the read recovers, without a restart.
        del engine.faults["get_weight_info"]
        pending = ctrl.pending_target
        assert pending is not None
        adapter = runner._adapter
        ctrl.confirm_validated(adapter.validate_pre_resume(pending))
        ctrl.confirm_resumed(adapter.resume(pending))
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(1))

    def test_missing_transfer_driver_does_not_taint(self):
        """A config error sends nothing, so there is no ambiguity to taint."""
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, adapter = make_runner(engine)
        runner.bootstrap()

        def refuse(_target):
            raise WeightTransferNotConfiguredError("test", "no driver")

        adapter.start_weight_update = refuse  # type: ignore[method-assign]
        with self.assertRaises(WeightTransferNotConfiguredError):
            runner.install_next(manifest_identity("v1"))
        self.assertIs(ctrl.state, LifecycleState.UPDATING)
        self.assertTrue(engine.paused)

    def test_version_mismatch_after_publish_taints(self):
        """Simulates someone else writing the version behind our back."""
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, adapter = make_runner(engine)
        runner.bootstrap()

        target = ctrl.next_target(manifest_identity("v1"))
        ctrl.begin_drain()
        adapter.begin_drain()
        ctrl.confirm_drained(adapter.await_drain())
        ctrl.begin_update(target)
        adapter.start_weight_update(target)
        ctrl.confirm_updated(adapter.complete_weight_update(target))
        ctrl.confirm_invalidated(adapter.invalidate_caches(target))

        engine.weight_version = "rc-77"  # external interference
        with self.assertRaises(VersionMismatchError):
            ctrl.confirm_validated(adapter.validate_pre_resume(target))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertTrue(engine.paused)

    def test_taint_preserves_in_flight_rollout_bindings(self):
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()
        ctrl.admit_rollout("R1")

        engine.faults["finish_weight_update"] = "worker died"
        with self.assertRaises(EngineTaintedError):
            runner.install_next(manifest_identity("v1"))

        # Amendment 5: the in-flight binding survives for forensics.
        self.assertEqual(ctrl.active_rollout_count, 0)
        self.assertEqual(len(ctrl.orphaned_rollouts), 1)
        self.assertEqual(ctrl.orphaned_rollouts[0].request_id, "R1")
        self.assertEqual(ctrl.orphaned_rollouts[0].weight_identity, IDENTITY_V0)

    def test_cycle_after_a_successful_one_still_works(self):
        """Sanity: failure handling has not made the happy path brittle."""
        engine = seeded_engine(IDENTITY_V0)
        runner, ctrl, _ = make_runner(engine)
        runner.bootstrap()
        for n in range(1, 5):
            result = runner.install_next(manifest_identity(f"v{n}"))
            self.assertEqual(result.committed.version, WeightVersion(n))
        self.assertIs(ctrl.state, LifecycleState.READY)


class TestDeadEngineDuringDrain(unittest.TestCase):
    """Phase 4B, on the fake engine: a drain failure is not a taint.

    The runner calls `await_drain` through `_observe`, which deliberately does not
    taint a failed read. Measured against a real SIGKILLed engine, the consequence
    is that the controller is left in DRAINING, untainted, with no legal exit --
    `CONFIRM_DRAINED` is the only transition out of DRAINING and a dead engine can
    never supply it. Only an operator `taint()` ends it.

    That is safe (DRAINING admits no rollouts and nothing was published) but it is
    not self-healing, so the behaviour is pinned here rather than left implicit.
    """

    def test_a_dead_engine_leaves_the_controller_in_draining_untainted(self):
        engine = seeded_engine(IDENTITY_V0)

        class DeadEngineAdapter(FakeVLLMAdapter):
            def await_drain(self):
                raise DrainFailedError(3, "transport error: [Errno 111] Connection refused")

        runner, ctrl, adapter = make_runner(
            engine, adapter=DeadEngineAdapter(engine), drain_polls=3
        )
        runner.bootstrap()
        target = ctrl.next_target(manifest_identity("v1"))

        with self.assertRaises(DrainFailedError):
            runner.run_cycle(target)

        # Not tainted: a failed read left nothing ambiguous, and nothing was
        # written. Fail-closed via DRAINING, not via TAINTED.
        self.assertFalse(ctrl.is_tainted)
        self.assertIs(ctrl.state, LifecycleState.DRAINING)
        # No version was published and no rollout can be admitted.
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        with self.assertRaises(NotServingError):
            ctrl.admit_rollout("R1")
        # And DRAINING really has one exit, so the operator has to decide.
        self.assertEqual(adapter.engine.weight_version, "rc-0")

    def test_an_operator_taint_is_the_way_out(self):
        engine = seeded_engine(IDENTITY_V0)

        class DeadEngineAdapter(FakeVLLMAdapter):
            def await_drain(self):
                raise DrainFailedError(3, "transport error: [Errno 111] Connection refused")

        runner, ctrl, _ = make_runner(engine, adapter=DeadEngineAdapter(engine), drain_polls=3)
        runner.bootstrap()
        with self.assertRaises(DrainFailedError):
            runner.run_cycle(ctrl.next_target(manifest_identity("v1")))

        ctrl.taint("engine unreachable: no drain evidence is coming")
        self.assertTrue(ctrl.is_tainted)
        self.assertIn("engine unreachable", str(ctrl.taint_reason))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)


class TestRunnerGuards(unittest.TestCase):
    def test_install_next_requires_bootstrap(self):
        runner, ctrl, _ = make_runner()
        with self.assertRaises(Exception) as ctx:
            runner.install_next(manifest_identity("v1"))
        self.assertIn("I1-ATOMICITY", str(ctx.exception))
        self.assertIs(ctrl.state, LifecycleState.UNINITIALIZED)

    def test_result_reports_drain_polls_and_duration(self):
        engine = seeded_engine(IDENTITY_V0)
        runner, _ctrl, _ = make_runner(engine)
        runner.bootstrap()
        result = runner.install_next(manifest_identity("v1"))
        self.assertGreaterEqual(result.drain_polls, 1)
        self.assertGreaterEqual(result.duration_s, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
