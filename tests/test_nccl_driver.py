# SPDX-License-Identifier: Apache-2.0
"""The NCCL driver and its target-aware client wrapper, tested without vLLM.

vLLM is not installed on a development machine or in CI, so every external
collaborator is injected: the sync client, the trainer engine, and the manifest
source. That is also what lets these tests assert the two things the Phase 3B
review called out, neither of which needs a GPU to prove:

* ``finish_weight_update`` receives the **target's label** even though upstream
  passes ``None`` (``nccl_engine.py:361`` vs ``clients.py:89``);
* a manifest that does not match the target fails **before any request is sent**.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from support import GOOD_BOOTSTRAP, IDENTITY_V0, IDENTITY_V1, TARGET_V1, good_drain

from rolloutcore import (
    BootstrapEvidence,
    InvalidateEvidence,
    InvariantViolation,
    LifecycleController,
    LifecycleRunner,
    LifecycleState,
    ParamSpec,
    ResumeEvidence,
    UpdateEvidence,
    UpdateTarget,
    ValidateEvidence,
    WeightIdentity,
    WeightIdentityMismatchError,
    WeightProvenance,
    WeightTransferDriver,
    WeightTransferInit,
    WeightTransferNotConfiguredError,
    WeightVersion,
)
from rolloutcore.adapters.nccl import (
    NCCLWeightTransferDriver,
    RolloutCoreWeightSyncClient,
    device_mismatch,
    dtype_name,
    param_spec,
)


@dataclass
class FakeDType:
    """Stands in for ``torch.bfloat16``: ``str()`` is what we normalise."""

    name: str

    def __str__(self) -> str:
        return f"torch.{self.name}"


@dataclass
class FakeParamMeta:
    name: str
    dtype: Any
    shape: tuple[int, ...]


@dataclass
class FakeSource:
    """A ``WeightSource`` whose ``metadata()`` is whatever the test wants."""

    metas: list[FakeParamMeta]

    def metadata(self) -> list[FakeParamMeta]:
        return list(self.metas)


@dataclass
class FakeSyncClient:
    """Records the control-plane calls upstream's engine would make."""

    calls: list[tuple[str, Any]] = field(default_factory=list)

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self.calls.append(("init", init_info))

    def start_weight_update(self) -> None:
        self.calls.append(("start", None))

    def update_weights(self, update_info: Any) -> None:
        self.calls.append(("update", update_info))

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self.calls.append(("finish", weight_version))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@dataclass
class FakeTrainerEngine:
    """Stands in for upstream's trainer engine: ``send_weights`` makes the calls.

    ``client`` is assigned by the builder from whatever client it was handed --
    the *wrapper* in production, so this fake exercises the same path upstream's
    engine does rather than bypassing it with the raw client.
    """

    client: Any = None
    broadcasts: int = 0
    shut_down: bool = False
    fail: bool = False

    def send_weights(self) -> None:
        if self.fail:
            raise RuntimeError("collective timed out mid-broadcast")
        if self.client is None:
            raise AssertionError("builder did not hand the engine a client")
        self.broadcasts += 1
        self.client.start_weight_update()
        self.client.update_weights({"names": ["x"]})
        # Upstream's line, verbatim in shape: no version supplied.
        self.client.finish_weight_update()

    def shutdown(self) -> None:
        self.shut_down = True


def source_for(identity: WeightIdentity) -> FakeSource:
    """A source whose manifest hashes to exactly ``identity``.

    ``support.manifest_identity(seed)`` builds three OPT-shaped parameters with
    the seed in one name, so seeding here reproduces the digest exactly and the
    fixture cannot silently drift from the target it claims to describe.
    """
    seed = {IDENTITY_V0: "v0", IDENTITY_V1: "v1"}.get(identity)
    if seed is None:
        raise AssertionError(f"no fixture manifest for {identity.describe()}")
    metas = [
        FakeParamMeta("model.embed_tokens.weight", FakeDType("bfloat16"), (151936, 2048)),
        FakeParamMeta(
            "model.layers.0.self_attn.q_proj.weight", FakeDType("bfloat16"), (2048, 2048)
        ),
        FakeParamMeta(f"model.layers.0.mlp.{seed}.weight", FakeDType("bfloat16"), (2048, 11008)),
    ]
    computed = WeightIdentity.from_param_specs([param_spec(m) for m in metas])
    if computed != identity:
        raise AssertionError(f"fixture manifest does not produce {identity.describe()}")
    return FakeSource(metas)


def make_driver(
    *,
    target: UpdateTarget = TARGET_V1,
    source: FakeSource | None = None,
    engine_fail: bool = False,
) -> tuple[NCCLWeightTransferDriver, FakeSyncClient, FakeTrainerEngine]:
    client = FakeSyncClient()
    engine = FakeTrainerEngine(fail=engine_fail)
    src = source if source is not None else source_for(target.identity)

    def builder(*, client: Any, trainer_init_info: Any, source: Any) -> FakeTrainerEngine:
        engine.client = client
        client.init_weight_transfer_engine({"init_info": trainer_init_info})
        return engine

    driver = NCCLWeightTransferDriver(
        base_url="http://engine.test",
        trainer_init_info={"rank_offset": 1, "world_size": 2},
        source=src,
        world_size=2,
        client=client,
        builder=builder,
    )
    return driver, client, engine


class TestNormalisation(unittest.TestCase):
    def test_torch_dtype_string_is_stripped(self):
        self.assertEqual(dtype_name(FakeDType("bfloat16")), "bfloat16")
        self.assertEqual(dtype_name("float16"), "float16")

    def test_param_meta_becomes_param_spec(self):
        self.assertEqual(
            param_spec(FakeParamMeta("w", FakeDType("float16"), (2, 3))),
            ParamSpec("w", "float16", (2, 3)),
        )

    def test_torch_and_string_manifests_hash_identically(self):
        """Without normalisation the driver could never match a bootstrap identity."""
        from_torch = WeightIdentity.from_param_specs(
            [param_spec(FakeParamMeta("w", FakeDType("bfloat16"), (4, 4)))]
        )
        from_strings = WeightIdentity.from_param_specs([ParamSpec("w", "bfloat16", (4, 4))])
        self.assertEqual(from_torch, from_strings)


class TestTargetAwareClient(unittest.TestCase):
    """The review's finding: upstream finalizes with no version."""

    def test_missing_version_becomes_the_pending_target_label(self):
        inner = FakeSyncClient()
        wrapper = RolloutCoreWeightSyncClient(inner)
        wrapper.set_target(TARGET_V1)
        wrapper.finish_weight_update()  # upstream's call: no argument
        self.assertEqual(inner.calls, [("finish", TARGET_V1.label)])

    def test_explicit_version_still_wins(self):
        inner = FakeSyncClient()
        wrapper = RolloutCoreWeightSyncClient(inner)
        wrapper.set_target(TARGET_V1)
        wrapper.finish_weight_update(weight_version="rc-99")
        self.assertEqual(inner.calls, [("finish", "rc-99")])

    def test_finalize_without_a_target_is_refused(self):
        wrapper = RolloutCoreWeightSyncClient(FakeSyncClient())
        with self.assertRaises(WeightTransferNotConfiguredError):
            wrapper.finish_weight_update()

    def test_other_calls_are_pure_delegation(self):
        inner = FakeSyncClient()
        wrapper = RolloutCoreWeightSyncClient(inner)
        wrapper.init_weight_transfer_engine({"init_info": {"world_size": 2}})
        wrapper.start_weight_update()
        wrapper.update_weights({"names": ["a"]})
        self.assertEqual(inner.names(), ["init", "start", "update"])

    def test_pending_label_is_readable(self):
        wrapper = RolloutCoreWeightSyncClient(FakeSyncClient())
        self.assertIsNone(wrapper.pending_label)
        wrapper.set_target(TARGET_V1)
        self.assertEqual(wrapper.pending_label, "rc-1")


class TestProtocolConformance(unittest.TestCase):
    def test_driver_satisfies_the_port(self):
        driver, _client, _engine = make_driver()
        self.assertIsInstance(driver, WeightTransferDriver)

    def test_driver_declares_that_it_moves_tensors(self):
        driver, _client, _engine = make_driver()
        self.assertTrue(driver.moves_tensors)
        self.assertEqual(driver.name, "nccl")


class TestInitialize(unittest.TestCase):
    def test_initialize_reports_the_transfer_group(self):
        driver, client, _engine = make_driver()
        init = driver.initialize()
        self.assertIsInstance(init, WeightTransferInit)
        self.assertTrue(init.initialised)
        self.assertEqual(init.backend, "nccl")
        self.assertEqual(init.world_size, 2)
        self.assertEqual(client.names(), ["init"], "the handshake goes through the wrapper")

    def test_transfer_before_initialize_is_refused(self):
        driver, _client, _engine = make_driver()
        with self.assertRaises(WeightTransferNotConfiguredError):
            driver.transfer(TARGET_V1)


class TestIdentityPrecheck(unittest.TestCase):
    """The "fail before mutation" invariant the review asked for."""

    def test_matching_identity_transfers_and_supplies_the_version(self):
        driver, client, engine = make_driver()
        driver.initialize()
        report = driver.transfer(TARGET_V1)
        self.assertTrue(report.data_plane_complete)
        self.assertTrue(report.finish_acknowledged)
        self.assertIsNone(report.chunks_transferred, "must not fabricate a chunk count")
        self.assertEqual(report.observed_identity, TARGET_V1.identity)
        self.assertEqual(client.names(), ["init", "start", "update", "finish"])
        self.assertEqual(client.calls[-1], ("finish", TARGET_V1.label))
        self.assertEqual(engine.broadcasts, 1)

    def test_mismatched_identity_fails_before_any_request(self):
        client = FakeSyncClient()
        engine = FakeTrainerEngine()

        def builder(*, client: Any, trainer_init_info: Any, source: Any) -> FakeTrainerEngine:
            engine.client = client
            client.init_weight_transfer_engine({"init_info": trainer_init_info})
            return engine

        driver = NCCLWeightTransferDriver(
            base_url="http://engine.test",
            trainer_init_info={"world_size": 2},
            source=source_for(IDENTITY_V0),
            client=client,
            builder=builder,
        )
        driver.initialize()
        before = list(client.calls)

        with self.assertRaises(WeightIdentityMismatchError) as ctx:
            driver.transfer(TARGET_V1)

        self.assertIsInstance(ctx.exception, InvariantViolation)
        self.assertIn("refusing before any request is sent", str(ctx.exception))
        self.assertEqual(client.calls, before, "nothing may be sent on a mismatch")
        self.assertEqual(engine.broadcasts, 0)

    def test_identity_is_computed_not_copied_from_the_target(self):
        driver, _client, _engine = make_driver()
        self.assertEqual(driver.identity(), TARGET_V1.identity)
        self.assertNotEqual(driver.identity(), IDENTITY_V0)

    def test_declared_provenance_is_folded_in(self):
        source = FakeSource([FakeParamMeta("a", FakeDType("f16"), (2, 2))])

        def identity_at(step: int) -> WeightIdentity:
            driver = NCCLWeightTransferDriver(
                base_url="http://engine.test",
                trainer_init_info={},
                source=source,
                provenance=WeightProvenance(step=step),
                client=FakeSyncClient(),
                builder=lambda **_kw: FakeTrainerEngine(),
            )
            return driver.identity()

        self.assertNotEqual(identity_at(100), identity_at(500))
        self.assertEqual(
            identity_at(100),
            WeightIdentity.from_param_specs(
                [ParamSpec("a", "f16", (2, 2))], source=WeightProvenance(step=100)
            ),
        )

    def test_declare_lets_one_driver_stage_a_second_step(self):
        """Provenance is per-version; a cached identity cannot follow the weights.

        Without `declare` the driver's identity is frozen at construction, so a
        target carrying a later step could never match it and `transfer` would
        refuse every update after the first as an identity mismatch.
        """
        source = FakeSource([FakeParamMeta("a", FakeDType("f16"), (2, 2))])
        client = FakeSyncClient()
        engine = FakeTrainerEngine()

        def builder(*, client: Any, trainer_init_info: Any, source: Any) -> FakeTrainerEngine:
            engine.client = client
            client.init_weight_transfer_engine({"init_info": trainer_init_info})
            return engine

        driver = NCCLWeightTransferDriver(
            base_url="http://engine.test",
            trainer_init_info={"rank_offset": 1, "world_size": 2},
            source=source,
            provenance=WeightProvenance(step=0),
            client=client,
            builder=builder,
        )
        step0 = driver.identity()
        self.assertEqual(driver.identity(), step0, "cached until re-declared")

        step1 = driver.declare(WeightProvenance(step=1))
        self.assertNotEqual(step1, step0)
        self.assertEqual(step1.manifest_digest, step0.manifest_digest, "same architecture")
        self.assertEqual(driver.identity(), step1, "the cache followed")

        # And the second step is now stageable, which is the point.
        driver.initialize()
        target = UpdateTarget(version=WeightVersion(2), identity=step1)
        report = driver.transfer(target)
        self.assertEqual(report.observed_identity, step1)

    def test_declare_returns_a_manifest_only_identity_when_given_nothing(self):
        """Declaring no provenance must be visible, not quietly inherited."""
        source = FakeSource([FakeParamMeta("a", FakeDType("f16"), (2, 2))])
        driver = NCCLWeightTransferDriver(
            base_url="http://engine.test",
            trainer_init_info={},
            source=source,
            provenance=WeightProvenance(step=9),
            client=FakeSyncClient(),
            builder=lambda **_kw: FakeTrainerEngine(),
        )
        self.assertEqual(driver.identity().exactness, "declared-source")
        self.assertEqual(driver.declare(WeightProvenance(step=10)).exactness, "declared-source")


class TestThreadDeviceRule(unittest.TestCase):
    """The rule behind five failed pod runs, testable with no GPU and no torch.

    PyTorch's current CUDA device is thread-local: a spawned thread starts on the
    default device, so a transfer issued from a worker thread streams the trainer's
    tensors on the wrong one. NCCL's own message for that is "Cuda failure 400
    invalid resource handle", which names neither the device nor the thread.
    """

    def test_same_device_is_fine(self):
        self.assertIsNone(device_mismatch(1, 1))
        self.assertIsNone(device_mismatch(0, 0))

    def test_unknown_on_either_side_is_not_a_mismatch(self):
        """No torch, or no CUDA, must not turn into a refusal."""
        self.assertIsNone(device_mismatch(None, 1))
        self.assertIsNone(device_mismatch(1, None))
        self.assertIsNone(device_mismatch(None, None))

    def test_a_different_device_names_both_and_the_fix(self):
        message = device_mismatch(1, 0)
        self.assertIsNotNone(message)
        assert message is not None  # for mypy
        self.assertIn("cuda:1", message)
        self.assertIn("cuda:0", message)
        self.assertIn("thread-local", message)
        self.assertIn("torch.cuda.set_device(1)", message)

    def test_transfer_refuses_before_any_request_on_a_wrong_thread(self):
        driver, client, _engine = make_driver()
        driver.initialize()
        # Pretend the rendezvous happened on device 1 and we are now elsewhere.
        driver._device = 1
        driver._current_device = staticmethod(lambda: 0)  # type: ignore[method-assign]
        with self.assertRaises(WeightTransferNotConfiguredError) as ctx:
            driver.transfer(TARGET_V1)
        self.assertIn("cuda:1", str(ctx.exception))
        self.assertNotIn("start_weight_update", client.names())


class TestFailurePropagation(unittest.TestCase):
    def test_a_failed_broadcast_propagates_without_finishing(self):
        driver, client, _engine = make_driver(engine_fail=True)
        driver.initialize()
        with self.assertRaises(RuntimeError):
            driver.transfer(TARGET_V1)
        self.assertNotIn("finish", client.names())

    def test_shutdown_is_forwarded(self):
        driver, _client, engine = make_driver()
        driver.initialize()
        driver.shutdown()
        self.assertTrue(engine.shut_down)

    def test_shutdown_before_initialize_is_harmless(self):
        driver, _client, _engine = make_driver()
        driver.shutdown()


class DriverBackedAdapter:
    """A real ``LifecycleAdapter`` whose update steps go through the driver.

    Duck-typed rather than inheriting the Protocol, so the only thing under test
    is the integration shape: bootstrap opens the rendezvous, and
    ``complete_weight_update`` returns evidence derived from the driver's report.
    """

    def __init__(self, driver: NCCLWeightTransferDriver, identity: WeightIdentity) -> None:
        self.driver = driver
        self.identity = identity

    def bootstrap(self) -> BootstrapEvidence:
        init = self.driver.initialize()
        return BootstrapEvidence(
            observed_engine_label="rc-0",
            weight_transfer_initialised=init.initialised,
            pre_seed_label="default",
            weight_identity=self.identity,
            backend="nccl",
            weight_transfer_driver=self.driver.name,
        )

    def begin_drain(self) -> None:
        return None

    def await_drain(self) -> Any:
        return good_drain()

    def start_weight_update(self, target: UpdateTarget) -> None:
        return None

    def complete_weight_update(self, target: UpdateTarget) -> UpdateEvidence:
        report = self.driver.transfer(target)
        return UpdateEvidence(
            target=target,
            weights_loaded=report.data_plane_complete,
            finish_acknowledged=report.finish_acknowledged,
            chunks_transferred=report.chunks_transferred,
            data_plane_complete=report.data_plane_complete,
            observed_identity=report.observed_identity,
        )

    def invalidate_caches(self, target: UpdateTarget) -> InvalidateEvidence:
        return InvalidateEvidence(True, True, True)

    def validate_pre_resume(self, target: UpdateTarget) -> ValidateEvidence:
        return ValidateEvidence(target=target, observed_engine_label=target.label, is_paused=True)

    def resume(self, target: UpdateTarget) -> ResumeEvidence:
        return ResumeEvidence(
            target=target,
            resume_acknowledged=True,
            is_paused=False,
            observed_engine_label=target.label,
        )


class TestRunsInsideTheRunner(unittest.TestCase):
    def test_a_full_cycle_commits_through_the_driver(self):
        self.assertEqual(GOOD_BOOTSTRAP.weight_identity, IDENTITY_V0)  # fixture sanity
        driver, client, engine = make_driver()
        ctrl = LifecycleController()
        runner = LifecycleRunner(
            ctrl, DriverBackedAdapter(driver, IDENTITY_V0), sleep=lambda _s: None
        )

        runner.bootstrap()
        self.assertIs(ctrl.state, LifecycleState.READY)
        binding = ctrl.admit_rollout("R1")
        ctrl.finish_rollout("R1")
        self.assertEqual(binding.version.label, "rc-0")

        result = runner.install_next(IDENTITY_V1)

        self.assertEqual(result.committed.version.label, "rc-1")
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(
            [s.value for s in result.states_visited],
            [
                "READY",
                "DRAINING",
                "QUIESCED",
                "UPDATING",
                "INVALIDATING",
                "VALIDATING",
                "RESUMING",
                "READY",
            ],
        )
        self.assertEqual(client.calls[-1], ("finish", "rc-1"))
        self.assertEqual(engine.broadcasts, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
