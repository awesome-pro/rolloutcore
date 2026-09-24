# SPDX-License-Identifier: Apache-2.0
"""Runnable demonstration of one full lifecycle cycle, no GPU required.

    python -m rolloutcore.demo

Runs ``READY -> DRAINING -> QUIESCED -> UPDATING -> INVALIDATING -> VALIDATING
-> RESUMING -> READY`` against the in-memory fake engine, with two
distinguishable weight identities modelling the plan's Model A / Model B. Prints
the engine calls made, the state journal, and the resulting rollout bindings.

Then it demonstrates the failure properties the design is built around: a
failed update is never published, a taint preserves the in-flight rollout
bindings for forensics, and a bootstrap that refuses an already-managed engine
does so before writing anything.
"""

from __future__ import annotations

import sys

from .adapters import FakeVLLMAdapter, FakeVLLMEngine, manifest_identity
from .lifecycle import LifecycleController, LifecycleState, RolloutBinding
from .runner import LifecycleRunner

W = 78


def rule(title: str = "") -> None:
    if title:
        print(f"\n{title}\n{'─' * W}")
    else:
        print("─" * W)


def show_binding(label: str, b: RolloutBinding) -> None:
    print(
        f"  {label:<6} request={b.request_id:<6} version={b.version.label:<7} "
        f"identity={b.weight_identity.short}  cache_salt={b.cache_salt}"
    )


def happy_path() -> None:
    rule("1. Full cycle: rollout(v0) -> drain -> update(v1) -> invalidate -> validate -> resume")

    engine = FakeVLLMEngine(world_size=2)
    engine.seed_fresh_with(manifest_identity("A"))
    adapter = FakeVLLMAdapter(engine)
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, adapter, sleep=lambda _s: None)

    print(
        f"  engine before bootstrap : weight_version={engine.weight_version!r} "
        f"(unmanaged; V1 is single-owner)"
    )
    runner.bootstrap()
    print(
        f"  engine after  bootstrap : weight_version={engine.weight_version!r}  "
        f"state={ctrl.state.value}"
    )

    before = ctrl.admit_rollout("R1")
    ctrl.finish_rollout("R1")
    show_binding("R1", before)

    target = ctrl.next_target(manifest_identity("B"))
    print(
        f"\n  installing: version={target.label} identity={target.identity.short} "
        f"(Model A -> Model B)"
    )

    result = runner.run_cycle(target)

    print("\n  state journal:")
    for rec in ctrl.journal:
        ident = f" identity={rec.identity.short}" if rec.identity else ""
        print(f"    {rec.seq:>2}. {rec.from_state:<13} -> {rec.to_state:<13} [{rec.event}]{ident}")

    print(f"\n  engine calls: {' -> '.join(engine.calls)}")
    print(
        f"  states visited ({len(result.states_visited)}): "
        f"{' -> '.join(s.value for s in result.states_visited)}"
    )

    after = ctrl.admit_rollout("R2")
    print()
    show_binding("R1", before)
    show_binding("R2", after)
    assert before.version != after.version
    assert before.weight_identity != after.weight_identity
    assert before.cache_salt != after.cache_salt, "cross-version cache reuse is possible!"
    print("\n  OK: distinct versions, distinct identities, distinct cache salts.")


def drain_disagreement_path() -> None:
    rule("2. Fail-closed: engine says drained, RolloutCore still counts work -> taint")

    engine = FakeVLLMEngine()
    engine.seed_fresh_with(manifest_identity("A"))
    adapter = FakeVLLMAdapter(engine)
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, adapter, sleep=lambda _s: None)
    runner.bootstrap()
    committed_before = ctrl.current_version
    assert committed_before is not None  # bootstrap committed rc-0

    # R1 is still in flight, but the engine reports the drain complete. The two
    # bookkeeping systems disagree and we cannot tell which is right, so the
    # controller taints rather than retrying.
    ctrl.admit_rollout("R1")

    target = ctrl.next_target(manifest_identity("B"))
    try:
        runner.run_cycle(target)
    except Exception as exc:
        print(f"  cycle raised: {type(exc).__name__}")
        print(f"               {str(exc).splitlines()[0]}")

    print(f"\n  controller state     : {ctrl.state.value}")
    print(f"  committed version    : {committed_before.label}  (NOT {target.label})")
    print(f"  engine paused        : {engine.paused}  (never resumed)")
    print(
        "  orphaned rollouts    : "
        f"{[o.request_id for o in ctrl.orphaned_rollouts]}  (retained for forensics)"
    )
    assert ctrl.current_version == committed_before
    assert engine.paused
    assert [o.request_id for o in ctrl.orphaned_rollouts] == ["R1"]
    print("\n  OK: nothing published, engine still paused, bindings preserved.")


def finalize_failure_path() -> None:
    rule("3. Fail-closed: a failed finalize publishes nothing and resumes nothing")

    engine = FakeVLLMEngine()
    engine.seed_fresh_with(manifest_identity("A"))
    adapter = FakeVLLMAdapter(engine)
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, adapter, sleep=lambda _s: None)
    runner.bootstrap()
    committed_before = ctrl.current_version
    assert committed_before is not None  # bootstrap committed rc-0

    engine.faults["finish_weight_update"] = "worker died mid-transfer"
    target = ctrl.next_target(manifest_identity("B"))
    try:
        runner.run_cycle(target)
    except Exception as exc:
        print(f"  cycle raised: {type(exc).__name__}")
        print(f"               {str(exc).splitlines()[0]}")

    print(f"\n  controller state     : {ctrl.state.value}")
    print(f"  committed version    : {committed_before.label}  (NOT {target.label})")
    print(f"  engine paused        : {engine.paused}  (never resumed)")
    print(
        f"  engine weight_version: {engine.weight_version!r}  "
        "(unchanged: vLLM writes the label only *after* the finalize RPC "
        "returns, so a failed finalize never advances it)"
    )
    assert ctrl.current_version == committed_before
    assert engine.paused
    assert engine.weight_version == "rc-0"
    print("\n  OK: nothing published, engine still paused.")


def cache_invalidation_failure_path() -> None:
    rule("4. The engine's label is NOT a commit marker")

    engine = FakeVLLMEngine()
    engine.seed_fresh_with(manifest_identity("A"))
    adapter = FakeVLLMAdapter(engine)
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, adapter, sleep=lambda _s: None)
    runner.bootstrap()
    committed_before = ctrl.current_version
    assert committed_before is not None  # bootstrap committed rc-0

    # The finalize succeeds, so the engine's label advances -- but the prefix
    # cache cannot be reset (blocks still held, 200 + {"success": false}).
    engine.prefix_reset_succeeds = False
    target = ctrl.next_target(manifest_identity("B"))
    try:
        runner.run_cycle(target)
    except Exception as exc:
        print(f"  cycle raised: {type(exc).__name__}")
        print(f"               {str(exc).splitlines()[0]}")

    print(f"\n  controller state     : {ctrl.state.value}")
    print(f"  RolloutCore committed: {committed_before.label}  (NOT {target.label})")
    print(
        f"  engine weight_version: {engine.weight_version!r}  "
        "<- the engine already claims the new version"
    )
    print(f"  engine paused        : {engine.paused}  (never resumed)")
    print("  => reading the engine's label back is not proof of a committed")
    print("     update; RolloutCore's own target is the commit record.")
    assert ctrl.current_version == committed_before
    assert engine.weight_version == target.label
    assert engine.paused
    print("\n  OK: tainted on the failed invalidation, nothing resumed.")


def refused_bootstrap_path() -> None:
    rule("5. A refused adoption leaves the other controller's engine untouched")

    engine = FakeVLLMEngine()
    engine.seed_fresh_with(manifest_identity("A"))
    # A first controller already owns this engine and has driven it to rc-7.
    engine.weight_version = "rc-7"
    engine.calls.clear()
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, FakeVLLMAdapter(engine), sleep=lambda _s: None)

    try:
        runner.bootstrap()
    except Exception as exc:
        print(f"  bootstrap raised: {type(exc).__name__}")
        print(f"                    {str(exc).splitlines()[0]}")

    print(f"\n  controller state     : {ctrl.state.value}")
    print(f"  engine weight_version: {engine.weight_version!r}  (unchanged)")
    print(f"  engine calls         : {' -> '.join(engine.calls)}")
    print("  => the refusal happens BEFORE the label write. Reading rc-7, writing")
    print("     rc-0 and refusing afterwards would have corrupted the owner's")
    print("     engine on the way out. Nothing was written, so this is retryable")
    print("     rather than tainted: the engine is healthy, it is just not ours.")

    assert ctrl.state is LifecycleState.UNINITIALIZED
    assert engine.weight_version == "rc-7"
    assert "update_weight_version" not in engine.calls
    assert "init_weight_transfer_engine" not in engine.calls

    # Once the owner releases it, the same controller can bootstrap normally.
    engine.weight_version = "default"
    runner.bootstrap()
    print(f"\n  after release        : {engine.weight_version!r}, state={ctrl.state.value}")
    # Compared by value: mypy narrows `ctrl.state` from the assert above and does
    # not invalidate that narrowing across an opaque method call, so an `is`
    # check here reads as a non-overlapping literal comparison.
    assert ctrl.state.value == LifecycleState.READY.value
    assert engine.weight_version == "rc-0"
    print("\n  OK: refused without side effects, then adopted cleanly.")


def main() -> int:
    print("=" * W)
    print("RolloutCore — lifecycle demo (fake engine, no GPU)".center(W))
    print("=" * W)
    happy_path()
    drain_disagreement_path()
    finalize_failure_path()
    cache_invalidation_failure_path()
    refused_bootstrap_path()
    rule()
    print("All assertions held.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
