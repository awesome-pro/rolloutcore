# Phase 3B notes — the target-aware weight-sync client

**Status:** planning only. Nothing here is implemented; Phase 3B starts after
Phase 3A passes on a real GPU.

---

## The problem the planner found

vLLM's trainer-side NCCL engine drives the whole update round trip, and it calls
`finish_weight_update` **without a version**:

- `vllm/distributed/weight_transfer/nccl_engine.py:361` — `self.client.finish_weight_update()`
- `vllm/distributed/weight_transfer/clients.py:89` — the client *does* accept one:
  `def finish_weight_update(self, weight_version: str | None = None)`, and posts
  `{"weight_version": ...}` only when it is not `None`.

So `engine.send_weights()` alone leaves the engine reporting its **old**
`weight_version`. RolloutCore's VALIDATING state compares that label for exact
equality with `UpdateTarget.label` and taints on any mismatch
(`docs/state-machine.md` §4). A naive `driver.transfer()` would therefore taint
on every successful transfer.

Equally unusable: calling `POST /update_weight_version` *afterwards* to patch the
label up. That publishes a version whose weights we did not prove, which is the
one thing the design forbids (I8: no automatic reconciliation).

## The fix: a wrapper client, not a patch-up call

```python
class RolloutCoreWeightSyncClient:
    """Delegate every control-plane call, but own the version at finalize."""

    def __init__(self, inner: HTTPVLLMWeightSyncClient) -> None:
        self._inner = inner
        self._pending_label: str | None = None

    def set_target(self, target: UpdateTarget) -> None:
        # Called by the driver *before* send_weights(); the target is already
        # committed as pending by the controller (UPDATING).
        self._pending_label = target.label

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._inner.init_weight_transfer_engine(init_info)

    def start_weight_update(self) -> None:
        self._inner.start_weight_update()

    def update_weights(self, update_info: Any) -> None:
        self._inner.update_weights(update_info)

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        # vLLM's engine passes None here (nccl_engine.py:361). Substitute the
        # target the controller opened this update with, and never invent one.
        label = weight_version or self._pending_label
        if label is None:
            raise WeightTransferNotConfiguredError(
                "finish_weight_update", "no target was set on the sync client"
            )
        self._inner.finish_weight_update(weight_version=label)
```

Sequence, end to end:

```
controller: QUIESCED --begin_update(target rc-1)--> UPDATING
driver.transfer(rc-1):
    client.set_target(rc-1)
    WeightTransferTrainerFactory.trainer_init(..., client=wrapper, source=ModuleSource(model))
        -> POST /init_weight_transfer_engine (real NCCLTrainerInitInfo rendezvous)
    engine.send_weights()
        -> POST /start_weight_update
        -> POST /update_weights   (interleaved with the NCCL broadcast)
        -> POST /finish_weight_update {"weight_version": "rc-1"}   <- the substitution
controller: confirm_updated(evidence) -> INVALIDATING -> VALIDATING
    validate_pre_resume sees rc-1 while still paused
```

## Two more requirements from the review, both worth pinning now

1. **Identity before mutation.** Compute the identity of what will actually be
   sent from `ModuleSource.metadata()` (upstream's `WeightSource` ABC,
   `vllm/distributed/weight_transfer/base.py:128`) plus a `WeightProvenance`, and
   compare it against `target.identity` **before** `trainer_init` posts anything.
   A mismatch must fail with no request sent, which is why
   `WeightTransferNotConfiguredError` is a `RolloutCoreError`: the runner's effect
   wrapper re-raises it without tainting, because nothing happened yet.
   (`WeightProvenance`, not `WeightSource`, precisely because the latter is
   upstream's type — see `docs/state-machine.md` §5.)

2. **`chunks_transferred` stays `None`** unless upstream exposes a trustworthy
   count. `NCCLTrainerWeightTransferEngine.send_weights()` does not, so the
   driver reports `None` rather than inventing a number. `UpdateEvidence` already
   treats it as diagnostic-only.

## Environment prerequisite: prove upstream first

Before writing any of the above, adapt `examples/rl/rlhf_http_nccl.py` to
`opt-125m` with GPU 0 = vLLM (TP=1) and GPU 1 = trainer, BF16, no FP8. If NCCL
fails there, the problem is the pod / CUDA / NCCL / vLLM, not RolloutCore. Only
once upstream's own transfer works does implementing
`NCCLWeightTransferDriver` become a RolloutCore question.

## What Phase 3B must report

- NCTCL rendezvous parameters actually used (rank/world_size/rendezvous mode).
- `weight_version` before and after: `rc-0` → `rc-1`, observed while still paused.
- The computed `WeightIdentity` versus `target.identity` (must be equal).
- Whether `send_weights()` returned cleanly, and what the engine reported.
- A demonstrable output change: `load-format=dummy` baseline vs. real weights,
  which is Phase 3C's headline result.
