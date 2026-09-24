# Phase 4B results: the failure paths, against a real engine

**Result: PASS. 21/21 checks.** Four scenarios, three engine lifetimes, against
real vLLM. Every fail-closed claim in this project had until now been tested only
against the fake engine, which models the failures we thought of.

| | |
|---|---|
| Date | 2026-09-24 16:1x UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` |
| RolloutCore | see `results/phase4b.json` |
| Hardware | 2× NVIDIA RTX 3090, one pod |
| Model | `facebook/opt-125m`, TP=1 on GPU 0 (`--load-format dummy`), bf16 trainer on GPU 1 |
| Artifact | `results/phase4b.json`, plus `phase4b-server-{1,2,3}.log` |

---

## A. A failed drain is recoverable, and does not taint

`/pause?mode=wait` given a 0.1 s socket timeout, `drain_reissues=0`, with a
256-token generation in flight. All seven checks pass:

```
A_drain_failed_with_DrainFailedError   drain did not complete after 1 attempt(s); last error: timed out
A_drain_failure_does_not_taint         controller not TAINTED
A_state_stays_DRAINING                 DRAINING is fail-closed: it admits no rollouts
A_stragglers_were_aborted              /abort_requests posted
A_recovery_reaches_READY               same controller, same server, no restart
A_recovery_installs_the_update         rc-1
A_recovery_serves                      ' the capital of the French Republic...'
```

The in-flight request's outcome is `abort`: vLLM aborted it and returned the
partial result rather than erroring, which is what makes the abort-and-reissue
strategy work. The recovery reuses the **same driver** (its NCCL session with the
engine was still live) and only swaps in a fresh adapter with a real timeout; the
server log shows a single transfer-engine creation for the whole scenario, which
confirms nothing was re-initialised.

This is the scenario that matters operationally: a drain that times out is a
retryable condition, not a reason to throw the engine away.

## B. An identity mismatch fails before any mutation

```
[I4-IDENTITY] the driver would stage 925369d663bc (manifest-only) but
8b378e8b3fe0 (manifest-only) is the target; refusing before any request is sent
```

`/start_weight_update` appears **zero** times in the server log for that attempt,
the engine is left paused, still at rc-1, and the controller is not tainted. The
driver computed its identity from the trainer's own manifest and compared it to
the target's *before* `send_weights()`, so "fail before mutating" is asserted
from the engine's own request log, not from our bookkeeping.

The handback check then measured the gap:

```
AlreadyManagedEngineError: engine already reports weight_version 'rc-1', which
RolloutCore wrote: another controller may own it. Bootstrap refused before
writing anything. A lease/recovery protocol is required to take over a managed
engine and V1 does not implement one.
```

So a driver-side failure mid-`UPDATING` leaves the engine paused at rc-1, the
controller stuck in UPDATING, and a fresh controller refused. Recovery in that
state is an operator action (the harness restores service with a raw `/resume`),
not a RolloutCore transition; see the finding below.

## C. A dead engine: two paths, and they differ

The engine's whole process group is SIGKILLed, then a cycle is attempted.

**C1: the drain path does not taint.** This was predicted from reading
`_observe` (`src/rolloutcore/runner.py:213`: reads are deliberately never
tainted, because during DRAINING the engine is paused and a failed read leaves
nothing ambiguous), and the run confirms it:

```
error     DrainFailedError: drain did not complete after 3 attempt(s); last error:
          POST .../pause?mode=wait&clear_cache=true: transport error: [Errno 111] Connection refused
state     DRAINING
tainted   false
committed rc-0
```

**C2: the mutating path taints, as designed.** A fresh controller cannot even
bootstrap, because the very first read fails with an unknown engine outcome:

```
EngineTaintedError: engine tainted in state UNINITIALIZED: bootstrap failed with an
unknown engine outcome: HTTPAdapterError('GET .../weight_info: transport error:
[Errno 111] Connection refused'). TAINTED is terminal in V1: restart the engine and
construct a new controller.
```

**C3: the taint is terminal and local:**

```
IllegalTransitionError: [T-ILLEGAL] event 'begin_drain' is not legal from state TAINTED
```

refused in under 50 ms, with no network I/O: `_require_legal` runs before any
precondition (`lifecycle.py:303`).

## D. Recovery is a fresh process

A third engine and a fourth controller: bootstrap succeeds, the label is `rc-0`,
and it serves. Taint is controller-scoped, and the engine's state died with its
process, so restart is a real recovery path, not a workaround.

## Findings

**1. A drain failure is fail-closed but not self-healing, and the error message
used to say the wrong thing.** DRAINING has exactly one exit, `CONFIRM_DRAINED`
(`lifecycle.py:130`), so an engine that never comes back leaves the controller
untainted in DRAINING forever. That is *safe* (DRAINING admits no rollouts and no
version was published), and it is the right classification, because taint is for
ambiguous *mutations* and a failed drain writes nothing. But nothing escalates it
either, so the exit is an operator `taint()`.

The run also showed the old `DrainFailedError` text was actively misleading in
this case: it said "The engine remains paused; resuming is an operator decision"
when the last error was `Connection refused`: the engine was not paused, it was
gone, and its pause state was what could not be confirmed. The message
now says the pause state is unconfirmed and names the two decisions available,
and `tests/test_cycle.py` pins the dead-engine outcome so it cannot drift.

**2. There is no hand-back path for an engine whose update failed.** A
driver-side failure mid-UPDATING leaves the engine paused, the controller without
a legal exit (`(UPDATING, CONFIRM_UPDATED)` is the only transition out), and a
fresh controller refused by the managed-label check. The check's own message is
the honest summary: *a lease/recovery protocol is required to take over a managed
engine and V1 does not implement one*. This is the clearest candidate for an
upstream proposal, and it is the same shape as the drain finding: the failure is
detected correctly and then has nowhere to go.

**3. The error taxonomy held up under real failures.** Every injected failure
produced a distinct, actionable exception, and the classification never had to be
guessed: `DrainFailedError` (retryable), `WeightIdentityMismatchError`
(non-tainting invariant), `EngineTaintedError` (terminal), `AlreadyManagedEngineError`
(refusal before writing), `IllegalTransitionError` (local, no I/O). The messages
named the invariant, the state, and in two cases the missing upstream feature.

## What Phase 4B does *not* prove

- **No mid-update kill.** C kills the engine *before* the cycle. A kill during the
  broadcast could hang the rendezvous rather than fail, and that is what Phases 4D
  and 6 went on to measure: 4D landed one at the update's edge and got a taint in
  0.177 s, while 6 landed one *inside* an 8B collective and it never returned. The
  bounded `NCCL_TIMEOUT` this section originally suggested does not exist in that
  path; see `docs/phase3b-runbook.md` §10.2.
- **Not a dead engine under load.** C1 has no in-flight rollouts; A's drain
  failure does. The two are not combined.
- **`opt-125m`, one node, TP=1.**
- **The 0.1 s timeout is a synthetic failure**: a real engine under real load
  produces slow drains, not socket timeouts.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_failures_2gpu.py
```
