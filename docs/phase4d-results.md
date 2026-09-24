# Phase 4D results: the engine dies mid-update

**Result: PASS. 8/8 checks.** Phase 4B killed the engine *before* a cycle; this
kills it *inside* one, at the first instant the controller reports `UPDATING`.
Detection took 0.1767 s against a 120 s budget, nothing was published, and the
controller tainted.

| | |
|---|---|
| Date | 2026-09-24 16:40 UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` |
| RolloutCore | `1d60230a` (see `results/phase4d.json`) |
| Hardware | 2× NVIDIA RTX 3090, one pod |
| Model | `facebook/opt-125m`, TP=1 on GPU 0 (`--load-format dummy`), bf16 trainer on GPU 1 |
| Artifact | `results/phase4d.json` |

---

## What was killed, and when

A healthy cycle ran first, both to have a baseline and to time the real update:
`2.1192 s` end to end. Then a second cycle to `rc-2` was started on a daemon
thread while the main thread watched `ctrl.state`, and the engine's whole process
group was SIGKILLed the moment the state read `UPDATING`:

```
state_at_kill                     UPDATING
seconds_from_cycle_start_to_kill  2.0039
healthy_cycle_seconds             2.1192
arm_timed_out                     false
```

The kill landed 0.115 s before the healthy cycle would have finished. Most of
that 2.12 s is the drain's poll interval, not work (the drain-only cycle in
Phase 4A took 1.375 s and its update phase was ~0.1 s), so the kill landed
essentially at the *opening* of `UPDATING`.

One limit: `complete_weight_update` is the whole round trip
(`/start_weight_update` → `/update_weights` ×N → `/finish_weight_update` **and**
the tensor broadcast (`port.py:23`)), so the failure here surfaced on an HTTP
socket, and this measurement cannot say whether the process died during the
collective or in the calls around it. On a 125M model the collective is ~100 ms
wide and the two are not separable. Phase 6 answers the question, with an 8B
broadcast and a kill aimed at half of it.

## What was detected

```
hung                     false
seconds_from_kill        0.1767
budget_seconds           120.0
outcome                  EngineTaintedError: engine tainted in state UPDATING:
                         complete_weight_update failed with an unknown engine
                         outcome: ConnectionError(ProtocolError('Connection
                         aborted.', ConnectionResetError(104, 'Connection reset
                         by peer'))). TAINTED is terminal in V1: restart the
                         engine and construct a new controller.
```

The path is the one `src/rolloutcore/runner.py:187-205` describes: an exception out
of a **mutating** call is an unknown engine outcome, so the controller taints
rather than guessing whether the weights landed. `ConnectionResetError(104)` is the
kernel saying the peer is gone, not a timeout, which is why detection was fast and
why the budget was never approached.

The controller's own record afterwards:

```
state      TAINTED
tainted    true
serving    false
committed  rc-1        ← the last version that was actually published
```

## Why this is the right outcome, not just a safe one

`rc-2` was mid-broadcast when the engine died. There are two possibilities (every
tensor landed, or some prefix of them did), and neither is observable from the
trainer side, because the acknowledgement (`/finish_weight_update`) is what
certifies completion and that socket is the thing that reset. Publishing `rc-2`
would therefore claim a weight set that may not exist, and *not* tainting would
leave the controller able to admit rollouts against an engine whose label no
longer describes its weights. Both are what I6 (*a failed update never publishes
its target version*) and I8 (*a version mismatch is a hard failure*) are for.

The recorded check `committed == rc-1` is the observable form of that: the
generation counter did not advance, so no trajectory can ever be bound to `rc-2`.

## Contrast with Phase 4B, and what it changes

| | kill before the cycle (4B) | kill inside the update (4D) |
|---|---|---|
| Path that fails | `begin_drain` (a read-ish effect) | `complete_weight_update` (mutating) |
| Taints? | **No**: `DRAINING` stays fail-closed | **Yes**: terminal |
| Detection | immediate (`Connection refused`) | 0.1767 s |
| Version published | none | none |
| Exit | operator taint, or engine comes back | fresh engine + fresh controller |

4B's finding stands and 4D sharpens it: the *classification* is right in both
cases (a failed drain writes nothing so it cannot be ambiguous, a failed update
writes tensor bytes so it is), but the two ends of the pipeline both land in
"correctly detected, nowhere to go". A taint is terminal, and the recovery is a
restart, which is what Phase 6 then priced at ~30 s.

## What Phase 4D does *not* prove

- **It does not bound the collective.** Nothing here shows what happens when the
  peer dies *inside* a long broadcast. The 125M window is too narrow to hit
  reliably, and a kill that lands in the wrong place would measure the HTTP tail
  instead. Phase 6's 8B deep kill is the measurement, and its result is not a
  fast failure.
- **One kill point.** Only the `UPDATING` entry, only `SIGKILL`, only at 125M.
- **No `INVALIDATING`/`VALIDATING`/`RESUMING` kill.** The three states after the
  update are not exercised; `VALIDATING` is deliberately last-before-resume for
  safety reasons, and a kill there is a different (and interesting) case.
- **`opt-125m`, one node, TP=1.**

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_updatekill_2gpu.py
```
