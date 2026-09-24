# SPDX-License-Identifier: Apache-2.0
"""Replay validation: did the committed weight source reproduce its own tokens?

A :class:`~rolloutcore.trajectory.Trajectory` claims that specific tokens came
from a specific declared weight source. A replay is the only way to test that
claim after the fact: re-run the record's prompt and compare. **Two modes, and a
token mismatch means something different in each** -- conflating them is how a
harness bug gets read as a weight finding.

* **Trace-forced** (:attr:`ReplayObservation.tokens_were_forced` is true). vLLM's
  ``SamplingParams.trace_decode_token_ids`` overwrites the sampled token at every
  decode step with a predetermined id while still computing real logprobs. The
  engine is *told* what to emit, so it should echo the sequence exactly. A
  mismatch here is an engine or harness fault -- the request was misconfigured, or
  the loop diverged -- and says nothing about the weights. The validator reports
  it as a protocol violation (``"echo: ..."``) rather than as a finding about the
  weight source.
* **Greedy** (the default). The recorded tokens are an *expectation*: with
  ``temperature=0`` the engine should produce them again, but nothing forces it
  to. A mismatch is the finding, because either the weights are not what the
  record declares or the decode is not reproducible.

Only the greedy mode can attribute a mismatch to the weights, and only the
trace-forced mode can attribute one to the harness -- so which mode ran is
recorded per observation and reported in the verdict.

Two honest limits, stated here rather than discovered later.

**Agreement is evidence, not proof.** A record that replays exactly is consistent
with the identity it carries; a record that does not is *inconsistent* with it.
The asymmetry is deliberate and is the whole reason the validator is worth
running: a mismatch is the strong direction. It falsifies the claim outright,
while agreement only fails to falsify it. Nothing here proves that a checkpoint
on disk is the checkpoint that ran -- that would need content hashing of the
tensors, which :class:`~rolloutcore.versions.WeightIdentity` explicitly does not do.

**Logprob equality is approximate by construction.** The default tolerance is
``1e-3``, chosen because bf16 kernels are not bit-reproducible across batch shapes
and only an SM90+ card with ``VLLM_BATCH_INVARIANT=1``
(``examples/rl/rlhf_async_new_apis.py:177``) can hope to make an exact comparison
meaningful. Below the tolerance the deltas are noise; above it they are a
divergence. Token mismatches have no tolerance -- an id either matches or it does
not.

The validator itself is a pure function over two immutable records, so it needs
no engine, no GPU and no network, and can be tested exhaustively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .trajectory import Trajectory

#: ``|Δ logprob|`` at or below this is treated as agreement. See the module docs
#: for why an exact comparison is not available without ``VLLM_BATCH_INVARIANT``.
DEFAULT_LOGPROB_TOLERANCE = 1e-3

_LANE_COMPARED = "compared"
_LANE_RECORD_MISSING = "record-missing"
_LANE_REPLAY_MISSING = "replay-missing"
_LANE_BOTH_MISSING = "both-missing"


class ReplayError(ValueError):
    """A replay that cannot be made meaningful."""


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """What to ask a replayed engine for, derived from one trajectory record.

    ``expected_token_ids`` is what the record says came out. What the *replay*
    does with them depends on the mode the caller chose, and the plan cannot tell
    which one ran -- ``ReplayObservation.tokens_were_forced`` carries that:

    * trace-forced: these ids are handed to the engine as a script, so they must
      come back unchanged. A token mismatch is then a protocol violation, not a
      weight finding.
    * greedy: these ids are the expectation the weights have to reproduce. A
      mismatch is the finding.

    An empty tuple means nothing is pinned either way, which is representable so a
    caller can say so explicitly; :meth:`from_trajectory` never produces it from a
    record that has tokens.
    """

    request_id: str
    prompt: str
    expected_token_ids: tuple[int, ...]
    version: str
    identity_digest: str
    cache_salt: str
    recorded_logprobs: tuple[float, ...] = ()

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> ReplayPlan:
        """Build a plan, or refuse if the record cannot support a replay claim.

        A ``manifest-only`` identity identifies an architecture, not a training
        step: Phase 3C measured the same digest for dummy-initialised weights and
        the real ones, and Phase 4C measured it surviving a corrupted checkpoint.
        Replaying against such a record and reporting "agreement" would be
        attributing the result to weights nobody identified.
        """
        if not trajectory.replay_ready:
            raise ReplayError(
                f"trajectory {trajectory.request_id!r} carries a manifest-only identity "
                f"({trajectory.identity.short}), which cannot separate two checkpoints of "
                f"the same architecture, so a replay could not be attributed to any "
                f"particular weight source; record a declared provenance "
                f"(checkpoint/run_id/step) before replaying"
            )
        return cls(
            request_id=trajectory.request_id,
            prompt=trajectory.prompt,
            expected_token_ids=tuple(trajectory.token_ids),
            version=trajectory.version,
            identity_digest=trajectory.identity.digest,
            cache_salt=trajectory.cache_salt,
            recorded_logprobs=tuple(trajectory.logprobs),
        )

    @property
    def has_expected_tokens(self) -> bool:
        """Whether the record pins any token at all, in either mode."""
        return bool(self.expected_token_ids)

    @property
    def max_tokens(self) -> int:
        """How many decode steps the replay must run: one per expected token."""
        return len(self.expected_token_ids)

    def describe(self) -> str:
        logprobs = (
            f"{len(self.recorded_logprobs)} recorded logprobs"
            if self.recorded_logprobs
            else "no recorded logprobs (token lane only)"
        )
        return (
            f"{self.request_id} {self.version} {self.identity_digest.split(':', 1)[-1][:12]} "
            f"expected {self.max_tokens} tokens, {logprobs}"
        )


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """What a replay run actually returned.

    ``version`` is the engine label *at replay time*, which is the label the
    engine reports when asked -- the same opaque string a trajectory binds at
    admission. ``identity_digest`` and ``prompt`` are optional because a bare
    completion API returns neither; when they are supplied the validator uses
    them, and when they are absent it says so rather than assuming they match.

    ``tokens_were_forced`` records which reproduction mode produced this
    observation: true for an in-process replay that passed
    ``trace_decode_token_ids``, false for a greedy request that merely expected
    the recorded tokens. It selects how a token mismatch is interpreted -- see
    :class:`ReplayPlan`.
    """

    request_id: str
    version: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] = ()
    identity_digest: str | None = None
    prompt: str | None = None
    tokens_were_forced: bool = False


@dataclass(frozen=True, slots=True)
class ReplayVerdict:
    """The outcome of one comparison, with every number the decision rested on."""

    request_id: str
    agrees: bool
    protocol: tuple[str, ...]
    token_mismatches: int
    first_divergence: int | None
    shared_prefix: int
    logprob_lane: str
    compared_logprobs: int
    mean_abs_delta: float | None
    max_abs_delta: float | None
    p99_abs_delta: float | None
    worst_index: int | None
    tolerance: float
    notes: tuple[str, ...]

    def describe(self) -> str:
        """One line, in the same voice as :meth:`Trajectory.describe`."""
        if self.logprob_lane == _LANE_COMPARED and self.max_abs_delta is not None:
            logprob = f"max|Δ|={self.max_abs_delta:.3e} over {self.compared_logprobs} logprobs"
        else:
            logprob = f"token-only ({self.logprob_lane})"
        divergence = (
            ""
            if self.first_divergence is None
            else f", first divergence at {self.first_divergence}"
        )
        # A protocol violation is why a comparison with matching tokens and a zero
        # logprob delta still disagrees, so it must be visible in the one line.
        protocol = "" if not self.protocol else f", {len(self.protocol)} protocol violation(s)"
        return (
            f"{self.request_id} {'AGREE' if self.agrees else 'DISAGREE'}: "
            f"shared prefix {self.shared_prefix}, {self.token_mismatches} token mismatch(es)"
            f"{divergence}, {logprob}{protocol}"
        )


def validate_replay(
    plan: ReplayPlan,
    observation: ReplayObservation,
    *,
    logprob_tolerance: float = DEFAULT_LOGPROB_TOLERANCE,
) -> ReplayVerdict:
    """Compare a replay against its plan. Pure: no I/O, no engine, no GPU.

    The caller pairs the two by ``request_id``; this function does not check that
    they agree, because a mismatch there is a caller bug rather than a finding
    about the engine -- the CLI matches by ``request_id`` by construction.

    Only the expected length is compared, so an engine that kept sampling past it
    is not a protocol violation -- but agreement still requires the lengths to
    match, because extra tokens mean the replay was not run the way the record
    describes.

    A token mismatch is reported two ways depending on
    :attr:`ReplayObservation.tokens_were_forced`: as an ``"echo"`` protocol
    violation when the engine was told what to emit (so the fault is not the
    weights), and as an ordinary finding -- ``token_mismatches`` and
    ``first_divergence`` -- when the tokens were only expected.
    """
    protocol: list[str] = []
    notes: list[str] = []

    if observation.version != plan.version:
        protocol.append(
            f"version: the record was bound to {plan.version} but the replay ran on "
            f"{observation.version}; a comparison across weight versions is meaningless"
        )
    if (
        observation.identity_digest is not None
        and observation.identity_digest != plan.identity_digest
    ):
        protocol.append(
            f"identity: the record declares {plan.identity_digest} but the replay reports "
            f"{observation.identity_digest}; these are not the same weight source"
        )
    if observation.prompt is not None and observation.prompt != plan.prompt:
        protocol.append(
            f"prompt: the record replays {plan.prompt!r} but the observation reports "
            f"{observation.prompt!r}"
        )
    if len(observation.token_ids) < len(plan.expected_token_ids):
        protocol.append(
            f"early stop: the engine returned {len(observation.token_ids)} tokens for an "
            f"expected sequence of {len(plan.expected_token_ids)}"
        )

    # ------------------------------------------------------------ tokens
    compared = min(len(plan.expected_token_ids), len(observation.token_ids))
    expected = plan.expected_token_ids[:compared]
    observed = observation.token_ids[:compared]
    mismatch_indexes = [i for i in range(compared) if expected[i] != observed[i]]
    token_mismatches = len(mismatch_indexes)

    if mismatch_indexes:
        first_divergence: int | None = mismatch_indexes[0]
    elif len(plan.expected_token_ids) != len(observation.token_ids):
        # The lengths differ but everything compared matched: divergence is the
        # first position that was never compared.
        first_divergence = compared
    else:
        first_divergence = None
    shared_prefix = first_divergence if first_divergence is not None else compared

    if observation.tokens_were_forced:
        # The ids were dictated, so the engine failing to reproduce them is a
        # fault in the replay, not evidence about the weights. Say which it is
        # rather than letting a misconfigured request look like a weight finding.
        if token_mismatches:
            protocol.append(
                f"echo: the engine was told to emit exactly these token ids and did not -- "
                f"{token_mismatches} position(s) differ, the first at {first_divergence}; "
                f"the replay is faulty and says nothing about the weights"
            )
        elif len(observation.token_ids) != len(plan.expected_token_ids):
            protocol.append(
                f"echo: the engine was told to emit exactly {len(plan.expected_token_ids)} "
                f"token ids and returned {len(observation.token_ids)}; the replay is faulty "
                f"and says nothing about the weights"
            )
    elif token_mismatches:
        notes.append(
            f"greedy replay: the tokens were expected, not forced, so the "
            f"{token_mismatches} mismatch(es) are the finding -- the weights did not "
            f"reproduce their own argmax"
        )

    if len(observation.token_ids) > len(plan.expected_token_ids):
        notes.append(
            f"the replay returned {len(observation.token_ids) - compared} token(s) beyond the "
            f"expected sequence; they were ignored, and agreement withholds because the replay "
            f"was not run the way the record describes"
        )

    # ----------------------------------------------------------- logprobs
    # "Covering every compared position" -- an empty side covers nothing, so a
    # token-only record and a token-only replay land in `both-missing` rather than
    # claiming a vacuous comparison.
    record_covers = bool(plan.recorded_logprobs) and len(plan.recorded_logprobs) >= compared
    observed_covers = bool(observation.logprobs) and len(observation.logprobs) >= compared
    if record_covers and observed_covers:
        logprob_lane = _LANE_COMPARED
    elif not record_covers and not observed_covers:
        logprob_lane = _LANE_BOTH_MISSING
    elif record_covers:
        logprob_lane = _LANE_REPLAY_MISSING
    else:
        logprob_lane = _LANE_RECORD_MISSING

    mean_abs_delta: float | None = None
    max_abs_delta: float | None = None
    p99_abs_delta: float | None = None
    worst_index: int | None = None
    compared_logprobs = 0

    if logprob_lane == _LANE_COMPARED:
        deltas = [abs(plan.recorded_logprobs[i] - observation.logprobs[i]) for i in range(compared)]
        compared_logprobs = len(deltas)
        if deltas:
            mean_abs_delta = sum(deltas) / len(deltas)
            max_abs_delta = max(deltas)
            worst_index = deltas.index(max_abs_delta)  # argmax, first occurrence
            ordered = sorted(deltas)
            # Nearest-rank percentile: the smallest value at or above the requested
            # rank, i.e. rank = ceil(p * n) with 1-based ranks, clamped into range.
            # No interpolation, so the figure is always one of the measured deltas.
            p99_abs_delta = ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]
    else:
        notes.append(f"logprobs were not compared ({logprob_lane}); the verdict is token-only")

    if (
        logprob_lane == _LANE_COMPARED
        and max_abs_delta is not None
        and max_abs_delta > logprob_tolerance
    ):
        notes.append(
            f"max |Δ logprob| {max_abs_delta:.6g} exceeds the tolerance "
            f"{logprob_tolerance:.6g} at position {worst_index}"
        )

    lengths_matched = len(plan.expected_token_ids) == len(observation.token_ids)
    logprobs_agree = logprob_lane != _LANE_COMPARED or (
        max_abs_delta is not None and max_abs_delta <= logprob_tolerance
    )
    agrees = not protocol and token_mismatches == 0 and lengths_matched and logprobs_agree

    return ReplayVerdict(
        request_id=plan.request_id,
        agrees=agrees,
        protocol=tuple(protocol),
        token_mismatches=token_mismatches,
        first_divergence=first_divergence,
        shared_prefix=shared_prefix,
        logprob_lane=logprob_lane,
        compared_logprobs=compared_logprobs,
        mean_abs_delta=mean_abs_delta,
        max_abs_delta=max_abs_delta,
        p99_abs_delta=p99_abs_delta,
        worst_index=worst_index,
        tolerance=logprob_tolerance,
        notes=tuple(notes),
    )
