# SPDX-License-Identifier: Apache-2.0
"""The adapter port: typed methods the controller's states map onto.

This is the boundary the controller/adapter split exists to create. States name
*what must be true*; these methods name *what to do about it*, in types.

Deliberately **not** a string-dispatch interface. ``CyclePlan.steps`` is
human-readable narration for logs and test output; an adapter that parsed it
would make the HTTP surface part of the controller's contract, and every future
transport change (gRPC, a Rust front-end, an in-process engine) would break the
state machine.

Each method is one vLLM operation group, and each returns the evidence the
corresponding transition checks:

============================  ==================================================
``bootstrap``                 ``POST /init_weight_transfer_engine``,
                              ``POST /update_weight_version``,
                              ``GET /weight_info``, ``GET /get_world_size``
``begin_drain``               ``POST /pause?mode=wait&clear_cache=true``
``await_drain``               waits for the above to return
``start_weight_update``       ``POST /start_weight_update``
``complete_weight_update``    ``POST /update_weights`` xN + ``POST /finish_weight_update``
``invalidate_caches``         ``POST /reset_prefix_cache`` + ``/reset_encoder_cache``
                              + ``/reset_mm_cache``
``validate_pre_resume``       ``GET /weight_info`` + ``GET /is_paused``
``resume``                    ``POST /resume`` + ``GET /is_paused``
============================  ==================================================

Note that the data plane is not here. Weights move over the trainer-side NCCL
engine, out of band; ``complete_weight_update`` covers only the control plane and
the acknowledgement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .evidence import (
    BootstrapEvidence,
    DrainEvidence,
    InvalidateEvidence,
    ResumeEvidence,
    UpdateEvidence,
    ValidateEvidence,
)
from .versions import UpdateTarget


@runtime_checkable
class LifecycleAdapter(Protocol):
    """Everything the controller needs from the outside world."""

    def bootstrap(self) -> BootstrapEvidence:
        """Bring a fresh engine under control and report what was observed.

        Must seed ``rc-0`` over an unmanaged label (V1 is single-owner) and
        report the pre-seed label in the evidence.
        """
        ...

    def begin_drain(self) -> None:
        """Issue the quiesce. Must not block until the drain completes.

        Returns as soon as the request is in flight so the caller can poll via
        :meth:`await_drain` -- a drain can legitimately take a long time, and
        the engine's own utility calls have no timeout
        (``vllm/v1/engine/core_client.py:996-1002``), so the waiting policy has
        to live on our side.
        """
        ...

    def await_drain(self) -> DrainEvidence:
        """Report on the in-flight drain without blocking indefinitely.

        Called repeatedly until it returns ``engine_drain_completed=True``. Each
        call should return promptly; a still-running drain reports
        ``engine_drain_completed=False`` rather than hanging.
        """
        ...

    def start_weight_update(self, target: UpdateTarget) -> None:
        """Open a weight-transfer session for ``target``."""
        ...

    def complete_weight_update(self, target: UpdateTarget) -> UpdateEvidence:
        """Transfer, load and finalize the weights; report the outcome."""
        ...

    def invalidate_caches(self, target: UpdateTarget) -> InvalidateEvidence:
        """Drop prefix, encoder and multimodal caches. Report each separately."""
        ...

    def validate_pre_resume(self, target: UpdateTarget) -> ValidateEvidence:
        """Read back version and pause state **without resuming**.

        Implementations must not call ``/resume`` here: the whole point of
        VALIDATING running before RESUMING is that a failure leaves the engine
        paused and therefore unable to serve unverified weights.
        """
        ...

    def resume(self, target: UpdateTarget) -> ResumeEvidence:
        """Issue the resume and confirm the engine is actually serving."""
        ...
