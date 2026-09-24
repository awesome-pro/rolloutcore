# SPDX-License-Identifier: Apache-2.0
"""The trainer-side weight-transfer seam.

RolloutCore's lifecycle control plane (``/pause``, ``/reset_*``, ``/resume``)
and vLLM's weight *data* plane are different things, and this module keeps them
apart:

* :class:`~rolloutcore.port.LifecycleAdapter` owns the control plane: quiesce,
  invalidate, validate, resume.
* :class:`WeightTransferDriver` owns the trainer side: the NCCL rendezvous,
  ``/init_weight_transfer_engine``, and the tensor push.

Why the driver owns *more* than the tensor push
-----------------------------------------------

Upstream does not split the update round trip. The trainer-side engine built by
``WeightTransferTrainerFactory.trainer_init``
(``vllm/distributed/weight_transfer/factory.py:167``) is what posts
``/init_weight_transfer_engine``, and its ``send_weights()`` drives
``/start_weight_update`` -> ``/update_weights`` -> ``/finish_weight_update``
*concurrently* with the collective broadcast
(``vllm/distributed/weight_transfer/base.py:551-627``). The reference example
says so explicitly: "Drives start_weight_update / update_weights /
finish_weight_update, concurrent with the NCCL broadcast"
(``examples/rl/rlhf_http_nccl.py:193-196``).

So RolloutCore must not post those three endpoints itself *and* ask a driver to
push tensors: the metadata POST blocks while the workers receive, and the
broadcast has to be in flight for it to ever return. The driver owns the round
trip; the adapter owns the evidence.

What a real driver must supply
------------------------------

``NCCLWeightTransferDriver`` (Phase 3B, deliberately not implemented here -- see
its docstring) wraps ``WeightTransferTrainerFactory``:

* ``trainer_init(init_info=NCCLTrainerInitInfo(master_address=..., master_port=...,
  world_size=..., rank=0, packed=True), client=HTTPVLLMWeightSyncClient(base_url),
  source=ModuleSource(model))`` opens the trainer endpoint *and* initializes the
  inference side -- the two happen in a ``ThreadPoolExecutor`` because the
  workers block inside the init call until the rendezvous completes
  (``vllm/distributed/weight_transfer/nccl_engine.py:272-318``).
* the worker-side payload is not ``{"backend": "nccl"}``. ``NCCLWeightTransferInitInfo``
  (``vllm/distributed/weight_transfer/nccl_common.py:71``) requires
  ``rank_offset`` and ``world_size`` plus exactly one rendezvous mode:
  ``master_address`` + ``master_port``, or ``nccl_unique_id_b64``; and
  ``worker_init_payload`` (``nccl_common.py:117``) is what serialises it,
  dropping the unset rendezvous field.

That is why this module has no "build the payload" helper: the payload is a
trainer-side object with real rendezvous data, and anything this module could
construct from a URL would be a fabrication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .errors import WeightTransferNotConfiguredError
from .versions import UpdateTarget, WeightIdentity


@dataclass(frozen=True, slots=True)
class WeightTransferInit:
    """What the driver observed while initializing the transfer engine.

    Produced by :meth:`WeightTransferDriver.initialize`. The driver performs the
    ``/init_weight_transfer_engine`` POST itself (it must run concurrently with
    opening the trainer endpoint), so this is a report, not a request payload.
    """

    #: Name of the driver, e.g. ``"nccl"``.
    driver: str
    #: Engine-side backend key the driver initialized, e.g. ``"nccl"``.
    backend: str
    #: Did the inference-side initialization complete?
    initialised: bool
    #: ``world_size_across_dp`` the transfer group was sized to, if known.
    world_size: int | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class WeightTransferReport:
    """Trainer-side facts after one ``send_weights()`` round trip.

    ``observed_identity`` is the identity of the tensors the driver *actually
    staged*, computed from its own manifest (``WeightSource.metadata()`` ->
    :class:`~rolloutcore.versions.ParamSpec`). It is the one place where the
    declared target identity can be checked against something other than itself:
    a mismatch against ``UpdateTarget.identity`` means the wrong checkpoint was
    pushed, and ``UpdateEvidence`` taints on it.
    """

    #: Did ``finish_weight_update`` return success?
    finish_acknowledged: bool
    #: Did every chunk of the broadcast report completion?
    data_plane_complete: bool
    #: Number of ``/update_weights`` metadata chunks sent. Diagnostic only.
    chunks_transferred: int | None = None
    #: Manifest identity of what was staged, if the driver can compute one.
    observed_identity: WeightIdentity | None = None
    note: str = ""


@runtime_checkable
class WeightTransferDriver(Protocol):
    """Trainer-side half of a weight update.

    Implementations are responsible for the rendezvous, the
    ``/init_weight_transfer_engine`` handshake and the tensor push. They must
    report honestly: a driver that did not move tensors must not return
    ``data_plane_complete=True``, because that is exactly the claim the
    controller then publishes as a committed generation.
    """

    #: Short name, reported in ``BootstrapEvidence.weight_transfer_driver``.
    name: str

    #: Whether this driver can actually install weights. A driver that moves no
    #: tensors sets this to ``False``, and the adapter then refuses
    #: ``start_weight_update`` instead of sending a payload that only looks like a
    #: transfer. Declared rather than inferred so the capability is part of the
    #: contract instead of an ``isinstance`` check at the call site.
    moves_tensors: bool

    def initialize(self) -> WeightTransferInit | None:
        """Rendezvous and initialize the inference-side transfer engine.

        Returning ``None`` means "this deployment has no weight-transfer path":
        bootstrap proceeds without one, and any later attempt to install weights
        fails with :class:`WeightTransferNotConfiguredError`. That is the
        control-plane-only mode used to validate the lifecycle against a real
        ``vllm serve`` before real weight replacement exists.
        """
        ...

    def transfer(self, target: UpdateTarget) -> WeightTransferReport:
        """Run the update round trip for ``target`` and report what happened."""
        ...

    def shutdown(self) -> None:
        """Tear down communicators. Must be safe to call more than once."""
        ...


class LifecycleOnlyDriver:
    """The default driver: no tensors move, and it says so.

    Bootstraps a controller for the lifecycle control plane only (a real
    ``vllm serve`` driven through drain/cache-reset/validate/resume). It
    deliberately cannot install weights: an empty ``update_info`` is not a weight
    transfer, and pretending otherwise is what this class exists to prevent.
    """

    name = "lifecycle-only"
    moves_tensors = False

    def initialize(self) -> WeightTransferInit | None:
        return None

    def transfer(self, target: UpdateTarget) -> WeightTransferReport:
        raise WeightTransferNotConfiguredError(
            f"install {target.describe()}",
            "the lifecycle-only driver moves no tensors, so there is no honest "
            "UpdateEvidence to report. Supply a driving WeightTransferDriver "
            "(Phase 3B: NCCLWeightTransferDriver wrapping "
            "vllm.distributed.weight_transfer.WeightTransferTrainerFactory) to run "
            "a weight update; until then the engine stays paused.",
        )

    def shutdown(self) -> None:
        return None


class NCCLWeightTransferDriver:
    """Phase 3B landing zone. **Not implemented** -- and it fails loudly.

    The implementation is a wrapper around upstream's trainer side, not new NCCL
    code::

        from vllm.distributed.weight_transfer import (
            ModuleSource, NCCLTrainerInitInfo, WeightTransferTrainerFactory,
        )
        from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient

        engine = WeightTransferTrainerFactory.trainer_init(
            init_info=NCCLTrainerInitInfo(
                master_address=..., master_port=...,
                world_size=<1 trainer + get_world_size()>, rank=0, packed=True,
            ),
            client=HTTPVLLMWeightSyncClient(base_url),
            source=ModuleSource(train_model),
        )
        # initialize() -> the call above
        # transfer(target)   -> engine.send_weights()

    It is not written yet on purpose. Upstream's ``trainer_init`` both opens the
    trainer endpoint and initializes the workers, and ``send_weights`` interleaves
    the three control-plane POSTs with the collective; a version of this class
    that has never been executed against a GPU would be a guess dressed up as an
    implementation, which is precisely the "pretend" the review flagged.
    """

    name = "nccl"
    moves_tensors = True

    def __init__(
        self,
        *,
        base_url: str,
        trainer_init_info: object,
        source: object,
    ) -> None:
        self.base_url = base_url
        self.trainer_init_info = trainer_init_info
        self.source = source

    def initialize(self) -> WeightTransferInit | None:
        raise WeightTransferNotConfiguredError(
            "initialize the NCCL transfer engine",
            "NCCLWeightTransferDriver is a Phase 3B landing zone and is not "
            "implemented; wrapping WeightTransferTrainerFactory.trainer_init "
            "(vllm/distributed/weight_transfer/factory.py:167) requires a GPU "
            "and a real NCCLTrainerInitInfo rendezvous",
        )

    def transfer(self, target: UpdateTarget) -> WeightTransferReport:
        raise WeightTransferNotConfiguredError(
            f"install {target.describe()} over NCCL",
            "NCCLWeightTransferDriver is a Phase 3B landing zone and is not implemented",
        )

    def shutdown(self) -> None:
        return None
