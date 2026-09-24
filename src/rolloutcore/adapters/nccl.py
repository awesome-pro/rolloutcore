# SPDX-License-Identifier: Apache-2.0
"""NCCL weight-transfer driver: the trainer-side half of a real update.

This is the implementation of the seam described in
``rolloutcore.weight_transfer``. It wraps exactly what vLLM already provides --
``WeightTransferTrainerFactory.trainer_init`` and ``send_weights`` -- and adds
the two things RolloutCore needs that upstream does not do:

1. **A version on the finalize.** Upstream's trainer engine calls
   ``client.finish_weight_update()`` with no version
   (``vllm/distributed/weight_transfer/nccl_engine.py:361``), so the engine keeps
   reporting its *old* label and RolloutCore's VALIDATING state would taint on
   every otherwise-successful transfer. :class:`RolloutCoreWeightSyncClient`
   substitutes the pending target's label, which is the controlled update the
   review asked for -- not a ``/update_weight_version`` reconciliation afterwards
   (that would publish a version whose weights we never proved).
2. **An identity check before mutation.** The identity of what will actually be
   staged is computed from ``source.metadata()`` and compared with
   ``target.identity`` *before* any request is sent. A mismatch raises with the
   engine untouched, which is why the error is an ``InvariantViolation`` rather
   than a taint: nothing happened, so there is nothing ambiguous to taint about.

vLLM is imported lazily, inside the default client/builder, so the package still
imports (and the test suite still runs) with vLLM absent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from ..errors import WeightIdentityMismatchError, WeightTransferNotConfiguredError
from ..versions import ParamSpec, UpdateTarget, WeightIdentity, WeightProvenance
from ..weight_transfer import WeightTransferInit, WeightTransferReport


class WeightSyncClientProtocol(Protocol):
    """The four control-plane calls upstream's trainer engine makes.

    Structural on purpose: ``HTTPVLLMWeightSyncClient`` (``clients.py:58``) and
    ``RayVLLMWeightSyncClient`` (``:96``) both satisfy it, and so does the
    wrapper below.
    """

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None: ...
    def start_weight_update(self) -> None: ...
    def update_weights(self, update_info: Any) -> None: ...
    def finish_weight_update(self, weight_version: str | None = None) -> None: ...


class TrainerEngineProtocol(Protocol):
    """The part of upstream's trainer engine the driver drives."""

    def send_weights(self) -> None: ...
    def shutdown(self) -> None: ...


class ManifestSource(Protocol):
    """The part of vLLM's ``WeightSource`` the driver needs.

    ``metadata()`` returns objects with ``name``/``dtype``/``shape``
    (``vllm.distributed.weight_transfer.base.ParamMeta``, ``base.py:120-125``).
    """

    def metadata(self) -> Sequence[Any]: ...


class RolloutCoreWeightSyncClient:
    """Wrap a vLLM weight-sync client and own the version at finalize.

    Every call delegates except :meth:`finish_weight_update`, where an explicit
    version still wins but upstream's ``None`` becomes the target RolloutCore
    opened this update with. The target is set by the driver immediately before
    ``send_weights()``, so the label written is the one VALIDATING will compare
    against.
    """

    def __init__(self, inner: WeightSyncClientProtocol) -> None:
        self._inner = inner
        self._pending_label: str | None = None

    @property
    def pending_label(self) -> str | None:
        return self._pending_label

    def set_target(self, target: UpdateTarget) -> None:
        self._pending_label = target.label

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._inner.init_weight_transfer_engine(init_info)

    def start_weight_update(self) -> None:
        self._inner.start_weight_update()

    def update_weights(self, update_info: Any) -> None:
        self._inner.update_weights(update_info)

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        label = weight_version or self._pending_label
        if label is None:
            raise WeightTransferNotConfiguredError(
                "finish_weight_update",
                "no target was set on the sync client, so the version to commit "
                "is unknown; the driver must call set_target() before send_weights()",
            )
        self._inner.finish_weight_update(weight_version=label)


def dtype_name(dtype: Any) -> str:
    """Normalise a parameter dtype to the string our manifest digest uses.

    vLLM reports ``torch.dtype`` objects; ``WeightIdentity`` hashes
    ``"bfloat16"``-style names (see the Phase 3A smoke, which computes the
    bootstrap identity from ``transformers`` the same way). Both sides must
    normalise identically or the digests differ for reasons unrelated to weights.
    """
    return str(dtype).replace("torch.", "")


def param_spec(meta: Any) -> ParamSpec:
    """``vLLM``'s ``ParamMeta`` -> our :class:`~rolloutcore.versions.ParamSpec`."""
    return ParamSpec(str(meta.name), dtype_name(meta.dtype), tuple(int(d) for d in meta.shape))


def _default_client_factory(base_url: str) -> WeightSyncClientProtocol:
    try:
        from vllm.distributed.weight_transfer import (  # type: ignore[import-not-found]
            HTTPVLLMWeightSyncClient,
        )
    except ImportError as exc:  # pragma: no cover - exercised only with vLLM absent
        raise WeightTransferNotConfiguredError(
            "initialize",
            f"vLLM is not importable ({exc}); the NCCL driver needs it installed",
        ) from exc
    client: WeightSyncClientProtocol = HTTPVLLMWeightSyncClient(base_url)
    return client


def _default_engine_builder(
    *,
    client: WeightSyncClientProtocol,
    trainer_init_info: Any,
    source: Any,
) -> TrainerEngineProtocol:
    try:
        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
    except ImportError as exc:  # pragma: no cover - exercised only with vLLM absent
        raise WeightTransferNotConfiguredError(
            "initialize",
            f"vLLM is not importable ({exc}); the NCCL driver needs it installed",
        ) from exc
    return WeightTransferTrainerFactory.trainer_init(  # type: ignore[no-any-return]
        init_info=trainer_init_info, client=client, source=source
    )


@dataclass
class NCCLWeightTransferDriver:
    """Drive one real weight update over NCCL, for one :class:`UpdateTarget`.

    ``trainer_init_info`` is upstream's ``NCCLTrainerInitInfo`` (rendezvous
    address, port, ``world_size``, ``rank``) and ``source`` is upstream's
    ``WeightSource`` (usually ``ModuleSource(model)``). Both are passed through
    untouched: this driver never builds NCCL itself.

    ``client`` and ``builder`` exist so the driver can be tested without vLLM
    installed; production code leaves them ``None`` and gets the upstream
    implementations lazily.
    """

    base_url: str
    trainer_init_info: Any
    source: ManifestSource
    #: Declared provenance folded into the identity (checkpoint/run/step).
    provenance: WeightProvenance | None = None
    #: Transfer-group size, for the record: 1 trainer + N inference workers.
    world_size: int | None = None
    client: WeightSyncClientProtocol | None = None
    builder: Callable[..., TrainerEngineProtocol] = _default_engine_builder

    name: ClassVar[str] = "nccl"
    moves_tensors: ClassVar[bool] = True

    _engine: TrainerEngineProtocol | None = field(default=None, init=False, repr=False)
    _sync: RolloutCoreWeightSyncClient | None = field(default=None, init=False, repr=False)
    _identity: WeightIdentity | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ identity

    def identity(self) -> WeightIdentity:
        """The identity of the tensors this driver would stage.

        Computed from the source's own manifest, not taken from the target: that
        independence is what makes the comparison in :meth:`transfer` evidence
        rather than a tautology. Offline -- no rendezvous, no request.
        """
        if self._identity is None:
            specs = [param_spec(meta) for meta in self.source.metadata()]
            self._identity = WeightIdentity.from_param_specs(specs, source=self.provenance)
        return self._identity

    # -------------------------------------------------------------- the driver

    def initialize(self) -> WeightTransferInit:
        """Rendezvous: open the trainer endpoint and initialize the engine.

        Upstream does both inside ``trainer_init`` because the workers block in
        ``/init_weight_transfer_engine`` until the group forms
        (``nccl_engine.py:272-318``); this call therefore inherits that blocking
        behaviour and must not be run concurrently with itself.
        """
        inner = self.client if self.client is not None else _default_client_factory(self.base_url)
        self._sync = RolloutCoreWeightSyncClient(inner)
        self._engine = self.builder(
            client=self._sync, trainer_init_info=self.trainer_init_info, source=self.source
        )
        return WeightTransferInit(
            driver=self.name,
            backend=self.name,
            initialised=True,
            world_size=self.world_size,
            note="trainer endpoint open; version handshake is owned by the wrapper client",
        )

    def transfer(self, target: UpdateTarget) -> WeightTransferReport:
        """Verify, then push. Raises before any request if the identity differs."""
        if self._engine is None or self._sync is None:
            raise WeightTransferNotConfiguredError(
                "transfer", "initialize() must run before transfer()"
            )
        computed = self.identity()
        if computed != target.identity:
            raise WeightIdentityMismatchError(expected=target.identity, computed=computed)

        self._sync.set_target(target)
        # Upstream drives /start_weight_update, /update_weights and
        # /finish_weight_update itself, interleaved with the broadcast. It raises
        # on failure, which the runner treats as an ambiguous mutating failure and
        # taints -- correct, because a half-broadcast is unprovable.
        self._engine.send_weights()
        return WeightTransferReport(
            finish_acknowledged=True,
            data_plane_complete=True,
            # Upstream exposes no trustworthy chunk count; do not invent one.
            chunks_transferred=None,
            observed_identity=computed,
            note=f"send_weights() returned for {target.label}",
        )

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
