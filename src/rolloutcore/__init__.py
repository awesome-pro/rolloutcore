# SPDX-License-Identifier: Apache-2.0
"""RolloutCore -- a versioned RL rollout runtime for vLLM.

* :mod:`rolloutcore.lifecycle` -- the pure state machine (Phase 1).
* :mod:`rolloutcore.port` -- the typed adapter interface (Phase 2 boundary).
* :mod:`rolloutcore.runner` -- drives a cycle using typed adapter methods.
* :mod:`rolloutcore.weight_transfer` -- the trainer-side weight-transfer seam
  (a driver owns the NCCL rendezvous and the update round trip).
* :mod:`rolloutcore.adapters` -- fake and HTTP implementations of the port.

See ``docs/state-machine.md`` for the design and ``docs/plan-delta.md`` for how
this relates to the original plan and to what vLLM main actually provides.
"""

from .errors import (
    AlreadyManagedEngineError,
    DrainDisagreementError,
    DrainFailedError,
    EngineTaintedError,
    EvidenceNotReady,
    IllegalTransitionError,
    InvariantViolation,
    NotServingError,
    RolloutCoreError,
    UnknownRolloutError,
    VersionMismatchError,
    WeightIdentityMismatchError,
    WeightTransferNotConfiguredError,
)
from .evidence import (
    BootstrapEvidence,
    CyclePlan,
    DrainEvidence,
    Evidence,
    InvalidateEvidence,
    OrphanedRollout,
    ResumeEvidence,
    TransitionRecord,
    UpdateEvidence,
    ValidateEvidence,
)
from .lifecycle import (
    ADMITTING_STATES,
    LEGAL_TRANSITIONS,
    PAUSED_STATES,
    ROLLOUT_COMPLETION_STATES,
    Event,
    LifecycleController,
    LifecycleState,
    RolloutBinding,
    enumerate_transition_matrix,
)
from .port import LifecycleAdapter
from .runner import CycleResult, LifecycleRunner
from .trajectory import Trajectory, TrajectoryError, TrajectoryRecorder
from .versions import (
    DIGEST_PREFIX,
    ENGINE_LABEL_PREFIX,
    INITIAL_VERSION,
    ParamSpec,
    UpdateTarget,
    VersionError,
    WeightIdentity,
    WeightIdentityError,
    WeightProvenance,
    WeightVersion,
)
from .weight_transfer import (
    LifecycleOnlyDriver,
    NCCLWeightTransferDriver,
    WeightTransferDriver,
    WeightTransferInit,
    WeightTransferReport,
)

__all__ = [
    "ADMITTING_STATES",
    "DIGEST_PREFIX",
    "ENGINE_LABEL_PREFIX",
    "INITIAL_VERSION",
    "LEGAL_TRANSITIONS",
    "PAUSED_STATES",
    "ROLLOUT_COMPLETION_STATES",
    "AlreadyManagedEngineError",
    "BootstrapEvidence",
    "CyclePlan",
    "CycleResult",
    "DrainDisagreementError",
    "DrainEvidence",
    "DrainFailedError",
    "EngineTaintedError",
    "Event",
    "Evidence",
    "EvidenceNotReady",
    "IllegalTransitionError",
    "InvalidateEvidence",
    "InvariantViolation",
    "LifecycleAdapter",
    "LifecycleController",
    "LifecycleOnlyDriver",
    "LifecycleRunner",
    "LifecycleState",
    "NCCLWeightTransferDriver",
    "NotServingError",
    "OrphanedRollout",
    "ParamSpec",
    "ResumeEvidence",
    "RolloutBinding",
    "RolloutCoreError",
    "Trajectory",
    "TrajectoryError",
    "TrajectoryRecorder",
    "TransitionRecord",
    "UnknownRolloutError",
    "UpdateEvidence",
    "UpdateTarget",
    "ValidateEvidence",
    "VersionError",
    "VersionMismatchError",
    "WeightIdentity",
    "WeightIdentityError",
    "WeightIdentityMismatchError",
    "WeightProvenance",
    "WeightTransferDriver",
    "WeightTransferInit",
    "WeightTransferNotConfiguredError",
    "WeightTransferReport",
    "WeightVersion",
    "enumerate_transition_matrix",
]
