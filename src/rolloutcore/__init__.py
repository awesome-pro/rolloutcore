# SPDX-License-Identifier: Apache-2.0
"""RolloutCore -- a versioned RL rollout runtime for vLLM.

* :mod:`rolloutcore.lifecycle` -- the pure state machine (Phase 1).
* :mod:`rolloutcore.port` -- the typed adapter interface (Phase 2 boundary).
* :mod:`rolloutcore.runner` -- drives a cycle using typed adapter methods.
* :mod:`rolloutcore.adapters` -- fake and HTTP implementations of the port.

See ``docs/state-machine.md`` for the design and ``docs/plan-delta.md`` for how
this relates to the original plan and to what vLLM main actually provides.
"""

from .errors import (
    DrainDisagreementError,
    EngineTaintedError,
    EvidenceNotReady,
    IllegalTransitionError,
    InvariantViolation,
    NotServingError,
    RolloutCoreError,
    UnknownRolloutError,
    VersionMismatchError,
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
from .versions import (
    DIGEST_PREFIX,
    ENGINE_LABEL_PREFIX,
    INITIAL_VERSION,
    ParamSpec,
    UpdateTarget,
    VersionError,
    WeightIdentity,
    WeightIdentityError,
    WeightVersion,
)

__all__ = [
    "ADMITTING_STATES",
    "DIGEST_PREFIX",
    "ENGINE_LABEL_PREFIX",
    "INITIAL_VERSION",
    "LEGAL_TRANSITIONS",
    "PAUSED_STATES",
    "ROLLOUT_COMPLETION_STATES",
    "BootstrapEvidence",
    "CyclePlan",
    "CycleResult",
    "DrainDisagreementError",
    "DrainEvidence",
    "EngineTaintedError",
    "Event",
    "Evidence",
    "EvidenceNotReady",
    "IllegalTransitionError",
    "InvalidateEvidence",
    "InvariantViolation",
    "LifecycleAdapter",
    "LifecycleController",
    "LifecycleRunner",
    "LifecycleState",
    "NotServingError",
    "OrphanedRollout",
    "ParamSpec",
    "ResumeEvidence",
    "RolloutBinding",
    "RolloutCoreError",
    "TransitionRecord",
    "UnknownRolloutError",
    "UpdateEvidence",
    "UpdateTarget",
    "ValidateEvidence",
    "VersionError",
    "VersionMismatchError",
    "WeightIdentity",
    "WeightIdentityError",
    "WeightVersion",
    "enumerate_transition_matrix",
]
