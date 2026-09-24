# SPDX-License-Identifier: Apache-2.0
"""RolloutCore -- a versioned RL rollout runtime for vLLM.

Phase 1: the pure-Python lifecycle state machine. No vLLM import, no I/O, no
GPU. See ``docs/state-machine.md`` for the design and ``docs/plan-delta.md`` for
how this relates to the original plan and to what vLLM main actually provides.
"""

from .errors import (
    EngineTaintedError,
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
    TransitionRecord,
    UpdateEvidence,
    ValidateEvidence,
)
from .lifecycle import (
    ADMITTING_STATES,
    LEGAL_TRANSITIONS,
    ROLLOUT_COMPLETION_STATES,
    Event,
    LifecycleController,
    LifecycleState,
    RolloutBinding,
    enumerate_transition_matrix,
)
from .versions import (
    ENGINE_LABEL_PREFIX,
    INITIAL_VERSION,
    WeightVersion,
    VersionError,
)

__all__ = [
    "ADMITTING_STATES",
    "BootstrapEvidence",
    "CyclePlan",
    "DrainEvidence",
    "ENGINE_LABEL_PREFIX",
    "EngineTaintedError",
    "Event",
    "Evidence",
    "INITIAL_VERSION",
    "IllegalTransitionError",
    "InvalidateEvidence",
    "InvariantViolation",
    "LEGAL_TRANSITIONS",
    "LifecycleController",
    "LifecycleState",
    "NotServingError",
    "ROLLOUT_COMPLETION_STATES",
    "RolloutBinding",
    "RolloutCoreError",
    "TransitionRecord",
    "UnknownRolloutError",
    "UpdateEvidence",
    "ValidateEvidence",
    "VersionError",
    "VersionMismatchError",
    "WeightVersion",
    "enumerate_transition_matrix",
]
