# SPDX-License-Identifier: Apache-2.0
"""Weight-version identity.

vLLM's ``weight_version`` is an *opaque, caller-supplied string* that the engine
never increments (``vllm/v1/engine/core.py:137`` initialises it to the literal
``"default"``; ``:1043`` assigns verbatim; there is no ``+=`` anywhere). The
engine also does not bind it to requests or responses -- PR #49040 deliberately
removed that, because one request may span multiple versions.

RolloutCore therefore owns the counter and serialises it into the label it hands
to the engine. This module is the single definition of that mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Prefix for every label RolloutCore writes into the engine's ``weight_version``.
#: Chosen so an engine that has never been driven by RolloutCore is trivially
#: distinguishable: a fresh engine reports ``"default"``, never ``"rc-0"``.
ENGINE_LABEL_PREFIX = "rc-"

#: The version a correctly bootstrapped engine must report before first use.
INITIAL_VERSION = 0


class VersionError(ValueError):
    """A weight version was constructed or parsed illegally."""


@dataclass(frozen=True, order=True, slots=True)
class WeightVersion:
    """A monotonically increasing, RolloutCore-owned weight version.

    Ordering is by ``value``, so ``WeightVersion(3) < WeightVersion(4)``.

    Deliberately *not* an int subclass: making it a distinct type means the
    engine's string label and RolloutCore's int counter cannot be silently
    interchanged, which is the bug class RFC #48306 section 2.2 is about.
    """

    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise VersionError(f"version value must be an int, got {type(self.value).__name__}")
        if self.value < 0:
            raise VersionError(f"version value must be >= 0, got {self.value}")

    @property
    def label(self) -> str:
        """The string written to / read from the engine's ``weight_version``."""
        return f"{ENGINE_LABEL_PREFIX}{self.value}"

    @property
    def cache_salt(self) -> str:
        """Prefix-cache isolation token for requests bound to this version.

        NOT universal cache versioning -- see ``docs/state-machine.md``. This
        value reaches ``generate_block_hash_extra_keys``
        (``vllm/v1/core/kv_cache_utils.py:632-634``) and therefore isolates the
        **prefix KV cache only**. It does *not* isolate the encoder cache, the
        multimodal processor cache, or LoRA adapter caches; those are covered by
        the explicit INVALIDATING step.
        """
        return self.label

    def next(self) -> WeightVersion:
        return WeightVersion(self.value + 1)

    @classmethod
    def parse_label(cls, label: str) -> WeightVersion:
        """Parse an engine label back into a version.

        Raises ``VersionError`` for anything not produced by this class, which
        is how "the engine is reporting a version we did not write" becomes a
        hard, typed failure rather than a silent mismatch.
        """
        if not isinstance(label, str) or not label.startswith(ENGINE_LABEL_PREFIX):
            raise VersionError(
                f"engine label {label!r} was not written by RolloutCore "
                f"(expected prefix {ENGINE_LABEL_PREFIX!r})"
            )
        suffix = label[len(ENGINE_LABEL_PREFIX) :]
        if not suffix.isdigit():
            raise VersionError(f"engine label {label!r} has a non-numeric version suffix")
        return cls(int(suffix))
