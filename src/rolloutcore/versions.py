# SPDX-License-Identifier: Apache-2.0
"""Version identity and weight identity.

Two distinct concepts live here, and keeping them apart is the point:

``WeightVersion``
    A **generation number** used for *lifecycle ordering*. Serialised into the
    engine's ``weight_version`` as ``rc-<n>``. It answers "which update came
    first?", nothing more. Two different weight sets can legitimately share a
    generation number if a controller is rebuilt, and a rebuild can reuse a
    number with different contents.

``WeightIdentity``
    An **immutable content digest** over the parameter manifest. It answers
    "which exact tensors were these?". Required for replay/provenance: without
    it, invariant I4 could only claim "the generation that produced this", not
    "the exact weights that produced this".

vLLM's own ``weight_version`` is an opaque, caller-supplied string that the
engine never increments (``vllm/v1/engine/core.py:137`` initialises it to the
literal ``"default"``; ``:1043`` assigns verbatim). Both of the above are
therefore RolloutCore's to own.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

#: Prefix for every label RolloutCore writes into the engine's ``weight_version``.
#: Chosen so an engine that has never been driven by RolloutCore is trivially
#: distinguishable: a fresh engine reports ``"default"``, never ``"rc-0"``.
ENGINE_LABEL_PREFIX = "rc-"

#: The version a correctly bootstrapped engine must report before first use.
INITIAL_VERSION = 0

#: Prefix for weight-identity digests.
DIGEST_PREFIX = "sha256:"

#: Field separator in the canonical manifest encoding. ASCII unit separator, so
#: it cannot collide with a parameter name or dtype.
_SEP = "\x1f"
_LINE = "\n"


class VersionError(ValueError):
    """A weight version was constructed or parsed illegally."""


class WeightIdentityError(ValueError):
    """A weight identity was constructed or parsed illegally."""


@dataclass(frozen=True, order=True, slots=True)
class WeightVersion:
    """A monotonically increasing generation number owned by RolloutCore.

    Ordering is by ``value``, so ``WeightVersion(3) < WeightVersion(4)``.

    Deliberately *not* an int subclass: making it a distinct type means the
    engine's string label and RolloutCore's counter cannot be silently
    interchanged, which is the bug class RFC #48306 section 2.2 is about.

    This is a **generation number, not a weight identity**. See
    :class:`WeightIdentity` and :class:`UpdateTarget`.
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
        """Prefix-cache isolation token for requests bound to this generation.

        NOT universal cache versioning, and NOT a content identity -- see
        ``docs/state-machine.md`` section 6. This value reaches
        ``generate_block_hash_extra_keys``
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


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One parameter's identity-relevant metadata.

    Mirrors ``vllm.distributed.weight_transfer.base.ParamMeta`` (``name``,
    ``dtype``, ``shape``), so a weight identity can be computed directly from a
    trainer-side ``WeightSource.metadata()`` without a translation layer.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise WeightIdentityError("param name must be a non-empty string")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise WeightIdentityError(f"param {self.name!r}: dtype must be a non-empty string")
        if not isinstance(self.shape, tuple):
            raise WeightIdentityError(f"param {self.name!r}: shape must be a tuple")
        for dim in self.shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise WeightIdentityError(
                    f"param {self.name!r}: shape entries must be non-negative ints, got {dim!r}"
                )

    def canonical(self) -> str:
        dims = ",".join(str(d) for d in self.shape)
        return f"{self.name}{_SEP}{self.dtype}{_SEP}{dims}"


@dataclass(frozen=True, slots=True)
class WeightIdentity:
    """An immutable content digest identifying exact weight tensors.

    The digest covers the *manifest* -- parameter names, dtypes and shapes --
    not the tensor bytes. That is a deliberate, documented limit: it is cheap
    (computable from ``ParamMeta`` alone, before any transfer), it catches
    accidental re-publication of the wrong checkpoint, and it is stable across
    process restarts. It will *not* catch a same-named, same-shaped checkpoint
    with different values; closing that requires hashing tensor contents, which
    is out of scope for V1 and noted as a limitation rather than implied away.
    """

    digest: str
    num_tensors: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not self.digest.startswith(DIGEST_PREFIX):
            raise WeightIdentityError(
                f"weight identity digest must start with {DIGEST_PREFIX!r}, got {self.digest!r}"
            )
        hexpart = self.digest[len(DIGEST_PREFIX) :]
        if len(hexpart) != 64 or any(c not in "0123456789abcdef" for c in hexpart):
            raise WeightIdentityError(
                f"weight identity digest must be 64 lowercase hex chars, got {self.digest!r}"
            )
        if self.num_tensors is not None and self.num_tensors < 0:
            raise WeightIdentityError("num_tensors must be >= 0")

    @property
    def short(self) -> str:
        """First 12 hex chars, for logs and trajectory metadata."""
        return self.digest[len(DIGEST_PREFIX) :][:12]

    @classmethod
    def from_param_specs(cls, specs: Iterable[ParamSpec]) -> WeightIdentity:
        """Compute an identity from a parameter manifest.

        The encoding is canonical: specs are sorted by name and each is rendered
        as ``name\\x1fdtype\\x1fdim,dim,...``. Sorting makes the digest
        independent of iteration order, which matters because a trainer-side
        ``WeightSource`` need not enumerate in a stable order.
        """
        ordered = sorted(specs, key=lambda s: s.name)
        names = [s.name for s in ordered]
        if len(set(names)) != len(names):
            raise WeightIdentityError("duplicate parameter names in manifest")

        hasher = hashlib.sha256()
        for spec in ordered:
            hasher.update(spec.canonical().encode("utf-8"))
            hasher.update(_LINE.encode("utf-8"))
        return cls(digest=f"{DIGEST_PREFIX}{hasher.hexdigest()}", num_tensors=len(ordered))

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str, tuple[int, ...]]]) -> WeightIdentity:
        """Convenience wrapper: ``(name, dtype, shape)`` triples."""
        return cls.from_param_specs(ParamSpec(*p) for p in pairs)

    @classmethod
    def parse(cls, digest: str) -> WeightIdentity:
        return cls(digest=digest)


@dataclass(frozen=True, slots=True)
class UpdateTarget:
    """What an update installs: a generation number *and* a weight identity.

    ``begin_update`` takes one of these rather than a bare ``WeightVersion`` so
    that a caller cannot open an update without saying which weights it intends
    to install. The separation matters: the generation number orders the
    lifecycle, the identity is what a trajectory is allowed to claim.
    """

    version: WeightVersion
    identity: WeightIdentity

    @property
    def label(self) -> str:
        return self.version.label

    def describe(self) -> str:
        return f"{self.version.label} ({self.identity.short})"
