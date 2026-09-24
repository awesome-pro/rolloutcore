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
    A digest over the **declared weight source**: the parameter manifest, and
    optionally the trainer-side provenance the manifest came from (checkpoint
    revision, run id, global step). It answers "which declared weight source was
    this?" -- required for replay/provenance, because without it invariant I4
    could only claim "the generation that produced this".

    Two tiers, and the difference matters:

    * *manifest-only* (``from_param_specs(specs)``) -- names, dtypes and shapes.
      Distinguishes architectures; it does **not** distinguish two checkpoints
      of the same architecture, which is the common case between training steps.
    * *provenance-qualified* (``source=WeightProvenance(...)``) -- the manifest
      plus where it came from. This is what I4 actually needs.

    The provenance type is deliberately **not** named ``WeightSource``: vLLM
    already has a ``WeightSource`` ABC
    (``vllm/distributed/weight_transfer/base.py:128``; ``ModuleSource`` at
    ``:244``) meaning "the trainer-side object that yields named tensors", which
    is a different concept and would collide at every use site.

    Neither tier is a byte-level hash of the tensors, and neither proves what the
    engine's memory contains. A trajectory states the **declared** source; the
    correspondence between that declaration and engine bytes is a separate
    differential/replay question (``docs/state-machine.md`` section 6).

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
class WeightProvenance:
    """The *declared* origin of a weight set. Trusted, caller-supplied.

    This is the piece that turns a manifest digest into a usable replay claim. A
    manifest alone cannot distinguish "checkpoint at step 100" from "checkpoint
    at step 500" when the architecture is unchanged -- which is precisely the
    normal case in RL training.

    Fields mirror what a trainer actually knows:
    ``checkpoint`` (a model/checkpoint URI with a revision, e.g.
    ``"Qwen/Qwen3-1.7B-Base@main"`` or a path to a run directory), ``run_id``
    (the trainer run) and ``step`` (the global step). At least one is required:
    a ``WeightProvenance()`` with nothing in it would silently degrade a
    provenance-qualified identity back to a manifest-only one.

    Named ``Provenance`` rather than ``Source`` because vLLM's own
    ``WeightSource`` (``vllm/distributed/weight_transfer/base.py:128``) is the
    trainer-side iterable of named tensors. This class is metadata *about* that
    source, so the two must not share a name.
    """

    checkpoint: str | None = None
    run_id: str | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("checkpoint", self.checkpoint), ("run_id", self.run_id)):
            if value is not None and (not isinstance(value, str) or not value):
                raise WeightIdentityError(f"weight provenance {name} must be a non-empty string")
        if self.step is not None and (
            not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0
        ):
            raise WeightIdentityError("weight provenance step must be a non-negative int")
        if self.checkpoint is None and self.run_id is None and self.step is None:
            raise WeightIdentityError(
                "weight provenance must declare at least one of checkpoint, run_id or "
                "step; an empty provenance is a manifest-only identity in disguise"
            )

    def canonical(self) -> str:
        """Stable encoding. ``None`` fields are omitted, not rendered as 'None'."""
        parts = []
        if self.checkpoint is not None:
            parts.append(f"checkpoint={self.checkpoint}")
        if self.run_id is not None:
            parts.append(f"run_id={self.run_id}")
        if self.step is not None:
            parts.append(f"step={self.step}")
        return _SEP.join(parts)

    def describe(self) -> str:
        return self.canonical().replace(_SEP, ", ")


@dataclass(frozen=True, slots=True)
class WeightIdentity:
    """An immutable digest identifying the **declared weight source**.

    The digest covers the parameter manifest (names, dtypes, shapes) and, when a
    :class:`WeightProvenance` is supplied, the declared provenance of that manifest.

    Deliberate limit, stated rather than implied away: this is **not** a hash of
    the tensor bytes, so it cannot detect a checkpoint whose manifest matches but
    whose values differ. That is why the claim is "the declared weight source",
    not "the exact tensors the engine is holding". Closing the byte-level gap
    needs content hashing or a differential/replay check, which is out of scope
    for V1.

    Use :attr:`exactness` to tell the two tiers apart at runtime, and prefer
    provenance-qualified identities for anything a trajectory will record.
    """

    digest: str
    num_tensors: int | None = None
    #: The manifest-only digest, retained even when provenance is folded in, so
    #: "same architecture, different step" is still diagnosable.
    manifest_digest: str | None = None
    #: Declared origin. ``None`` means manifest-only.
    source: WeightProvenance | None = None

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
        if self.manifest_digest is not None and not self.manifest_digest.startswith(DIGEST_PREFIX):
            raise WeightIdentityError("manifest_digest must start with the digest prefix")

    @property
    def exactness(self) -> str:
        """``"declared-source"`` or ``"manifest-only"``.

        ``"manifest-only"`` is the honest label for an identity that cannot
        distinguish two same-architecture checkpoints; it is what a trajectory
        must not use to claim a specific training step.
        """
        return "declared-source" if self.source is not None else "manifest-only"

    @property
    def short(self) -> str:
        """First 12 hex chars, for logs and trajectory metadata."""
        return self.digest[len(DIGEST_PREFIX) :][:12]

    def describe(self) -> str:
        """Human-readable provenance, for logs and evidence messages."""
        if self.source is None:
            return f"{self.short} (manifest-only)"
        return f"{self.short} (declared {self.source.describe()})"

    @classmethod
    def from_param_specs(
        cls, specs: Iterable[ParamSpec], *, source: WeightProvenance | None = None
    ) -> WeightIdentity:
        """Compute an identity from a parameter manifest, plus optional source.

        The encoding is canonical: specs are sorted by name and each is rendered
        as ``name\\x1fdtype\\x1fdim,dim,...``. Sorting makes the digest
        independent of iteration order, which matters because a trainer-side
        ``WeightSource`` need not enumerate in a stable order.

        Without ``source`` the result is the manifest-only digest, byte-for-byte
        what this method has always produced. With ``source`` the provenance is
        folded in and :attr:`exactness` reports ``"declared-source"``.
        """
        ordered = sorted(specs, key=lambda s: s.name)
        names = [s.name for s in ordered]
        if len(set(names)) != len(names):
            raise WeightIdentityError("duplicate parameter names in manifest")

        hasher = hashlib.sha256()
        for spec in ordered:
            hasher.update(spec.canonical().encode("utf-8"))
            hasher.update(_LINE.encode("utf-8"))
        manifest_digest = f"{DIGEST_PREFIX}{hasher.hexdigest()}"

        if source is None:
            return cls(
                digest=manifest_digest,
                num_tensors=len(ordered),
                manifest_digest=manifest_digest,
            )

        combined = hashlib.sha256()
        combined.update(source.canonical().encode("utf-8"))
        combined.update(_LINE.encode("utf-8"))
        combined.update(manifest_digest.encode("utf-8"))
        combined.update(_LINE.encode("utf-8"))
        return cls(
            digest=f"{DIGEST_PREFIX}{combined.hexdigest()}",
            num_tensors=len(ordered),
            manifest_digest=manifest_digest,
            source=source,
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str, tuple[int, ...]]],
        *,
        source: WeightProvenance | None = None,
    ) -> WeightIdentity:
        """Convenience wrapper: ``(name, dtype, shape)`` triples."""
        return cls.from_param_specs((ParamSpec(*p) for p in pairs), source=source)

    @classmethod
    def parse(cls, digest: str) -> WeightIdentity:
        return cls(digest=digest)


@dataclass(frozen=True, slots=True)
class UpdateTarget:
    """What an update installs: a generation number *and* a weight identity.

    ``begin_update`` takes one of these rather than a bare ``WeightVersion`` so
    that a caller cannot open an update without saying which weights it intends
    to install. The separation matters: the generation number orders the
    lifecycle, the identity names the declared weight source a trajectory may
    claim. A manifest-only identity supports only the weaker claim -- see
    :attr:`WeightIdentity.exactness`.
    """

    version: WeightVersion
    identity: WeightIdentity

    @property
    def label(self) -> str:
        return self.version.label

    def describe(self) -> str:
        return f"{self.version.label} ({self.identity.describe()})"
