# SPDX-License-Identifier: Apache-2.0
"""Trajectory records: binding a rollout's output to the weights that made it.

vLLM has no rollout/trajectory domain model -- ``trajectory`` does not appear
anywhere in the tree -- and PR #49040 deliberately removed binding a weight
version to a request, because one request may span two versions. So the record a
trainer needs, "these tokens came from *this* weight source", has to live here.

A :class:`Trajectory` is that record: the controller's :class:`RolloutBinding`
(the version and identity the rollout was admitted against) plus what came out.
Two properties are the point of it:

* :attr:`~Trajectory.replay_ready` is ``False`` for a ``manifest-only`` identity.
  Phase 3C measured why: the digest covers parameter names, dtypes and shapes, so
  a dummy-initialised ``opt-125m`` and the real one share ``925369d663bc``, and
  Phase 4C measured the same digest surviving a *corrupted* checkpoint. Such a
  record identifies an architecture, not a training step, and must not be used to
  claim one.
* :attr:`~Trajectory.admission_agrees` checks the identity against something
  outside RolloutCore: the label the engine reported when the rollout was
  admitted. The binding is RolloutCore's own bookkeeping, so on its own it cannot
  be evidence about the engine.

The engine's label *at completion* is recorded too, and is deliberately **not**
required to match. Phase 4A ran a rollout that was admitted at ``rc-0`` and
finished after the update to ``rc-1``; the engine then reports ``rc-1`` for a
request whose tokens were produced entirely by ``rc-0``. That divergence is the
reason a trajectory carries the binding rather than the engine's last word.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lifecycle import RolloutBinding
from .versions import WeightIdentity, WeightProvenance, WeightVersion

#: ``"declared-source"`` -- the only exactness that supports a replay claim.
REPLAY_READY_EXACTNESS = "declared-source"


class TrajectoryError(ValueError):
    """A trajectory record that cannot support the claim it would make."""


def _identity_to_json(identity: WeightIdentity) -> dict[str, Any]:
    source = identity.source
    return {
        "digest": identity.digest,
        "manifest_digest": identity.manifest_digest,
        "num_tensors": identity.num_tensors,
        "source": (
            None
            if source is None
            else {"checkpoint": source.checkpoint, "run_id": source.run_id, "step": source.step}
        ),
    }


def _identity_from_json(data: dict[str, Any]) -> WeightIdentity:
    """Rebuild an identity *with* its provenance.

    ``WeightIdentity.parse`` cannot do this -- it takes a digest and nothing else,
    so provenance would be silently dropped and a declared-source trajectory would
    come back as manifest-only, quietly failing :attr:`Trajectory.replay_ready`.
    """
    source = data.get("source")
    return WeightIdentity(
        digest=str(data["digest"]),
        manifest_digest=data.get("manifest_digest"),
        num_tensors=data.get("num_tensors"),
        source=None if source is None else WeightProvenance(**source),
    )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One rollout, and the declared weight source that produced it.

    ``prompt`` is required: a record that cannot be replayed is the thing this
    type exists to avoid, and the prompt is half of what a replay needs.
    """

    binding: RolloutBinding
    prompt: str
    text: str
    token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None
    seconds: float | None = None
    #: What the engine reported when the rollout was admitted. ``None`` when the
    #: observation was not taken, which makes :attr:`admission_agrees` unknown
    #: rather than true.
    engine_version_at_admission: str | None = None
    #: What the engine reported once the rollout finished. Expected to differ from
    #: :attr:`version` for a rollout that spanned an update -- see the module docs.
    engine_version_at_completion: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise TrajectoryError("a trajectory needs a non-empty prompt to be replayable")
        if not isinstance(self.text, str):
            raise TrajectoryError("trajectory text must be a string")
        if not isinstance(self.token_ids, tuple) or any(
            not isinstance(t, int) or isinstance(t, bool) for t in self.token_ids
        ):
            raise TrajectoryError("trajectory token_ids must be a tuple of ints")
        if self.seconds is not None and self.seconds < 0:
            raise TrajectoryError("trajectory seconds must be >= 0")

    # ------------------------------------------------------------- views

    @property
    def request_id(self) -> str:
        return self.binding.request_id

    @property
    def version(self) -> str:
        """The version the rollout was **bound** to, not the engine's last word."""
        return self.binding.version.label

    @property
    def identity(self) -> WeightIdentity:
        return self.binding.weight_identity

    @property
    def exactness(self) -> str:
        return self.identity.exactness

    @property
    def provenance(self) -> WeightProvenance | None:
        return self.identity.source

    @property
    def cache_salt(self) -> str:
        return self.binding.cache_salt

    @property
    def replay_ready(self) -> bool:
        """Whether this record can support a claim about *which* weights ran.

        False for a manifest-only identity: it cannot separate two checkpoints of
        the same architecture, which is the normal case in RL.
        """
        return self.exactness == REPLAY_READY_EXACTNESS

    @property
    def admission_agrees(self) -> bool | None:
        """Did the engine agree with the binding when the rollout was admitted?

        ``None`` when no observation was recorded -- unknown, not fine.
        """
        if self.engine_version_at_admission is None:
            return None
        return self.engine_version_at_admission == self.version

    @property
    def spans_an_update(self) -> bool:
        """Did the engine's label change under this rollout?"""
        if self.engine_version_at_completion is None:
            return False
        return self.engine_version_at_completion != self.version

    def describe(self) -> str:
        provenance = "manifest-only" if self.provenance is None else self.provenance.describe()
        return f"{self.request_id} {self.version} ({provenance}) {len(self.token_ids)} tokens" + (
            "" if self.replay_ready else " [NOT replay-ready]"
        )

    # -------------------------------------------------------------- codec

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "version": self.version,
            "identity": _identity_to_json(self.identity),
            "exactness": self.exactness,
            "replay_ready": self.replay_ready,
            "cache_salt": self.cache_salt,
            "admitted_seq": self.binding.admitted_seq,
            "prompt": self.prompt,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "finish_reason": self.finish_reason,
            "seconds": self.seconds,
            "engine_version_at_admission": self.engine_version_at_admission,
            "engine_version_at_completion": self.engine_version_at_completion,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Trajectory:
        binding = RolloutBinding(
            request_id=str(data["request_id"]),
            version=WeightVersion.parse_label(str(data["version"])),
            weight_identity=_identity_from_json(dict(data["identity"])),
            cache_salt=str(data["cache_salt"]),
            admitted_seq=int(data["admitted_seq"]),
        )
        return cls(
            binding=binding,
            prompt=str(data["prompt"]),
            text=str(data["text"]),
            token_ids=tuple(int(t) for t in data.get("token_ids", ())),
            finish_reason=data.get("finish_reason"),
            seconds=data.get("seconds"),
            engine_version_at_admission=data.get("engine_version_at_admission"),
            engine_version_at_completion=data.get("engine_version_at_completion"),
        )


@dataclass
class TrajectoryRecorder:
    """Collects trajectories and writes them as JSONL.

    Locked, for the same reason the controller is: in a real run the record is
    appended by whichever thread owns the request while the lifecycle runner is
    mid-cycle.
    """

    _items: list[Trajectory] = field(default_factory=list, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def record(self, trajectory: Trajectory) -> Trajectory:
        with self._lock:
            self._items.append(trajectory)
        return trajectory

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        with self._lock:
            return tuple(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def manifest_only(self) -> tuple[Trajectory, ...]:
        """Records that cannot support a claim about which weights produced them."""
        return tuple(t for t in self.trajectories if not t.replay_ready)

    def summary(self) -> str:
        trajectories = self.trajectories
        versions = sorted({t.version for t in trajectories})
        exact = sum(1 for t in trajectories if t.replay_ready)
        return (
            f"{len(trajectories)} trajectories over {versions or ['-']}; "
            f"{exact} replay-ready, {len(trajectories) - exact} manifest-only"
        )

    def write_jsonl(self, path: Path) -> Path:
        """One JSON object per line, in record order."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for trajectory in self.trajectories:
                handle.write(json.dumps(trajectory.to_json(), sort_keys=True) + "\n")
        return path

    @staticmethod
    def read_jsonl(path: Path) -> tuple[Trajectory, ...]:
        out = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append(Trajectory.from_json(json.loads(line)))
        return tuple(out)
