# SPDX-License-Identifier: Apache-2.0
"""Adapters: the I/O side of the controller/adapter split.

* :mod:`rolloutcore.adapters.fake_engine` -- an in-memory model of vLLM's dev
  endpoint surface, with injectable faults.
* :mod:`rolloutcore.adapters.fake` -- ``FakeVLLMAdapter``, the reference
  implementation of the port, used for no-GPU cycle tests and the demo.
* :mod:`rolloutcore.adapters.http` -- ``HttpVLLMAdapter``, a thin stdlib client
  over a real ``vllm serve``.
* :mod:`rolloutcore.adapters.nccl` -- ``NCCLWeightTransferDriver``, the real
  trainer-side weight transfer (imports vLLM lazily, inside its methods).
"""

from .fake import FakeVLLMAdapter, manifest_identity, seeded_engine
from .fake_engine import FakeEngineError, FakeVLLMEngine
from .http import HTTPAdapterError, HttpVLLMAdapter, Response, Transport, UrllibTransport
from .nccl import NCCLWeightTransferDriver, RolloutCoreWeightSyncClient

__all__ = [
    "FakeEngineError",
    "FakeVLLMAdapter",
    "FakeVLLMEngine",
    "HTTPAdapterError",
    "HttpVLLMAdapter",
    "NCCLWeightTransferDriver",
    "Response",
    "RolloutCoreWeightSyncClient",
    "Transport",
    "UrllibTransport",
    "manifest_identity",
    "seeded_engine",
]
