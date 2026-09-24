#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify every ``file.py:LINE`` anchor in the docs against a vLLM checkout.

RolloutCore's documents make source-level claims about vLLM, and a claim whose
line number has drifted is worse than no claim: it looks verified. This script
re-derives every anchor from the checkout and fails if any does not resolve.

    python scripts/verify_anchors.py --vllm /path/to/vllm

Anchors are the backticked ``path.py:<line>`` / ``path.py:<first>-<last>`` tokens
in the scanned files (docs *and* Python source). A bare basename
(``core.py:<line>``) is accepted if *some* file with that name contains the line;
ambiguous basenames are reported so they can be qualified rather than silently
trusted.

Exits 0 when every anchor resolves, 1 otherwise. With no ``--vllm`` (or a missing
directory) it prints a skip notice and exits 0, so it is safe in CI.

**Limitation.** This checks that the anchored lines *exist*, not that they say
what the surrounding prose claims. An anchor into the wrong-but-real region of a
file passes. Widening the scan to ``src/`` caught two citations that ran one line
past EOF and one that pointed at ``iter_groups`` while naming ``ParamMeta``;
nothing mechanical would have caught the last one, so spot-check anchors you
touch.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: ``path.py:<line>``, ``path.py:<first>-<last>``, ``core.py:<line>``.
#: Written with a placeholder so this file's own examples are not scanned as
#: claims -- the verifier is in its own scan set.
ANCHOR = re.compile(r"`([A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?`")

ROOT = Path(__file__).resolve().parent.parent


def scanned_files(root: Path) -> list[Path]:
    """Docs *and* source: the adapters cite upstream lines in their docstrings,
    and an unverified claim in `src/` is no better than one in a docstring."""
    return sorted(
        [
            *root.glob("*.md"),
            *root.glob("docs/*.md"),
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            *root.glob("diagnostics/*.py"),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vllm", default="", help="path to a vLLM checkout")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.vllm:
        print("verify_anchors: no --vllm checkout given; skipping")
        return 0
    vllm = Path(args.vllm).expanduser().resolve()
    if not vllm.is_dir():
        print(f"verify_anchors: {vllm} is not a directory; skipping")
        return 0

    all_py = sorted(vllm.rglob("*.py"))
    by_name: dict[str, list[Path]] = {}
    by_suffix: dict[str, list[Path]] = {}
    for path in all_py:
        by_name.setdefault(path.name, []).append(path)
        rel = path.relative_to(vllm).as_posix()
        for i in range(rel.count("/") + 1):
            by_suffix.setdefault("/".join(rel.split("/")[i:]), []).append(path)

    lengths: dict[Path, int] = {}
    total = ambiguous = 0
    bad: list[str] = []
    for src in scanned_files(ROOT):
        where = src.relative_to(ROOT).as_posix()
        for n, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
            for raw_path, first, last in ANCHOR.findall(line):
                total += 1
                if "/" in raw_path:
                    direct = vllm / raw_path
                    candidates = [direct] if direct.exists() else by_suffix.get(raw_path, [])
                    if not candidates:  # e.g. only a basename of a longer path
                        candidates = by_name.get(Path(raw_path).name, [])
                else:
                    candidates = by_name.get(raw_path, [])
                if not candidates:
                    bad.append(f"{where}:{n} {raw_path}:{first} (file not found)")
                    continue
                if len(candidates) > 1:
                    ambiguous += 1
                hi = int(last or first)
                if int(first) < 1 or hi < int(first):
                    bad.append(f"{where}:{n} {raw_path}:{first}-{last} (bad range)")
                    continue
                resolved = []
                for cand in candidates:
                    if cand not in lengths:
                        try:
                            lengths[cand] = len(cand.read_text(encoding="utf-8").splitlines())
                        except OSError:  # pragma: no cover - unreadable checkout
                            lengths[cand] = 0
                    if hi <= lengths[cand]:
                        resolved.append(cand)
                if not resolved:
                    worst = max(lengths[c] for c in candidates)
                    bad.append(
                        f"{where}:{n} {raw_path}:{first}-{last} "
                        f"(beyond EOF: file has {worst} lines)"
                    )

    if bad:
        print(f"verify_anchors: {len(bad)} unresolved anchor(s):")
        for item in bad[:40]:
            print(f"  {item}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        return 1

    if not args.quiet:
        print(
            f"verify_anchors: {total} anchors resolved against {vllm} "
            f"({ambiguous} via an ambiguous basename, all in range)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
