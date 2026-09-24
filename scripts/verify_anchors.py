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

**Cross-tree basenames.** Five basenames exist in both trees (``runner.py``,
``nccl.py``, ``http.py``, ``weight_transfer.py``, ``__init__.py``). A bare
``runner.py:NNN`` in this repo's docs means *this repo's* runner, but vLLM's
``benchmarks/attention_benchmarks/runner.py`` also has 200+ lines, so a
vLLM-first lookup would verify it against the wrong file. Repo-local basenames
are therefore resolved against this repo first, and every anchor that resolves in
both trees -- or only upstream while a same-named local file exists -- is counted
and named in the output, so the ambiguity is visible instead of silent.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
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
            *root.glob("diagnostics/*.md"),
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            *root.glob("diagnostics/*.py"),
        ]
    )


#: A lookup over a tree: basename -> files, and every path suffix -> files.
Index = tuple[dict[str, list[Path]], dict[str, list[Path]]]


def _index(roots: Iterable[Path], base: Path) -> Index:
    """``(by_name, by_suffix)`` for every ``.py`` under ``roots``, relative to ``base``."""
    by_name: dict[str, list[Path]] = {}
    by_suffix: dict[str, list[Path]] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            by_name.setdefault(path.name, []).append(path)
            parts = path.relative_to(base).as_posix().split("/")
            for i in range(len(parts)):
                by_suffix.setdefault("/".join(parts[i:]), []).append(path)
    return by_name, by_suffix


def _lookup(base: Path, index: Index, raw_path: str) -> tuple[list[Path], bool]:
    """``(candidates, qualified)`` for one tree.

    *Qualified* means the anchor's **path** matched -- directly or as a suffix --
    rather than only its basename, so it names one file. A basename-only match is
    a guess whenever the other tree has a file with that name too.
    """
    by_name, by_suffix = index
    if "/" in raw_path:
        direct = base / raw_path
        if direct.exists():
            return [direct], True
        if by_suffix.get(raw_path):
            return by_suffix[raw_path], True
        return by_name.get(Path(raw_path).name, []), False
    return by_name.get(raw_path, []), False


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
    vllm_index = (by_name, by_suffix)
    # Anchors may also point at *this* repo: a docstring explaining where a
    # guarantee is enforced (`lifecycle.py:465-472`) should rot as loudly as one
    # about vLLM. Where a basename exists in both trees the local file wins (see
    # the module docstring), and every cross-tree resolution is reported.
    repo_index = _index(
        [ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "diagnostics"], ROOT
    )

    lengths: dict[Path, int] = {}

    def lines_of(path: Path) -> int:
        if path not in lengths:
            try:
                lengths[path] = len(path.read_text(encoding="utf-8").splitlines())
            except OSError:  # pragma: no cover - unreadable checkout
                lengths[path] = 0
        return lengths[path]

    def in_range(candidates: list[Path], first: int, hi: int) -> list[Path]:
        return [c for c in candidates if 1 <= first <= hi <= lines_of(c)]

    total = ambiguous = local = cross_tree = 0
    cross_names: set[str] = set()
    cross_spots: list[str] = []
    bad: list[str] = []
    for src in scanned_files(ROOT):
        where = src.relative_to(ROOT).as_posix()
        for n, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
            for raw_path, first_s, last_s in ANCHOR.findall(line):
                total += 1
                first, hi = int(first_s), int(last_s or first_s)
                if first < 1 or hi < first:
                    bad.append(f"{where}:{n} {raw_path}:{first_s}-{last_s} (bad range)")
                    continue
                upstream, up_qualified = _lookup(vllm, vllm_index, raw_path)
                here, here_qualified = _lookup(ROOT, repo_index, raw_path)
                resolved_here = in_range(here, first, hi)
                resolved_upstream = in_range(upstream, first, hi)
                guessed = False
                if here_qualified and resolved_here:
                    candidates = resolved_here
                    local += 1
                elif up_qualified and resolved_upstream:
                    candidates = resolved_upstream
                elif resolved_here:
                    candidates = resolved_here
                    local += 1
                    guessed = bool(upstream)
                elif resolved_upstream:
                    candidates = resolved_upstream
                    guessed = bool(here)
                else:
                    pool = here or upstream
                    if not pool:
                        bad.append(f"{where}:{n} {raw_path}:{first_s} (file not found)")
                        continue
                    bad.append(
                        f"{where}:{n} {raw_path}:{first_s}-{last_s} "
                        f"(beyond EOF in {'RolloutCore' if here else 'vLLM'}: "
                        f"file has {max(lines_of(c) for c in pool)} lines)"
                    )
                    continue
                if guessed:
                    cross_tree += 1
                    cross_names.add(Path(raw_path).name)
                    cross_spots.append(f"{where}:{n} {raw_path}:{first_s}")
                if len(candidates) > 1:
                    ambiguous += 1

    if bad:
        print(f"verify_anchors: {len(bad)} unresolved anchor(s):")
        for item in bad[:40]:
            print(f"  {item}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        return 1

    if not args.quiet:
        print(
            f"verify_anchors: {total} anchors resolved "
            f"({total - local} against {vllm}, {local} against this repo; "
            f"{ambiguous} via an ambiguous basename, all in range)"
        )
        if cross_tree:
            print(
                f"verify_anchors: {cross_tree} anchor(s) name a basename that exists "
                f"in both trees ({', '.join(sorted(cross_names))}); each was checked "
                f"against whichever tree has that line, which is not proof it names "
                f"the intended file:"
            )
            for spot in cross_spots[:20]:
                print(f"  {spot}")
            if len(cross_spots) > 20:
                print(f"  ... and {len(cross_spots) - 20} more")
    return 0
    if bad:
        print(f"verify_anchors: {len(bad)} unresolved anchor(s):")
        for item in bad[:40]:
            print(f"  {item}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        return 1

    if not args.quiet:
        print(
            f"verify_anchors: {total} anchors resolved "
            f"({total - local} against {vllm}, {local} against this repo; "
            f"{ambiguous} via an ambiguous basename, all in range)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
