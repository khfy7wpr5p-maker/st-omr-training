#!/usr/bin/env python3
"""CLI runner for the bounded Meter V4-1 numerator specialist experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from st_omr_training.meter_v4_1_numerator_specialist_run import (  # noqa: E402
    repository_binding_v4_1,
    run_meter_v4_1_numerator_specialist,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--parent-v4-0-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError("Git HEAD is not canonical lowercase SHA-1")
    return value


def main() -> int:
    args = _parser().parse_args()
    repository_root = Path(args.repository_root).resolve()
    if repository_root != REPOSITORY_ROOT.resolve():
        raise RuntimeError("--repository-root must be the repository containing this exact runner")
    git_commit_sha = _git_head(repository_root)
    repository_binding = repository_binding_v4_1(git_commit_sha)

    print(json.dumps({
        "experiment": "meter-v4-1-learned-numerator-specialist-v1",
        "git_commit_sha": git_commit_sha,
        "repository_sha256_binding": repository_binding,
        "parent_v4_0_root": str(Path(args.parent_v4_0_root)),
        "output_root": str(Path(args.output_root)),
        "teacher_adaptation_validation_evaluated": False,
        "d10_opened": False,
        "test_opened": False,
    }, indent=2), flush=True)

    result = run_meter_v4_1_numerator_specialist(
        parent_v4_0_root=args.parent_v4_0_root,
        output_root=args.output_root,
        git_commit_sha=git_commit_sha,
        repository_sha=repository_binding,
        progress=lambda message: print(message, flush=True),
    )

    print("\n==============================================", flush=True)
    print("V4-1 OOF SUMMARY", flush=True)
    print("==============================================", flush=True)
    print(json.dumps(result["oof_summary"], indent=2, ensure_ascii=False), flush=True)
    print("\n==============================================", flush=True)
    print("V4-1 DECISION", flush=True)
    print("==============================================", flush=True)
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False), flush=True)
    print("\n==============================================", flush=True)
    print("SAFETY", flush=True)
    print("==============================================", flush=True)
    print(json.dumps({
        "determinism_repeat_pass": result["determinism"]["repeat_pass"],
        "teacher_adaptation_validation_evaluated": result["data_surface"]["teacher_adaptation_validation_evaluated"],
        "teacher_adaptation_validation_images_decoded": result["data_surface"]["teacher_adaptation_validation_images_decoded"],
        "d10_opened": result["data_surface"]["d10_opened"],
        "test_opened": result["data_surface"]["test_opened"],
        "runtime_connected": result["runtime_connected"],
        "production_promotion_authorized": result["production_promotion_authorized"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
