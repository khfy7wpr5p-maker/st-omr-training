#!/usr/bin/env python3
"""CLI runner for Meter V4-2 full-train candidate + development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from st_omr_training.meter_v4_2_full_train_dev_screen_run import (  # noqa: E402
    repository_binding_v4_2,
    run_meter_v4_2_full_train_dev_screen,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--parent-v4-0-root", required=True)
    parser.add_argument("--parent-v4-1-root", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--permission", required=True)
    parser.add_argument("--privacy", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _git_head(root: Path) -> str:
    value = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError("Git HEAD is not canonical lowercase SHA-1")
    return value


def main() -> int:
    args = _parser().parse_args()
    repository_root = Path(args.repository_root).resolve()
    if repository_root != REPOSITORY_ROOT.resolve():
        raise RuntimeError("--repository-root must point to this exact repository")
    git_sha = _git_head(repository_root)
    binding = repository_binding_v4_2(git_sha)
    print(json.dumps({
        "experiment": "meter-v4-2-full-train-dev-screen-v1",
        "git_commit_sha": git_sha,
        "repository_sha256_binding": binding,
        "parent_v4_0_root": args.parent_v4_0_root,
        "parent_v4_1_root": args.parent_v4_1_root,
        "development_validation_images_planned": 9,
        "development_validation_used_for_training": False,
        "d10_opened": False,
        "test_opened": False,
    }, indent=2), flush=True)
    result = run_meter_v4_2_full_train_dev_screen(
        parent_v4_0_root=args.parent_v4_0_root,
        parent_v4_1_root=args.parent_v4_1_root,
        pilot_path=args.pilot,
        choices_path=args.choices,
        permission_path=args.permission,
        privacy_path=args.privacy,
        output_root=args.output_root,
        git_commit_sha=git_sha,
        repository_sha=binding,
        progress=lambda message: print(message, flush=True),
    )
    print("\n==============================================", flush=True)
    print("V4-2 DEVELOPMENT SUMMARY", flush=True)
    print("==============================================", flush=True)
    print(json.dumps(result["development_validation"], indent=2, ensure_ascii=False), flush=True)
    print("\n==============================================", flush=True)
    print("V4-2 DECISION", flush=True)
    print("==============================================", flush=True)
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False), flush=True)
    print("\n==============================================", flush=True)
    print("V4-2 SAFETY", flush=True)
    print("==============================================", flush=True)
    print(json.dumps({
        **result["safety"],
        "deterministic_repeat_pass": result["full_train"]["deterministic_repeat_pass"],
        "candidate_checkpoint_sha256": result["candidate_checkpoint"]["sha256"],
        "candidate_model_state_sha256": result["candidate_checkpoint"]["model_state_sha256"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
