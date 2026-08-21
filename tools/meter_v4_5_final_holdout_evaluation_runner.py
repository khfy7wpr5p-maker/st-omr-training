#!/usr/bin/env python3
"""CLI for Meter V4-5 one-time independent final-holdout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _git_sha(repository_root: Path) -> str:
    value = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError("repository HEAD is not canonical lowercase SHA-1")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--completion-receipt", required=True)
    parser.add_argument("--human-review-evidence", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--v4-2-result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    repository_root = Path(args.repository_root).resolve()
    if not repository_root.is_dir() or repository_root.is_symlink():
        raise RuntimeError("repository root must be a regular directory")
    sys.path.insert(0, str(repository_root))

    from st_omr_training.meter_v4_5_final_holdout_evaluation import (
        run_meter_v4_5_one_time_final_holdout_evaluation,
    )

    git_sha = _git_sha(repository_root)
    result = run_meter_v4_5_one_time_final_holdout_evaluation(
        candidate_root=args.candidate_root,
        manifest_path=args.manifest,
        completion_receipt_path=args.completion_receipt,
        human_review_evidence_path=args.human_review_evidence,
        preregistration_path=args.preregistration,
        v4_2_result_path=args.v4_2_result,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        git_commit_sha=git_sha,
    )
    summary = result["final_holdout"]
    print(json.dumps({
        "decision": result["decision"],
        "record_count": summary["record_count"],
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "per_class_recall": summary["per_class_recall"],
        "confusion": summary["confusion"],
        "safety": result["safety"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
