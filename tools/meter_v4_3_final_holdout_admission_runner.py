#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from st_omr_training.meter_v4_3_final_holdout_admission import build_manifest, write_manifest_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--v4-1-result", required=True)
    parser.add_argument("--v4-2-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        candidate_root=args.candidate_root,
        v4_1_result_path=args.v4_1_result,
        v4_2_result_path=args.v4_2_result,
    )
    write_manifest_atomic(manifest, args.output)

    excluded = manifest["excluded"]
    reason_counts = Counter(row["reason"] for row in excluded)
    class_excluded = Counter(row["numerator_class"] for row in excluded)
    print("==============================================")
    print("V4-3 FINAL HOLDOUT ADMISSION")
    print("==============================================")
    print(json.dumps({
        "candidate_count": manifest["candidate_count"],
        "candidate_classes": manifest["candidate_classes"],
        "previously_observed_family_count": manifest["previously_observed_family_count"],
        "excluded_count": len(excluded),
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "excluded_by_class": dict(sorted(class_excluded.items())),
        "selected_count": manifest["selected_count"],
        "selected_classes": manifest["selected_classes"],
        "selection_sha256": manifest["selection_sha256"],
        "bbox_annotation_complete": manifest["bbox_annotation_complete"],
        "model_evaluated": manifest["model_evaluated"],
        "candidate_checkpoint_opened": manifest["candidate_checkpoint_opened"],
        "production_promotion_authorized": manifest["production_promotion_authorized"],
        "output": str(Path(args.output)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
