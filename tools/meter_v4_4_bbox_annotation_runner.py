#!/usr/bin/env python3
"""CLI preflight/QA runner for Meter V4-4 annotation-only stage."""

from __future__ import annotations

import argparse
import json

from st_omr_training.meter_v4_4_final_holdout_bbox_annotation import (
    AnnotationSession,
    EXPECTED_SELECTION_SHA256,
    generate_review_contact_sheets,
    write_completion_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("preflight", "qa"), default="preflight")
    args = parser.parse_args()

    if args.mode == "preflight":
        session = AnnotationSession(
            candidate_root=args.candidate_root,
            manifest_path=args.manifest,
        )
        print(
            json.dumps(
                {
                    "stage": "meter-v4-4-final-holdout-bbox-annotation",
                    "selection_sha256": EXPECTED_SELECTION_SHA256,
                    "selected_count": len(session.samples),
                    "annotated_count": session.annotated_count,
                    "resume_index": session.resume_index(),
                    "review_flag_count": len(session.progress["review_flags"]),
                    "image_binding_sha256": session.binding["image_binding_sha256"],
                    "model_evaluated": False,
                    "inference_count": 0,
                    "candidate_checkpoint_opened": False,
                    "test_opened": False,
                    "runtime_connected": False,
                    "production_promotion_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    receipt = write_completion_receipt(
        candidate_root=args.candidate_root,
        manifest_path=args.manifest,
    )
    sheets = generate_review_contact_sheets(
        candidate_root=args.candidate_root,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "mechanical_qa": "PASS",
                "completion_receipt": str(receipt),
                "contact_sheets": [str(path) for path in sheets],
                "human_visual_review_required": True,
                "model_evaluated": False,
                "inference_count": 0,
                "candidate_checkpoint_opened": False,
                "production_promotion_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
