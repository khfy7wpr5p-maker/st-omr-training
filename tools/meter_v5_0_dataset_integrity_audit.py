#!/usr/bin/env python3
"""Audit METER_V2_1500 selection manifests before annotation/training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from st_omr_training.meter_v5_0_dataset_integrity import audit_manifests, write_audit_receipt


def _read_family_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected", [])
    return {str(row["family_id"]) for row in selected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter-2", required=True, type=Path)
    parser.add_argument("--meter-3", required=True, type=Path)
    parser.add_argument("--meter-4", required=True, type=Path)
    parser.add_argument("--consumed-selection-json", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    consumed = set()
    if args.consumed_selection_json:
        consumed = _read_family_ids(args.consumed_selection_json)

    result = audit_manifests(
        {"2/4": args.meter_2, "3/4": args.meter_3, "4/4": args.meter_4},
        consumed_family_ids=consumed,
    )
    receipt_sha = write_audit_receipt(result, args.receipt)
    print(f"DATASET_INTEGRITY={result['status']}")
    print(f"RECEIPT_SHA256={receipt_sha}")
    for reason in result["reasons"]:
        print(f"REASON={reason}")
    print("TRAINING_AUTHORIZED=" + str(result["training_authorized"]))
    print("BBOX_ANNOTATION_AUTHORIZED=" + str(result["bbox_annotation_authorized"]))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
