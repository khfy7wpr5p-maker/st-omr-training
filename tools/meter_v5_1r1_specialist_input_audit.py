#!/usr/bin/env python3
"""Run the bounded METER V5-1R1 specialist-input audit.

Private checkpoints remain local/Drive-side. This runner never trains or tunes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from st_omr_training.meter_v5_1r1_specialist_input_audit import (
    run_specialist_input_audit_from_checkpoints_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pilot-evidence", required=True, type=Path)
    parser.add_argument("--digit2-checkpoint", required=True, type=Path)
    parser.add_argument("--digit3-checkpoint", required=True, type=Path)
    parser.add_argument("--digit4-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_specialist_input_audit_from_checkpoints_v1(
        dataset_root=args.dataset_root,
        pilot_evidence_path=args.pilot_evidence,
        digit2_checkpoint=args.digit2_checkpoint,
        digit3_checkpoint=args.digit3_checkpoint,
        digit4_checkpoint=args.digit4_checkpoint,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "PASS_SCALE_ANNOTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
