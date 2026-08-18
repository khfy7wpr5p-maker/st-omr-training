#!/usr/bin/env python3
"""Run the frozen TRAIN-only System Geometry topology audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from st_omr_training.system_geometry_train_topology_audit_v1 import audit_d6_train_topology

FOLDER = "stage7d6-staff-structure-derivatives-f33e70ec24a60ebab547ed7d4a395902129b0e23"


def _discover() -> Path:
    candidates = (
        Path("/content/gdrive_r2/MyDrive/ST-OMR-SYNTHETIC") / FOLDER,
        Path("/content/gdrive_r2/ST-OMR-SYNTHETIC") / FOLDER,
        Path("/content/drive/MyDrive/ST-OMR-SYNTHETIC") / FOLDER,
        Path("/content/drive/ST-OMR-SYNTHETIC") / FOLDER,
    )
    for path in candidates:
        if (path / "manifest.json").is_file() and (path / "labels").is_dir():
            return path
    raise SystemExit(
        "D6 root not found automatically. Re-run with --d6-root /path/to/" + FOLDER
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d6-root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/content/system-geometry-train-topology-audit-v1/report.json"),
    )
    args = parser.parse_args()
    root = args.d6_root if args.d6_root is not None else _discover()
    report = audit_d6_train_topology(root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("SYSTEM GEOMETRY TRAIN TOPOLOGY AUDIT COMPLETE")
    print("D6 ROOT        :", root)
    print("DECISION       :", report["decision"])
    print("TRAIN PAGES    :", report["train_pages"])
    print("SYSTEMS TOTAL  :", report["systems_total"])
    print("STAFFS TOTAL   :", report["staffs_total"])
    print("MULTI-SYSTEM   :", report["multi_system_pages"])
    print("MULTI-STAFF SYS:", report["multi_staff_systems"])
    print("SYSTEM/PAGE    :", report["system_counts_per_page"])
    print("STAFF/PAGE     :", report["staff_counts_per_page"])
    print("STAFF/SYSTEM   :", report["staffs_per_system"])
    print("MEAS/SYSTEM    :", report["measures_per_system"])
    print("SYS GAP / SPACE:", report["adjacent_system_gap_staff_spacing_units"])
    print("SYS X OVERLAP  :", report["adjacent_system_horizontal_overlap_ratio"])
    print("INTRA SYS GAP  :", report["intra_system_staff_gap_staff_spacing_units"])
    print("VALIDATION     : LABELS NOT OPENED")
    print("TEST           : CLOSED")
    print("TRAINING       : NONE")
    print("TUNING         : NONE")
    print("REPORT         :", args.out)


if __name__ == "__main__":
    main()
