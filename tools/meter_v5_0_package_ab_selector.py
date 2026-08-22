#!/usr/bin/env python3
"""Generate deterministic package_ab-only METER V5-0 selection manifests.

Selection only: no source files are copied and no bbox/model work is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from st_omr_training.meter_v5_0_package_ab_selector import (
    build_package_ab_selection,
    load_family_blacklist,
    load_master_index,
    sha256_file,
    write_selection_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-index", required=True)
    parser.add_argument("--historical-blacklist", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    master_index = Path(args.master_index)
    blacklist_path = Path(args.historical_blacklist)
    output_dir = Path(args.output_dir)

    rows = load_master_index(master_index)
    blacklist = load_family_blacklist([blacklist_path])
    selected, receipt = build_package_ab_selection(rows, blacklist=blacklist)

    manifest_paths = write_selection_manifests(selected, output_dir)
    receipt["master_index_sha256"] = sha256_file(master_index)
    receipt["historical_blacklist_sha256"] = sha256_file(blacklist_path)
    receipt["manifest_sha256"] = {
        meter: sha256_file(path) for meter, path in manifest_paths.items()
    }

    receipt_path = output_dir / "SELECTION_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"SELECTION_RECEIPT={receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
