"""Live TR-POLY-09B8P OSSQ system-pair materialization runner.

The runner reconstructs only the seven B8O READY scores in an ephemeral OSSQ
checkout, using the exact B8N source PDFs and exact pinned preprocessor.  It
prints or verifies a hash-only receipt; raw PDFs/images/MusicXML are never
uploaded as evidence by this runner.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

from st_omr_training.poly_v2_ossq_pairing_preflight import B8O_PAIRING_SOURCE_SPECS
from st_omr_training.poly_v2_ossq_system_pair_materialization import (
    B8P_EXPECTED_B8N_RECEIPT_SHA256,
    B8P_EXPECTED_B8O_RECEIPT_SHA256,
    B8P_READY_SCORE_IDS,
    build_b8p_materialization_receipt,
    receipt_to_json,
)

ROOT = Path(__file__).resolve().parents[1]
B8N_RECEIPT = ROOT / "evidence" / "ossq_b8n_source_byte_pins.json"
B8P_RECEIPT = ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"
USER_AGENT = "ScoreMosaic-ST-OMR/1.0 (+https://github.com/khfy7wpr5p-maker/st-omr-training)"
MAX_PDF_BYTES = 64 * 1024 * 1024


def fetch_pdf(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.1"})
    with urlopen(request, timeout=120) as response:  # nosec B310 - B8N-pinned HTTPS URLs only
        data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise RuntimeError(f"invalid or oversized source PDF: {url}")
    return data


def run_uv(preprocessor_root: Path, *args: str) -> None:
    command = ["uv", "run", "--frozen", "python", *args]
    print("B8P_RUN=" + " ".join(command))
    subprocess.run(command, cwd=preprocessor_root, check=True, env={**os.environ, "PYTHONPATH": "."})


def prepare_sources(dataset_root: Path) -> tuple[dict[str, str], list[str]]:
    b8n = json.loads(B8N_RECEIPT.read_text(encoding="utf-8"))
    if b8n.get("receipt_sha256") != B8P_EXPECTED_B8N_RECEIPT_SHA256:
        raise RuntimeError("committed B8N receipt identity differs from B8P freeze")
    specs = {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}
    source_hashes: dict[str, str] = {}
    task_lines: list[str] = []
    for pin in b8n["pins"]:
        ready_ids = [score_id for score_id in pin["score_ids"] if score_id in B8P_READY_SCORE_IDS]
        if not ready_ids:
            continue
        data = fetch_pdf(pin["source_url"])
        if len(data) != pin["byte_count"] or sha256(data).hexdigest() != pin["source_sha256"]:
            raise RuntimeError(f"B8N source bytes drifted for {pin['imslp_id']}")
        for score_id in ready_ids:
            spec = specs[score_id]
            score_dir = dataset_root / "scores" / spec.work_path
            if not score_dir.is_dir():
                raise RuntimeError(f"OSSQ work path missing for score {score_id}")
            (score_dir / f"sq{score_id}_scanned.pdf").write_bytes(data)
            source_hashes[score_id] = pin["source_sha256"]
            task_lines.append(f"{spec.work_path}/sq{score_id}.mscx")
    if set(source_hashes) != set(B8P_READY_SCORE_IDS):
        raise RuntimeError("downloaded B8P source population differs from exact READY population")
    return source_hashes, task_lines


def materialize(dataset_root: Path, preprocessor_root: Path, task_file: Path) -> None:
    logs = dataset_root / "logs"
    logs.mkdir(exist_ok=True)
    d = str(dataset_root.resolve())
    l = str(task_file.resolve())
    run_uv(preprocessor_root, "omrdp/ossq/convert_musicxml_to_lmxe.py", "-d", d, "-v", "0", "--multi-process", "0", "--remove-stem-direction", "1", "-l", l)
    run_uv(preprocessor_root, "omrdp/ossq/convert_pdf_to_images.py", "-d", d, "-t", "scanned", "-l", l)
    run_uv(preprocessor_root, "omrdp/ossq/yolo_detect_systems.py", "-d", d, "-t", "scanned", "-n", "16", "--device", "cpu", "--reproduce", "1", "-l", l)
    run_uv(preprocessor_root, "omrdp/ossq/yolo_crop_systems.py", "-d", d, "-t", "scanned", "-v", "1", "--confidence-threshold", "0.3", "--merge-threshold", "0.7", "--reproduce", "1", "-l", l)
    run_uv(preprocessor_root, "omrdp/ossq/yolo_detect_staff_heights.py", "-d", d, "-t", "scanned", "-n", "32", "--device", "cpu", "-v", "0", "--reproduce", "1", "-l", l)
    run_uv(preprocessor_root, "omrdp/ossq/yolo_resize_systems.py", "-d", d, "-t", "scanned", "--target_height", "18", "--reproduce", "1", "-l", l)
    run_uv(preprocessor_root, "omrdp/ossq/align_systems_lmxe.py", "-d", d, "-t", "scanned", "-l", l)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--preprocessor-root", required=True, type=Path)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    preprocessor_root = args.preprocessor_root.resolve()
    if not (preprocessor_root / "uv.lock").is_file():
        raise RuntimeError("pinned preprocessor uv.lock is missing")

    source_hashes, task_lines = prepare_sources(dataset_root)
    task_file = dataset_root / "b8p_ready_scores.txt"
    task_file.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    materialize(dataset_root, preprocessor_root, task_file)

    receipt = build_b8p_materialization_receipt(
        dataset_root=dataset_root,
        b8n_source_sha256_by_score=source_hashes,
        b8n_receipt_sha256=B8P_EXPECTED_B8N_RECEIPT_SHA256,
        b8o_receipt_sha256=B8P_EXPECTED_B8O_RECEIPT_SHA256,
    )
    canonical = receipt_to_json(receipt)
    if B8P_RECEIPT.exists():
        committed = B8P_RECEIPT.read_text(encoding="utf-8").strip()
        if committed != canonical:
            raise RuntimeError("live B8P materialization differs from committed receipt")
        print("B8P_MODE=VERIFIED")
        print("B8P_LIVE_PAIRS_MATCH_COMMITTED_RECEIPT=true")
    else:
        print("B8P_MODE=DISCOVERY")
        print("B8P_RECEIPT_JSON=" + canonical)
    print(f"B8P_RECEIPT_SHA256={receipt.receipt_sha256}")
    print(f"B8P_PAIR_COUNT={len(receipt.pairs)}")
    print("B8P_PAIR_COUNT_BY_SCORE=" + ",".join(f"{score}:{count}" for score, count in receipt.pair_count_by_score))
    print("B8P_RAW_PDF_BYTES_PERSISTED_AS_EVIDENCE=false")
    print("B8P_RAW_PAIR_BYTES_PERSISTED_AS_EVIDENCE=false")
    print("B8P_INDEPENDENT_PAIRING_REVIEW_AUTHORITY=false")
    print("B8P_STAGE8_ADMISSION_AUTHORITY=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
