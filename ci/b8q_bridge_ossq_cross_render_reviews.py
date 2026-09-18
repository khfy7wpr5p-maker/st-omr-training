"""Materialize or verify the canonical B8Q pair-review receipt from frozen B8R evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from st_omr_training.poly_v2_ossq_b8r_review_bridge import (
    B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256,
    B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256,
    build_b8q_from_b8r,
)
from st_omr_training.poly_v2_ossq_pair_review_admission import (
    PairReviewMethod,
    review_receipt_to_json,
    verified_review_records,
)


ROOT = Path(__file__).resolve().parents[1]
B8P_RECEIPT = ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"
B8R_RECEIPT = ROOT / "evidence" / "ossq_b8r_cross_render_audit.json"
B8Q_RECEIPT = ROOT / "evidence" / "ossq_b8q_pair_review_admission.json"


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise RuntimeError(f"required committed evidence is missing: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="ascii"))


def main() -> int:
    b8p_payload = _load_json(B8P_RECEIPT)
    b8r_payload = _load_json(B8R_RECEIPT)
    if not isinstance(b8r_payload, dict):
        raise RuntimeError("committed B8R evidence must be a JSON object")
    if b8r_payload.get("receipt_sha256") != B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256:
        raise RuntimeError("committed B8R receipt identity differs from bridge freeze")

    receipt = build_b8q_from_b8r(
        b8p_payload=b8p_payload,
        b8r_payload=b8r_payload,
    )
    canonical = review_receipt_to_json(receipt)

    if B8Q_RECEIPT.exists():
        committed = B8Q_RECEIPT.read_text(encoding="ascii").strip()
        if committed != canonical:
            raise RuntimeError("live B8Q bridge output differs from committed receipt")
        print("B8Q_MODE=VERIFIED")
        print("B8Q_LIVE_BRIDGE_MATCHES_COMMITTED_RECEIPT=true")
    else:
        print("B8Q_MODE=DISCOVERY")
        print("B8Q_RECEIPT_JSON=" + canonical)

    verified = verified_review_records(receipt)
    if any(item.method is not PairReviewMethod.INDEPENDENT_CROSS_RENDER for item in verified):
        raise RuntimeError("B8Q verified population contains an unexpected review method")

    print(f"B8Q_SOURCE_B8R_RECEIPT_SHA256={B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256}")
    print(f"B8Q_RECEIPT_SHA256={receipt.receipt_sha256}")
    print(
        "B8Q_REVIEW_COUNTS="
        + ",".join(f"{name}:{count}" for name, count in receipt.review_count_by_decision)
    )
    print(f"B8Q_VERIFIED_COUNT={len(verified)}")
    print("B8Q_VERIFIED_METHOD=independent-cross-render-v1")
    print(f"B8Q_REVIEWER_IDENTITY_SHA256={B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256}")
    print(f"B8Q_INDEPENDENT_PAIRING_REVIEW_AUTHORITY={str(receipt.independent_pairing_review_authority).lower()}")
    print(f"B8Q_STAGE8_ADMISSION_AUTHORITY={str(receipt.stage8_admission_authority).lower()}")
    print(
        "B8Q_TRAIN_VALIDATION_ASSIGNMENT_AUTHORITY="
        + str(receipt.train_validation_assignment_authority).lower()
    )
    print(f"B8Q_TEST_ARTIFACT_BYTES_ACCESSED={str(receipt.test_artifact_bytes_accessed).lower()}")
    print(f"B8Q_PRODUCTION_AUTHORITY={str(receipt.production_authority).lower()}")
    print(f"B8Q_COMMERCIAL_USE_AUTHORITY={str(receipt.commercial_use_authority).lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
