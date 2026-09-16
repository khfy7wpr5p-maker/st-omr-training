"""CI runner for TR-POLY-09B8N live OSSQ source-byte pinning.

Discovery mode: if the committed receipt does not exist, fetch the exact five
B8M-reviewed PDFs and print the hash-only receipt to the job log.

Verification mode: if the receipt exists, fetch the same exact URLs again and
fail unless the live bytes exactly match the committed receipt.

No raw PDF bytes are written to the repository or uploaded as workflow artifacts.
"""

from __future__ import annotations

from pathlib import Path
import sys

from st_omr_training.poly_v2_ossq_source_byte_pin import (
    fetch_and_pin_b8m_sources,
    receipt_from_json,
    receipt_to_json,
    verify_live_receipt,
)


RECEIPT_PATH = Path("evidence/ossq_b8n_source_byte_pins.json")


def main() -> int:
    actual = fetch_and_pin_b8m_sources()
    actual_json = receipt_to_json(actual)
    if not RECEIPT_PATH.exists():
        print("B8N_MODE=DISCOVERY")
        print(f"B8N_RECEIPT_SHA256={actual.receipt_sha256}")
        print(f"B8N_RECEIPT_JSON={actual_json}")
        print("B8N_RAW_PDF_BYTES_PERSISTED=false")
        return 0

    expected = receipt_from_json(RECEIPT_PATH.read_text(encoding="utf-8"))
    verify_live_receipt(expected, actual)
    print("B8N_MODE=VERIFIED")
    print(f"B8N_RECEIPT_SHA256={actual.receipt_sha256}")
    print("B8N_LIVE_BYTES_MATCH_COMMITTED_RECEIPT=true")
    print("B8N_RAW_PDF_BYTES_PERSISTED=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
