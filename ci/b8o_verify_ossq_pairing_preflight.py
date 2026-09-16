from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

from st_omr_training.poly_v2_ossq_pairing_preflight import (
    B8O_PAIRING_SOURCE_SPECS,
    OssqPairingSourcePayload,
    build_b8o_pairing_preflight,
    parse_scanned_alignment,
    receipt_to_json,
)
from st_omr_training.poly_v2_ossq_source_byte_pin import receipt_from_json


ROOT = Path(__file__).resolve().parents[1]
B8N_RECEIPT_PATH = ROOT / "evidence" / "ossq_b8n_source_byte_pins.json"
B8O_RECEIPT_PATH = ROOT / "evidence" / "ossq_b8o_pairing_preflight.json"
MAX_ALIGNMENT_BYTES = 1024 * 1024
MAX_MUSICXML_BYTES = 16 * 1024 * 1024
USER_AGENT = "ScoreMosaic-ST-OMR/1.0 (+https://github.com/khfy7wpr5p-maker/st-omr-training)"


def fetch_bytes(url: str, *, max_bytes: int) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/plain,application/xml,*/*;q=0.1"},
        method="GET",
    )
    with urlopen(request, timeout=60) as response:  # nosec B310 - exact pinned GitHub raw URLs only
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"upstream payload exceeds bounded size: {url}")
    if not data:
        raise RuntimeError(f"upstream payload is empty: {url}")
    return data


def main() -> None:
    b8n_receipt = receipt_from_json(B8N_RECEIPT_PATH.read_text(encoding="utf-8"))
    payloads: dict[str, OssqPairingSourcePayload] = {}
    for spec in B8O_PAIRING_SOURCE_SPECS:
        alignment = fetch_bytes(spec.alignment_raw_url, max_bytes=MAX_ALIGNMENT_BYTES)
        musicxml = fetch_bytes(spec.cleaned_musicxml_raw_url, max_bytes=MAX_MUSICXML_BYTES)
        print(f"B8O_INSPECT_SCORE={spec.score_id}")
        try:
            parse_scanned_alignment(alignment)
        except Exception:
            # Emit only the exact score identity and bounded textual alignment
            # metadata for diagnosis. No PDF/image bytes are logged.
            print("B8O_ALIGNMENT_DIAGNOSTIC=" + repr(alignment.decode("utf-8", errors="replace")))
            raise
        payloads[spec.score_id] = OssqPairingSourcePayload(
            alignment_bytes=alignment,
            cleaned_musicxml_bytes=musicxml,
        )

    receipt = build_b8o_pairing_preflight(
        b8n_receipt=b8n_receipt,
        payloads_by_score=payloads,
    )
    canonical = receipt_to_json(receipt)

    if B8O_RECEIPT_PATH.exists():
        committed = B8O_RECEIPT_PATH.read_text(encoding="utf-8").strip()
        if committed != canonical:
            raise RuntimeError("live B8O upstream evidence differs from committed receipt")
        print("B8O_MODE=VERIFIED")
        print(f"B8O_RECEIPT_SHA256={receipt.receipt_sha256}")
        print("B8O_LIVE_METADATA_MATCH_COMMITTED_RECEIPT=true")
    else:
        print("B8O_MODE=DISCOVERY")
        print(f"B8O_RECEIPT_SHA256={receipt.receipt_sha256}")
        print("B8O_READY_SCORE_IDS=" + ",".join(receipt.ready_score_ids))
        print("B8O_BLOCKED_SCORE_IDS=" + ",".join(receipt.blocked_score_ids))
        print("B8O_RECEIPT_JSON=" + canonical)

    print("B8O_RAW_UPSTREAM_BYTES_PERSISTED=false")
    print("B8O_PAIRING_REVIEW_AUTHORITY=false")
    print("B8O_STAGE8_ADMISSION_AUTHORITY=false")


if __name__ == "__main__":
    main()
