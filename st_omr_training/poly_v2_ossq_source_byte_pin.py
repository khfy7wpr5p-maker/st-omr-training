"""TR-POLY-09B8N exact source-byte pinning for the first OSSQ real-data batch.

B8M established a conservative research-training rights candidate set for five
exact IMSLP source PDFs. B8N binds those reviewed source identities to the
actual bytes fetched from the exact source URLs. Raw PDFs are temporary inputs
only: this module emits hash-only receipts and does not persist or redistribute
PDF bytes.

This package is not Stage 8 admission by itself. It does not create training
images, MusicXML pairings, TRAIN/VALIDATION splits, Stage 8-1 sample receipts,
model checkpoints, TEST access, production authority, or commercial authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Callable, Final, Iterable
from urllib.request import Request, urlopen

from .poly_v2_ossq_rights_evidence_batch1 import B8M_BATCH1_EVIDENCE


B8N_SOURCE_BYTE_PIN_VERSION: Final[str] = "st-omr-poly-v2-ossq-source-byte-pin-v1"
B8N_MAX_SOURCE_BYTES: Final[int] = 64 * 1024 * 1024
B8N_EXPECTED_SOURCE_COUNT: Final[int] = 5
B8N_EXPECTED_SCORE_COUNT: Final[int] = 8
B8N_USER_AGENT: Final[str] = (
    "ScoreMosaic-ST-OMR/1.0 (+https://github.com/khfy7wpr5p-maker/st-omr-training)"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OssqSourceBytePinError(ValueError):
    """Raised when B8N source-byte identity or receipt validation fails closed."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OssqSourceBytePinError(f"{name} must be lowercase SHA-256 text")
    return value


def b8m_rights_evidence_manifest_sha256() -> str:
    """Bind B8N to the exact merged B8M evidence population and decisions."""

    payload = {
        "version": B8N_SOURCE_BYTE_PIN_VERSION,
        "evidence": [item.canonical_sha256() for item in B8M_BATCH1_EVIDENCE],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OssqSourceBytePin:
    imslp_id: str
    score_ids: tuple[str, ...]
    source_url: str
    source_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.imslp_id, str) or not self.imslp_id.startswith("#") or not self.imslp_id[1:].isdigit():
            raise OssqSourceBytePinError("imslp_id must be an IMSLP file identifier")
        if not self.score_ids or any(not isinstance(value, str) or not value.isdigit() for value in self.score_ids):
            raise OssqSourceBytePinError("score_ids must contain decimal text")
        if tuple(sorted(self.score_ids, key=int)) != self.score_ids or len(set(self.score_ids)) != len(self.score_ids):
            raise OssqSourceBytePinError("score_ids must be unique and numerically sorted")
        if not isinstance(self.source_url, str) or not self.source_url.startswith("https://"):
            raise OssqSourceBytePinError("source_url must use https")
        if "imslp" not in self.source_url.lower():
            raise OssqSourceBytePinError("source_url must identify the reviewed IMSLP source")
        _require_sha256("source_sha256", self.source_sha256)
        if not isinstance(self.byte_count, int) or isinstance(self.byte_count, bool) or not 5 <= self.byte_count <= B8N_MAX_SOURCE_BYTES:
            raise OssqSourceBytePinError("byte_count is outside the B8N source-document bounds")

    def canonical_sha256(self) -> str:
        payload = asdict(self)
        payload["score_ids"] = list(self.score_ids)
        return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OssqSourceBytePinReceipt:
    rights_evidence_manifest_sha256: str
    pins: tuple[OssqSourceBytePin, ...]
    receipt_sha256: str
    version: str = B8N_SOURCE_BYTE_PIN_VERSION
    raw_source_bytes_persisted: bool = False
    raw_source_bytes_redistributed: bool = False
    stage8_admission_authority: bool = False
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False
    commercial_use_authority: bool = False

    def __post_init__(self) -> None:
        _require_sha256("rights_evidence_manifest_sha256", self.rights_evidence_manifest_sha256)
        _require_sha256("receipt_sha256", self.receipt_sha256)
        if self.rights_evidence_manifest_sha256 != b8m_rights_evidence_manifest_sha256():
            raise OssqSourceBytePinError("B8N receipt is not bound to the exact B8M rights-evidence population")
        if self.version != B8N_SOURCE_BYTE_PIN_VERSION:
            raise OssqSourceBytePinError("B8N receipt version mismatch")
        if len(self.pins) != B8N_EXPECTED_SOURCE_COUNT:
            raise OssqSourceBytePinError("B8N receipt must contain exactly five reviewed source pins")
        if tuple(sorted(self.pins, key=lambda item: int(item.imslp_id[1:]))) != self.pins:
            raise OssqSourceBytePinError("B8N pins must be numerically IMSLP-id sorted")
        if len({item.imslp_id for item in self.pins}) != len(self.pins):
            raise OssqSourceBytePinError("B8N receipt contains duplicate IMSLP source ids")
        score_ids = tuple(sorted((score_id for pin in self.pins for score_id in pin.score_ids), key=int))
        if len(score_ids) != B8N_EXPECTED_SCORE_COUNT or len(set(score_ids)) != B8N_EXPECTED_SCORE_COUNT:
            raise OssqSourceBytePinError("B8N receipt score population differs from the eight B8M candidates")
        expected_scores = tuple(sorted((score_id for item in B8M_BATCH1_EVIDENCE for score_id in item.score_ids), key=int))
        if score_ids != expected_scores:
            raise OssqSourceBytePinError("B8N receipt score population differs from B8M")
        expected_by_imslp = {item.imslp_id: item for item in B8M_BATCH1_EVIDENCE}
        for pin in self.pins:
            evidence = expected_by_imslp.get(pin.imslp_id)
            if evidence is None:
                raise OssqSourceBytePinError(f"B8N receipt contains unreviewed IMSLP source {pin.imslp_id}")
            if pin.score_ids != evidence.score_ids or pin.source_url != evidence.source_pdf_url:
                raise OssqSourceBytePinError(f"B8N pin identity drifted for {pin.imslp_id}")
        if any(
            (
                self.raw_source_bytes_persisted,
                self.raw_source_bytes_redistributed,
                self.stage8_admission_authority,
                self.test_artifact_bytes_accessed,
                self.production_authority,
                self.commercial_use_authority,
            )
        ):
            raise OssqSourceBytePinError("B8N may not persist/redistribute source bytes or grant admission/product authority")
        if self.receipt_sha256 != _receipt_sha256_without_self(self):
            raise OssqSourceBytePinError("B8N receipt_sha256 does not match canonical receipt payload")

    def canonical_payload(self) -> dict[str, object]:
        return _receipt_payload_without_self(self)


def _receipt_payload_without_self(receipt: OssqSourceBytePinReceipt) -> dict[str, object]:
    return {
        "version": receipt.version,
        "rights_evidence_manifest_sha256": receipt.rights_evidence_manifest_sha256,
        "pins": [
            {
                "imslp_id": pin.imslp_id,
                "score_ids": list(pin.score_ids),
                "source_url": pin.source_url,
                "source_sha256": pin.source_sha256,
                "byte_count": pin.byte_count,
            }
            for pin in receipt.pins
        ],
        "raw_source_bytes_persisted": receipt.raw_source_bytes_persisted,
        "raw_source_bytes_redistributed": receipt.raw_source_bytes_redistributed,
        "stage8_admission_authority": receipt.stage8_admission_authority,
        "test_artifact_bytes_accessed": receipt.test_artifact_bytes_accessed,
        "production_authority": receipt.production_authority,
        "commercial_use_authority": receipt.commercial_use_authority,
    }


def _receipt_sha256_without_self(receipt: OssqSourceBytePinReceipt) -> str:
    return sha256(_canonical_json_bytes(_receipt_payload_without_self(receipt))).hexdigest()


def build_source_byte_pin_receipt(pins: Iterable[OssqSourceBytePin]) -> OssqSourceBytePinReceipt:
    values = tuple(sorted(tuple(pins), key=lambda item: int(item.imslp_id[1:])))
    placeholder = object.__new__(OssqSourceBytePinReceipt)
    object.__setattr__(placeholder, "rights_evidence_manifest_sha256", b8m_rights_evidence_manifest_sha256())
    object.__setattr__(placeholder, "pins", values)
    object.__setattr__(placeholder, "receipt_sha256", "0" * 64)
    object.__setattr__(placeholder, "version", B8N_SOURCE_BYTE_PIN_VERSION)
    object.__setattr__(placeholder, "raw_source_bytes_persisted", False)
    object.__setattr__(placeholder, "raw_source_bytes_redistributed", False)
    object.__setattr__(placeholder, "stage8_admission_authority", False)
    object.__setattr__(placeholder, "test_artifact_bytes_accessed", False)
    object.__setattr__(placeholder, "production_authority", False)
    object.__setattr__(placeholder, "commercial_use_authority", False)
    digest = _receipt_sha256_without_self(placeholder)
    return OssqSourceBytePinReceipt(
        rights_evidence_manifest_sha256=b8m_rights_evidence_manifest_sha256(),
        pins=values,
        receipt_sha256=digest,
    )


def pin_source_payloads(payloads_by_imslp: dict[str, bytes]) -> OssqSourceBytePinReceipt:
    """Hash exact caller-supplied source PDFs and emit a hash-only B8N receipt."""

    if not isinstance(payloads_by_imslp, dict):
        raise TypeError("payloads_by_imslp must be a dict")
    expected = {item.imslp_id for item in B8M_BATCH1_EVIDENCE}
    if set(payloads_by_imslp) != expected:
        raise OssqSourceBytePinError("source payload population differs from the exact B8M evidence batch")
    pins: list[OssqSourceBytePin] = []
    evidence_by_imslp = {item.imslp_id: item for item in B8M_BATCH1_EVIDENCE}
    for imslp_id in sorted(expected, key=lambda value: int(value[1:])):
        data = payloads_by_imslp[imslp_id]
        if not isinstance(data, bytes) or not data:
            raise OssqSourceBytePinError(f"source bytes for {imslp_id} must be non-empty bytes")
        if len(data) > B8N_MAX_SOURCE_BYTES:
            raise OssqSourceBytePinError(f"source bytes for {imslp_id} exceed the B8N limit")
        if not data.startswith(b"%PDF-"):
            raise OssqSourceBytePinError(f"source bytes for {imslp_id} are not a PDF")
        evidence = evidence_by_imslp[imslp_id]
        pins.append(
            OssqSourceBytePin(
                imslp_id=imslp_id,
                score_ids=evidence.score_ids,
                source_url=evidence.source_pdf_url,
                source_sha256=sha256(data).hexdigest(),
                byte_count=len(data),
            )
        )
    return build_source_byte_pin_receipt(pins)


def _default_fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": B8N_USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.1",
        },
        method="GET",
    )
    with urlopen(request, timeout=60) as response:  # nosec B310 - exact B8M-reviewed HTTPS URLs only
        data = response.read(B8N_MAX_SOURCE_BYTES + 1)
    if len(data) > B8N_MAX_SOURCE_BYTES:
        raise OssqSourceBytePinError("downloaded source document exceeds the B8N byte limit")
    return data


def fetch_and_pin_b8m_sources(*, fetcher: Callable[[str], bytes] = _default_fetch) -> OssqSourceBytePinReceipt:
    """Fetch only the five exact B8M-approved HTTPS URLs and return hash-only identity."""

    if not callable(fetcher):
        raise TypeError("fetcher must be callable")
    payloads: dict[str, bytes] = {}
    for evidence in B8M_BATCH1_EVIDENCE:
        payloads[evidence.imslp_id] = fetcher(evidence.source_pdf_url)
    return pin_source_payloads(payloads)


def receipt_to_json(receipt: OssqSourceBytePinReceipt) -> str:
    if not isinstance(receipt, OssqSourceBytePinReceipt):
        raise TypeError("receipt must be OssqSourceBytePinReceipt")
    payload = {**receipt.canonical_payload(), "receipt_sha256": receipt.receipt_sha256}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def receipt_from_json(text: str) -> OssqSourceBytePinReceipt:
    if not isinstance(text, str) or not text.strip():
        raise OssqSourceBytePinError("receipt JSON must be non-empty text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OssqSourceBytePinError("receipt JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise OssqSourceBytePinError("receipt JSON root must be an object")
    required = {
        "version",
        "rights_evidence_manifest_sha256",
        "pins",
        "receipt_sha256",
        "raw_source_bytes_persisted",
        "raw_source_bytes_redistributed",
        "stage8_admission_authority",
        "test_artifact_bytes_accessed",
        "production_authority",
        "commercial_use_authority",
    }
    if set(payload) != required:
        raise OssqSourceBytePinError("receipt JSON fields differ from the B8N schema")
    if not isinstance(payload["pins"], list):
        raise OssqSourceBytePinError("receipt pins must be a list")
    pins: list[OssqSourceBytePin] = []
    for item in payload["pins"]:
        if not isinstance(item, dict) or set(item) != {"imslp_id", "score_ids", "source_url", "source_sha256", "byte_count"}:
            raise OssqSourceBytePinError("receipt pin fields differ from the B8N schema")
        if not isinstance(item["score_ids"], list):
            raise OssqSourceBytePinError("receipt pin score_ids must be a list")
        pins.append(
            OssqSourceBytePin(
                imslp_id=item["imslp_id"],
                score_ids=tuple(item["score_ids"]),
                source_url=item["source_url"],
                source_sha256=item["source_sha256"],
                byte_count=item["byte_count"],
            )
        )
    return OssqSourceBytePinReceipt(
        rights_evidence_manifest_sha256=payload["rights_evidence_manifest_sha256"],
        pins=tuple(pins),
        receipt_sha256=payload["receipt_sha256"],
        version=payload["version"],
        raw_source_bytes_persisted=payload["raw_source_bytes_persisted"],
        raw_source_bytes_redistributed=payload["raw_source_bytes_redistributed"],
        stage8_admission_authority=payload["stage8_admission_authority"],
        test_artifact_bytes_accessed=payload["test_artifact_bytes_accessed"],
        production_authority=payload["production_authority"],
        commercial_use_authority=payload["commercial_use_authority"],
    )


def verify_live_receipt(expected: OssqSourceBytePinReceipt, actual: OssqSourceBytePinReceipt) -> None:
    if not isinstance(expected, OssqSourceBytePinReceipt) or not isinstance(actual, OssqSourceBytePinReceipt):
        raise TypeError("expected and actual must be OssqSourceBytePinReceipt")
    if receipt_to_json(actual) != receipt_to_json(expected):
        raise OssqSourceBytePinError("live OSSQ source bytes differ from the committed B8N receipt")
