"""TR-POLY-09B8O exact OSSQ image/MusicXML pairing-source preflight.

B8N pinned the exact source-PDF bytes for the first five reviewed IMSLP
sources. B8O binds those source identities to the exact camera-ready OSSQ
alignment metadata and cleaned MusicXML annotation bytes for the eight score
records in that source batch.

This module deliberately stops before claiming that an image/MusicXML pair is
verified. It identifies which score records have enough upstream alignment
metadata to proceed to deterministic system-image materialization. Raw PDF,
alignment and MusicXML bytes are caller-supplied and are not persisted here.
Opaque alignment markers whose semantics are not established by the pinned
preprocessor remain blocked rather than being guessed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha1, sha256
import json
import re
from typing import Final, Mapping
from urllib.parse import quote

from .poly_v2_corpus_source_selection import OSSQ_OMR_CAMERA_READY_SHA
from .poly_v2_ossq_source_byte_pin import OssqSourceBytePinReceipt


B8O_PAIRING_PREFLIGHT_VERSION: Final[str] = "st-omr-poly-v2-ossq-pairing-preflight-v1"
B8O_EXPECTED_B8N_RECEIPT_SHA256: Final[str] = (
    "9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c"
)
B8O_PREPROCESSOR_COMMIT_SHA: Final[str] = "bdea0d1829c9db84480ebd2e0385f6f5fe324274"
B8O_EXPECTED_SCORE_COUNT: Final[int] = 8
B8O_EXPECTED_READY_SCORE_COUNT: Final[int] = 7
B8O_EXPECTED_BLOCKED_SCORE_IDS: Final[tuple[str, ...]] = ("7397765",)
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OssqPairingPreflightError(ValueError):
    """Raised when B8O evidence or identity validation fails closed."""


class PairingMaterializationState(str, Enum):
    READY = "ready-for-materialization"
    BLOCKED_EMPTY_ALIGNMENT = "blocked-empty-alignment"
    BLOCKED_UNINTERPRETED_ALIGNMENT_MARKER = "blocked-uninterpreted-alignment-marker"


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
        raise OssqPairingPreflightError(f"{name} must be lowercase SHA-256 text")
    return value


def _require_git_sha1(name: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA1_RE.fullmatch(value) is None:
        raise OssqPairingPreflightError(f"{name} must be lowercase Git SHA-1 text")
    return value


def git_blob_sha1(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("Git blob input must be bytes")
    header = f"blob {len(data)}\0".encode("ascii")
    return sha1(header + data).hexdigest()  # nosec B324 - Git object identity, not security use


@dataclass(frozen=True, slots=True)
class OssqPairingSourceSpec:
    score_id: str
    imslp_id: str
    work_path: str
    alignment_git_blob_sha1: str
    cleaned_musicxml_git_blob_sha1: str

    def __post_init__(self) -> None:
        if not isinstance(self.score_id, str) or not self.score_id.isdigit():
            raise OssqPairingPreflightError("score_id must be decimal text")
        if not isinstance(self.imslp_id, str) or not self.imslp_id.startswith("#") or not self.imslp_id[1:].isdigit():
            raise OssqPairingPreflightError("imslp_id must be an IMSLP file identifier")
        if not isinstance(self.work_path, str) or not self.work_path or self.work_path.startswith("/") or ".." in self.work_path.split("/"):
            raise OssqPairingPreflightError("work_path must be a bounded repository-relative path")
        _require_git_sha1("alignment_git_blob_sha1", self.alignment_git_blob_sha1)
        _require_git_sha1("cleaned_musicxml_git_blob_sha1", self.cleaned_musicxml_git_blob_sha1)

    @property
    def alignment_raw_url(self) -> str:
        path = quote(f"scores/{self.work_path}/sq{self.score_id}_scanned.csv", safe="/")
        return f"https://raw.githubusercontent.com/MALerLab/ossq-omr/{OSSQ_OMR_CAMERA_READY_SHA}/{path}"

    @property
    def cleaned_musicxml_raw_url(self) -> str:
        path = quote(f"scores/{self.work_path}/sq{self.score_id}_cleaned.musicxml", safe="/")
        return f"https://raw.githubusercontent.com/MALerLab/ossq-omr/{OSSQ_OMR_CAMERA_READY_SHA}/{path}"


B8O_PAIRING_SOURCE_SPECS: Final[tuple[OssqPairingSourceSpec, ...]] = (
    OssqPairingSourceSpec(
        score_id="7070781",
        imslp_id="#64141",
        work_path="Mozart,_Wolfgang_Amadeus/String_Quartet_No.18_in_A_major,_K.464_(Op._10,_No._5)",
        alignment_git_blob_sha1="a42c5f0858b389155507d8af641225d3f220dc8a",
        cleaned_musicxml_git_blob_sha1="703da90053e0f78d64377f9e6a7438ab90c1828f",
    ),
    OssqPairingSourceSpec(
        score_id="7075297",
        imslp_id="#64141",
        work_path="Mozart,_Wolfgang_Amadeus/String_Quartet_No.18_in_A_major,_K.464_(Op._10,_No._5)",
        alignment_git_blob_sha1="91a64aebd4d82ed6c06cbbffb19b5fcf4570495e",
        cleaned_musicxml_git_blob_sha1="3ab1673d8d359e67c647ee9b1c9a9cc00b461111",
    ),
    OssqPairingSourceSpec(
        score_id="7078259",
        imslp_id="#64141",
        work_path="Mozart,_Wolfgang_Amadeus/String_Quartet_No.18_in_A_major,_K.464_(Op._10,_No._5)",
        alignment_git_blob_sha1="982bed3a6cf2216209a58ed637f57af8a90cc9fb",
        cleaned_musicxml_git_blob_sha1="9d00bc5739da458e6fe9f4e68956dafad6b0f5ae",
    ),
    OssqPairingSourceSpec(
        score_id="7093885",
        imslp_id="#64141",
        work_path="Mozart,_Wolfgang_Amadeus/String_Quartet_No.18_in_A_major,_K.464_(Op._10,_No._5)",
        alignment_git_blob_sha1="90466d5f302e7782b6784170c5ac0be2f7d5e6f9",
        cleaned_musicxml_git_blob_sha1="46b5e8fbb7b9bd59b315697a877523861937cc6f",
    ),
    OssqPairingSourceSpec(
        score_id="7103818",
        imslp_id="#64136",
        work_path="Mozart,_Wolfgang_Amadeus/String_Quartet_No.14_in_G_major,_K.387_(Op._10,_No._1)",
        alignment_git_blob_sha1="14ba054c5e0645048320c81ae8af0e07d1959557",
        cleaned_musicxml_git_blob_sha1="a58b55d4ea687982c0d5365be54710e687bced17",
    ),
    OssqPairingSourceSpec(
        score_id="7108150",
        imslp_id="#242305",
        work_path="Brahms,_Johannes/String_Quartet_No.1,_Op.51_No.1",
        alignment_git_blob_sha1="5db572988155cba10045485d28eb99588c10dc9d",
        cleaned_musicxml_git_blob_sha1="e57a7b229aac3ea43e4328cb2edec059a637f7fe",
    ),
    OssqPairingSourceSpec(
        score_id="7397765",
        imslp_id="#04047",
        work_path="Schubert,_Franz/String_Quartet_in_D_minor,_D.810,_Op.14_(“Death_and_the_Maiden”)",
        alignment_git_blob_sha1="9f741e97d7b63b77b64788c3167e69617beefec1",
        cleaned_musicxml_git_blob_sha1="5c9b4360e86c2890ed2603f2b9e8620b35f1a73b",
    ),
    OssqPairingSourceSpec(
        score_id="8071278",
        imslp_id="#04755",
        work_path="Beethoven,_Ludwig_van/String_Quartet_No.1,_Op.18_No.1",
        alignment_git_blob_sha1="5d3314d2dae8a402e0216c5cfd7abc95475e7d0a",
        cleaned_musicxml_git_blob_sha1="62181f8e389214252d43cd91cbeaa8d014db2a27",
    ),
)


@dataclass(frozen=True, slots=True)
class ScannedAlignmentProfile:
    page_start: int | None
    page_end: int | None
    block_count: int
    row_count: int
    value_count: int
    value_sum: int
    marker_tokens: tuple[str, ...]

    @property
    def has_alignment_values(self) -> bool:
        return self.value_count > 0

    @property
    def has_uninterpreted_markers(self) -> bool:
        return any(marker != "x" for marker in self.marker_tokens)


def parse_scanned_alignment(data: bytes) -> ScannedAlignmentProfile:
    if not isinstance(data, bytes) or not data:
        raise OssqPairingPreflightError("alignment metadata must be non-empty bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OssqPairingPreflightError("alignment metadata must be UTF-8") from exc
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or ":" not in lines[0]:
        raise OssqPairingPreflightError("alignment metadata must begin with a page-range line")
    if lines[0].count(":") != 1:
        raise OssqPairingPreflightError("alignment page-range line is malformed")
    start_text, end_text = (part.strip() for part in lines[0].split(":", 1))

    def parse_bound(name: str, value: str) -> int | None:
        if value == "":
            return None
        if not value.isdigit() or int(value) < 1:
            raise OssqPairingPreflightError(f"alignment {name} must be a positive integer")
        return int(value)

    page_start = parse_bound("page_start", start_text)
    page_end = parse_bound("page_end", end_text)
    if page_start is not None and page_end is not None and page_end < page_start:
        raise OssqPairingPreflightError("alignment page range is reversed")

    block_count = 0
    row_count = 0
    values: list[int] = []
    markers: set[str] = set()
    in_block = False
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            in_block = False
            continue
        if not in_block:
            block_count += 1
            in_block = True
        row_count += 1
        cells = [cell.strip() for cell in line.split(",")]
        row_has_token = False
        for cell in cells:
            if cell == "":
                continue
            row_has_token = True
            if cell in {"x", "a"}:
                # `x` is visibly used alongside numeric alignment values in the
                # camera-ready OSSQ files. `a` is present in the Schubert D.810
                # file, but its semantics are not established by the pinned
                # preprocessor. Both remain byte-bound; `a` fails readiness.
                markers.add(cell)
                continue
            if not cell.isdigit() or int(cell) < 1:
                raise OssqPairingPreflightError(
                    "alignment cells must be positive integers or a pinned OSSQ marker"
                )
            values.append(int(cell))
        if not row_has_token:
            raise OssqPairingPreflightError("alignment row contains no alignment tokens")
    return ScannedAlignmentProfile(
        page_start=page_start,
        page_end=page_end,
        block_count=block_count,
        row_count=row_count,
        value_count=len(values),
        value_sum=sum(values),
        marker_tokens=tuple(sorted(markers)),
    )


@dataclass(frozen=True, slots=True)
class OssqPairingSourcePayload:
    alignment_bytes: bytes
    cleaned_musicxml_bytes: bytes


@dataclass(frozen=True, slots=True)
class OssqPairingSourceEvidence:
    score_id: str
    imslp_id: str
    source_document_sha256: str
    alignment_git_blob_sha1: str
    alignment_sha256: str
    cleaned_musicxml_git_blob_sha1: str
    cleaned_musicxml_sha256: str
    alignment_profile: ScannedAlignmentProfile
    state: PairingMaterializationState

    def canonical_payload(self) -> dict[str, object]:
        return {
            "score_id": self.score_id,
            "imslp_id": self.imslp_id,
            "source_document_sha256": self.source_document_sha256,
            "alignment_git_blob_sha1": self.alignment_git_blob_sha1,
            "alignment_sha256": self.alignment_sha256,
            "cleaned_musicxml_git_blob_sha1": self.cleaned_musicxml_git_blob_sha1,
            "cleaned_musicxml_sha256": self.cleaned_musicxml_sha256,
            "alignment_profile": asdict(self.alignment_profile),
            "state": self.state.value,
        }


def inspect_pairing_source(
    spec: OssqPairingSourceSpec,
    *,
    source_document_sha256: str,
    payload: OssqPairingSourcePayload,
) -> OssqPairingSourceEvidence:
    if not isinstance(spec, OssqPairingSourceSpec):
        raise TypeError("spec must be OssqPairingSourceSpec")
    _require_sha256("source_document_sha256", source_document_sha256)
    if not isinstance(payload, OssqPairingSourcePayload):
        raise TypeError("payload must be OssqPairingSourcePayload")
    if git_blob_sha1(payload.alignment_bytes) != spec.alignment_git_blob_sha1:
        raise OssqPairingPreflightError(f"alignment Git blob identity differs for score {spec.score_id}")
    if git_blob_sha1(payload.cleaned_musicxml_bytes) != spec.cleaned_musicxml_git_blob_sha1:
        raise OssqPairingPreflightError(f"cleaned MusicXML Git blob identity differs for score {spec.score_id}")
    profile = parse_scanned_alignment(payload.alignment_bytes)
    if not payload.cleaned_musicxml_bytes.strip():
        raise OssqPairingPreflightError(f"cleaned MusicXML is empty for score {spec.score_id}")
    head = payload.cleaned_musicxml_bytes[:131072]
    if b"<score-partwise" not in head and b"<score-timewise" not in head:
        raise OssqPairingPreflightError(f"cleaned MusicXML envelope is missing for score {spec.score_id}")
    if profile.has_uninterpreted_markers:
        state = PairingMaterializationState.BLOCKED_UNINTERPRETED_ALIGNMENT_MARKER
    elif profile.has_alignment_values:
        state = PairingMaterializationState.READY
    else:
        state = PairingMaterializationState.BLOCKED_EMPTY_ALIGNMENT
    return OssqPairingSourceEvidence(
        score_id=spec.score_id,
        imslp_id=spec.imslp_id,
        source_document_sha256=source_document_sha256,
        alignment_git_blob_sha1=spec.alignment_git_blob_sha1,
        alignment_sha256=sha256(payload.alignment_bytes).hexdigest(),
        cleaned_musicxml_git_blob_sha1=spec.cleaned_musicxml_git_blob_sha1,
        cleaned_musicxml_sha256=sha256(payload.cleaned_musicxml_bytes).hexdigest(),
        alignment_profile=profile,
        state=state,
    )


@dataclass(frozen=True, slots=True)
class OssqPairingPreflightReceipt:
    b8n_source_receipt_sha256: str
    source_commit_sha: str
    preprocessor_commit_sha: str
    entries: tuple[OssqPairingSourceEvidence, ...]
    ready_score_ids: tuple[str, ...]
    blocked_score_ids: tuple[str, ...]
    receipt_sha256: str
    version: str = B8O_PAIRING_PREFLIGHT_VERSION
    image_bytes_materialized: bool = False
    pairing_review_authority: bool = False
    stage8_admission_authority: bool = False
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False
    commercial_use_authority: bool = False

    def __post_init__(self) -> None:
        _require_sha256("b8n_source_receipt_sha256", self.b8n_source_receipt_sha256)
        _require_sha256("receipt_sha256", self.receipt_sha256)
        if self.b8n_source_receipt_sha256 != B8O_EXPECTED_B8N_RECEIPT_SHA256:
            raise OssqPairingPreflightError("B8O receipt is not bound to the exact B8N source-byte receipt")
        if self.source_commit_sha != OSSQ_OMR_CAMERA_READY_SHA:
            raise OssqPairingPreflightError("B8O OSSQ source commit drifted")
        if self.preprocessor_commit_sha != B8O_PREPROCESSOR_COMMIT_SHA:
            raise OssqPairingPreflightError("B8O preprocessor commit drifted")
        if self.version != B8O_PAIRING_PREFLIGHT_VERSION:
            raise OssqPairingPreflightError("B8O receipt version mismatch")
        if len(self.entries) != B8O_EXPECTED_SCORE_COUNT:
            raise OssqPairingPreflightError("B8O receipt must cover exactly eight score records")
        if tuple(sorted(self.entries, key=lambda item: int(item.score_id))) != self.entries:
            raise OssqPairingPreflightError("B8O entries must be score-id sorted")
        if len({item.score_id for item in self.entries}) != len(self.entries):
            raise OssqPairingPreflightError("B8O receipt contains duplicate score ids")
        expected_ids = tuple(spec.score_id for spec in B8O_PAIRING_SOURCE_SPECS)
        if tuple(item.score_id for item in self.entries) != expected_ids:
            raise OssqPairingPreflightError("B8O score population differs from the exact first source batch")
        computed_ready = tuple(item.score_id for item in self.entries if item.state is PairingMaterializationState.READY)
        computed_blocked = tuple(item.score_id for item in self.entries if item.state is not PairingMaterializationState.READY)
        if self.ready_score_ids != computed_ready or self.blocked_score_ids != computed_blocked:
            raise OssqPairingPreflightError("B8O readiness populations do not match entry states")
        if len(self.ready_score_ids) != B8O_EXPECTED_READY_SCORE_COUNT:
            raise OssqPairingPreflightError("B8O current camera-ready readiness population drifted")
        if self.blocked_score_ids != B8O_EXPECTED_BLOCKED_SCORE_IDS:
            raise OssqPairingPreflightError("B8O current blocked population drifted")
        if any(
            (
                self.image_bytes_materialized,
                self.pairing_review_authority,
                self.stage8_admission_authority,
                self.test_artifact_bytes_accessed,
                self.production_authority,
                self.commercial_use_authority,
            )
        ):
            raise OssqPairingPreflightError("B8O preflight may not grant materialization/review/admission/product authority")
        if self.receipt_sha256 != _receipt_sha256_without_self(self):
            raise OssqPairingPreflightError("B8O receipt_sha256 does not match canonical payload")

    def canonical_payload(self) -> dict[str, object]:
        return _receipt_payload_without_self(self)


def _receipt_payload_without_self(receipt: OssqPairingPreflightReceipt) -> dict[str, object]:
    return {
        "version": receipt.version,
        "b8n_source_receipt_sha256": receipt.b8n_source_receipt_sha256,
        "source_commit_sha": receipt.source_commit_sha,
        "preprocessor_commit_sha": receipt.preprocessor_commit_sha,
        "entries": [item.canonical_payload() for item in receipt.entries],
        "ready_score_ids": list(receipt.ready_score_ids),
        "blocked_score_ids": list(receipt.blocked_score_ids),
        "image_bytes_materialized": receipt.image_bytes_materialized,
        "pairing_review_authority": receipt.pairing_review_authority,
        "stage8_admission_authority": receipt.stage8_admission_authority,
        "test_artifact_bytes_accessed": receipt.test_artifact_bytes_accessed,
        "production_authority": receipt.production_authority,
        "commercial_use_authority": receipt.commercial_use_authority,
    }


def _receipt_sha256_without_self(receipt: OssqPairingPreflightReceipt) -> str:
    return sha256(_canonical_json_bytes(_receipt_payload_without_self(receipt))).hexdigest()


def build_b8o_pairing_preflight(
    *,
    b8n_receipt: OssqSourceBytePinReceipt,
    payloads_by_score: Mapping[str, OssqPairingSourcePayload],
) -> OssqPairingPreflightReceipt:
    if not isinstance(b8n_receipt, OssqSourceBytePinReceipt):
        raise TypeError("b8n_receipt must be OssqSourceBytePinReceipt")
    if b8n_receipt.receipt_sha256 != B8O_EXPECTED_B8N_RECEIPT_SHA256:
        raise OssqPairingPreflightError("B8O requires the exact merged B8N source receipt")
    expected_ids = tuple(spec.score_id for spec in B8O_PAIRING_SOURCE_SPECS)
    if set(payloads_by_score) != set(expected_ids):
        raise OssqPairingPreflightError("B8O payload population differs from the exact eight-score source batch")
    pin_by_score: dict[str, tuple[str, str]] = {}
    for pin in b8n_receipt.pins:
        for score_id in pin.score_ids:
            if score_id in pin_by_score:
                raise OssqPairingPreflightError("B8N source receipt reuses a score id")
            pin_by_score[score_id] = (pin.imslp_id, pin.source_sha256)
    if set(pin_by_score) != set(expected_ids):
        raise OssqPairingPreflightError("B8N source population differs from B8O")

    entries: list[OssqPairingSourceEvidence] = []
    for spec in B8O_PAIRING_SOURCE_SPECS:
        pin_imslp_id, source_sha256 = pin_by_score[spec.score_id]
        if pin_imslp_id != spec.imslp_id:
            raise OssqPairingPreflightError(f"B8N IMSLP source differs for score {spec.score_id}")
        entries.append(
            inspect_pairing_source(
                spec,
                source_document_sha256=source_sha256,
                payload=payloads_by_score[spec.score_id],
            )
        )
    entries_tuple = tuple(entries)
    ready = tuple(item.score_id for item in entries_tuple if item.state is PairingMaterializationState.READY)
    blocked = tuple(item.score_id for item in entries_tuple if item.state is not PairingMaterializationState.READY)

    placeholder = object.__new__(OssqPairingPreflightReceipt)
    object.__setattr__(placeholder, "b8n_source_receipt_sha256", b8n_receipt.receipt_sha256)
    object.__setattr__(placeholder, "source_commit_sha", OSSQ_OMR_CAMERA_READY_SHA)
    object.__setattr__(placeholder, "preprocessor_commit_sha", B8O_PREPROCESSOR_COMMIT_SHA)
    object.__setattr__(placeholder, "entries", entries_tuple)
    object.__setattr__(placeholder, "ready_score_ids", ready)
    object.__setattr__(placeholder, "blocked_score_ids", blocked)
    object.__setattr__(placeholder, "receipt_sha256", "0" * 64)
    object.__setattr__(placeholder, "version", B8O_PAIRING_PREFLIGHT_VERSION)
    object.__setattr__(placeholder, "image_bytes_materialized", False)
    object.__setattr__(placeholder, "pairing_review_authority", False)
    object.__setattr__(placeholder, "stage8_admission_authority", False)
    object.__setattr__(placeholder, "test_artifact_bytes_accessed", False)
    object.__setattr__(placeholder, "production_authority", False)
    object.__setattr__(placeholder, "commercial_use_authority", False)
    digest = _receipt_sha256_without_self(placeholder)
    return OssqPairingPreflightReceipt(
        b8n_source_receipt_sha256=b8n_receipt.receipt_sha256,
        source_commit_sha=OSSQ_OMR_CAMERA_READY_SHA,
        preprocessor_commit_sha=B8O_PREPROCESSOR_COMMIT_SHA,
        entries=entries_tuple,
        ready_score_ids=ready,
        blocked_score_ids=blocked,
        receipt_sha256=digest,
    )


def receipt_to_json(receipt: OssqPairingPreflightReceipt) -> str:
    if not isinstance(receipt, OssqPairingPreflightReceipt):
        raise TypeError("receipt must be OssqPairingPreflightReceipt")
    payload = {**receipt.canonical_payload(), "receipt_sha256": receipt.receipt_sha256}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
