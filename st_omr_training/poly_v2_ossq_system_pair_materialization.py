"""TR-POLY-09B8P real OSSQ system-image/MusicXML materialization evidence.

B8P consumes only the seven B8O READY score records. It validates a reproduced
OSSQ/preprocessor output tree and emits a hash-only receipt for each final
systemwise PNG <-> MusicXML pair. Upstream YOLO `system.ignores` and
`system.exceptions` are preserved as explicit exclusions; they are never
silently intersected away.

B8P does not independently approve pairing, admit Stage 8 data, assign
TRAIN/VALIDATION, open TEST, or authorize training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from typing import Final, Mapping

from .poly_v2_corpus_source_selection import OSSQ_OMR_CAMERA_READY_SHA
from .poly_v2_ossq_pairing_preflight import (
    B8O_PREPROCESSOR_COMMIT_SHA,
    B8O_PAIRING_SOURCE_SPECS,
)

B8P_MATERIALIZATION_VERSION: Final[str] = "st-omr-poly-v2-ossq-system-pair-materialization-v1"
B8P_EXPECTED_B8N_RECEIPT_SHA256: Final[str] = "9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c"
B8P_EXPECTED_B8O_RECEIPT_SHA256: Final[str] = "b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a"
B8P_READY_SCORE_IDS: Final[tuple[str, ...]] = (
    "7070781", "7075297", "7078259", "7093885", "7103818", "7108150", "8071278"
)
B8P_BLOCKED_SCORE_IDS: Final[tuple[str, ...]] = ("7397765",)
_SEGMENT_RE = re.compile(r"^sq(?P<score>[0-9]+):(?P<page>[0-9]{4}):(?P<system>[0-9]{4})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_YOLO_INFO_VERSION = "0.1.0"


class OssqSystemPairMaterializationError(ValueError):
    """Raised when B8P materialization evidence fails closed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OssqSystemPairMaterializationError(f"{name} must be lowercase SHA-256")
    return value


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != _PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise OssqSystemPairMaterializationError("system image is not a valid PNG envelope")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise OssqSystemPairMaterializationError("system image dimensions must be positive")
    return width, height


def _validate_musicxml(data: bytes) -> None:
    if not data.strip():
        raise OssqSystemPairMaterializationError("system MusicXML is empty")
    head = data[:131072]
    if b"<score-partwise" not in head and b"<score-timewise" not in head:
        raise OssqSystemPairMaterializationError("system MusicXML envelope is missing")


def _parse_system_exclusions(yolo_bytes: bytes, *, score_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse the bounded camera-ready YOLOInfo surface without a YAML dependency.

    Only the exact scalar/list shape used by YOLOInfo v0.1.0 is accepted for
    `system.ignores` and `system.exceptions`. Unknown top-level content is
    tolerated because the receipt separately hash-binds the complete YAML.
    """
    try:
        text = yolo_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OssqSystemPairMaterializationError("yolo_info must be UTF-8") from exc

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    version = None
    data_id = None
    image_type = None
    system_start = None
    for index, raw in enumerate(lines):
        if raw.startswith("version:"):
            version = raw.split(":", 1)[1].strip()
        elif raw.startswith("data_id:"):
            data_id = raw.split(":", 1)[1].strip()
        elif raw.startswith("image_type:"):
            image_type = raw.split(":", 1)[1].strip()
        elif raw == "system:":
            system_start = index + 1

    if version != _YOLO_INFO_VERSION:
        raise OssqSystemPairMaterializationError("unsupported yolo_info version")
    if data_id != f"sq{score_id}":
        raise OssqSystemPairMaterializationError("yolo_info data_id differs from score")
    if image_type != "scanned":
        raise OssqSystemPairMaterializationError("yolo_info image_type must be scanned")
    if system_start is None:
        raise OssqSystemPairMaterializationError("yolo_info system section is missing")

    sections: dict[str, list[str]] = {"ignores": [], "exceptions": []}
    active: str | None = None
    for raw in lines[system_start:]:
        if raw and not raw.startswith(" "):
            break
        if raw.startswith("  ") and not raw.startswith("    "):
            stripped = raw.strip()
            if stripped in ("ignores:", "exceptions:"):
                active = stripped[:-1]
            else:
                active = None
            continue
        if raw.startswith("    - "):
            if active not in sections:
                raise OssqSystemPairMaterializationError("unexpected yolo_info system list")
            segment_id = raw[6:].strip()
            match = _SEGMENT_RE.fullmatch(segment_id)
            if match is None or match.group("score") != score_id:
                raise OssqSystemPairMaterializationError("invalid yolo_info system exclusion segment")
            sections[active].append(segment_id)
        elif raw.strip() and active is not None:
            raise OssqSystemPairMaterializationError("malformed yolo_info system exclusion list")

    ignores = tuple(sorted(set(sections["ignores"])))
    exceptions = tuple(sorted(set(sections["exceptions"])))
    if set(ignores) & set(exceptions):
        raise OssqSystemPairMaterializationError("system ignore/exception sets overlap")
    return ignores, exceptions


@dataclass(frozen=True, slots=True)
class OssqSystemPairEvidence:
    score_id: str
    imslp_id: str
    segment_id: str
    source_document_sha256: str
    yolo_info_sha256: str
    image_sha256: str
    image_byte_count: int
    image_width: int
    image_height: int
    musicxml_sha256: str
    musicxml_byte_count: int

    def __post_init__(self) -> None:
        match = _SEGMENT_RE.fullmatch(self.segment_id)
        if match is None or match.group("score") != self.score_id:
            raise OssqSystemPairMaterializationError("segment_id does not bind the declared score")
        for name in ("source_document_sha256", "yolo_info_sha256", "image_sha256", "musicxml_sha256"):
            _require_sha256(name, getattr(self, name))
        if self.image_byte_count < 1 or self.musicxml_byte_count < 1:
            raise OssqSystemPairMaterializationError("materialized pair byte counts must be positive")
        if self.image_width < 1 or self.image_height < 1:
            raise OssqSystemPairMaterializationError("materialized image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class OssqSystemExclusionEvidence:
    score_id: str
    yolo_info_sha256: str
    declared_ignore_ids: tuple[str, ...]
    declared_exception_ids: tuple[str, ...]
    applied_ignore_ids: tuple[str, ...]
    applied_exception_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("yolo_info_sha256", self.yolo_info_sha256)
        for segment_id in (
            *self.declared_ignore_ids,
            *self.declared_exception_ids,
            *self.applied_ignore_ids,
            *self.applied_exception_ids,
        ):
            match = _SEGMENT_RE.fullmatch(segment_id)
            if match is None or match.group("score") != self.score_id:
                raise OssqSystemPairMaterializationError("exclusion segment does not bind score")
        if not set(self.applied_ignore_ids).issubset(set(self.declared_ignore_ids)):
            raise OssqSystemPairMaterializationError("applied ignores must be declared upstream")
        if not set(self.applied_exception_ids).issubset(set(self.declared_exception_ids)):
            raise OssqSystemPairMaterializationError("applied exceptions must be declared upstream")


@dataclass(frozen=True, slots=True)
class OssqSystemPairMaterializationReceipt:
    version: str
    source_commit_sha: str
    preprocessor_commit_sha: str
    b8n_source_receipt_sha256: str
    b8o_pairing_preflight_receipt_sha256: str
    ready_score_ids: tuple[str, ...]
    blocked_score_ids: tuple[str, ...]
    pairs: tuple[OssqSystemPairEvidence, ...]
    exclusions: tuple[OssqSystemExclusionEvidence, ...]
    pair_count_by_score: tuple[tuple[str, int], ...]
    raw_pdf_bytes_persisted: bool
    raw_pair_bytes_persisted_as_evidence: bool
    independent_pairing_review_authority: bool
    stage8_admission_authority: bool
    train_validation_assignment_authority: bool
    test_artifact_bytes_accessed: bool
    production_authority: bool
    commercial_use_authority: bool
    receipt_sha256: str

    def payload_without_fingerprint(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_commit_sha": self.source_commit_sha,
            "preprocessor_commit_sha": self.preprocessor_commit_sha,
            "b8n_source_receipt_sha256": self.b8n_source_receipt_sha256,
            "b8o_pairing_preflight_receipt_sha256": self.b8o_pairing_preflight_receipt_sha256,
            "ready_score_ids": list(self.ready_score_ids),
            "blocked_score_ids": list(self.blocked_score_ids),
            "pairs": [asdict(item) for item in self.pairs],
            "exclusions": [asdict(item) for item in self.exclusions],
            "pair_count_by_score": [list(item) for item in self.pair_count_by_score],
            "raw_pdf_bytes_persisted": self.raw_pdf_bytes_persisted,
            "raw_pair_bytes_persisted_as_evidence": self.raw_pair_bytes_persisted_as_evidence,
            "independent_pairing_review_authority": self.independent_pairing_review_authority,
            "stage8_admission_authority": self.stage8_admission_authority,
            "train_validation_assignment_authority": self.train_validation_assignment_authority,
            "test_artifact_bytes_accessed": self.test_artifact_bytes_accessed,
            "production_authority": self.production_authority,
            "commercial_use_authority": self.commercial_use_authority,
        }


def _source_spec_by_id() -> dict[str, object]:
    return {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}


def build_b8p_materialization_receipt(
    *,
    dataset_root: Path,
    b8n_source_sha256_by_score: Mapping[str, str],
    b8n_receipt_sha256: str,
    b8o_receipt_sha256: str,
) -> OssqSystemPairMaterializationReceipt:
    dataset_root = Path(dataset_root)
    if b8n_receipt_sha256 != B8P_EXPECTED_B8N_RECEIPT_SHA256:
        raise OssqSystemPairMaterializationError("B8N receipt identity differs from frozen B8P input")
    if b8o_receipt_sha256 != B8P_EXPECTED_B8O_RECEIPT_SHA256:
        raise OssqSystemPairMaterializationError("B8O receipt identity differs from frozen B8P input")
    if set(b8n_source_sha256_by_score) != set(B8P_READY_SCORE_IDS):
        raise OssqSystemPairMaterializationError("source hash population must equal the exact B8P READY population")

    specs = _source_spec_by_id()
    pairs: list[OssqSystemPairEvidence] = []
    exclusions: list[OssqSystemExclusionEvidence] = []
    counts: list[tuple[str, int]] = []
    seen_segments: set[str] = set()

    for score_id in B8P_READY_SCORE_IDS:
        spec = specs[score_id]
        score_dir = dataset_root / "scores" / spec.work_path
        yolo_info = score_dir / "images" / "scanned" / f"sq{score_id}_yolo_infos.yaml"
        if not yolo_info.is_file():
            raise OssqSystemPairMaterializationError(f"missing yolo_info for score {score_id}")
        yolo_bytes = yolo_info.read_bytes()
        yolo_info_sha = sha256(yolo_bytes).hexdigest()
        declared_ignores, declared_exceptions = _parse_system_exclusions(yolo_bytes, score_id=score_id)

        image_dir = score_dir / "images" / "scanned" / "systemwise"
        musicxml_dir = score_dir / "musicxml" / "scanned" / "systemwise"
        image_paths = sorted(image_dir.glob(f"sq{score_id}:*.png"))
        xml_paths = sorted(musicxml_dir.glob(f"sq{score_id}:*.musicxml"))
        image_stems = {path.stem for path in image_paths}
        xml_stems = {path.stem for path in xml_paths}
        if not image_stems or not xml_stems:
            raise OssqSystemPairMaterializationError(f"zero final materialized pair population for score {score_id}")

        ignore_set = set(declared_ignores)
        exception_set = set(declared_exceptions)
        declared_exclusion_set = ignore_set | exception_set
        applied_ignores = tuple(sorted(image_stems & ignore_set))
        applied_exceptions = tuple(sorted(image_stems & exception_set))
        expected_pair_stems = image_stems - declared_exclusion_set

        if xml_stems != expected_pair_stems:
            unexpected_unpaired_images = sorted((image_stems - xml_stems) - declared_exclusion_set)
            xml_without_image = sorted(xml_stems - image_stems)
            excluded_xml = sorted(xml_stems & declared_exclusion_set)
            raise OssqSystemPairMaterializationError(
                f"system image/MusicXML population differs from pinned upstream exclusions for score {score_id}; "
                f"unexpected_unpaired_images={unexpected_unpaired_images}; "
                f"xml_without_image={xml_without_image}; excluded_xml={excluded_xml}"
            )

        exclusions.append(OssqSystemExclusionEvidence(
            score_id=score_id,
            yolo_info_sha256=yolo_info_sha,
            declared_ignore_ids=declared_ignores,
            declared_exception_ids=declared_exceptions,
            applied_ignore_ids=applied_ignores,
            applied_exception_ids=applied_exceptions,
        ))

        score_count = 0
        source_sha = _require_sha256("source_document_sha256", b8n_source_sha256_by_score[score_id])
        for stem in sorted(xml_stems):
            match = _SEGMENT_RE.fullmatch(stem)
            if match is None or match.group("score") != score_id:
                raise OssqSystemPairMaterializationError(f"invalid system segment id: {stem}")
            if stem in seen_segments:
                raise OssqSystemPairMaterializationError(f"duplicate system segment id: {stem}")
            seen_segments.add(stem)
            image_bytes = (image_dir / f"{stem}.png").read_bytes()
            xml_bytes = (musicxml_dir / f"{stem}.musicxml").read_bytes()
            width, height = _png_dimensions(image_bytes)
            _validate_musicxml(xml_bytes)
            pairs.append(OssqSystemPairEvidence(
                score_id=score_id,
                imslp_id=spec.imslp_id,
                segment_id=stem,
                source_document_sha256=source_sha,
                yolo_info_sha256=yolo_info_sha,
                image_sha256=sha256(image_bytes).hexdigest(),
                image_byte_count=len(image_bytes),
                image_width=width,
                image_height=height,
                musicxml_sha256=sha256(xml_bytes).hexdigest(),
                musicxml_byte_count=len(xml_bytes),
            ))
            score_count += 1
        if score_count < 1:
            raise OssqSystemPairMaterializationError(f"zero materialized pairs for score {score_id}")
        counts.append((score_id, score_count))

    provisional = OssqSystemPairMaterializationReceipt(
        version=B8P_MATERIALIZATION_VERSION,
        source_commit_sha=OSSQ_OMR_CAMERA_READY_SHA,
        preprocessor_commit_sha=B8O_PREPROCESSOR_COMMIT_SHA,
        b8n_source_receipt_sha256=b8n_receipt_sha256,
        b8o_pairing_preflight_receipt_sha256=b8o_receipt_sha256,
        ready_score_ids=B8P_READY_SCORE_IDS,
        blocked_score_ids=B8P_BLOCKED_SCORE_IDS,
        pairs=tuple(pairs),
        exclusions=tuple(exclusions),
        pair_count_by_score=tuple(counts),
        raw_pdf_bytes_persisted=False,
        raw_pair_bytes_persisted_as_evidence=False,
        independent_pairing_review_authority=False,
        stage8_admission_authority=False,
        train_validation_assignment_authority=False,
        test_artifact_bytes_accessed=False,
        production_authority=False,
        commercial_use_authority=False,
        receipt_sha256="0" * 64,
    )
    fingerprint = sha256(_canonical_bytes(provisional.payload_without_fingerprint())).hexdigest()
    return OssqSystemPairMaterializationReceipt(
        version=provisional.version,
        source_commit_sha=provisional.source_commit_sha,
        preprocessor_commit_sha=provisional.preprocessor_commit_sha,
        b8n_source_receipt_sha256=provisional.b8n_source_receipt_sha256,
        b8o_pairing_preflight_receipt_sha256=provisional.b8o_pairing_preflight_receipt_sha256,
        ready_score_ids=provisional.ready_score_ids,
        blocked_score_ids=provisional.blocked_score_ids,
        pairs=provisional.pairs,
        exclusions=provisional.exclusions,
        pair_count_by_score=provisional.pair_count_by_score,
        raw_pdf_bytes_persisted=provisional.raw_pdf_bytes_persisted,
        raw_pair_bytes_persisted_as_evidence=provisional.raw_pair_bytes_persisted_as_evidence,
        independent_pairing_review_authority=provisional.independent_pairing_review_authority,
        stage8_admission_authority=provisional.stage8_admission_authority,
        train_validation_assignment_authority=provisional.train_validation_assignment_authority,
        test_artifact_bytes_accessed=provisional.test_artifact_bytes_accessed,
        production_authority=provisional.production_authority,
        commercial_use_authority=provisional.commercial_use_authority,
        receipt_sha256=fingerprint,
    )


def receipt_to_json(receipt: OssqSystemPairMaterializationReceipt) -> str:
    payload = receipt.payload_without_fingerprint()
    payload["receipt_sha256"] = receipt.receipt_sha256
    return _canonical_bytes(payload).decode("ascii")
