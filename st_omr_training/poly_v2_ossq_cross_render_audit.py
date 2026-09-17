"""TR-POLY-09B8R independent cross-render audit for OSSQ system pairs.

B8R does not trust B8P filenames as pairing evidence. It independently renders
each MusicXML candidate, extracts bounded structural signatures from both the
scanned system image and the independent render, and evaluates the complete
within-score all-to-all similarity matrix.

A pair is only a VERIFIED_CANDIDATE when it is the unique mutual nearest
neighbour under two different structural signatures and the score has no
cross-pair mutual-nearest conflict. This module emits hash-only evidence and
does not itself grant Stage 8 admission, TRAIN/VALIDATION assignment, TEST,
production, or commercial authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
import json
import math
import re
from typing import Final, Mapping, Sequence

B8R_AUDIT_VERSION: Final[str] = "st-omr-poly-v2-ossq-independent-cross-render-audit-v1"
B8R_EXPECTED_B8P_RECEIPT_SHA256: Final[str] = (
    "3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf"
)
B8R_READY_SCORE_IDS: Final[tuple[str, ...]] = (
    "7070781", "7075297", "7078259", "7093885", "7103818", "7108150", "8071278"
)
B8R_EXPECTED_PAIR_COUNT_BY_SCORE: Final[tuple[tuple[str, int], ...]] = (
    ("7070781", 9),
    ("7075297", 28),
    ("7078259", 26),
    ("7093885", 26),
    ("7103818", 87),
    ("7108150", 117),
    ("8071278", 119),
)
B8R_EXPECTED_PAIR_COUNT: Final[int] = sum(count for _, count in B8R_EXPECTED_PAIR_COUNT_BY_SCORE)

PRIMARY_BINS: Final[int] = 64
SECONDARY_WIDTH: Final[int] = 16
SECONDARY_HEIGHT: Final[int] = 8
SIMILARITY_SCALE: Final[int] = 1_000_000
STAFF_ROW_OCCUPANCY_PPM: Final[int] = 550_000
STAFF_ROW_EXPANSION: Final[int] = 1

_SEGMENT_RE = re.compile(r"^sq(?P<score>[0-9]+):(?P<page>[0-9]{4}):(?P<system>[0-9]{4})$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class OssqCrossRenderAuditError(ValueError):
    """Raised when B8R cross-render evidence fails closed."""


class CrossRenderDecision(str, Enum):
    VERIFIED_CANDIDATE = "verified-candidate"
    REVIEW_REQUIRED = "review-required"


@dataclass(frozen=True, slots=True)
class StructuralSignature:
    primary: tuple[int, ...]
    secondary: tuple[int, ...]
    image_width: int
    image_height: int
    otsu_threshold: int
    suppressed_staff_rows: int
    ink_fraction_ppm: int

    def __post_init__(self) -> None:
        if len(self.primary) != PRIMARY_BINS:
            raise OssqCrossRenderAuditError("primary signature has unexpected length")
        if len(self.secondary) != SECONDARY_WIDTH * SECONDARY_HEIGHT:
            raise OssqCrossRenderAuditError("secondary signature has unexpected length")
        for vector in (self.primary, self.secondary):
            if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000 for value in vector):
                raise OssqCrossRenderAuditError("signature values must be bounded integers")
        if self.image_width < 1 or self.image_height < 1:
            raise OssqCrossRenderAuditError("signature image dimensions must be positive")
        if not 0 <= self.otsu_threshold <= 255:
            raise OssqCrossRenderAuditError("otsu threshold is outside byte range")
        if not 0 <= self.suppressed_staff_rows <= self.image_height:
            raise OssqCrossRenderAuditError("suppressed_staff_rows is invalid")
        if not 0 <= self.ink_fraction_ppm <= SIMILARITY_SCALE:
            raise OssqCrossRenderAuditError("ink_fraction_ppm is invalid")


@dataclass(frozen=True, slots=True)
class CrossRenderPairObservation:
    score_id: str
    segment_id: str
    image_sha256: str
    musicxml_sha256: str
    independent_render_png_sha256: str
    primary_similarity_ppm: int
    secondary_similarity_ppm: int
    primary_scan_rank: int
    primary_render_rank: int
    secondary_scan_rank: int
    secondary_render_rank: int
    score_cross_conflict_count: int
    decision: CrossRenderDecision
    evidence_sha256: str

    def __post_init__(self) -> None:
        match = _SEGMENT_RE.fullmatch(self.segment_id)
        if match is None or match.group("score") != self.score_id:
            raise OssqCrossRenderAuditError("observation segment does not bind score")
        if self.score_id not in B8R_READY_SCORE_IDS:
            raise OssqCrossRenderAuditError("observation score is outside B8R population")
        for name in ("image_sha256", "musicxml_sha256", "independent_render_png_sha256", "evidence_sha256"):
            _require_sha256(name, getattr(self, name))
        for name in ("primary_similarity_ppm", "secondary_similarity_ppm"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= SIMILARITY_SCALE:
                raise OssqCrossRenderAuditError(f"{name} is outside similarity range")
        for name in ("primary_scan_rank", "primary_render_rank", "secondary_scan_rank", "secondary_render_rank"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise OssqCrossRenderAuditError(f"{name} must be a positive integer")
        if not isinstance(self.score_cross_conflict_count, int) or self.score_cross_conflict_count < 0:
            raise OssqCrossRenderAuditError("score_cross_conflict_count must be non-negative")
        if not isinstance(self.decision, CrossRenderDecision):
            raise OssqCrossRenderAuditError("decision must be CrossRenderDecision")


@dataclass(frozen=True, slots=True)
class OssqCrossRenderAuditReceipt:
    version: str
    b8p_receipt_sha256: str
    reviewer_identity_sha256: str
    policy_fingerprint: str
    observations: tuple[CrossRenderPairObservation, ...]
    decision_counts: tuple[tuple[str, int], ...]
    score_cross_conflicts: tuple[tuple[str, int], ...]
    raw_pdf_bytes_persisted_as_evidence: bool
    raw_pair_bytes_persisted_as_evidence: bool
    test_artifact_bytes_accessed: bool
    stage8_admission_authority: bool
    train_validation_assignment_authority: bool
    production_authority: bool
    commercial_use_authority: bool
    receipt_sha256: str

    def payload_without_fingerprint(self) -> dict[str, object]:
        return {
            "version": self.version,
            "b8p_receipt_sha256": self.b8p_receipt_sha256,
            "reviewer_identity_sha256": self.reviewer_identity_sha256,
            "policy_fingerprint": self.policy_fingerprint,
            "observations": [_observation_payload(item) for item in self.observations],
            "decision_counts": [list(item) for item in self.decision_counts],
            "score_cross_conflicts": [list(item) for item in self.score_cross_conflicts],
            "raw_pdf_bytes_persisted_as_evidence": self.raw_pdf_bytes_persisted_as_evidence,
            "raw_pair_bytes_persisted_as_evidence": self.raw_pair_bytes_persisted_as_evidence,
            "test_artifact_bytes_accessed": self.test_artifact_bytes_accessed,
            "stage8_admission_authority": self.stage8_admission_authority,
            "train_validation_assignment_authority": self.train_validation_assignment_authority,
            "production_authority": self.production_authority,
            "commercial_use_authority": self.commercial_use_authority,
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise OssqCrossRenderAuditError(f"{name} must be lowercase SHA-256")
    return value


def audit_policy_fingerprint() -> str:
    payload = {
        "version": B8R_AUDIT_VERSION,
        "primary": {"type": "staff-suppressed-column-ink-profile", "bins": PRIMARY_BINS},
        "secondary": {"type": "staff-suppressed-coarse-occupancy", "width": SECONDARY_WIDTH, "height": SECONDARY_HEIGHT},
        "threshold": "per-image-otsu",
        "staff_row_occupancy_ppm": STAFF_ROW_OCCUPANCY_PPM,
        "staff_row_expansion": STAFF_ROW_EXPANSION,
        "matching": "within-score-all-to-all-unique-mutual-nearest-on-both-signatures",
        "cross_conflict_policy": "any-wrong-mutual-nearest-on-both-signatures-blocks-verification-for-score",
        "similarity": "cosine-ppm",
    }
    return sha256(_canonical_bytes(payload)).hexdigest()


def reviewer_identity_sha256(*, verovio_version: str, cairosvg_version: str, pillow_version: str) -> str:
    for name, value in (("verovio_version", verovio_version), ("cairosvg_version", cairosvg_version), ("pillow_version", pillow_version)):
        if not isinstance(value, str) or not value:
            raise OssqCrossRenderAuditError(f"{name} must be non-empty")
    payload = {
        "reviewer": "st-omr-b8r-independent-cross-render",
        "audit_version": B8R_AUDIT_VERSION,
        "policy_fingerprint": audit_policy_fingerprint(),
        "verovio_version": verovio_version,
        "cairosvg_version": cairosvg_version,
        "pillow_version": pillow_version,
    }
    return sha256(_canonical_bytes(payload)).hexdigest()


def _otsu_threshold(histogram: Sequence[int]) -> int:
    if len(histogram) != 256 or any(not isinstance(value, int) or value < 0 for value in histogram):
        raise OssqCrossRenderAuditError("histogram must contain 256 non-negative integer bins")
    total = sum(histogram)
    if total < 1:
        raise OssqCrossRenderAuditError("cannot threshold an empty image")
    weighted_total = sum(index * count for index, count in enumerate(histogram))
    weighted_background = 0
    background_count = 0
    best_score = -1.0
    best_threshold = 127
    for threshold, count in enumerate(histogram):
        background_count += count
        if background_count == 0:
            continue
        foreground_count = total - background_count
        if foreground_count == 0:
            break
        weighted_background += threshold * count
        mean_background = weighted_background / background_count
        mean_foreground = (weighted_total - weighted_background) / foreground_count
        score = background_count * foreground_count * (mean_background - mean_foreground) ** 2
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _resample_profile(values: Sequence[int], bins: int) -> tuple[int, ...]:
    if bins < 1 or not values:
        raise OssqCrossRenderAuditError("profile resampling requires non-empty values and positive bins")
    length = len(values)
    output: list[float] = []
    for target in range(bins):
        lo = target * length / bins
        hi = (target + 1) * length / bins
        start = int(math.floor(lo))
        end = int(math.ceil(hi))
        weighted = 0.0
        for source in range(start, min(end, length)):
            overlap = max(0.0, min(hi, source + 1.0) - max(lo, float(source)))
            weighted += values[source] * overlap
        output.append(weighted / (hi - lo))
    maximum = max(output)
    if maximum <= 0:
        return tuple(0 for _ in output)
    return tuple(int(round(value * 10_000 / maximum)) for value in output)


def extract_structural_signature(png_bytes: object) -> StructuralSignature:
    if not isinstance(png_bytes, bytes) or not png_bytes:
        raise OssqCrossRenderAuditError("signature input must be non-empty PNG bytes")
    try:
        from PIL import Image, ImageChops
    except Exception as exc:
        raise OssqCrossRenderAuditError("Pillow is required for B8R signature extraction") from exc
    try:
        with Image.open(BytesIO(png_bytes)) as opened:
            if opened.format != "PNG":
                raise OssqCrossRenderAuditError("signature input must decode as PNG")
            image = opened.convert("L")
            image.load()
    except OssqCrossRenderAuditError:
        raise
    except Exception as exc:
        raise OssqCrossRenderAuditError("could not decode PNG for B8R") from exc

    width, height = image.size
    if width < 1 or height < 1:
        raise OssqCrossRenderAuditError("decoded PNG has invalid dimensions")
    threshold = _otsu_threshold(image.histogram())
    raw = tuple(image.getdata())
    ink = [1 if value <= threshold else 0 for value in raw]
    ink_count = sum(ink)
    ink_fraction_ppm = int(round(ink_count * SIMILARITY_SCALE / (width * height)))

    staff_rows: set[int] = set()
    for row in range(height):
        row_ink = sum(ink[row * width : (row + 1) * width])
        occupancy_ppm = int(round(row_ink * SIMILARITY_SCALE / width))
        if occupancy_ppm >= STAFF_ROW_OCCUPANCY_PPM:
            for expanded in range(max(0, row - STAFF_ROW_EXPANSION), min(height, row + STAFF_ROW_EXPANSION + 1)):
                staff_rows.add(expanded)

    cleaned = [255] * (width * height)
    for row in range(height):
        if row in staff_rows:
            continue
        offset = row * width
        for column in range(width):
            if ink[offset + column]:
                cleaned[offset + column] = 0
    cleaned_image = Image.new("L", (width, height))
    cleaned_image.putdata(cleaned)
    bbox = ImageChops.invert(cleaned_image).getbbox()
    crop = image if bbox is None else cleaned_image.crop(bbox)

    crop_width, crop_height = crop.size
    crop_pixels = tuple(crop.getdata())
    column_ink = [
        sum(1 for row in range(crop_height) if crop_pixels[row * crop_width + column] < 128)
        for column in range(crop_width)
    ]
    primary = _resample_profile(column_ink, PRIMARY_BINS)
    coarse = crop.resize((SECONDARY_WIDTH, SECONDARY_HEIGHT), resample=Image.Resampling.BOX)
    secondary = tuple(max(0, min(10_000, int(round((255 - value) * 10_000 / 255)))) for value in coarse.getdata())
    return StructuralSignature(primary, secondary, width, height, threshold, len(staff_rows), ink_fraction_ppm)


def cosine_similarity_ppm(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right) or not left:
        raise OssqCrossRenderAuditError("cosine vectors must have the same non-zero length")
    dot = sum(int(a) * int(b) for a, b in zip(left, right))
    left_norm = sum(int(a) * int(a) for a in left)
    right_norm = sum(int(b) * int(b) for b in right)
    if left_norm == 0 or right_norm == 0:
        return 0
    value = dot / math.sqrt(left_norm * right_norm)
    return max(0, min(SIMILARITY_SCALE, int(round(value * SIMILARITY_SCALE))))


def _similarity_matrix(left_vectors: Sequence[Sequence[int]], right_vectors: Sequence[Sequence[int]]) -> list[list[int]]:
    if len(left_vectors) != len(right_vectors) or not left_vectors:
        raise OssqCrossRenderAuditError("similarity matrix requires equal non-empty populations")
    width = len(left_vectors[0])
    if width < 1:
        raise OssqCrossRenderAuditError("similarity vectors may not be empty")
    if any(len(vector) != width for vector in (*left_vectors, *right_vectors)):
        raise OssqCrossRenderAuditError("similarity vectors must share one width")
    left_norms = [sum(int(value) * int(value) for value in vector) for vector in left_vectors]
    right_norms = [sum(int(value) * int(value) for value in vector) for vector in right_vectors]
    matrix: list[list[int]] = []
    for left_index, left in enumerate(left_vectors):
        row: list[int] = []
        for right_index, right in enumerate(right_vectors):
            if left_norms[left_index] == 0 or right_norms[right_index] == 0:
                row.append(0)
                continue
            dot = sum(int(a) * int(b) for a, b in zip(left, right))
            value = dot / math.sqrt(left_norms[left_index] * right_norms[right_index])
            row.append(max(0, min(SIMILARITY_SCALE, int(round(value * SIMILARITY_SCALE)))))
        matrix.append(row)
    return matrix


def _rank_desc(values: Sequence[int], target_index: int) -> int:
    target = values[target_index]
    return 1 + sum(value > target for value in values)


def _unique_max(values: Sequence[int], target_index: int) -> bool:
    target = values[target_index]
    return all(index == target_index or target > value for index, value in enumerate(values))


def _observation_payload(item: CrossRenderPairObservation) -> dict[str, object]:
    payload = asdict(item)
    payload["decision"] = item.decision.value
    return payload


def _evidence_sha256(payload: Mapping[str, object]) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _validate_pair_input(pair: Mapping[str, object]) -> tuple[str, str, str, str, str]:
    score_id = pair.get("score_id")
    segment_id = pair.get("segment_id")
    image_sha = pair.get("image_sha256")
    xml_sha = pair.get("musicxml_sha256")
    render_sha = pair.get("independent_render_png_sha256")
    if not isinstance(score_id, str) or score_id not in B8R_READY_SCORE_IDS:
        raise OssqCrossRenderAuditError("pair score is outside B8R population")
    if not isinstance(segment_id, str):
        raise OssqCrossRenderAuditError("pair segment_id must be a string")
    match = _SEGMENT_RE.fullmatch(segment_id)
    if match is None or match.group("score") != score_id:
        raise OssqCrossRenderAuditError("pair segment_id does not bind score")
    for name, value in (("image_sha256", image_sha), ("musicxml_sha256", xml_sha), ("independent_render_png_sha256", render_sha)):
        _require_sha256(name, value)
    return score_id, segment_id, str(image_sha), str(xml_sha), str(render_sha)


def build_cross_render_audit(*, b8p_receipt_sha256: str, pairs: Sequence[Mapping[str, object]], scanned_signatures: Mapping[str, StructuralSignature], render_signatures: Mapping[str, StructuralSignature], reviewer_identity_sha: str) -> OssqCrossRenderAuditReceipt:
    if b8p_receipt_sha256 != B8R_EXPECTED_B8P_RECEIPT_SHA256:
        raise OssqCrossRenderAuditError("B8P receipt identity differs from B8R freeze")
    _require_sha256("reviewer_identity_sha", reviewer_identity_sha)
    if len(pairs) != B8R_EXPECTED_PAIR_COUNT:
        raise OssqCrossRenderAuditError("B8R requires exactly 412 pair inputs")

    parsed = [_validate_pair_input(pair) for pair in pairs]
    if len({segment for _, segment, *_ in parsed}) != len(parsed):
        raise OssqCrossRenderAuditError("B8R pair input contains duplicate segments")
    if set(scanned_signatures) != {item[1] for item in parsed}:
        raise OssqCrossRenderAuditError("scanned signature population differs from pair population")
    if set(render_signatures) != {item[1] for item in parsed}:
        raise OssqCrossRenderAuditError("render signature population differs from pair population")

    count_by_score = {score: 0 for score in B8R_READY_SCORE_IDS}
    for score_id, *_ in parsed:
        count_by_score[score_id] += 1
    if tuple((score, count_by_score[score]) for score in B8R_READY_SCORE_IDS) != B8R_EXPECTED_PAIR_COUNT_BY_SCORE:
        raise OssqCrossRenderAuditError("B8R score counts differ from frozen population")

    parsed_by_segment = {item[1]: item for item in parsed}
    observations: list[CrossRenderPairObservation] = []
    conflicts_by_score: list[tuple[str, int]] = []
    policy = audit_policy_fingerprint()

    for score_id in B8R_READY_SCORE_IDS:
        segment_ids = sorted(item[1] for item in parsed if item[0] == score_id)
        scan_primary = [scanned_signatures[segment].primary for segment in segment_ids]
        scan_secondary = [scanned_signatures[segment].secondary for segment in segment_ids]
        render_primary = [render_signatures[segment].primary for segment in segment_ids]
        render_secondary = [render_signatures[segment].secondary for segment in segment_ids]
        primary_matrix = _similarity_matrix(scan_primary, render_primary)
        secondary_matrix = _similarity_matrix(scan_secondary, render_secondary)

        def mutual_unique(matrix: Sequence[Sequence[int]], row: int, column: int) -> bool:
            row_values = matrix[row]
            column_values = [matrix[index][column] for index in range(len(matrix))]
            return _unique_max(row_values, column) and _unique_max(column_values, row)

        cross_conflicts = sum(
            1
            for row in range(len(segment_ids))
            for column in range(len(segment_ids))
            if row != column and mutual_unique(primary_matrix, row, column) and mutual_unique(secondary_matrix, row, column)
        )
        conflicts_by_score.append((score_id, cross_conflicts))

        for index, segment_id in enumerate(segment_ids):
            score, _, image_sha, xml_sha, render_sha = parsed_by_segment[segment_id]
            primary_column = [primary_matrix[row][index] for row in range(len(segment_ids))]
            secondary_column = [secondary_matrix[row][index] for row in range(len(segment_ids))]
            primary_scan_rank = _rank_desc(primary_matrix[index], index)
            secondary_scan_rank = _rank_desc(secondary_matrix[index], index)
            primary_render_rank = _rank_desc(primary_column, index)
            secondary_render_rank = _rank_desc(secondary_column, index)
            verified = cross_conflicts == 0 and mutual_unique(primary_matrix, index, index) and mutual_unique(secondary_matrix, index, index)
            decision = CrossRenderDecision.VERIFIED_CANDIDATE if verified else CrossRenderDecision.REVIEW_REQUIRED
            evidence_payload = {
                "version": B8R_AUDIT_VERSION,
                "policy_fingerprint": policy,
                "reviewer_identity_sha256": reviewer_identity_sha,
                "score_id": score,
                "segment_id": segment_id,
                "image_sha256": image_sha,
                "musicxml_sha256": xml_sha,
                "independent_render_png_sha256": render_sha,
                "primary_similarity_ppm": primary_matrix[index][index],
                "secondary_similarity_ppm": secondary_matrix[index][index],
                "primary_scan_rank": primary_scan_rank,
                "primary_render_rank": primary_render_rank,
                "secondary_scan_rank": secondary_scan_rank,
                "secondary_render_rank": secondary_render_rank,
                "score_cross_conflict_count": cross_conflicts,
                "decision": decision.value,
            }
            observations.append(CrossRenderPairObservation(score, segment_id, image_sha, xml_sha, render_sha, primary_matrix[index][index], secondary_matrix[index][index], primary_scan_rank, primary_render_rank, secondary_scan_rank, secondary_render_rank, cross_conflicts, decision, _evidence_sha256(evidence_payload)))

    ordered = tuple(sorted(observations, key=lambda item: item.segment_id))
    decision_counts = tuple((decision.value, sum(item.decision is decision for item in ordered)) for decision in CrossRenderDecision)
    provisional = OssqCrossRenderAuditReceipt(B8R_AUDIT_VERSION, B8R_EXPECTED_B8P_RECEIPT_SHA256, reviewer_identity_sha, policy, ordered, decision_counts, tuple(conflicts_by_score), False, False, False, False, False, False, False, "0" * 64)
    fingerprint = sha256(_canonical_bytes(provisional.payload_without_fingerprint())).hexdigest()
    return OssqCrossRenderAuditReceipt(provisional.version, provisional.b8p_receipt_sha256, provisional.reviewer_identity_sha256, provisional.policy_fingerprint, provisional.observations, provisional.decision_counts, provisional.score_cross_conflicts, False, False, False, False, False, False, False, fingerprint)


def receipt_to_json(receipt: OssqCrossRenderAuditReceipt) -> str:
    payload = receipt.payload_without_fingerprint()
    payload["receipt_sha256"] = receipt.receipt_sha256
    return _canonical_bytes(payload).decode("ascii")
