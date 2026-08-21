"""Bounded Meter V4-0 numerator-only representation audit.

V4-0 isolates the numerator of positive Teacher Gold TRAIN Meter examples and
measures family-generalizing pixel-space separability with a deterministic
normalized class-centroid probe.  It has no trainable parameters or optimizer.

D10, Teacher Gold adaptation-validation evaluation, sealed TEST, runtime,
Resolver, checkpoint replacement, and production promotion remain closed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Final

from PIL import Image


METER_V4_0_NUMERATOR_AUDIT: Final[str] = "meter-v4-0-numerator-representation-audit-v1"
NUMERATOR_CLASSES: Final[tuple[str, ...]] = ("2", "3", "4")
METER_TO_NUMERATOR: Final[dict[str, str]] = {"2/4": "2", "3/4": "3", "4/4": "4"}
ROI_WIDTH: Final[int] = 256
ROI_HEIGHT: Final[int] = 192


class MeterV4_0AuditError(RuntimeError):
    """Raised when the bounded V4-0 audit contract is violated."""


def _fail(message: str) -> None:
    raise MeterV4_0AuditError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class NumeratorAuditConfigV4_0:
    output_size: int = 64
    folds: int = 3
    horizontal_padding_milli: int = 150
    vertical_padding_milli: int = 50
    numerator_fraction_milli: int = 500

    def __post_init__(self) -> None:
        bounds = {
            "output_size": (self.output_size, 32, 128),
            "folds": (self.folds, 2, 9),
            "horizontal_padding_milli": (self.horizontal_padding_milli, 0, 500),
            "vertical_padding_milli": (self.vertical_padding_milli, 0, 250),
            "numerator_fraction_milli": (self.numerator_fraction_milli, 350, 650),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside Meter V4-0 bounds")
        frozen = {
            "output_size": 64,
            "folds": 3,
            "horizontal_padding_milli": 150,
            "vertical_padding_milli": 50,
            "numerator_fraction_milli": 500,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0: Final[NumeratorAuditConfigV4_0] = NumeratorAuditConfigV4_0()


@dataclass(frozen=True, slots=True)
class AuditRecordIdentityV4_0:
    record_id: str
    family_id: str
    meter_class: str

    def __post_init__(self) -> None:
        _hex64("record_id", self.record_id)
        if not isinstance(self.family_id, str) or not self.family_id or len(self.family_id) > 256:
            raise ValueError("family_id must be bounded non-empty string")
        if self.meter_class not in METER_TO_NUMERATOR:
            raise ValueError("meter_class must be 2/4, 3/4, or 4/4")

    @property
    def numerator_class(self) -> str:
        return METER_TO_NUMERATOR[self.meter_class]

    @property
    def class_index(self) -> int:
        return NUMERATOR_CLASSES.index(self.numerator_class)


@dataclass(frozen=True, slots=True)
class FoldAssignmentV4_0:
    record_id: str
    family_id: str
    meter_class: str
    fold: int


@dataclass(frozen=True, slots=True)
class OofPredictionV4_0:
    record_id: str
    family_id: str
    fold: int
    true_class: str
    predicted_class: str
    cosine_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class ClassificationSummaryV4_0:
    record_count: int
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class CentroidProbeResultV4_0:
    predictions: tuple[OofPredictionV4_0, ...]
    summary: ClassificationSummaryV4_0


@dataclass(frozen=True, slots=True)
class AuditDecisionV4_0:
    decision: str
    strong_signal: bool
    reasons: tuple[str, ...]


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def numerator_crop_bounds_v4_0(
    meter_bbox: Mapping[str, object],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> tuple[int, int, int, int]:
    """Return deterministic pixel bounds for the numerator half of a full Meter bbox."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(meter_bbox, Mapping) or set(meter_bbox) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail("meter_bbox must contain exactly x_min/y_min/x_max/y_max")
    x_min = _finite("meter_bbox.x_min", meter_bbox.get("x_min"))
    y_min = _finite("meter_bbox.y_min", meter_bbox.get("y_min"))
    x_max = _finite("meter_bbox.x_max", meter_bbox.get("x_max"))
    y_max = _finite("meter_bbox.y_max", meter_bbox.get("y_max"))
    if not (0 <= x_min < x_max <= ROI_WIDTH and 0 <= y_min < y_max <= ROI_HEIGHT):
        _fail("meter_bbox is outside the canonical 256x192 Teacher Gold ROI")

    width = x_max - x_min
    height = y_max - y_min
    x_pad = width * (config.horizontal_padding_milli / 1000.0)
    y_pad = height * (config.vertical_padding_milli / 1000.0)
    numerator_bottom = y_min + height * (config.numerator_fraction_milli / 1000.0)

    left = max(0, int(math.floor(x_min - x_pad)))
    top = max(0, int(math.floor(y_min - y_pad)))
    right = min(ROI_WIDTH, int(math.ceil(x_max + x_pad)))
    bottom = min(ROI_HEIGHT, int(math.ceil(numerator_bottom + y_pad)))
    if not (0 <= left < right <= ROI_WIDTH and 0 <= top < bottom <= ROI_HEIGHT):
        _fail("derived numerator crop is empty or outside the Teacher Gold ROI")
    return left, top, right, bottom


def render_numerator_crop_v4_0(
    roi_image: Image.Image,
    meter_bbox: Mapping[str, object],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> Image.Image:
    """Extract and aspect-fit the numerator to a deterministic gray8 64x64 canvas."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(roi_image, Image.Image) or roi_image.mode != "L" or roi_image.size != (ROI_WIDTH, ROI_HEIGHT):
        _fail("Teacher Gold ROI must be exact gray8 256x192 image")
    left, top, right, bottom = numerator_crop_bounds_v4_0(meter_bbox, config)
    crop = roi_image.crop((left, top, right, bottom))
    scale = min(config.output_size / crop.width, config.output_size / crop.height)
    resized_w = max(1, min(config.output_size, int(round(crop.width * scale))))
    resized_h = max(1, min(config.output_size, int(round(crop.height * scale))))
    resized = crop.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (config.output_size, config.output_size), 255)
    x0 = (config.output_size - resized_w) // 2
    y0 = (config.output_size - resized_h) // 2
    canvas.paste(resized, (x0, y0))
    return canvas


def normalized_ink_vector_v4_0(
    crop_image: Image.Image,
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> tuple[float, ...]:
    """Convert gray8 crop to a unit-L2 ink vector, white=0 and black=1."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(crop_image, Image.Image) or crop_image.mode != "L" or crop_image.size != (64, 64):
        _fail("V4-0 numerator crop must be exact gray8 64x64")
    vector = tuple((255 - value) / 255.0 for value in crop_image.tobytes())
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        _fail("V4-0 numerator crop has no finite ink signal")
    normalized = tuple(value / norm for value in vector)
    if len(normalized) != 4096 or any(not math.isfinite(value) for value in normalized):
        _fail("V4-0 normalized ink vector is invalid")
    return normalized


def fold_plan_v4_0(
    identities: Sequence[AuditRecordIdentityV4_0],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> tuple[FoldAssignmentV4_0, ...]:
    """Create the frozen class-balanced family-disjoint 3-fold plan."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if len(identities) != 27:
        _fail("V4-0 requires exactly 27 positive Teacher Gold TRAIN records")
    if len({item.record_id for item in identities}) != 27:
        _fail("V4-0 record ids must be unique")
    if len({item.family_id for item in identities}) != 27:
        _fail("V4-0 requires one positive record per family")
    counts = Counter(item.meter_class for item in identities)
    if counts != Counter({"2/4": 9, "3/4": 9, "4/4": 9}):
        _fail("V4-0 positive TRAIN classes must be exactly balanced 9/9/9")

    assignments: list[FoldAssignmentV4_0] = []
    for meter_class in ("2/4", "3/4", "4/4"):
        class_items = [item for item in identities if item.meter_class == meter_class]
        ranked = sorted(
            class_items,
            key=lambda item: sha256(
                _canonical_json(
                    {
                        "version": METER_V4_0_NUMERATOR_AUDIT,
                        "fold_policy": "class-balanced-family-hash-rank-mod3-v1",
                        "meter_class": meter_class,
                        "family_id": item.family_id,
                    }
                )
            ).hexdigest(),
        )
        for rank, item in enumerate(ranked):
            assignments.append(
                FoldAssignmentV4_0(
                    record_id=item.record_id,
                    family_id=item.family_id,
                    meter_class=item.meter_class,
                    fold=rank % config.folds,
                )
            )

    assignments.sort(key=lambda item: (item.fold, item.meter_class, item.family_id, item.record_id))
    fold_class_counts = Counter((item.fold, item.meter_class) for item in assignments)
    for fold in range(3):
        for meter_class in ("2/4", "3/4", "4/4"):
            if fold_class_counts[(fold, meter_class)] != 3:
                _fail("V4-0 fold plan must hold out exactly three families per class per fold")
    return tuple(assignments)


def _unit_centroid(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(vectors) != 6 or any(len(vector) != 4096 for vector in vectors):
        _fail("each V4-0 class centroid must consume exactly six 4096-element vectors")
    mean = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(4096)]
    norm = math.sqrt(sum(value * value for value in mean))
    if not math.isfinite(norm) or norm <= 1e-12:
        _fail("V4-0 class centroid has no finite signal")
    return tuple(value / norm for value in mean)


def _cosine(unit_a: Sequence[float], unit_b: Sequence[float]) -> float:
    if len(unit_a) != 4096 or len(unit_b) != 4096:
        _fail("V4-0 cosine vectors must have length 4096")
    score = sum(a * b for a, b in zip(unit_a, unit_b))
    if not math.isfinite(score):
        _fail("V4-0 cosine score is not finite")
    return score


def centroid_oof_probe_v4_0(
    identities: Sequence[AuditRecordIdentityV4_0],
    vectors_by_record_id: Mapping[str, Sequence[float]],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> CentroidProbeResultV4_0:
    """Run the zero-training family-disjoint normalized-centroid OOF probe."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    assignments = fold_plan_v4_0(identities, config)
    identity_by_id = {item.record_id: item for item in identities}
    if set(vectors_by_record_id) != set(identity_by_id):
        _fail("V4-0 vector set must match the exact 27 selected record ids")
    for record_id, vector in vectors_by_record_id.items():
        if len(vector) != 4096 or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in vector):
            _fail(f"V4-0 vector is invalid for record {record_id}")

    predictions: list[OofPredictionV4_0] = []
    for fold in range(3):
        train_ids = [item.record_id for item in assignments if item.fold != fold]
        holdout_ids = [item.record_id for item in assignments if item.fold == fold]
        if len(train_ids) != 18 or len(holdout_ids) != 9:
            _fail("V4-0 fold cardinality changed")
        centroids: dict[str, tuple[float, ...]] = {}
        for numerator_class in NUMERATOR_CLASSES:
            class_vectors = [
                vectors_by_record_id[record_id]
                for record_id in train_ids
                if identity_by_id[record_id].numerator_class == numerator_class
            ]
            centroids[numerator_class] = _unit_centroid(class_vectors)

        for record_id in holdout_ids:
            identity = identity_by_id[record_id]
            vector = vectors_by_record_id[record_id]
            scores = {
                label: _cosine(vector, centroids[label])
                for label in NUMERATOR_CLASSES
            }
            predicted = max(NUMERATOR_CLASSES, key=lambda label: scores[label])
            predictions.append(
                OofPredictionV4_0(
                    record_id=record_id,
                    family_id=identity.family_id,
                    fold=fold,
                    true_class=identity.numerator_class,
                    predicted_class=predicted,
                    cosine_scores=scores,
                )
            )

    predictions.sort(key=lambda row: (row.fold, row.true_class, row.family_id, row.record_id))
    truth = [NUMERATOR_CLASSES.index(row.true_class) for row in predictions]
    guessed = [NUMERATOR_CLASSES.index(row.predicted_class) for row in predictions]
    return CentroidProbeResultV4_0(
        predictions=tuple(predictions),
        summary=classification_summary_v4_0(truth, guessed),
    )


def classification_summary_v4_0(
    truth: Sequence[int],
    predicted: Sequence[int],
) -> ClassificationSummaryV4_0:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth/predicted must be non-empty equal-length sequences")
    if any(value not in {0, 1, 2} for value in truth) or any(value not in {0, 1, 2} for value in predicted):
        raise ValueError("V4-0 classes must be 0,1,2")
    confusion = [[0, 0, 0] for _ in range(3)]
    for actual, guess in zip(truth, predicted):
        confusion[actual][guess] += 1
    correct = sum(confusion[index][index] for index in range(3))
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for index, label in enumerate(NUMERATOR_CLASSES):
        tp = confusion[index][index]
        fn = sum(confusion[index]) - tp
        fp = sum(confusion[row][index] for row in range(3)) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = recall
        f1s.append(f1)
    return ClassificationSummaryV4_0(
        record_count=len(truth),
        accuracy=correct / len(truth),
        macro_f1=sum(f1s) / 3.0,
        per_class_recall=recalls,
        confusion=tuple(tuple(row) for row in confusion),
    )


def audit_decision_v4_0(summary: ClassificationSummaryV4_0) -> AuditDecisionV4_0:
    if summary.record_count != 27:
        raise ValueError("V4-0 OOF decision requires exactly 27 predictions")
    reasons: list[str] = []
    if summary.accuracy < 25 / 27:
        reasons.append("OOF_ACCURACY_BELOW_25_OF_27")
    for label in NUMERATOR_CLASSES:
        if summary.per_class_recall[label] < 8 / 9:
            reasons.append(f"OOF_{label}_RECALL_BELOW_8_OF_9")
    strong = not reasons
    return AuditDecisionV4_0(
        decision="REPRESENTATION_SIGNAL_STRONG" if strong else "REPRESENTATION_SIGNAL_WEAK_OR_DATA_LIMITED",
        strong_signal=strong,
        reasons=tuple(reasons),
    )


def optimizer_step_allowed() -> bool:
    return False


def sealed_test_access_allowed() -> bool:
    return False


def d10_access_allowed() -> bool:
    return False


def teacher_adaptation_validation_evaluation_allowed() -> bool:
    return False


def runtime_connection_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
