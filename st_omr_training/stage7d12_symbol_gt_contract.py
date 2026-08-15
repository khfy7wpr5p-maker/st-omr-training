"""Stage 7-D12 NoteHead/Rest/Accidental deterministic GT contract.

D12 is a data/ground-truth stage only.  It freezes the development-only source
surface and the symbol-label/linkage rules needed before any NoteHead, Rest, or
Accidental specialist optimizer is allowed to exist.

The contract deliberately does not load a model, checkpoint, optimizer, or TEST
record.  Synthetic spatial authority remains pinned Verovio geometry replayed
through the accepted final-PNG transform; symbolic authority remains canonical
ST music / deterministic MusicXML.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Final


STAGE7D12_VERSION: Final[str] = "stage7d12-symbol-specialist-gt-contract-v1"
STAGE7D12_SCHEMA: Final[str] = "stage7d12-symbol-specialist-gt-contract-v1"

EXPECTED_DEVELOPMENT_SAMPLE_COUNTS: Final[dict[str, int]] = {
    "train": 1230,
    "validation": 153,
}
EXPECTED_DEVELOPMENT_FAMILY_COUNTS: Final[dict[str, int]] = {
    "train": 410,
    "validation": 51,
}
TEST_SPECIALIST_RECORDS: Final[int] = 0
OPTIMIZER_STEPS: Final[int] = 0

NOTEHEAD_FILL_CLASSES: Final[tuple[str, ...]] = ("open", "filled")
REST_CLASSES: Final[tuple[str, ...]] = ("half", "quarter", "eighth")
ACCIDENTAL_CLASSES: Final[tuple[str, ...]] = ("sharp", "flat", "natural")

# These are renderer object classes, not training labels.  D12 extraction may
# use only these explicit visible-object classes and must reject ambiguous
# linkage rather than infer a label from appearance.
RENDERER_OBJECT_CLASSES: Final[dict[str, str]] = {
    "notehead": "notehead",
    "rest": "rest",
    "accidental": "accid",
}

# One shared deterministic derivative bundle is allowed, but future optimizers
# remain separate specialists.  This preserves the accepted D4 task graph.
FUTURE_TRAINING_BOUNDARY: Final[dict[str, object]] = {
    "shared_derivative_bundle": True,
    "separate_specialist_models": True,
    "notehead_model": "NoteHeadSet",
    "rest_model": "RestSet",
    "accidental_model": "AccidentalSet",
    "joint_optimizer": False,
}


class Stage7D12ContractError(ValueError):
    """Raised when a D12 contract boundary is violated."""


@dataclass(frozen=True, slots=True)
class SymbolTargetContract:
    set_name: str
    task_id: str
    classes: tuple[str, ...]
    required_labels: tuple[str, ...]
    renderer_class: str
    canonical_event_link_required: bool = True

    def __post_init__(self) -> None:
        if not self.set_name.endswith("Set"):
            raise Stage7D12ContractError("set_name must end in Set")
        if not self.task_id or not self.task_id.isascii():
            raise Stage7D12ContractError("task_id must be non-empty ASCII")
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise Stage7D12ContractError("classes must be non-empty and unique")
        if not self.required_labels or len(set(self.required_labels)) != len(
            self.required_labels
        ):
            raise Stage7D12ContractError(
                "required_labels must be non-empty and unique"
            )
        if not self.renderer_class or not self.renderer_class.isascii():
            raise Stage7D12ContractError("renderer_class must be non-empty ASCII")
        if not self.canonical_event_link_required:
            raise Stage7D12ContractError(
                "D12 symbol geometry requires canonical event linkage"
            )


SYMBOL_TARGETS: Final[tuple[SymbolTargetContract, ...]] = (
    SymbolTargetContract(
        set_name="NoteHeadSet",
        task_id="notehead",
        classes=NOTEHEAD_FILL_CLASSES,
        required_labels=(
            "notehead_bbox",
            "notehead_center",
            "fill_class",
            "canonical_event_id",
        ),
        renderer_class=RENDERER_OBJECT_CLASSES["notehead"],
    ),
    SymbolTargetContract(
        set_name="RestSet",
        task_id="rest",
        classes=REST_CLASSES,
        required_labels=(
            "rest_bbox",
            "rest_class",
            "duration_class",
            "canonical_event_id",
        ),
        renderer_class=RENDERER_OBJECT_CLASSES["rest"],
    ),
    SymbolTargetContract(
        set_name="AccidentalSet",
        task_id="accidental",
        classes=ACCIDENTAL_CLASSES,
        required_labels=(
            "accidental_bbox",
            "accidental_class",
            "canonical_event_id",
        ),
        renderer_class=RENDERER_OBJECT_CLASSES["accidental"],
    ),
)


# Canonical event IDs are audit identities created from already-authoritative
# symbolic ordering.  They are not renderer IDs and never become pitch labels.
_EVENT_ID = re.compile(
    r"^m(?P<measure>[1-9][0-9]*)-e(?P<event>[0-9]+)(?:-n(?P<member>[0-9]+))?$"
)


def canonical_event_id(
    *, measure_number: int, event_index: int, chord_member_index: int | None = None
) -> str:
    if not isinstance(measure_number, int) or isinstance(measure_number, bool) or measure_number < 1:
        raise Stage7D12ContractError("measure_number must be a positive integer")
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
        raise Stage7D12ContractError("event_index must be a non-negative integer")
    base = f"m{measure_number}-e{event_index}"
    if chord_member_index is None:
        return base
    if (
        not isinstance(chord_member_index, int)
        or isinstance(chord_member_index, bool)
        or chord_member_index < 0
    ):
        raise Stage7D12ContractError(
            "chord_member_index must be a non-negative integer"
        )
    return f"{base}-n{chord_member_index}"


def validate_canonical_event_id(value: object) -> str:
    if not isinstance(value, str) or _EVENT_ID.fullmatch(value) is None:
        raise Stage7D12ContractError("invalid canonical_event_id")
    return value


def development_split(row: Mapping[str, object]) -> str | None:
    """Return TRAIN/VALIDATION split while sealing TEST before all other reads.

    Callers must invoke this before reading source IDs, paths, hashes, labels, or
    geometry.  TEST returns ``None`` after touching only ``split``.
    """

    split = row.get("split")
    if split == "test":
        return None
    if split not in EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        raise Stage7D12ContractError("invalid D12 development split")
    assert isinstance(split, str)
    return split


def notehead_fill_class(duration: object) -> str:
    """Map supported canonical note duration names to visual head fill class."""

    if duration in {"whole", "half"}:
        return "open"
    if duration in {"quarter", "eighth"}:
        return "filled"
    raise Stage7D12ContractError("unsupported V1 note duration")


def rest_class(duration: object) -> str:
    if duration not in REST_CLASSES:
        raise Stage7D12ContractError("unsupported V1 rest duration")
    assert isinstance(duration, str)
    return duration


def accidental_class(value: object) -> str:
    if value not in ACCIDENTAL_CLASSES:
        raise Stage7D12ContractError("unsupported visible accidental")
    assert isinstance(value, str)
    return value


# Geometry may be admitted only when canonical and renderer cardinalities agree
# for the exact measure/kind.  Ordering or proximity is never allowed to hide a
# missing/extra glyph.  The concrete extractor must add stricter uniqueness and
# bbox containment checks before persistence.
def require_link_cardinality(
    *, kind: str, canonical_count: int, renderer_count: int
) -> None:
    if kind not in RENDERER_OBJECT_CLASSES:
        raise Stage7D12ContractError("unknown D12 symbol kind")
    for name, value in (
        ("canonical_count", canonical_count),
        ("renderer_count", renderer_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise Stage7D12ContractError(f"{name} must be a non-negative integer")
    if canonical_count != renderer_count:
        raise Stage7D12ContractError(
            f"{kind} canonical/renderer cardinality mismatch"
        )


D12_ACCEPTANCE_GATES: Final[dict[str, object]] = {
    "source_sample_counts": EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
    "source_family_counts": EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
    "test_specialist_records": TEST_SPECIALIST_RECORDS,
    "optimizer_steps": OPTIMIZER_STEPS,
    "source_png_sha_rechecked": True,
    "accepted_d6_label_sha_rechecked": True,
    "renderer_geometry_fingerprint_bound": True,
    "final_png_transform_fingerprint_bound": True,
    "canonical_renderer_cardinality_exact": True,
    "ambiguous_or_unlinked_symbol": "reject_sample",
    "bbox_must_be_finite_positive_and_inside_bound_measure": True,
    "canonical_event_id_unique_within_source_sample": True,
    "persisted_artifacts_independently_reopened": True,
    "complete_marker_after_independent_verification_only": True,
    # Class-count minima are intentionally not invented before the deterministic
    # D12 inventory exists.  D12 must publish TRAIN/VALIDATION counts first; a
    # future training package freezes readiness/balance thresholds before any
    # optimizer step.
    "training_class_balance_thresholds": "deferred_until_verified_d12_inventory",
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stage7d12_contract_payload() -> dict[str, object]:
    return {
        "schema": STAGE7D12_SCHEMA,
        "version": STAGE7D12_VERSION,
        "targets": [asdict(target) for target in SYMBOL_TARGETS],
        "renderer_object_classes": dict(RENDERER_OBJECT_CLASSES),
        "future_training_boundary": dict(FUTURE_TRAINING_BOUNDARY),
        "acceptance_gates": dict(D12_ACCEPTANCE_GATES),
    }


def stage7d12_contract_fingerprint() -> str:
    return sha256(_canonical_json_bytes(stage7d12_contract_payload())).hexdigest()
