"""Stage 7-D8 validation-only diagnostics for the accepted Structure specialist.

D8 does not train or mutate a model. It binds the exact authoritative D7
checkpoint/metrics/verification bundle, reproduces the accepted Structure
validation result, then diagnoses all Structure channels on VALIDATION only.
The sealed TEST split remains inaccessible through the inherited D7 loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Final, Mapping, Sequence

import torch
from torch.nn import functional as F

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d7_specialist_training import (
    FROZEN_D7_CONFIG,
    STAGE7D7_METRICS_SCHEMA,
    STAGE7D7_VERIFICATION_SCHEMA,
    STAGE7D7_VERSION,
    STRUCTURE_CHANNELS,
    _evaluate,
    _stack_records,
    build_specialist_model,
    load_verified_stage7d7_records,
    stage7d7_profile_fingerprint,
)
from .training_model import assert_finite_tensor, assert_model_finite, model_state_sha256


STAGE7D8_VERSION: Final[str] = "stage7d8-structure-validation-diagnostics-v1"
STAGE7D8_REPORT_SCHEMA: Final[str] = "stage7d8-structure-diagnostic-report-v1"
THRESHOLD_MILLIS: Final[tuple[int, ...]] = tuple(range(50, 1000, 50))
TOLERANCE_RADII: Final[tuple[int, ...]] = (1, 2)
DEFAULT_THRESHOLD_MILLIS: Final[int] = 500

EXPECTED_D7_RUN_ID: Final[str] = (
    "4ce2903206c7965471bb9569d379d8d9d1022d9248d80886638acfe0bd822598"
)
EXPECTED_D7_REPOSITORY_SHA: Final[str] = (
    "25bdf2b3146faba54a93c00f05537f522c75b532"
)
EXPECTED_D7_PROFILE_FINGERPRINT: Final[str] = (
    "7b7fbc79c748da0f1195bc9273fe012e0b1128b3a1e491bb484653d47cb5201a"
)
EXPECTED_D7_CHECKPOINT_SHA256: Final[str] = (
    "5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc"
)
EXPECTED_D7_METRICS_SHA256: Final[str] = (
    "43cd98a75c2db740b4af6ee3c8826122fa387347820d2e7d2c639ac2fe30f792"
)
EXPECTED_D7_VERIFICATION_SHA256: Final[str] = (
    "cdc0733af1bd6c7336f5bd2a0cb12fcae269120d8b5a9a564f08db860ee21a0a"
)
EXPECTED_D7_STAFF_STATE_SHA256: Final[str] = (
    "3131548548521229e6acd6fee8cffc66081cb54125645f9eff5a488de7603af8"
)
EXPECTED_D7_STRUCTURE_STATE_SHA256: Final[str] = (
    "0d11b2ae414959b678ccc22a6b8cfcc1edc1ecadc3c73ed6ab5a0cda6e593907"
)
EXPECTED_D7_STRUCTURE_VALIDATION_LOSS: Final[float] = 0.49127569106908947
EXPECTED_D7_STRUCTURE_DICE: Final[dict[str, float]] = {
    "system_region": 0.93046746804164,
    "measure_region": 0.8445145579484793,
    "barline": 0.2667824041384917,
    "clef_g2": 0.8228637140530807,
    "meter_2_4": 0.34398488560691476,
    "meter_3_4": 0.34151152062874574,
    "meter_4_4": 0.3092358358777486,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Stage7D8DiagnosticError(RuntimeError):
    """Raised when D8 provenance, tensors, metrics, or output fail closed."""


def _fail(message: str) -> None:
    raise Stage7D8DiagnosticError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D8DiagnosticError("D8 payload is not canonical JSON serializable") from exc


def _read_exact_bytes(path: Path, expected_sha256: str, maximum: int, name: str) -> bytes:
    if _HEX64.fullmatch(expected_sha256) is None:
        _fail(f"{name} expected SHA-256 is invalid")
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside the D8 bound")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        _fail(f"{name} SHA-256 mismatch")
    return raw


def _read_exact_canonical_json(
    path: Path,
    expected_sha256: str,
    maximum: int,
    name: str,
) -> dict[str, object]:
    raw = _read_exact_bytes(path, expected_sha256, maximum, name)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda value: _fail(f"non-finite constant in {name}: {value}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D8DiagnosticError(f"{name} is not valid canonical JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _require_exact(value: object, expected: object, name: str) -> None:
    if value != expected:
        _fail(f"{name} differs from accepted D7 evidence")


@dataclass(frozen=True, slots=True)
class VerifiedD7Bundle:
    checkpoint_path: Path
    metrics_path: Path
    verification_path: Path
    structure_model: torch.nn.Module


@dataclass(frozen=True, slots=True)
class Stage7D8RunResult:
    report_path: Path
    report_sha256: str
    complete_path: Path
    report: dict[str, object]


def verify_accepted_d7_bundle(run_directory: str | Path) -> VerifiedD7Bundle:
    """Verify and safely reload the exact accepted external D7 artifact bundle."""
    root = Path(run_directory)
    if root.is_symlink() or not root.is_dir():
        _fail("D7 run directory must be a regular directory")
    if root.name != EXPECTED_D7_RUN_ID:
        _fail("D7 run directory identity mismatch")

    checkpoint_path = root / f"checkpoint-{EXPECTED_D7_CHECKPOINT_SHA256}.pt"
    metrics_path = root / f"metrics-{EXPECTED_D7_METRICS_SHA256}.json"
    verification_path = root / f"verification-{EXPECTED_D7_VERIFICATION_SHA256}.json"
    complete_path = root / "COMPLETE"

    _read_exact_bytes(
        checkpoint_path,
        EXPECTED_D7_CHECKPOINT_SHA256,
        64 * 1024 * 1024,
        "D7 checkpoint",
    )
    metrics = _read_exact_canonical_json(
        metrics_path,
        EXPECTED_D7_METRICS_SHA256,
        8 * 1024 * 1024,
        "D7 metrics",
    )
    verification = _read_exact_canonical_json(
        verification_path,
        EXPECTED_D7_VERIFICATION_SHA256,
        4 * 1024 * 1024,
        "D7 verification",
    )
    complete = _read_exact_bytes(
        complete_path,
        sha256(
            (
                f"{EXPECTED_D7_VERIFICATION_SHA256}  {verification_path.name}\n"
                f"{EXPECTED_D7_METRICS_SHA256}  {metrics_path.name}\n"
                f"{EXPECTED_D7_CHECKPOINT_SHA256}  {checkpoint_path.name}\n"
            ).encode("ascii")
        ).hexdigest(),
        4096,
        "D7 COMPLETE",
    )
    expected_complete = (
        f"{EXPECTED_D7_VERIFICATION_SHA256}  {verification_path.name}\n"
        f"{EXPECTED_D7_METRICS_SHA256}  {metrics_path.name}\n"
        f"{EXPECTED_D7_CHECKPOINT_SHA256}  {checkpoint_path.name}\n"
    ).encode("ascii")
    if complete != expected_complete:
        _fail("D7 COMPLETE content mismatch")

    _require_exact(metrics.get("schema_version"), STAGE7D7_METRICS_SCHEMA, "D7 metrics schema")
    _require_exact(metrics.get("stage7d7_version"), STAGE7D7_VERSION, "D7 metrics version")
    _require_exact(metrics.get("repository_sha"), EXPECTED_D7_REPOSITORY_SHA, "D7 repository SHA")
    _require_exact(
        metrics.get("profile_fingerprint"),
        EXPECTED_D7_PROFILE_FINGERPRINT,
        "D7 profile fingerprint",
    )
    _require_exact(metrics.get("sealed_test_split_opened"), False, "D7 TEST-open marker")
    dataset = _require_mapping(metrics.get("dataset"), "D7 dataset")
    _require_exact(dataset.get("train_samples"), 1230, "D7 train sample count")
    _require_exact(dataset.get("validation_samples"), 153, "D7 validation sample count")
    _require_exact(dataset.get("test_records"), 0, "D7 test record count")

    structure_metrics = _require_mapping(metrics.get("structure"), "D7 structure metrics")
    _require_exact(
        structure_metrics.get("state_sha256"),
        EXPECTED_D7_STRUCTURE_STATE_SHA256,
        "D7 structure state hash",
    )
    _require_exact(structure_metrics.get("best_epoch"), 8, "D7 structure best epoch")
    _require_exact(structure_metrics.get("optimizer_steps"), 1640, "D7 structure optimizer steps")

    _require_exact(
        verification.get("schema_version"),
        STAGE7D7_VERIFICATION_SCHEMA,
        "D7 verification schema",
    )
    _require_exact(
        verification.get("stage7d7_version"),
        STAGE7D7_VERSION,
        "D7 verification version",
    )
    _require_exact(verification.get("test_records"), 0, "D7 verification test records")
    _require_exact(verification.get("test_opened"), False, "D7 verification TEST-open marker")
    _require_exact(
        verification.get("checkpoint_reload_verified"),
        True,
        "D7 checkpoint reload marker",
    )
    _require_exact(
        verification.get("checkpoint_sha256"),
        EXPECTED_D7_CHECKPOINT_SHA256,
        "D7 verification checkpoint hash",
    )
    _require_exact(
        verification.get("metrics_sha256"),
        EXPECTED_D7_METRICS_SHA256,
        "D7 verification metrics hash",
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7D8DiagnosticError("D7 checkpoint cannot be safely loaded") from exc
    if not isinstance(checkpoint, dict):
        _fail("D7 checkpoint root is invalid")
    _require_exact(
        checkpoint.get("schema_version"),
        "stage7d7-specialist-checkpoint-v1",
        "D7 checkpoint schema",
    )
    _require_exact(
        checkpoint.get("repository_sha"),
        EXPECTED_D7_REPOSITORY_SHA,
        "D7 checkpoint repository SHA",
    )
    _require_exact(
        checkpoint.get("profile_fingerprint"),
        EXPECTED_D7_PROFILE_FINGERPRINT,
        "D7 checkpoint profile fingerprint",
    )

    staff_model = build_specialist_model("staff", FROZEN_D7_CONFIG)
    structure_model = build_specialist_model("structure", FROZEN_D7_CONFIG)
    try:
        staff_model.load_state_dict(checkpoint["staff_state_dict"], strict=True)
        structure_model.load_state_dict(checkpoint["structure_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise Stage7D8DiagnosticError("D7 checkpoint state cannot be strictly restored") from exc
    assert_model_finite(staff_model)
    assert_model_finite(structure_model)
    if model_state_sha256(staff_model) != EXPECTED_D7_STAFF_STATE_SHA256:
        _fail("D7 staff state hash mismatch after reload")
    if model_state_sha256(structure_model) != EXPECTED_D7_STRUCTURE_STATE_SHA256:
        _fail("D7 structure state hash mismatch after reload")
    structure_model.eval()
    return VerifiedD7Bundle(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        verification_path=verification_path,
        structure_model=structure_model,
    )


def _validate_probability_inputs(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    channels: Sequence[str],
) -> None:
    if not isinstance(probabilities, torch.Tensor) or not isinstance(targets, torch.Tensor):
        _fail("D8 probabilities and targets must be tensors")
    if probabilities.dtype != torch.float32 or targets.dtype != torch.float32:
        _fail("D8 probabilities and targets must be float32")
    if probabilities.ndim != 4 or targets.ndim != 4 or probabilities.shape != targets.shape:
        _fail("D8 probabilities and targets must share [B,C,H,W] shape")
    if probabilities.shape[1] != len(channels) or not channels:
        _fail("D8 channel count mismatch")
    if len(set(channels)) != len(channels) or any(not isinstance(name, str) or not name for name in channels):
        _fail("D8 channel names must be unique non-empty strings")
    assert_finite_tensor("D8 probabilities", probabilities)
    assert_finite_tensor("D8 targets", targets)
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        _fail("D8 probabilities must be in [0,1]")
    if bool((targets < 0).any()) or bool((targets > 1).any()):
        _fail("D8 targets must be in [0,1]")


def _metric(numerator: int, denominator: int) -> float:
    if denominator < 0 or numerator < 0:
        _fail("D8 metric count is negative")
    if denominator == 0:
        return 1.0
    value = numerator / denominator
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        _fail("D8 metric is non-finite or outside [0,1]")
    return value


def _threshold_key(millis: int) -> str:
    return f"{millis / 1000.0:.2f}"


def _new_count_state(channel_count: int, threshold_count: int) -> dict[str, object]:
    return {
        "total_pixels": [0] * channel_count,
        "positive_pixels": [0] * channel_count,
        "positive_records": [0] * channel_count,
        "positive_probability_sum": [0.0] * channel_count,
        "negative_probability_sum": [0.0] * channel_count,
        "tp": [[0] * threshold_count for _ in range(channel_count)],
        "fp": [[0] * threshold_count for _ in range(channel_count)],
        "fn": [[0] * threshold_count for _ in range(channel_count)],
    }


def _accumulate_probability_batch(
    state: dict[str, object],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    channels: Sequence[str],
    threshold_millis: Sequence[int],
) -> None:
    _validate_probability_inputs(probabilities, targets, channels)
    target_positive = targets >= 0.5
    flat_target = target_positive.flatten(2)
    batch_pixels = probabilities.shape[0] * probabilities.shape[2] * probabilities.shape[3]

    total_pixels = state["total_pixels"]
    positive_pixels = state["positive_pixels"]
    positive_records = state["positive_records"]
    positive_probability_sum = state["positive_probability_sum"]
    negative_probability_sum = state["negative_probability_sum"]
    tp_state = state["tp"]
    fp_state = state["fp"]
    fn_state = state["fn"]
    if not all(isinstance(value, list) for value in (
        total_pixels,
        positive_pixels,
        positive_records,
        positive_probability_sum,
        negative_probability_sum,
        tp_state,
        fp_state,
        fn_state,
    )):
        _fail("D8 accumulator state is invalid")

    for index in range(len(channels)):
        positive_mask = target_positive[:, index]
        negative_mask = ~positive_mask
        positive_count = int(positive_mask.sum().item())
        total_pixels[index] += int(batch_pixels)
        positive_pixels[index] += positive_count
        positive_records[index] += int(flat_target[:, index].any(dim=1).sum().item())
        positive_probability_sum[index] += float(probabilities[:, index][positive_mask].sum().item())
        negative_probability_sum[index] += float(probabilities[:, index][negative_mask].sum().item())

    for threshold_index, millis in enumerate(threshold_millis):
        if not isinstance(millis, int) or isinstance(millis, bool) or not 1 <= millis <= 999:
            _fail("D8 threshold millis is invalid")
        predicted_positive = probabilities >= (millis / 1000.0)
        tp = (predicted_positive & target_positive).sum(dim=(0, 2, 3))
        fp = (predicted_positive & ~target_positive).sum(dim=(0, 2, 3))
        fn = (~predicted_positive & target_positive).sum(dim=(0, 2, 3))
        for channel_index in range(len(channels)):
            tp_state[channel_index][threshold_index] += int(tp[channel_index].item())
            fp_state[channel_index][threshold_index] += int(fp[channel_index].item())
            fn_state[channel_index][threshold_index] += int(fn[channel_index].item())


def _finalize_probability_state(
    state: dict[str, object],
    channels: Sequence[str],
    threshold_millis: Sequence[int],
) -> dict[str, object]:
    if DEFAULT_THRESHOLD_MILLIS not in threshold_millis:
        _fail("D8 threshold grid must contain 0.50")
    default_index = tuple(threshold_millis).index(DEFAULT_THRESHOLD_MILLIS)
    output: dict[str, object] = {}
    for channel_index, channel in enumerate(channels):
        total = int(state["total_pixels"][channel_index])
        positive = int(state["positive_pixels"][channel_index])
        negative = total - positive
        if total <= 0 or positive < 0 or negative < 0:
            _fail("D8 accumulated pixel counts are invalid")
        threshold_rows: dict[str, object] = {}
        best: tuple[float, int, dict[str, float | int]] | None = None
        for threshold_index, millis in enumerate(threshold_millis):
            tp = int(state["tp"][channel_index][threshold_index])
            fp = int(state["fp"][channel_index][threshold_index])
            fn = int(state["fn"][channel_index][threshold_index])
            precision = _metric(tp, tp + fp)
            recall = _metric(tp, tp + fn)
            dice = _metric(2 * tp, 2 * tp + fp + fn)
            predicted_positive = tp + fp
            row: dict[str, float | int] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "dice": dice,
                "predicted_positive_fraction": _metric(predicted_positive, total),
            }
            threshold_rows[_threshold_key(millis)] = row
            ranking = (dice, -abs(millis - DEFAULT_THRESHOLD_MILLIS), -millis)
            if best is None or ranking > (best[0], -abs(best[1] - DEFAULT_THRESHOLD_MILLIS), -best[1]):
                best = (dice, millis, row)
        if best is None:
            _fail("D8 threshold grid is empty")
        default_row = threshold_rows[_threshold_key(DEFAULT_THRESHOLD_MILLIS)]
        best_dice, best_millis, best_row = best
        positive_sum = float(state["positive_probability_sum"][channel_index])
        negative_sum = float(state["negative_probability_sum"][channel_index])
        output[channel] = {
            "positive_records": int(state["positive_records"][channel_index]),
            "positive_pixels": positive,
            "total_pixels": total,
            "positive_pixel_fraction": _metric(positive, total),
            "mean_probability_on_positive_pixels": (positive_sum / positive) if positive else 0.0,
            "mean_probability_on_negative_pixels": (negative_sum / negative) if negative else 0.0,
            "threshold_0_50": default_row,
            "best_threshold": best_millis / 1000.0,
            "best_threshold_metrics": best_row,
            "best_threshold_dice_gain_over_0_50": best_dice - float(default_row["dice"]),
            "threshold_sweep": threshold_rows,
        }
    return output


def diagnose_probability_tensor(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    channels: Sequence[str],
    threshold_millis: Sequence[int] = THRESHOLD_MILLIS,
) -> dict[str, object]:
    """Pure tensor diagnostic used by D8 and unit tests; it never mutates a model."""
    state = _new_count_state(len(channels), len(threshold_millis))
    _accumulate_probability_batch(state, probabilities, targets, channels, threshold_millis)
    return _finalize_probability_state(state, channels, threshold_millis)


def _tolerant_counts(
    predicted_positive: torch.Tensor,
    target_positive: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(radius, int) or isinstance(radius, bool) or not 1 <= radius <= 8:
        _fail("D8 tolerance radius is invalid")
    kernel = radius * 2 + 1
    predicted_float = predicted_positive.to(torch.float32)
    target_float = target_positive.to(torch.float32)
    dilated_target = F.max_pool2d(target_float, kernel_size=kernel, stride=1, padding=radius) >= 0.5
    dilated_prediction = F.max_pool2d(predicted_float, kernel_size=kernel, stride=1, padding=radius) >= 0.5
    matched_prediction = (predicted_positive & dilated_target).sum(dim=(0, 2, 3))
    predicted_count = predicted_positive.sum(dim=(0, 2, 3))
    matched_target = (target_positive & dilated_prediction).sum(dim=(0, 2, 3))
    target_count = target_positive.sum(dim=(0, 2, 3))
    return matched_prediction, predicted_count, matched_target, target_count


def tolerant_f1_for_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    channels: Sequence[str],
    threshold_millis_by_channel: Mapping[str, int],
    radius: int,
) -> dict[str, dict[str, float]]:
    """Measure near-miss localization with a bounded pixel tolerance."""
    _validate_probability_inputs(probabilities, targets, channels)
    thresholds: list[float] = []
    for channel in channels:
        millis = threshold_millis_by_channel.get(channel)
        if not isinstance(millis, int) or isinstance(millis, bool) or not 1 <= millis <= 999:
            _fail("D8 per-channel tolerance threshold is invalid")
        thresholds.append(millis / 1000.0)
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32).view(1, len(channels), 1, 1)
    predicted_positive = probabilities >= threshold_tensor
    target_positive = targets >= 0.5
    matched_prediction, predicted_count, matched_target, target_count = _tolerant_counts(
        predicted_positive,
        target_positive,
        radius,
    )
    output: dict[str, dict[str, float]] = {}
    for index, channel in enumerate(channels):
        precision = _metric(int(matched_prediction[index].item()), int(predicted_count[index].item()))
        recall = _metric(int(matched_target[index].item()), int(target_count[index].item()))
        f1 = _metric(
            int(2 * matched_prediction[index].item() * matched_target[index].item()),
            int(
                matched_prediction[index].item() * target_count[index].item()
                + matched_target[index].item() * predicted_count[index].item()
            ),
        ) if precision + recall > 0 else 0.0
        # Recompute from bounded [0,1] precision/recall to avoid count-asymmetry surprises.
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        output[channel] = {"precision": precision, "recall": recall, "f1": f1}
    return output


def _accumulate_tolerant_batch(
    state: dict[tuple[str, int, str], list[int]],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    channels: Sequence[str],
    threshold_millis_by_channel: Mapping[str, int],
    label: str,
) -> None:
    thresholds = torch.tensor(
        [threshold_millis_by_channel[channel] / 1000.0 for channel in channels],
        dtype=torch.float32,
    ).view(1, len(channels), 1, 1)
    predicted_positive = probabilities >= thresholds
    target_positive = targets >= 0.5
    for radius in TOLERANCE_RADII:
        matched_prediction, predicted_count, matched_target, target_count = _tolerant_counts(
            predicted_positive,
            target_positive,
            radius,
        )
        for index, channel in enumerate(channels):
            key = (channel, radius, label)
            counts = state.setdefault(key, [0, 0, 0, 0])
            counts[0] += int(matched_prediction[index].item())
            counts[1] += int(predicted_count[index].item())
            counts[2] += int(matched_target[index].item())
            counts[3] += int(target_count[index].item())


def _finalize_tolerant_state(
    state: Mapping[tuple[str, int, str], Sequence[int]],
    channels: Sequence[str],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for channel in channels:
        channel_rows: dict[str, object] = {}
        for label in ("threshold_0_50", "best_threshold"):
            radius_rows: dict[str, object] = {}
            for radius in TOLERANCE_RADII:
                matched_prediction, predicted_count, matched_target, target_count = state[
                    (channel, radius, label)
                ]
                precision = _metric(matched_prediction, predicted_count)
                recall = _metric(matched_target, target_count)
                f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
                radius_rows[str(radius)] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            channel_rows[label] = radius_rows
        output[channel] = channel_rows
    return output


def _fresh_output_root(path: Path, repository_root: Path) -> None:
    resolved = path.resolve()
    repo = repository_root.resolve()
    if resolved == repo or repo in resolved.parents:
        _fail("D8 output must stay outside repository")
    if path.exists() or path.is_symlink():
        _fail("D8 output root must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def run_stage7d8_structure_diagnostics(
    *,
    corpus_root: str | Path,
    derivative_root: str | Path,
    d7_run_directory: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> Stage7D8RunResult:
    """Run the authoritative D8 validation-only diagnostic and write one report."""
    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    runtime = verify_stage7c_runtime()
    bundle = verify_accepted_d7_bundle(d7_run_directory)
    records = load_verified_stage7d7_records(corpus_root, derivative_root)
    validation_records = tuple(record for record in records if record.split == "validation")
    if len(validation_records) != 153:
        _fail("D8 requires exactly 153 accepted validation records")
    if len({record.family_id for record in validation_records}) != 51:
        _fail("D8 requires exactly 51 validation families")

    baseline_loss, baseline_dice = _evaluate(
        bundle.structure_model,
        validation_records,
        "structure",
        FROZEN_D7_CONFIG,
    )
    if abs(baseline_loss - EXPECTED_D7_STRUCTURE_VALIDATION_LOSS) > 1e-9:
        _fail("D8 did not reproduce accepted D7 Structure validation loss")
    for channel in STRUCTURE_CHANNELS:
        if abs(baseline_dice[channel] - EXPECTED_D7_STRUCTURE_DICE[channel]) > 1e-9:
            _fail(f"D8 did not reproduce accepted D7 {channel} Dice")

    count_state = _new_count_state(len(STRUCTURE_CHANNELS), len(THRESHOLD_MILLIS))
    with torch.no_grad():
        for start in range(0, len(validation_records), FROZEN_D7_CONFIG.batch_size):
            batch = validation_records[start : start + FROZEN_D7_CONFIG.batch_size]
            images, targets = _stack_records(batch, "structure", FROZEN_D7_CONFIG)
            probabilities = torch.sigmoid(bundle.structure_model(images))
            assert_finite_tensor("D8 Structure probabilities", probabilities)
            _accumulate_probability_batch(
                count_state,
                probabilities,
                targets,
                STRUCTURE_CHANNELS,
                THRESHOLD_MILLIS,
            )
    channels = _finalize_probability_state(count_state, STRUCTURE_CHANNELS, THRESHOLD_MILLIS)

    default_thresholds = {channel: DEFAULT_THRESHOLD_MILLIS for channel in STRUCTURE_CHANNELS}
    best_thresholds = {
        channel: int(round(float(channels[channel]["best_threshold"]) * 1000.0))
        for channel in STRUCTURE_CHANNELS
    }
    tolerant_state: dict[tuple[str, int, str], list[int]] = {}
    with torch.no_grad():
        for start in range(0, len(validation_records), FROZEN_D7_CONFIG.batch_size):
            batch = validation_records[start : start + FROZEN_D7_CONFIG.batch_size]
            images, targets = _stack_records(batch, "structure", FROZEN_D7_CONFIG)
            probabilities = torch.sigmoid(bundle.structure_model(images))
            _accumulate_tolerant_batch(
                tolerant_state,
                probabilities,
                targets,
                STRUCTURE_CHANNELS,
                default_thresholds,
                "threshold_0_50",
            )
            _accumulate_tolerant_batch(
                tolerant_state,
                probabilities,
                targets,
                STRUCTURE_CHANNELS,
                best_thresholds,
                "best_threshold",
            )
    tolerant = _finalize_tolerant_state(tolerant_state, STRUCTURE_CHANNELS)

    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    ending_runtime = verify_stage7c_runtime()
    if ending_sha != repository_sha or ending_origin != repository_origin:
        _fail("repository identity changed during D8 diagnostic")
    if ending_runtime != runtime:
        _fail("runtime identity changed during D8 diagnostic")

    report: dict[str, object] = {
        "schema_version": STAGE7D8_REPORT_SCHEMA,
        "stage7d8_version": STAGE7D8_VERSION,
        "repository_sha": repository_sha,
        "repository_origin": repository_origin,
        "runtime": runtime,
        "accepted_d7": {
            "run_id": EXPECTED_D7_RUN_ID,
            "repository_sha": EXPECTED_D7_REPOSITORY_SHA,
            "profile_fingerprint": EXPECTED_D7_PROFILE_FINGERPRINT,
            "checkpoint_sha256": EXPECTED_D7_CHECKPOINT_SHA256,
            "metrics_sha256": EXPECTED_D7_METRICS_SHA256,
            "verification_sha256": EXPECTED_D7_VERIFICATION_SHA256,
            "structure_state_sha256": EXPECTED_D7_STRUCTURE_STATE_SHA256,
        },
        "surface": {
            "train_tensor_records": 0,
            "validation_tensor_records": 153,
            "validation_families": 51,
            "test_records": 0,
            "optimizer_steps": 0,
            "threshold_millis": list(THRESHOLD_MILLIS),
            "tolerance_radii_pixels": list(TOLERANCE_RADII),
        },
        "baseline_reproduction": {
            "validation_loss": baseline_loss,
            "channel_dice": baseline_dice,
        },
        "channels": channels,
        "tolerant_localization": tolerant,
        "sealed_test_split_opened": False,
        "model_mutated": False,
    }
    report_bytes = _canonical_json(report)
    report_sha = sha256(report_bytes).hexdigest()
    root = Path(output_root)
    _fresh_output_root(root, Path(repository_root))
    report_path = root / f"structure-diagnostic-{report_sha}.json"
    complete_path = root / "COMPLETE"
    report_path.write_bytes(report_bytes)
    complete_path.write_bytes(f"{report_sha}  {report_path.name}\n".encode("ascii"))
    if sha256(report_path.read_bytes()).hexdigest() != report_sha:
        _fail("persisted D8 report hash mismatch")
    return Stage7D8RunResult(
        report_path=report_path,
        report_sha256=report_sha,
        complete_path=complete_path,
        report=report,
    )
