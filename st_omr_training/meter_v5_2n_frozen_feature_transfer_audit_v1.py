"""Meter V5-2N frozen-feature transfer audit.

Read-only diagnostic over the existing V5 adaptation TRAIN slots and historical
M4A TRAIN surface. The exact frozen 2-AI/3-AI feature extractors are inspected
without gradients or parameter updates. No new crop or spatial semantics are
introduced.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Callable, Final, Mapping

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as ret_legacy
from . import meter_v5_2d_positive_collapse_forensics_v1 as forensic


SCHEMA: Final[str] = "st-omr-meter-v5-2n-frozen-feature-transfer-audit-v1"
REPORT_NAME: Final[str] = "v5_2n_frozen_feature_transfer_audit_v1.json"
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_V5_POSITIVE: Final[int] = 90
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
EXPECTED_HISTORICAL_LABEL_COUNTS: Final[dict[str, int]] = {
    "2": 1527,
    "3": 1587,
    "4": 6396,
    "NONE": 17454,
}
EXPECTED_FEATURE_DIM: Final[int] = 64
HISTORICAL_BATCH_SIZE: Final[int] = 256

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2NError(RuntimeError):
    """Raised whenever a frozen-feature audit invariant fails closed."""


def _fail(message: str) -> None:
    raise MeterV5_2NError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": True,
        "checkpoint_write": False,
        "image_read": True,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "old_d11_glyph_window_reused": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
        "architecture_selected": False,
        "residual_topology_selected": False,
        "repair_training_authorized": False,
    }


def _finite_tensor(tensor, *, name: str) -> None:
    torch, _nn = v52b._import_torch()
    if tensor.numel() == 0:
        _fail(f"empty tensor: {name}")
    if not bool(torch.isfinite(tensor).all().item()):
        _fail(f"non-finite tensor: {name}")


def _cosine(a, b) -> float:
    torch, _nn = v52b._import_torch()
    a64 = a.detach().cpu().to(dtype=torch.float64).reshape(-1)
    b64 = b.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if a64.shape != b64.shape or a64.numel() == 0:
        _fail("cosine vector shape mismatch")
    na = float(torch.linalg.vector_norm(a64).item())
    nb = float(torch.linalg.vector_norm(b64).item())
    if na <= 0.0 or nb <= 0.0 or not math.isfinite(na) or not math.isfinite(nb):
        _fail("cosine requires finite nonzero vectors")
    value = float(torch.dot(a64, b64).item()) / (na * nb)
    return min(1.0, max(-1.0, value))


def _margin_stats(values) -> dict[str, float]:
    torch, _nn = v52b._import_torch()
    x = values.detach().cpu().to(dtype=torch.float64).reshape(-1)
    _finite_tensor(x, name="centroid-margin")
    q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float64)
    quantiles = torch.quantile(x, q)
    return {
        "min": float(x.min().item()),
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


def feature_transfer_metrics_v1(
    *,
    source_features,
    source_targets,
    v5_features,
    v5_targets,
    head_weight,
) -> dict[str, object]:
    """Pure descriptive feature-geometry metrics; performs no fitting."""
    torch, _nn = v52b._import_torch()
    source_features = source_features.detach().cpu().to(dtype=torch.float64)
    v5_features = v5_features.detach().cpu().to(dtype=torch.float64)
    source_targets = source_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    v5_targets = v5_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    head_weight = head_weight.detach().cpu().to(dtype=torch.float64).reshape(-1)

    if source_features.ndim != 2 or v5_features.ndim != 2:
        _fail("feature tensors must be rank-2")
    if source_features.shape[1] != v5_features.shape[1]:
        _fail("source/V5 feature dimensions differ")
    if source_features.shape[1] != EXPECTED_FEATURE_DIM:
        _fail(f"digit specialist feature dimension changed: {source_features.shape[1]}")
    if len(source_features) != len(source_targets) or len(v5_features) != len(v5_targets):
        _fail("feature/target cardinality mismatch")
    if head_weight.numel() != EXPECTED_FEATURE_DIM:
        _fail("frozen head feature dimension changed")
    _finite_tensor(source_features, name="source-features")
    _finite_tensor(v5_features, name="v5-features")
    if not bool(((source_targets == 0.0) | (source_targets == 1.0)).all().item()):
        _fail("source targets are not binary")
    if not bool(((v5_targets == 0.0) | (v5_targets == 1.0)).all().item()):
        _fail("V5 targets are not binary")

    source_pos = source_features[source_targets == 1.0]
    source_neg = source_features[source_targets == 0.0]
    v5_pos = v5_features[v5_targets == 1.0]
    v5_neg = v5_features[v5_targets == 0.0]
    if min(len(source_pos), len(source_neg), len(v5_pos), len(v5_neg)) <= 0:
        _fail("both binary classes are required on both domains")

    source_pos_centroid = source_pos.mean(dim=0)
    source_neg_centroid = source_neg.mean(dim=0)
    v5_pos_centroid = v5_pos.mean(dim=0)
    v5_neg_centroid = v5_neg.mean(dim=0)

    d_pos = ((v5_features - source_pos_centroid) ** 2).sum(dim=1)
    d_neg = ((v5_features - source_neg_centroid) ** 2).sum(dim=1)
    margins = torch.where(v5_targets == 1.0, d_neg - d_pos, d_pos - d_neg)
    positive_margins = margins[v5_targets == 1.0]
    negative_margins = margins[v5_targets == 0.0]

    delta_source = source_pos_centroid - source_neg_centroid
    delta_v5 = v5_pos_centroid - v5_neg_centroid
    source_delta_norm = float(torch.linalg.vector_norm(delta_source).item())
    v5_delta_norm = float(torch.linalg.vector_norm(delta_v5).item())
    if source_delta_norm <= 0.0 or v5_delta_norm <= 0.0:
        _fail("class-separation vector collapsed")

    return {
        "feature_dim": int(source_features.shape[1]),
        "source_count": int(len(source_features)),
        "source_positive_count": int(len(source_pos)),
        "source_negative_count": int(len(source_neg)),
        "v5_count": int(len(v5_features)),
        "v5_positive_count": int(len(v5_pos)),
        "v5_negative_count": int(len(v5_neg)),
        "nearest_historical_centroid": {
            "positive_correct_fraction": float((positive_margins > 0.0).to(dtype=torch.float64).mean().item()),
            "negative_correct_fraction": float((negative_margins > 0.0).to(dtype=torch.float64).mean().item()),
            "overall_correct_fraction": float((margins > 0.0).to(dtype=torch.float64).mean().item()),
            "tie_count": int((margins == 0.0).sum().item()),
            "positive_margin": _margin_stats(positive_margins),
            "negative_margin": _margin_stats(negative_margins),
            "overall_margin": _margin_stats(margins),
        },
        "class_separation": {
            "source_delta_l2": source_delta_norm,
            "v5_delta_l2": v5_delta_norm,
            "v5_over_source_delta_l2": v5_delta_norm / source_delta_norm,
            "source_v5_delta_cosine": _cosine(delta_source, delta_v5),
            "frozen_head_source_delta_cosine": _cosine(head_weight, delta_source),
            "frozen_head_v5_delta_cosine": _cosine(head_weight, delta_v5),
        },
    }


def _frozen_models(*, digit2_frozen: Path, digit3_frozen: Path) -> dict[str, object]:
    expected = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    paths = {"2": digit2_frozen, "3": digit3_frozen}
    models: dict[str, object] = {}
    for digit in ("2", "3"):
        if v52b._sha_file(paths[digit]) != expected[digit]:
            _fail(f"frozen {digit}-AI SHA changed")
        model = ret_legacy._frozen_model(paths[digit], digit=digit)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            if parameter.grad is not None:
                _fail("unexpected pre-existing gradient on frozen specialist")
        models[digit] = model
    return models


def _features_from_images(model, images):
    torch, _nn = v52b._import_torch()
    model.eval()
    with torch.no_grad():
        features = model.features(images).flatten(start_dim=1).cpu()
    _finite_tensor(features, name="frozen-features")
    if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_DIM:
        _fail(f"unexpected frozen feature shape: {tuple(features.shape)}")
    return features


def _v5_surface(root: Path, models: Mapping[str, object]):
    torch, _nn = v52b._import_torch()
    manifest_path, rows, _audit = v52b.verify_slot_manifest_v1(root)
    train = [row for row in rows if row.get("data_role") == "adaptation_train"]
    if len(train) != EXPECTED_V5_COUNT or len({row.get("sample_id") for row in train}) != 270:
        _fail("V5 adaptation TRAIN identity changed")
    ann_dir = root / v51.ANNOTATIONS_DIR
    images = torch.stack(
        [v52b._tensor_from_crop(ann_dir / row["crop_relpath"]) for row in train],
        dim=0,
    )
    features: dict[str, object] = {}
    targets: dict[str, object] = {}
    frozen_metrics: dict[str, object] = {}
    for digit in ("2", "3"):
        target = torch.tensor(
            [float(row[f"label_digit{digit}"]) for row in train],
            dtype=torch.float32,
        )
        if int(target.sum().item()) != EXPECTED_V5_POSITIVE:
            _fail(f"{digit}-AI V5 positive count changed")
        targets[digit] = target
        features[digit] = _features_from_images(models[digit], images)
        with torch.no_grad():
            probabilities = torch.sigmoid(models[digit](images)).cpu()
        frozen_metrics[digit] = v52b._binary_counts(
            probabilities,
            target,
            v52b.FROZEN_THRESHOLDS[digit],
        )
    return manifest_path, train, features, targets, frozen_metrics


def _historical_surface(
    *,
    m4a_root: Path,
    d10_root: Path,
    models: Mapping[str, object],
    progress: ProgressCallback | None,
):
    torch, _nn = v52b._import_torch()
    rows, d10_meter = forensic._historical_train_records(
        m4a_root=m4a_root,
        d10_root=d10_root,
    )
    counts = Counter(str(row.get("digit_label")) for row in rows)
    if len(rows) != EXPECTED_HISTORICAL_COUNT or dict(counts) != EXPECTED_HISTORICAL_LABEL_COUNTS:
        _fail(f"historical M4A TRAIN identity changed: total={len(rows)} counts={dict(counts)}")

    features_parts: dict[str, list[object]] = {"2": [], "3": []}
    labels: list[str] = []
    batch_tensors: list[object] = []
    batch_labels: list[str] = []
    processed = 0
    current_path: Path | None = None
    current_image: Image.Image | None = None

    def flush() -> None:
        nonlocal processed, batch_tensors, batch_labels
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors, dim=0).to(dtype=torch.float32).unsqueeze(1) / 255.0
        for digit in ("2", "3"):
            features_parts[digit].append(_features_from_images(models[digit], images))
        labels.extend(batch_labels)
        processed += len(batch_labels)
        if progress is not None and (
            processed == len(batch_labels)
            or processed % 2048 == 0
            or processed == EXPECTED_HISTORICAL_COUNT
        ):
            progress(processed, EXPECTED_HISTORICAL_COUNT, "v5-2n-historical-frozen-features")
        batch_tensors = []
        batch_labels = []

    try:
        for row in rows:
            source_id = str(row.get("source_record_id"))
            d10_row = d10_meter.get(source_id)
            if not isinstance(d10_row, Mapping):
                _fail(f"historical TRAIN missing D10 record: {source_id}")
            relpath = d10_row.get("image_path")
            if not isinstance(relpath, str) or not relpath:
                _fail(f"historical D10 image path missing: {source_id}")
            image_path = d10_root / relpath
            if image_path != current_path:
                if current_image is not None:
                    current_image.close()
                if image_path.is_symlink() or not image_path.is_file():
                    _fail(f"historical D10 image missing/non-regular: {image_path}")
                current_image = Image.open(image_path).convert("L")
                current_path = image_path
            assert current_image is not None
            canvas = ret_legacy._historical_canvas_from_bbox(current_image, row.get("bbox"))
            tensor = torch.tensor(list(canvas.getdata()), dtype=torch.uint8).reshape(64, 64)
            batch_tensors.append(tensor)
            batch_labels.append(str(row.get("digit_label")))
            if len(batch_tensors) == HISTORICAL_BATCH_SIZE:
                flush()
        flush()
    finally:
        if current_image is not None:
            current_image.close()

    if processed != EXPECTED_HISTORICAL_COUNT or len(labels) != EXPECTED_HISTORICAL_COUNT:
        _fail("historical feature extraction incomplete")
    features = {digit: torch.cat(features_parts[digit], dim=0) for digit in ("2", "3")}
    targets = {
        digit: torch.tensor([1.0 if label == digit else 0.0 for label in labels], dtype=torch.float32)
        for digit in ("2", "3")
    }
    return features, targets


def run_frozen_feature_transfer_audit_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the read-only V5-2N frozen-representation diagnostic."""
    root = Path(v5_data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    report_path = ann_dir / REPORT_NAME
    if report_path.exists():
        _fail(f"refusing to overwrite existing V5-2N evidence: {report_path}")

    models = _frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    manifest_path, _rows, v5_features, v5_targets, frozen_metrics = _v5_surface(root, models)
    source_features, source_targets = _historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )

    per_specialist: dict[str, object] = {}
    for digit in ("2", "3"):
        head_weight = models[digit].head.weight.detach().cpu().reshape(-1)
        metrics = feature_transfer_metrics_v1(
            source_features=source_features[digit],
            source_targets=source_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            head_weight=head_weight,
        )
        metrics["frozen_head_v5_adaptation_train_metrics"] = frozen_metrics[digit]
        metrics["frozen_threshold"] = v52b.FROZEN_THRESHOLDS[digit]
        per_specialist[digit] = metrics

    report: dict[str, object] = {
        "schema": SCHEMA,
        "question": "do_frozen_64d_digit_features_transfer_to_existing_v5_adaptation_train",
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "m4a_manifest_sha256": ret_legacy.M4A_MANIFEST_SHA256,
        "d10_manifest_sha256": ret_legacy.D10_MANIFEST_SHA256,
        "frozen_checkpoint_sha256": {
            "2": v52b.DIGIT2_SHA256,
            "3": v52b.DIGIT3_SHA256,
        },
        "v5_adaptation_train_slot_count": EXPECTED_V5_COUNT,
        "m4a_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "distance_metric": "squared_euclidean_to_historical_binary_centroids",
        "fit_or_training_performed": False,
        "numeric_pass_threshold_preregistered": False,
        "per_specialist": per_specialist,
        **safety_boundary(),
    }
    v51._atomic_write_json(report_path, report)
    return report
