"""Meter V5-2K parameter-gradient balance audit.

Diagnostic-only audit of the domain-normalized objective proposed after V5-2J.
It reads exact frozen 2-AI/3-AI checkpoints and only the already-approved V5
adaptation TRAIN + historical M4A TRAIN surfaces. It computes parameter
gradients with torch.autograd.grad, performs zero optimizer steps, writes no
checkpoint, and introduces no new spatial semantics.
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
from . import meter_v5_2j_domain_normalized_balance_audit_v1 as v52j


SCHEMA: Final[str] = "st-omr-meter-v5-2k-parameter-gradient-balance-audit-v1"
REPORT_NAME: Final[str] = "v5_2k_parameter_gradient_balance_audit_v1.json"
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
EXPECTED_V5_POSITIVE_PER_SPECIALIST: Final[int] = 90
POS_WEIGHT: Final[float] = 1.0
HISTORICAL_BATCH_SIZE: Final[int] = 256
COMMON_MINIMAX_ITERATIONS: Final[int] = 120

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2KError(RuntimeError):
    """Raised when V5-2K evidence or safety invariants fail closed."""


def _fail(message: str) -> None:
    raise MeterV5_2KError(message)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"V5-2K requires numeric field: {field}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"V5-2K requires finite field: {field}")
    return result


def _verify_v52j(root: Path) -> tuple[Path, dict[str, object], dict[str, float]]:
    path = root / v51.ANNOTATIONS_DIR / v52j.REPORT_NAME
    report = v52b._read_json(path)
    expected = {
        "schema": v52j.SCHEMA,
        "objective_contract": "mean(V5_BCE_w1)+lambda_source*mean(HISTORICAL_BCE_w1)",
        "positive_weight": 1.0,
        "domain_weight_selected": False,
        "repair_training_authorized": False,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": False,
        "checkpoint_write": False,
        "image_read": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            _fail(f"V5-2J carried-forward field changed: {key}")

    zeros = report.get("per_specialist_zero_crossing_lambda_source")
    minimax = report.get("minimax_reference")
    if not isinstance(zeros, Mapping) or not isinstance(minimax, Mapping):
        _fail("V5-2J reference coefficients missing")
    z2 = _finite(zeros.get("2"), field="v52j.zero.2")
    z3 = _finite(zeros.get("3"), field="v52j.zero.3")
    mm = _finite(minimax.get("lambda_source"), field="v52j.minimax")
    if not (0.0 < min(z2, z3) <= mm <= max(z2, z3)):
        _fail("V5-2J minimax reference is outside the zero-crossing interval")
    if minimax.get("reference_only") is not True or minimax.get("training_setting_selected") is not False:
        _fail("V5-2J minimax reference boundary changed")
    return path, report, {
        "logit_zero_2": z2,
        "logit_minimax": mm,
        "logit_zero_3": z3,
    }


def _verify_v5_train(root: Path) -> tuple[Path, list[dict[str, str]]]:
    manifest_path, rows, _audit = v52b.verify_slot_manifest_v1(root)
    train = [row for row in rows if row.get("data_role") == "adaptation_train"]
    if len(train) != EXPECTED_V5_COUNT:
        _fail(f"V5 adaptation TRAIN slot count changed: {len(train)}")
    if len({row.get("sample_id") for row in train}) != 270:
        _fail("V5 adaptation TRAIN sample identity changed")
    for digit in ("2", "3"):
        positives = sum(int(row[f"label_digit{digit}"]) for row in train)
        if positives != EXPECTED_V5_POSITIVE_PER_SPECIALIST:
            _fail(f"{digit}-AI V5 positive count changed: {positives}")
    return manifest_path, train


def _named_params(model) -> list[tuple[str, object]]:
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not params:
        _fail("model has no trainable parameters for gradient audit")
    return params


def _grad_dict(names: list[str], grads: tuple[object, ...]) -> dict[str, object]:
    if len(names) != len(grads):
        _fail("gradient/name cardinality mismatch")
    return {name: grad.detach().cpu().to(dtype=grad.dtype).clone() for name, grad in zip(names, grads)}


def _v5_gradients(
    *,
    root: Path,
    train_rows: list[dict[str, str]],
    models: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    torch, _nn = v52b._import_torch()
    ann_dir = root / v51.ANNOTATIONS_DIR
    images = torch.stack(
        [v52b._tensor_from_crop(ann_dir / row["crop_relpath"]) for row in train_rows],
        dim=0,
    )
    result: dict[str, dict[str, object]] = {}
    losses: dict[str, float] = {}
    for digit in ("2", "3"):
        model = models[digit]
        model.eval()
        named = _named_params(model)
        names = [name for name, _parameter in named]
        params = [parameter for _name, parameter in named]
        labels = torch.tensor(
            [float(row[f"label_digit{digit}"]) for row in train_rows],
            dtype=torch.float32,
        )
        logits = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="mean",
        )
        if not bool(torch.isfinite(loss).item()):
            _fail(f"{digit}-AI V5 mean loss is non-finite")
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        if any(parameter.grad is not None for parameter in params):
            _fail("torch.autograd.grad unexpectedly populated parameter .grad on V5 surface")
        result[digit] = _grad_dict(names, grads)
        losses[digit] = float(loss.detach().item())
    return result, losses


def _historical_gradients(
    *,
    historical_rows: list[dict[str, object]],
    d10_meter: Mapping[str, Mapping[str, object]],
    d10_root: Path,
    models: Mapping[str, object],
    progress: ProgressCallback | None,
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    torch, _nn = v52b._import_torch()
    if len(historical_rows) != EXPECTED_HISTORICAL_COUNT:
        _fail("historical TRAIN count changed before gradient replay")

    named_by_digit = {digit: _named_params(models[digit]) for digit in ("2", "3")}
    accum: dict[str, dict[str, object]] = {}
    loss_sums = {"2": 0.0, "3": 0.0}
    for digit in ("2", "3"):
        accum[digit] = {
            name: torch.zeros_like(parameter.detach().cpu(), dtype=torch.float64)
            for name, parameter in named_by_digit[digit]
        }
        models[digit].eval()

    batch_tensors: list[object] = []
    batch_labels: list[str] = []
    current_path: Path | None = None
    current_image: Image.Image | None = None
    processed = 0

    def flush() -> None:
        nonlocal batch_tensors, batch_labels, processed
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors, dim=0).to(dtype=torch.float32).unsqueeze(1) / 255.0
        for digit in ("2", "3"):
            named = named_by_digit[digit]
            names = [name for name, _parameter in named]
            params = [parameter for _name, parameter in named]
            labels = torch.tensor(
                [1.0 if label == digit else 0.0 for label in batch_labels],
                dtype=torch.float32,
            )
            logits = models[digit](images)
            loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="sum",
            )
            if not bool(torch.isfinite(loss_sum).item()):
                _fail(f"{digit}-AI historical loss sum is non-finite")
            grads = torch.autograd.grad(loss_sum, params, retain_graph=False, create_graph=False)
            if any(parameter.grad is not None for parameter in params):
                _fail("torch.autograd.grad unexpectedly populated parameter .grad on historical surface")
            for name, grad in zip(names, grads):
                accum[digit][name] += grad.detach().cpu().to(dtype=torch.float64)
            loss_sums[digit] += float(loss_sum.detach().item())
        processed += len(batch_labels)
        if progress is not None and (
            processed == len(batch_labels)
            or processed % 512 == 0
            or processed == EXPECTED_HISTORICAL_COUNT
        ):
            progress(processed, EXPECTED_HISTORICAL_COUNT, "v5-2k-historical-parameter-gradients")
        batch_tensors = []
        batch_labels = []

    try:
        for row in historical_rows:
            source_id = str(row.get("source_record_id"))
            d10_row = d10_meter.get(source_id)
            if not isinstance(d10_row, Mapping):
                _fail(f"historical TRAIN references missing D10 record: {source_id}")
            image_relpath = d10_row.get("image_path")
            if not isinstance(image_relpath, str) or not image_relpath:
                _fail(f"D10 image path missing: {source_id}")
            image_path = d10_root / image_relpath
            if image_path != current_path:
                if current_image is not None:
                    current_image.close()
                if image_path.is_symlink() or not image_path.is_file():
                    _fail(f"D10 source image missing/non-regular: {image_path}")
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

    if processed != EXPECTED_HISTORICAL_COUNT:
        _fail(f"historical gradient replay incomplete: {processed}")
    mean_grads: dict[str, dict[str, object]] = {}
    mean_losses: dict[str, float] = {}
    for digit in ("2", "3"):
        mean_grads[digit] = {
            name: tensor / EXPECTED_HISTORICAL_COUNT
            for name, tensor in accum[digit].items()
        }
        mean_losses[digit] = loss_sums[digit] / EXPECTED_HISTORICAL_COUNT
    return mean_grads, mean_losses


def _group_names(gradient_names: set[str]) -> dict[str, list[str]]:
    features = sorted(name for name in gradient_names if name.startswith("features."))
    head = sorted(name for name in gradient_names if name.startswith("head."))
    if set(features) | set(head) != gradient_names or not features or not head:
        _fail("unexpected digit specialist parameter grouping")
    return {"all": sorted(gradient_names), "features": features, "head": head}


def _pair_metrics(
    v5_grad: Mapping[str, object],
    source_grad: Mapping[str, object],
    *,
    names: list[str],
    reference_lambdas: Mapping[str, float],
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    if not names:
        _fail("empty gradient group")
    v5_sq = 0.0
    source_sq = 0.0
    dot = 0.0
    for name in names:
        a = v5_grad[name].to(dtype=torch.float64).reshape(-1)
        b = source_grad[name].to(dtype=torch.float64).reshape(-1)
        if a.shape != b.shape:
            _fail(f"gradient shape mismatch: {name}")
        v5_sq += float(torch.dot(a, a).item())
        source_sq += float(torch.dot(b, b).item())
        dot += float(torch.dot(a, b).item())
    if v5_sq <= 0.0 or source_sq <= 0.0:
        _fail("zero parameter-gradient norm encountered")
    v5_norm = math.sqrt(v5_sq)
    source_norm = math.sqrt(source_sq)
    cosine = dot / (v5_norm * source_norm)
    cosine = min(1.0, max(-1.0, cosine))
    lambda_star = max(0.0, -dot / source_sq)

    refs: dict[str, object] = {}
    for label, value in reference_lambdas.items():
        lam = float(value)
        if not math.isfinite(lam) or lam < 0.0:
            _fail(f"invalid reference lambda: {label}")
        combined_sq = max(0.0, v5_sq + 2.0 * lam * dot + lam * lam * source_sq)
        combined_norm = math.sqrt(combined_sq)
        refs[label] = {
            "lambda_source": lam,
            "combined_gradient_l2": combined_norm,
            "combined_over_v5_gradient_l2": combined_norm / v5_norm,
            "projection_on_v5_normalized": (v5_sq + lam * dot) / v5_sq,
            "projection_on_source_normalized": (dot + lam * source_sq) / source_sq,
        }

    optimal_sq = max(0.0, v5_sq + 2.0 * lambda_star * dot + lambda_star * lambda_star * source_sq)
    return {
        "parameter_count": sum(int(v5_grad[name].numel()) for name in names),
        "v5_gradient_l2": v5_norm,
        "historical_gradient_l2": source_norm,
        "dot_product": dot,
        "cosine_similarity": cosine,
        "gradient_conflict": dot < 0.0,
        "minimum_norm_lambda_source": lambda_star,
        "minimum_combined_gradient_l2": math.sqrt(optimal_sq),
        "minimum_combined_over_v5_gradient_l2": math.sqrt(optimal_sq) / v5_norm,
        "references": refs,
    }


def _normalized_combined_squared(metrics: Mapping[str, object], lam: float) -> float:
    v = float(metrics["v5_gradient_l2"])
    s = float(metrics["historical_gradient_l2"])
    dot = float(metrics["dot_product"])
    return max(0.0, (v * v + 2.0 * lam * dot + lam * lam * s * s) / (v * v))


def _common_minimax_reference(per_digit_all: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    stars = [float(per_digit_all[digit]["minimum_norm_lambda_source"]) for digit in ("2", "3")]
    low = min(stars)
    high = max(stars)
    if high == low:
        candidate = low
    else:
        left, right = low, high
        for _ in range(COMMON_MINIMAX_ITERATIONS):
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            f1 = max(_normalized_combined_squared(per_digit_all[digit], m1) for digit in ("2", "3"))
            f2 = max(_normalized_combined_squared(per_digit_all[digit], m2) for digit in ("2", "3"))
            if f1 <= f2:
                right = m2
            else:
                left = m1
        candidate = (left + right) / 2.0
    residuals = {
        digit: math.sqrt(_normalized_combined_squared(per_digit_all[digit], candidate))
        for digit in ("2", "3")
    }
    return {
        "lambda_source": candidate,
        "normalized_combined_gradient_l2": residuals,
        "max_normalized_combined_gradient_l2": max(residuals.values()),
        "search_interval_from_per_specialist_minima": {"min": low, "max": high},
        "reference_only": True,
        "training_setting_selected": False,
    }


def run_parameter_gradient_balance_audit_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(v5_data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    output_path = ann_dir / REPORT_NAME
    if output_path.exists():
        _fail(f"refusing to overwrite existing V5-2K evidence: {output_path}")

    v52j_path, _v52j_report, references = _verify_v52j(root)
    manifest_path, v5_rows = _verify_v5_train(root)
    historical_rows, d10_meter = forensic._historical_train_records(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
    )
    counts = Counter(str(row.get("digit_label")) for row in historical_rows)
    if dict(counts) != forensic.EXPECTED_M4A_TRAIN_COUNTS:
        _fail("historical TRAIN label identity changed")

    frozen_paths = {"2": Path(digit2_frozen), "3": Path(digit3_frozen)}
    expected_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    models: dict[str, object] = {}
    for digit in ("2", "3"):
        if v52b._sha_file(frozen_paths[digit]) != expected_sha[digit]:
            _fail(f"frozen {digit}-AI checkpoint SHA changed")
        models[digit] = ret_legacy._frozen_model(frozen_paths[digit], digit=digit)

    v5_grads, v5_losses = _v5_gradients(root=root, train_rows=v5_rows, models=models)
    source_grads, source_losses = _historical_gradients(
        historical_rows=historical_rows,
        d10_meter=d10_meter,
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )

    per_specialist: dict[str, object] = {}
    all_metrics: dict[str, Mapping[str, object]] = {}
    for digit in ("2", "3"):
        if set(v5_grads[digit]) != set(source_grads[digit]):
            _fail(f"{digit}-AI V5/source parameter keys differ")
        groups = _group_names(set(v5_grads[digit]))
        group_metrics = {
            group: _pair_metrics(
                v5_grads[digit],
                source_grads[digit],
                names=names,
                reference_lambdas=references,
            )
            for group, names in groups.items()
        }
        all_metrics[digit] = group_metrics["all"]
        per_specialist[digit] = {
            "v5_mean_bce_pos_weight_1": v5_losses[digit],
            "historical_mean_bce_pos_weight_1": source_losses[digit],
            "groups": group_metrics,
        }

    common = _common_minimax_reference(all_metrics)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "objective_contract": "mean(V5_BCE_w1)+lambda_source*mean(HISTORICAL_BCE_w1)",
        "positive_weight": POS_WEIGHT,
        "v5_2j_report_sha256": v52b._sha_file(v52j_path),
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "m4a_manifest_sha256": forensic.ret_v2.M4A_MANIFEST_SHA256,
        "d10_manifest_sha256": forensic.ret_v2.D10_MANIFEST_SHA256,
        "frozen_checkpoint_sha256": dict(expected_sha),
        "v5_adaptation_train_slot_count": len(v5_rows),
        "m4a_train_record_count": len(historical_rows),
        "m4a_train_label_counts": dict(forensic.EXPECTED_M4A_TRAIN_COUNTS),
        "reference_lambdas_from_v5_2j": dict(references),
        "per_specialist": per_specialist,
        "common_parameter_gradient_minimax_reference": common,
        "interpretation": {
            "parameter_gradient_balance_measured": True,
            "logit_reference_only_before_this_audit": True,
            "training_trajectory_stability_proven": False,
            "retention_pass_proven": False,
            "v5_learning_pass_proven": False,
            "common_reference_is_training_setting": False,
        },
        "domain_weight_selected": False,
        "repair_training_authorized": False,
        "training": False,
        "autograd_grad_used": True,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": True,
        "checkpoint_write": False,
        "image_read": True,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }
    v51._atomic_write_json(output_path, report)
    return report


def training_allowed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
