"""Meter V5-2I exact repair pilot for 2-AI and 3-AI.

This is the single user-authorized gradient pilot preregistered by V5-2H.
It uses only the existing 540 V5 adaptation_train slots plus a deterministic
12:1 replay of 6,480 historical M4A TRAIN examples through the already-frozen
historical pixel path. No new spatial derivation is introduced.

Gate order is fail-closed:
1) train exactly one fixed configuration;
2) historical M4A VALIDATION retention gate;
3) only if retention PASS, run the existing first-30 V5 diagnostic surface.

No second configuration, threshold tuning, V5 reserve TRAIN, V5 VAL,
FINAL_HOLDOUT, 4-AI mutation, Resolver wiring, or production promotion.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Callable, Final, Mapping

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as ret_legacy
from . import meter_v5_2c_historical_retention_v2 as ret_v2
from . import meter_v5_2d_positive_collapse_forensics_v1 as forensic
from . import meter_v5_2h_repair_recipe_v1 as recipe_v1


SCHEMA: Final[str] = "st-omr-meter-v5-2i-repair-pilot-v1"
APPROVAL_TOKEN: Final[str] = "V5_2I_REPAIR_PILOT_12_TO_1_APPROVED"
REPLAY_MANIFEST_NAME: Final[str] = "v5_2i_historical_replay_manifest.json"
TRAINING_REPORT_NAME: Final[str] = "v5_2i_repair_training_report.json"
RETENTION_REPORT_NAME: Final[str] = "v5_2i_historical_retention.json"
DIAGNOSTIC_REPORT_NAME: Final[str] = "v5_2i_diagnostic_gate.json"
FINAL_REPORT_NAME: Final[str] = "v5_2i_repair_pilot_report.json"
CANDIDATE_DIR_NAME: Final[str] = "v5_2i_repair_candidates"

EXPECTED_V5_SLOT_COUNT: Final[int] = recipe_v1.V5_ADAPTATION_SLOTS
EXPECTED_HISTORICAL_REPLAY_COUNT: Final[int] = recipe_v1.HISTORICAL_REPLAY_COUNT
EXPECTED_COMBINED_COUNT: Final[int] = recipe_v1.COMBINED_EXAMPLE_COUNT
EXPECTED_HISTORICAL_LABEL_COUNTS: Final[dict[str, int]] = dict(
    recipe_v1.HISTORICAL_LABEL_COUNTS
)
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2IError(RuntimeError):
    """Raised whenever the authorized repair pilot deviates from preregistration."""


def _fail(message: str) -> None:
    raise MeterV5_2IError(message)


def gate_order() -> tuple[str, str]:
    return ("historical_retention", "v5_first30_diagnostic")


def safety_boundary() -> dict[str, object]:
    return {
        "single_fixed_repair_pilot_authorized": True,
        "automatic_second_configuration": False,
        "replay_ratio": recipe_v1.REPLAY_RATIO,
        "positive_weight": recipe_v1.POS_WEIGHT,
        "epochs": recipe_v1.EPOCHS,
        "expected_optimizer_steps_per_specialist": recipe_v1.EXPECTED_OPTIMIZER_STEPS,
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


def _canonical_row(row: Mapping[str, object]) -> str:
    return json.dumps(
        dict(row),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _selection_rank(row: Mapping[str, object], *, label: str) -> str:
    payload = f"{recipe_v1.SEED}|{label}|{_canonical_row(row)}".encode("utf-8")
    return sha256(payload).hexdigest()


def select_historical_replay_v1(
    train_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Deterministically select the preregistered 6,480 source examples."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in train_rows:
        label = str(row.get("digit_label"))
        if label not in EXPECTED_HISTORICAL_LABEL_COUNTS:
            _fail(f"unexpected historical label: {label}")
        groups[label].append(row)

    selected: list[dict[str, object]] = []
    for label in ("2", "3", "4", "NONE"):
        required = EXPECTED_HISTORICAL_LABEL_COUNTS[label]
        ranked = sorted(
            groups[label],
            key=lambda row: (_selection_rank(row, label=label), _canonical_row(row)),
        )
        if len(ranked) < required:
            _fail(f"not enough historical {label} rows for preregistered replay")
        selected.extend(ranked[:required])

    selected.sort(
        key=lambda row: (
            str(row.get("digit_label")),
            _selection_rank(row, label=str(row.get("digit_label"))),
            _canonical_row(row),
        )
    )
    counts = Counter(str(row.get("digit_label")) for row in selected)
    if len(selected) != EXPECTED_HISTORICAL_REPLAY_COUNT:
        _fail("historical replay selection count changed")
    if dict(counts) != EXPECTED_HISTORICAL_LABEL_COUNTS:
        _fail(f"historical replay allocation changed: {dict(counts)}")
    if len({_canonical_row(row) for row in selected}) != len(selected):
        _fail("historical replay selection contains duplicate manifest rows")
    return selected


def _write_json_no_overwrite(path: Path, payload: object) -> None:
    if path.exists():
        _fail(f"refusing to overwrite existing V5-2I evidence: {path}")
    v51._atomic_write_json(path, payload)


def _write_replay_manifest(
    ann_dir: Path,
    selected: list[dict[str, object]],
) -> Path:
    rows = []
    for row in selected:
        label = str(row.get("digit_label"))
        rows.append(
            {
                "digit_label": label,
                "source_record_id": str(row.get("source_record_id")),
                "row_sha256": sha256(_canonical_row(row).encode("utf-8")).hexdigest(),
                "selection_rank": _selection_rank(row, label=label),
            }
        )
    payload = {
        "schema": SCHEMA,
        "selection_policy": "sha256-ranked-stratified-no-replacement",
        "seed": recipe_v1.SEED,
        "source_manifest_sha256": ret_v2.M4A_MANIFEST_SHA256,
        "selected_count": len(selected),
        "selected_label_counts": dict(EXPECTED_HISTORICAL_LABEL_COUNTS),
        "source_examples_per_v5_example": recipe_v1.REPLAY_RATIO,
        "rows": rows,
    }
    path = ann_dir / REPLAY_MANIFEST_NAME
    _write_json_no_overwrite(path, payload)
    return path


def _prepare_historical_replay_tensors(
    *,
    selected: list[dict[str, object]],
    d10_meter: Mapping[str, Mapping[str, object]],
    d10_root: Path,
    progress: ProgressCallback | None,
):
    torch, _nn = v52b._import_torch()
    tensors: list[object | None] = [None] * len(selected)
    indexes_by_path: dict[Path, list[int]] = defaultdict(list)
    for index, row in enumerate(selected):
        source_id = str(row.get("source_record_id"))
        d10_row = d10_meter.get(source_id)
        if not isinstance(d10_row, Mapping):
            _fail(f"selected historical row missing D10 source: {source_id}")
        relpath = d10_row.get("image_path")
        if not isinstance(relpath, str) or not relpath:
            _fail(f"D10 image path missing: {source_id}")
        indexes_by_path[d10_root / relpath].append(index)

    processed = 0
    for image_path in sorted(indexes_by_path, key=lambda p: p.as_posix()):
        if image_path.is_symlink() or not image_path.is_file():
            _fail(f"historical source image missing/non-regular: {image_path}")
        with Image.open(image_path) as source:
            source.load()
            gray = source.convert("L")
            for index in indexes_by_path[image_path]:
                row = selected[index]
                canvas = ret_legacy._historical_canvas_from_bbox(gray, row.get("bbox"))
                values = torch.tensor(list(canvas.getdata()), dtype=torch.float32)
                tensors[index] = values.reshape(1, 64, 64) / 255.0
                processed += 1
                if progress is not None and (
                    processed == 1
                    or processed % 250 == 0
                    or processed == len(selected)
                ):
                    progress(processed, len(selected), "staging-v5-2i-historical-replay")

    if any(value is None for value in tensors):
        _fail("historical replay tensor staging incomplete")
    return torch.stack([value for value in tensors if value is not None], dim=0)


def _candidate_path(target_dir: Path, digit: str) -> Path:
    return target_dir / f"digit{digit}_v5_2i_repair_candidate.pt"


def _load_repair_candidate(
    path: Path,
    *,
    digit: str,
    slot_manifest_sha256: str,
    replay_manifest_sha256: str,
):
    torch, _nn = v52b._import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_2IError(f"cannot load V5-2I candidate {digit}-AI") from exc
    if not isinstance(payload, Mapping):
        _fail("V5-2I checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    state = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        _fail("V5-2I checkpoint missing state/metadata")
    expected_source = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
    expected = {
        "schema": SCHEMA,
        "role": f"digit-{digit}-v5-2i-repair-candidate",
        "source_checkpoint_sha256": expected_source,
        "slot_manifest_sha256": slot_manifest_sha256,
        "replay_manifest_sha256": replay_manifest_sha256,
        "replay_ratio": recipe_v1.REPLAY_RATIO,
        "positive_weight": recipe_v1.POS_WEIGHT,
        "optimizer": recipe_v1.OPTIMIZER,
        "learning_rate": recipe_v1.LEARNING_RATE,
        "weight_decay": recipe_v1.WEIGHT_DECAY,
        "batch_size": recipe_v1.BATCH_SIZE,
        "epochs": recipe_v1.EPOCHS,
        "seed": recipe_v1.SEED,
        "optimizer_steps": recipe_v1.EXPECTED_OPTIMIZER_STEPS,
        "threshold": v52b.FROZEN_THRESHOLDS[digit],
        "threshold_tuned": False,
        "diagnostic_seed_gradient_updates": 0,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            _fail(f"V5-2I candidate {digit}-AI metadata changed: {key}")
    model = v52b._build_digit_model().cpu()
    model.load_state_dict(dict(state), strict=True)
    if metadata.get("state_fingerprint") != v52b._state_fingerprint(model):
        _fail(f"V5-2I candidate {digit}-AI state fingerprint mismatch")
    model.eval()
    return model


def _metrics_for_subset(model, images, labels, *, threshold: float) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(images)).cpu()
    if not bool(torch.isfinite(probabilities).all().item()):
        _fail("non-finite candidate probabilities")
    return v52b._binary_counts(probabilities, labels.cpu(), threshold)


def train_exact_repair_pilot_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run exactly the single preregistered V5-2H repair configuration."""
    if confirmation != APPROVAL_TOKEN:
        _fail("exact V5-2I repair pilot approval token missing")
    if recipe_v1.REPLAY_RATIO != 12 or recipe_v1.POS_WEIGHT != 1.0:
        _fail("V5-2H repair recipe changed")
    if recipe_v1.EPOCHS != 1 or recipe_v1.BATCH_SIZE != 64:
        _fail("V5-2H epoch/batch recipe changed")
    if recipe_v1.OPTIMIZER != "AdamW":
        _fail("V5-2H optimizer changed")

    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    manifest_path, rows, _slot_audit = v52b.verify_slot_manifest_v1(root)
    train_rows = [row for row in rows if row.get("data_role") == "adaptation_train"]
    if len(train_rows) != EXPECTED_V5_SLOT_COUNT:
        _fail("V5 adaptation TRAIN slot count changed")

    frozen_paths = {"2": Path(digit2_frozen), "3": Path(digit3_frozen)}
    expected_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    for digit in ("2", "3"):
        if v52b._sha_file(frozen_paths[digit]) != expected_sha[digit]:
            _fail(f"frozen {digit}-AI SHA changed")

    target_dir = ann_dir / CANDIDATE_DIR_NAME
    if target_dir.exists():
        _fail(f"refusing to reuse existing V5-2I candidate directory: {target_dir}")

    historical_train, d10_meter = forensic._historical_train_records(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
    )
    selected = select_historical_replay_v1(historical_train)
    replay_manifest_path = _write_replay_manifest(ann_dir, selected)
    replay_manifest_sha = v52b._sha_file(replay_manifest_path)
    slot_manifest_sha = v52b._sha_file(manifest_path)

    torch, nn = v52b._import_torch()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    v5_images = torch.stack(
        [v52b._tensor_from_crop(ann_dir / row["crop_relpath"]) for row in train_rows],
        dim=0,
    )
    hist_images = _prepare_historical_replay_tensors(
        selected=selected,
        d10_meter=d10_meter,
        d10_root=Path(d10_root),
        progress=progress,
    )
    if len(v5_images) != EXPECTED_V5_SLOT_COUNT or len(hist_images) != EXPECTED_HISTORICAL_REPLAY_COUNT:
        _fail("combined repair tensor counts changed")

    target_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "approval_token_verified": True,
        "slot_manifest_sha256": slot_manifest_sha,
        "replay_manifest_sha256": replay_manifest_sha,
        "source_checkpoint_sha256": dict(expected_sha),
        "recipe": recipe_v1.recipe(),
        "v5_train_slot_count": len(train_rows),
        "historical_replay_count": len(selected),
        "combined_example_count": len(train_rows) + len(selected),
        "diagnostic_seed_gradient_updates": 0,
        "optimizer_steps_per_specialist": 0,
        "threshold_tuning": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "candidates": {},
    }

    for digit in ("2", "3"):
        model = ret_legacy._frozen_model(frozen_paths[digit], digit=digit)
        model.train()
        v5_labels = torch.tensor(
            [int(row[f"label_digit{digit}"]) for row in train_rows],
            dtype=torch.float32,
        )
        hist_labels = torch.tensor(
            [1.0 if str(row.get("digit_label")) == digit else 0.0 for row in selected],
            dtype=torch.float32,
        )
        images = torch.cat((v5_images, hist_images), dim=0)
        labels = torch.cat((v5_labels, hist_labels), dim=0)
        if len(images) != EXPECTED_COMBINED_COUNT:
            _fail("combined repair dataset count changed")

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([recipe_v1.POS_WEIGHT], dtype=torch.float32)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=recipe_v1.LEARNING_RATE,
            weight_decay=recipe_v1.WEIGHT_DECAY,
        )
        seed = recipe_v1.SEED + int(digit)
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(images), generator=generator)
        if sorted(order.tolist()) != list(range(len(images))):
            _fail("V5-2I combined epoch order is not a full permutation")

        total_loss = 0.0
        total_seen = 0
        optimizer_steps = 0
        for start in range(0, len(images), recipe_v1.BATCH_SIZE):
            batch_index = order[start : start + recipe_v1.BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[batch_index])
            loss = criterion(logits, labels[batch_index])
            if not bool(torch.isfinite(loss).item()):
                _fail(f"{digit}-AI V5-2I produced non-finite loss")
            loss.backward()
            optimizer.step()
            count = int(batch_index.numel())
            total_loss += float(loss.detach().item()) * count
            total_seen += count
            optimizer_steps += 1
            if progress is not None and (
                optimizer_steps == 1
                or optimizer_steps % 10 == 0
                or optimizer_steps == recipe_v1.EXPECTED_OPTIMIZER_STEPS
            ):
                progress(
                    optimizer_steps,
                    recipe_v1.EXPECTED_OPTIMIZER_STEPS,
                    f"training-v5-2i-{digit}-AI",
                )

        if optimizer_steps != recipe_v1.EXPECTED_OPTIMIZER_STEPS:
            _fail(f"{digit}-AI optimizer step count changed: {optimizer_steps}")
        report["optimizer_steps_per_specialist"] = optimizer_steps

        model.eval()
        v5_metrics = _metrics_for_subset(
            model,
            v5_images,
            v5_labels,
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        replay_metrics = _metrics_for_subset(
            model,
            hist_images,
            hist_labels,
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        state_fp = v52b._state_fingerprint(model)
        candidate_path = _candidate_path(target_dir, digit)
        payload = {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "metadata": {
                "schema": SCHEMA,
                "role": f"digit-{digit}-v5-2i-repair-candidate",
                "source_checkpoint_sha256": expected_sha[digit],
                "slot_manifest_sha256": slot_manifest_sha,
                "replay_manifest_sha256": replay_manifest_sha,
                "replay_ratio": recipe_v1.REPLAY_RATIO,
                "positive_weight": recipe_v1.POS_WEIGHT,
                "optimizer": recipe_v1.OPTIMIZER,
                "learning_rate": recipe_v1.LEARNING_RATE,
                "weight_decay": recipe_v1.WEIGHT_DECAY,
                "batch_size": recipe_v1.BATCH_SIZE,
                "epochs": recipe_v1.EPOCHS,
                "seed": recipe_v1.SEED,
                "optimizer_steps": optimizer_steps,
                "state_fingerprint": state_fp,
                "threshold": v52b.FROZEN_THRESHOLDS[digit],
                "threshold_tuned": False,
                "diagnostic_seed_gradient_updates": 0,
                "v5_validation_opened": False,
                "final_holdout_locked": True,
            },
        }
        torch.save(payload, candidate_path)
        report["candidates"][digit] = {
            "candidate_path": str(candidate_path),
            "candidate_sha256": v52b._sha_file(candidate_path),
            "state_fingerprint": state_fp,
            "mean_training_loss": total_loss / total_seen,
            "v5_adaptation_train_metrics": v5_metrics,
            "historical_replay_train_metrics": replay_metrics,
            "optimizer_steps": optimizer_steps,
        }

    _write_json_no_overwrite(ann_dir / TRAINING_REPORT_NAME, report)
    return report


def run_historical_retention_gate_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    training_report = v52b._read_json(ann_dir / TRAINING_REPORT_NAME)
    if training_report.get("schema") != SCHEMA:
        _fail("V5-2I training report missing/wrong schema")
    slot_manifest_sha = str(training_report.get("slot_manifest_sha256"))
    replay_manifest_sha = str(training_report.get("replay_manifest_sha256"))

    frozen_paths = {"2": Path(digit2_frozen), "3": Path(digit3_frozen), "4": Path(digit4_frozen)}
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    expected_frozen = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256, "4": v52b.DIGIT4_SHA256}
    for digit in ("2", "3", "4"):
        if v52b._sha_file(frozen_paths[digit]) != expected_frozen[digit]:
            _fail(f"historical retention frozen {digit}-AI SHA changed")
    for digit in ("2", "3"):
        expected_candidate_sha = training_report.get("candidates", {}).get(digit, {}).get("candidate_sha256")
        if expected_candidate_sha != v52b._sha_file(candidate_paths[digit]):
            _fail(f"historical retention {digit}-AI candidate differs from training report")

    validation, d10_meter = ret_legacy._load_manifests(
        m4a_root=Path(m4a_root), d10_root=Path(d10_root)
    )
    images, labels = ret_legacy._prepare_inputs(
        validation=validation,
        d10_meter=d10_meter,
        d10_root=Path(d10_root),
        progress=progress,
    )

    frozen_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3", "4"):
        probabilities = ret_legacy._probabilities(
            ret_legacy._frozen_model(frozen_paths[digit], digit=digit),
            images,
            progress=progress,
            phase=f"v5-2i-frozen-{digit}-AI-retention-self-check",
        )
        metrics = ret_legacy._binary_counts(
            probabilities,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )
        frozen_metrics[digit] = metrics
        expected = ret_v2.EXPECTED_FROZEN_COUNTS[digit]
        if any(metrics[key] != expected[key] for key in ("tp", "fp", "fn", "tn")):
            _fail(f"historical validation pixel-path reproduction failed for {digit}-AI: {metrics}")

    candidate_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        model = _load_repair_candidate(
            candidate_paths[digit],
            digit=digit,
            slot_manifest_sha256=slot_manifest_sha,
            replay_manifest_sha256=replay_manifest_sha,
        )
        probabilities = ret_legacy._probabilities(
            model,
            images,
            progress=progress,
            phase=f"v5-2i-candidate-{digit}-AI-retention",
        )
        candidate_metrics[digit] = ret_legacy._binary_counts(
            probabilities,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )

    gate = ret_legacy.evaluate_retention_gate_v1(
        frozen_metrics=frozen_metrics, candidate_metrics=candidate_metrics
    )
    report = {
        "schema": SCHEMA,
        "gate_kind": "historical-retention-first",
        "historical_pixel_path_reproduced": True,
        "validation_record_count": 3372,
        "frozen_metrics": frozen_metrics,
        "candidate_metrics": candidate_metrics,
        "gate": gate["gate"],
        "reasons": gate["reasons"],
        "per_digit_retention": gate["per_digit"],
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuning": False,
        "v5_diagnostic_authorized": gate["gate"] == "PASS",
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
    }
    _write_json_no_overwrite(ann_dir / RETENTION_REPORT_NAME, report)
    return report


def run_v5_diagnostic_gate_v1(
    data_root: str | Path,
    *,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    digit4_frozen: str | Path,
) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    retention = v52b._read_json(ann_dir / RETENTION_REPORT_NAME)
    if retention.get("schema") != SCHEMA or retention.get("gate") != "PASS":
        _fail("V5 diagnostic cannot run before historical retention PASS")
    training_report = v52b._read_json(ann_dir / TRAINING_REPORT_NAME)
    slot_manifest_sha = str(training_report.get("slot_manifest_sha256"))
    replay_manifest_sha = str(training_report.get("replay_manifest_sha256"))
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    for digit in ("2", "3"):
        expected_sha = training_report.get("candidates", {}).get(digit, {}).get("candidate_sha256")
        if expected_sha != v52b._sha_file(candidate_paths[digit]):
            _fail(f"V5 diagnostic {digit}-AI candidate differs from training report")

    model2 = _load_repair_candidate(
        candidate_paths["2"], digit="2", slot_manifest_sha256=slot_manifest_sha, replay_manifest_sha256=replay_manifest_sha
    )
    model3 = _load_repair_candidate(
        candidate_paths["3"], digit="3", slot_manifest_sha256=slot_manifest_sha, replay_manifest_sha256=replay_manifest_sha
    )
    if v52b._sha_file(Path(digit4_frozen)) != v52b.DIGIT4_SHA256:
        _fail("4-AI frozen checkpoint SHA changed")
    model4 = ret_legacy._frozen_model(Path(digit4_frozen), digit="4")

    _manifest_path, rows, _slot_audit = v52b.verify_slot_manifest_v1(root)
    diagnostic = [row for row in rows if row.get("data_role") == "diagnostic_seed"]
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in diagnostic:
        grouped.setdefault(row["sample_id"], {})[row["slot_role"]] = row
    if len(grouped) != 30 or any(set(slots) != {"numerator", "denominator"} for slots in grouped.values()):
        _fail("first-30 diagnostic manifest identity changed")

    per_meter_pass: Counter[str] = Counter()
    denominator_exact4 = 0
    samples: list[dict[str, object]] = []
    for sample_id in sorted(grouped, key=lambda sid: int(grouped[sid]["numerator"]["sample_index"])):
        slots = grouped[sample_id]
        meter = slots["numerator"]["meter"]
        expected_num = meter.split("/")[0]
        slot_results: dict[str, dict[str, object]] = {}
        for role in ("numerator", "denominator"):
            crop_path = ann_dir / slots[role]["crop_relpath"]
            probabilities = {
                "2": v52b._probability(model2, crop_path),
                "3": v52b._probability(model3, crop_path),
                "4": v52b._probability(model4, crop_path),
            }
            hits = [digit for digit in ("2", "3", "4") if probabilities[digit] >= v52b.FROZEN_THRESHOLDS[digit]]
            slot_results[role] = {"probabilities": probabilities, "hits": hits}
        numerator_ok = slot_results["numerator"]["hits"] == [expected_num]
        denominator_ok = slot_results["denominator"]["hits"] == ["4"]
        meter_pass = bool(numerator_ok and denominator_ok)
        if meter_pass:
            per_meter_pass[meter] += 1
        if denominator_ok:
            denominator_exact4 += 1
        samples.append({"sample_id": sample_id, "meter": meter, "numerator": slot_results["numerator"], "denominator": slot_results["denominator"], "meter_pass": meter_pass})

    required = {"2/4": 8, "3/4": 8, "4/4": 9}
    reasons = [f"{meter}_PASS_BELOW_{minimum}_OF_10" for meter, minimum in required.items() if per_meter_pass[meter] < minimum]
    if denominator_exact4 < 26:
        reasons.append("DENOMINATOR_EXACT4_BELOW_26_OF_30")
    report = {
        "schema": SCHEMA,
        "gate_kind": "v5-first30-after-retention-pass",
        "diagnostic_seed_count": 30,
        "diagnostic_seed_gradient_updates": 0,
        "per_meter_pass": {meter: per_meter_pass[meter] for meter in ("2/4", "3/4", "4/4")},
        "denominator_exact4": denominator_exact4,
        "gate": "PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuning": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
        "samples": samples,
    }
    _write_json_no_overwrite(ann_dir / DIAGNOSTIC_REPORT_NAME, report)
    return report


def run_exact_repair_pilot_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    for name in (TRAINING_REPORT_NAME, RETENTION_REPORT_NAME, DIAGNOSTIC_REPORT_NAME, FINAL_REPORT_NAME, REPLAY_MANIFEST_NAME):
        if (ann_dir / name).exists():
            _fail(f"V5-2I evidence already exists; refusing rerun: {ann_dir / name}")
    if (ann_dir / CANDIDATE_DIR_NAME).exists():
        _fail("V5-2I candidate directory already exists; refusing rerun")

    training = train_exact_repair_pilot_v1(
        root,
        m4a_root=m4a_root,
        d10_root=d10_root,
        digit2_frozen=digit2_frozen,
        digit3_frozen=digit3_frozen,
        confirmation=confirmation,
        progress=progress,
    )
    candidates = training["candidates"]
    digit2_candidate = str(candidates["2"]["candidate_path"])
    digit3_candidate = str(candidates["3"]["candidate_path"])

    retention = run_historical_retention_gate_v1(
        root,
        m4a_root=m4a_root,
        d10_root=d10_root,
        digit2_frozen=digit2_frozen,
        digit3_frozen=digit3_frozen,
        digit4_frozen=digit4_frozen,
        digit2_candidate=digit2_candidate,
        digit3_candidate=digit3_candidate,
        progress=progress,
    )

    diagnostic: dict[str, object] | None = None
    if retention["gate"] == "PASS":
        diagnostic = run_v5_diagnostic_gate_v1(
            root,
            digit2_candidate=digit2_candidate,
            digit3_candidate=digit3_candidate,
            digit4_frozen=digit4_frozen,
        )

    overall_pass = retention["gate"] == "PASS" and diagnostic is not None and diagnostic["gate"] == "PASS"
    final = {
        "schema": SCHEMA,
        "recipe": recipe_v1.recipe(),
        "training_completed": True,
        "optimizer_steps_per_specialist": training["optimizer_steps_per_specialist"],
        "candidate_sha256": {"2": candidates["2"]["candidate_sha256"], "3": candidates["3"]["candidate_sha256"]},
        "historical_retention_gate": retention["gate"],
        "historical_retention_reasons": retention["reasons"],
        "v5_diagnostic_executed": diagnostic is not None,
        "v5_diagnostic_gate": diagnostic["gate"] if diagnostic is not None else "NOT_RUN",
        "v5_diagnostic_reasons": diagnostic["reasons"] if diagnostic is not None else ["HISTORICAL_RETENTION_NOT_PASS"],
        "overall_gate": "PASS" if overall_pass else "HOLD",
        "automatic_second_configuration": False,
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
    _write_json_no_overwrite(ann_dir / FINAL_REPORT_NAME, final)
    return final


def automatic_second_configuration_allowed() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
