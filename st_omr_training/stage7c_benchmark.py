"""Guarded runtime benchmark for the frozen Stage 7-C CPU training profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter
from typing import Callable, Final

import torch

from .dataset_manifest import DatasetSplit
from .stage7c_dataset import build_and_persist_stage7c_baseline_dataset
from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7c_profile import (
    STAGE7C_FROZEN_MODEL_CONFIG,
    STAGE7C_FROZEN_PREPROCESS_CONFIG,
    STAGE7C_FROZEN_RUN_CONFIG,
    STAGE7C_FROZEN_RUN_FINGERPRINT,
    STAGE7C_FROZEN_TRAINER_CONFIG,
)
from .training_data import (
    TrainingSampleRef,
    load_image_tensor,
    load_training_samples,
    make_training_batch,
)
from .training_model import (
    BaselineSTOMRModel,
    build_baseline_model,
    train_one_smoke_step,
    validation_loss,
)
from .training_tokens import BOS_TOKEN_ID


STAGE7C_BENCHMARK_VERSION: Final[str] = "stage7c-runtime-benchmark-v1"
STAGE7C_SAFE_RUNTIME_BUDGET_SECONDS: Final[float] = 4.0 * 60.0 * 60.0
STAGE7C_BENCHMARK_SAFETY_FACTOR: Final[float] = 2.0
STAGE7C_FIXED_OVERHEAD_SECONDS: Final[float] = 5.0 * 60.0
_MEASUREMENT_REPEATS: Final[int] = 3


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_progress(payload: dict[str, object]) -> None:
    print(_canonical_json(payload), file=sys.stderr, flush=True)


def _require_positive_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return normalized


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def estimate_runtime_seconds(
    *,
    dataset_seconds: float,
    train_step_seconds: float,
    validation_batch_seconds: float,
    decode_sample_seconds: float,
    training_steps: int,
    validation_batches: int,
    validation_passes: int,
    validation_samples: int,
    fixed_overhead_seconds: float,
    safety_factor: float,
) -> dict[str, float]:
    """Scale measured frozen work units and apply a conservative multiplicative margin."""

    dataset = _require_positive_number("dataset_seconds", dataset_seconds)
    train_step = _require_positive_number("train_step_seconds", train_step_seconds)
    validation_batch = _require_positive_number(
        "validation_batch_seconds", validation_batch_seconds
    )
    decode_sample = _require_positive_number("decode_sample_seconds", decode_sample_seconds)
    train_count = _require_positive_int("training_steps", training_steps)
    validation_batch_count = _require_positive_int(
        "validation_batches", validation_batches
    )
    validation_pass_count = _require_positive_int(
        "validation_passes", validation_passes
    )
    validation_sample_count = _require_positive_int(
        "validation_samples", validation_samples
    )
    overhead = _require_positive_number("fixed_overhead_seconds", fixed_overhead_seconds)
    margin = _require_positive_number("safety_factor", safety_factor)
    if margin < 1.0:
        raise ValueError("safety_factor must be at least 1.0")

    projected = (
        dataset
        + train_step * train_count
        + validation_batch * validation_batch_count * validation_pass_count
        + decode_sample * validation_sample_count
        + overhead
    )
    return {
        "projected_seconds": float(projected),
        "safety_adjusted_seconds": float(projected * margin),
    }


def _batch_groups(
    samples: tuple[TrainingSampleRef, ...],
    batch_size: int,
) -> tuple[tuple[TrainingSampleRef, ...], ...]:
    return tuple(
        samples[index : index + batch_size]
        for index in range(0, len(samples), batch_size)
    )


def _heaviest_group(
    samples: tuple[TrainingSampleRef, ...],
    batch_size: int,
) -> tuple[TrainingSampleRef, ...]:
    groups = _batch_groups(samples, batch_size)
    if not groups:
        raise RuntimeError("benchmark received an empty sample set")
    return max(
        groups,
        key=lambda group: (
            len(group) * max(len(sample.target_token_ids) for sample in group),
            tuple(sample.sample_id for sample in group),
        ),
    )


def _median_seconds(operation: Callable[[], object], repeats: int) -> float:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    observed: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        elapsed = perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("benchmark produced an invalid elapsed time")
        observed.append(elapsed)
    return float(median(observed))


def _full_length_decode_seconds(
    model: BaselineSTOMRModel,
    sample: TrainingSampleRef,
) -> float:
    image = load_image_tensor(sample, STAGE7C_FROZEN_PREPROCESS_CONFIG).unsqueeze(0).cpu()
    current = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long)
    model.eval()
    started = perf_counter()
    with torch.no_grad():
        conditioning, hidden = model.begin_incremental_decode(image)
        for _ in range(STAGE7C_FROZEN_RUN_CONFIG.max_decode_tokens - 1):
            logits, hidden = model.decode_incremental_step(current, conditioning, hidden)
            current = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    elapsed = perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("full-length decode benchmark produced an invalid elapsed time")
    return float(elapsed)


def run_stage7c_benchmark(workspace: str | Path) -> dict[str, object]:
    """Measure exact frozen work units without opening the sealed test split."""

    repository_root = Path(__file__).resolve().parents[1]
    target = Path(workspace).expanduser().resolve()
    if target == repository_root or repository_root in target.parents:
        raise ValueError("benchmark workspace must be outside the Git repository")
    if target.exists():
        raise FileExistsError("benchmark workspace must be fresh")

    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    runtime = verify_stage7c_runtime()
    _write_progress({"event": "benchmark_started", "repository_sha": repository_sha})

    target.mkdir(parents=True, exist_ok=False)
    dataset_root = target / "dataset"
    dataset_started = perf_counter()
    build = build_and_persist_stage7c_baseline_dataset(
        dataset_root,
        progress=_write_progress,
    )
    train_samples = load_training_samples(
        build,
        dataset_root,
        DatasetSplit.TRAIN,
        max_samples=STAGE7C_FROZEN_RUN_CONFIG.max_train_samples,
    )
    validation_samples = load_training_samples(
        build,
        dataset_root,
        DatasetSplit.VALIDATION,
        max_samples=STAGE7C_FROZEN_RUN_CONFIG.max_validation_samples,
    )
    dataset_seconds = perf_counter() - dataset_started
    _write_progress(
        {
            "event": "benchmark_dataset_ready",
            "dataset_seconds": dataset_seconds,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
        }
    )

    batch_size = STAGE7C_FROZEN_RUN_CONFIG.batch_size
    train_group = _heaviest_group(train_samples, batch_size)
    validation_group = _heaviest_group(validation_samples, batch_size)
    train_batch_prep_seconds = _median_seconds(
        lambda: make_training_batch(train_group, STAGE7C_FROZEN_PREPROCESS_CONFIG),
        _MEASUREMENT_REPEATS,
    )
    validation_batch_prep_seconds = _median_seconds(
        lambda: make_training_batch(
            validation_group,
            STAGE7C_FROZEN_PREPROCESS_CONFIG,
        ),
        _MEASUREMENT_REPEATS,
    )
    train_batch = make_training_batch(train_group, STAGE7C_FROZEN_PREPROCESS_CONFIG)
    validation_batch = make_training_batch(
        validation_group, STAGE7C_FROZEN_PREPROCESS_CONFIG
    )
    model = build_baseline_model(
        STAGE7C_FROZEN_MODEL_CONFIG,
        seed=STAGE7C_FROZEN_TRAINER_CONFIG.master_seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE7C_FROZEN_TRAINER_CONFIG.learning_rate_micros / 1_000_000,
        weight_decay=STAGE7C_FROZEN_TRAINER_CONFIG.weight_decay_micros / 1_000_000,
        foreach=False,
        fused=False,
    )

    train_one_smoke_step(model, train_batch, optimizer, STAGE7C_FROZEN_TRAINER_CONFIG)
    train_model_step_seconds = _median_seconds(
        lambda: train_one_smoke_step(
            model, train_batch, optimizer, STAGE7C_FROZEN_TRAINER_CONFIG
        ),
        _MEASUREMENT_REPEATS,
    )
    validation_loss(model, validation_batch)
    validation_model_batch_seconds = _median_seconds(
        lambda: validation_loss(model, validation_batch),
        _MEASUREMENT_REPEATS,
    )
    decode_sample = max(
        validation_samples,
        key=lambda sample: (len(sample.target_token_ids), sample.sample_id),
    )
    decode_sample_seconds = _full_length_decode_seconds(model, decode_sample)
    train_step_seconds = train_batch_prep_seconds + train_model_step_seconds
    validation_batch_seconds = (
        validation_batch_prep_seconds + validation_model_batch_seconds
    )

    train_groups = _batch_groups(train_samples, batch_size)
    validation_groups = _batch_groups(validation_samples, batch_size)
    training_steps = len(train_groups) * STAGE7C_FROZEN_RUN_CONFIG.epochs
    validation_passes = STAGE7C_FROZEN_RUN_CONFIG.epochs + 1
    estimate = estimate_runtime_seconds(
        dataset_seconds=dataset_seconds,
        train_step_seconds=train_step_seconds,
        validation_batch_seconds=validation_batch_seconds,
        decode_sample_seconds=decode_sample_seconds,
        training_steps=training_steps,
        validation_batches=len(validation_groups),
        validation_passes=validation_passes,
        validation_samples=len(validation_samples),
        fixed_overhead_seconds=STAGE7C_FIXED_OVERHEAD_SECONDS,
        safety_factor=STAGE7C_BENCHMARK_SAFETY_FACTOR,
    )
    within_budget = estimate["safety_adjusted_seconds"] <= STAGE7C_SAFE_RUNTIME_BUDGET_SECONDS
    result = {
        "schema_version": STAGE7C_BENCHMARK_VERSION,
        "repository_origin": repository_origin,
        "repository_sha": repository_sha,
        "runtime": runtime,
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "dataset": {
            "build_id": build.build_id,
            "manifest_sha256": build.manifest_sha256,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
        },
        "frozen_run_fingerprint": STAGE7C_FROZEN_RUN_FINGERPRINT,
        "work": {
            "epochs": STAGE7C_FROZEN_RUN_CONFIG.epochs,
            "training_steps": training_steps,
            "validation_batches": len(validation_groups),
            "validation_passes": validation_passes,
            "validation_samples": len(validation_samples),
            "decode_tokens_per_sample": STAGE7C_FROZEN_RUN_CONFIG.max_decode_tokens,
        },
        "measurements_seconds": {
            "dataset_build_persist_and_load": dataset_seconds,
            "heaviest_training_batch_preprocess_median": train_batch_prep_seconds,
            "heaviest_training_model_step_median": train_model_step_seconds,
            "heaviest_training_step_median": train_step_seconds,
            "heaviest_validation_batch_preprocess_median": (
                validation_batch_prep_seconds
            ),
            "heaviest_validation_model_batch_median": validation_model_batch_seconds,
            "heaviest_validation_batch_median": validation_batch_seconds,
            "full_length_incremental_decode_sample": decode_sample_seconds,
            "fixed_overhead": STAGE7C_FIXED_OVERHEAD_SECONDS,
        },
        "estimate": {
            **estimate,
            "safety_factor": STAGE7C_BENCHMARK_SAFETY_FACTOR,
            "safe_budget_seconds": STAGE7C_SAFE_RUNTIME_BUDGET_SECONDS,
            "within_budget": within_budget,
        },
        "sealed_test_split_opened": False,
    }
    _write_progress(
        {
            "event": "benchmark_completed",
            "safety_adjusted_seconds": estimate["safety_adjusted_seconds"],
            "within_budget": within_budget,
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the exact frozen Stage 7-C workload before real training."
    )
    parser.add_argument("--workspace", required=True, help="fresh directory outside the repository")
    parser.add_argument("--output", required=True, help="benchmark JSON output path")
    arguments = parser.parse_args(argv)
    output = Path(arguments.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("benchmark output path must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)

    result = run_stage7c_benchmark(arguments.workspace)
    serialized = _canonical_json(result)
    output.write_text(serialized, encoding="ascii", newline="\n")
    print(serialized)
    if not result["estimate"]["within_budget"]:
        raise SystemExit("Stage 7-C runtime estimate exceeds the safe execution budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
