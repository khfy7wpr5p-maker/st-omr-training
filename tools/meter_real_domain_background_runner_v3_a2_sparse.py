#!/usr/bin/env python3
"""Durable Colab runner for Meter V3-A2 without full D10 materialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback
from typing import Mapping

from tools.meter_real_domain_background_runner_v1 import DurableStatus, _copy_atomic, _sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--permission", type=Path, required=True)
    parser.add_argument("--privacy", type=Path, required=True)
    parser.add_argument("--teacher-bundle", type=Path, required=True)
    parser.add_argument("--d10-drive-root", type=Path, required=True)
    parser.add_argument("--d10-sparse-cache-root", type=Path, required=True)
    parser.add_argument("--d10-manifest-sha256", required=True)
    parser.add_argument("--d10-artifact-binding-sha256", required=True)
    parser.add_argument("--d11-drive-checkpoint", type=Path, required=True)
    parser.add_argument("--d11-local-checkpoint", type=Path, required=True)
    parser.add_argument("--d11-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace, status: DurableStatus) -> dict[str, object]:
    from st_omr_training.meter_real_domain_adaptation_v3_a2 import (
        FROZEN_ADAPTATION_CONFIG_V3_A2,
    )
    from st_omr_training.meter_real_domain_adaptation_v3_a2_run import (
        run_meter_real_domain_adaptation_v3_a2,
    )
    from st_omr_training.meter_teacher_gold_admission_v1 import (
        build_meter_teacher_gold_bundle_v1,
        verify_meter_teacher_gold_bundle_v1,
    )
    from st_omr_training.meter_v3_a1_sparse_d10 import (
        patched_stage7d11_for_sparse_v3_a1,
        prepare_sparse_meter_d10_v3_a1,
    )

    config = FROZEN_ADAPTATION_CONFIG_V3_A2
    status.update(
        "teacher_gold_started",
        {
            "phase": "teacher_gold",
            "phase_index": 1,
            "phase_total": 9,
            "epochs_total": config.epochs,
            "full_cache_copy": False,
            "experiment": "meter-v3-a2-positive-margin",
        },
    )
    if args.teacher_bundle.is_dir() and (args.teacher_bundle / "COMPLETE").is_file():
        teacher_receipt = verify_meter_teacher_gold_bundle_v1(args.teacher_bundle)
    else:
        if args.teacher_bundle.exists():
            raise RuntimeError(
                "Incomplete teacher bundle exists; move it aside explicitly before rebuilding"
            )
        teacher_receipt = build_meter_teacher_gold_bundle_v1(
            pilot_path=args.pilot,
            choices_path=args.choices,
            permission_evidence_path=args.permission,
            privacy_review_evidence_path=args.privacy,
            output_root=args.teacher_bundle,
            repository_root=args.repository_root,
        )
    status.update(
        "teacher_gold_complete",
        {
            "teacher_records": teacher_receipt.record_count,
            "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
            "full_cache_copy": False,
        },
    )

    prepared = prepare_sparse_meter_d10_v3_a1(
        source_root=args.d10_drive_root,
        cache_root=args.d10_sparse_cache_root,
        expected_manifest_sha256=args.d10_manifest_sha256,
        expected_artifact_binding_sha256=args.d10_artifact_binding_sha256,
        replay_per_class=config.synthetic_replay_per_class,
        replay_seed=config.master_seed,
        progress=lambda event, payload: status.update(event, payload),
    )
    if prepared.staged_image_count != 1_736:
        raise RuntimeError("V3-A2 sparse transport must stage exactly 1,736 images")
    if prepared.meter_label_count != 11_064:
        raise RuntimeError("V3-A2 sparse transport must validate exactly 11,064 Meter labels")

    status.update(
        "d11_local_copy_started",
        {
            "phase": "d11_local_copy",
            "phase_index": 3,
            "phase_total": 9,
            "full_cache_copy": False,
        },
    )
    _copy_atomic(args.d11_drive_checkpoint, args.d11_local_checkpoint)
    observed_d11 = _sha(args.d11_local_checkpoint.read_bytes())
    if observed_d11 != args.d11_sha256:
        raise RuntimeError("local D11 checkpoint SHA-256 mismatch")
    status.update(
        "d11_local_copy_complete",
        {
            "d11_checkpoint_sha256": observed_d11,
            "full_cache_copy": False,
        },
    )

    def progress(event: str, payload: Mapping[str, object]) -> None:
        translated = dict(payload)
        if event == "phase_started":
            translated["phase_index"] = min(9, int(payload.get("phase_index", 1)) + 3)
            translated["phase_total"] = 9
        translated.setdefault("epochs_total", config.epochs)
        translated["full_cache_copy"] = False
        translated["experiment"] = "meter-v3-a2-positive-margin"
        status.update(event, translated)

    with patched_stage7d11_for_sparse_v3_a1(prepared):
        return run_meter_real_domain_adaptation_v3_a2(
            teacher_bundle_root=args.teacher_bundle,
            d10_root=args.d10_drive_root,
            base_checkpoint_path=args.d11_local_checkpoint,
            output_root=args.output_root,
            repository_root=args.repository_root,
            expected_d10_manifest_sha256=args.d10_manifest_sha256,
            expected_d10_artifact_binding_sha256=args.d10_artifact_binding_sha256,
            progress=progress,
            resume=True,
        )


def main() -> int:
    args = _parser().parse_args()
    status = DurableStatus(args.status_path)
    try:
        metrics = _run(args, status)
        status.finish(
            state="COMPLETE",
            payload={
                "result": metrics["status"],
                "epoch": metrics["best"]["epoch"],
                "epochs_total": metrics["configuration"]["epochs"],
                "run_id": metrics["run_id"],
                "output_root": str(args.output_root),
                "bbox_frozen_exact": metrics["bbox_frozen_exact"],
                "full_cache_copy": False,
                "test_opened": metrics["test_opened"],
                "runtime_connected": metrics["runtime_connected"],
                "production_promotion_authorized": metrics[
                    "production_promotion_authorized"
                ],
            },
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        status.finish(
            state="FAILED",
            payload={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_root": str(args.output_root),
                "full_cache_copy": False,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
