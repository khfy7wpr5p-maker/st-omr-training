"""Stage 7-D3 validation-only error analysis for the accepted D2 checkpoint.

D3 never performs optimizer steps. It re-verifies the frozen corpus, loads only
VALIDATION artifacts after D1, verifies the exact accepted D2 checkpoint, then
emits hash-addressed diagnostics. TRAIN and TEST rows are skipped before any D3
artifact path or byte is derived.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Final

import torch

from .dataset_manifest import DatasetSplit
from .degradation import sample_degradation_config
from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d2_execution import (
    MAX_BUILD_BYTES,
    MAX_MANIFEST_BYTES,
    _load_canonical_json,
    _verify_d1_receipt,
)
from .stage7d2_profile import (
    EXPECTED_D1_ARTIFACT_BINDING_SHA256,
    STAGE7D2_FROZEN_MODEL_CONFIG,
    STAGE7D2_FROZEN_PREPROCESS_CONFIG,
    STAGE7D2_FROZEN_RUN_CONFIG,
    STAGE7D2_FROZEN_RUN_FINGERPRINT,
    STAGE7D2_VERIFICATION_SCHEMA,
)
from .synthetic_curriculum_acceptance import (
    EXPECTED_BUILD_ID,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
)
from .synthetic_curriculum_corpus_gate import verify_stage7d_corpus
from .training_data import TrainingSampleRef, preprocess_grayscale_png
from .training_model import (
    assert_model_finite,
    build_baseline_model,
    model_config_fingerprint,
    model_state_sha256,
)
from .training_run import _greedy_decode_sample
from .training_tokens import TokenizationError, tokenize_musicxml
from .validation_diagnostics import (
    SampleDiagnostic,
    analyze_validation_sample,
    build_validation_diagnostic_report,
)


STAGE7D3_EXECUTION_SCHEMA: Final[str] = "st-omr-stage7d3-execution-v1"
STAGE7D3_VERIFICATION_SCHEMA: Final[str] = "st-omr-stage7d3-verification-v1"

EXPECTED_D2_SOURCE_REPOSITORY_SHA: Final[str] = "877da93367d338707f62ddc709ca861a8f4c16cd"
EXPECTED_D2_RUN_ID: Final[str] = "14d63841254c03463ad76bbed83df95045742c23f71ad91d7b0c5dc19495a373"
EXPECTED_D2_CHECKPOINT_SHA256: Final[str] = "239cf3dbdf80235bfc7e4a68fe5fecc03e8cd6fefc8a9ff6e27a2ca879ed5291"
EXPECTED_D2_CHECKPOINT_STATE_SHA256: Final[str] = "466cefcd40887cb0578b7bbc87c6a1b5f676dc0272ab5eee1142e45e7da8e17d"
EXPECTED_D2_METRICS_SHA256: Final[str] = "e80b8aed13cc8c7aafae283f4306f1f60821fbf75faaaf568ddff7b132c318bd"
EXPECTED_D2_VERIFICATION_SHA256: Final[str] = "6743425d42da77dfacef50388e879d45aa01f01b740cfd2deb381a55436500c3"
EXPECTED_D2_BEST_EPOCH: Final[int] = 20
MAX_CHECKPOINT_BYTES: Final[int] = 64 * 1024 * 1024
MAX_VERIFICATION_BYTES: Final[int] = 128 * 1024
_PROGRESS_INTERVAL: Final[int] = 8
_HEX = frozenset("0123456789abcdef")


class Stage7D3ExecutionError(RuntimeError):
    """Raised when D3 cannot prove its validation-only diagnostic contract."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha(name: str, value: object, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise Stage7D3ExecutionError(f"{name} is not canonical lowercase SHA hex")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bounded(path: Path, *, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Stage7D3ExecutionError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        raise Stage7D3ExecutionError(f"{label} size is outside the D3 bound")
    return path.read_bytes()


def _classify_degradation(sample: dict[str, object]) -> str:
    config = sample.get("degradation_config")
    if not isinstance(config, dict):
        raise Stage7D3ExecutionError("validation degradation_config is missing")
    seed = config.get("seed")
    raster_width = config.get("raster_width")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(raster_width, int)
        or isinstance(raster_width, bool)
    ):
        raise Stage7D3ExecutionError("validation degradation seed/width is invalid")
    for profile in ("clean", "light", "medium"):
        expected = asdict(
            sample_degradation_config(seed, profile, raster_width=raster_width)
        )
        if config == expected:
            return profile
    raise Stage7D3ExecutionError("validation degradation config is outside frozen profiles")


def _skip_non_validation(sample: dict[str, object], index: int) -> bool:
    """Return True only for TRAIN/TEST, reading no other field on skipped rows."""

    split_text = sample.get("split")
    if split_text in {DatasetSplit.TRAIN.value, DatasetSplit.TEST.value}:
        return True
    if split_text != DatasetSplit.VALIDATION.value:
        raise Stage7D3ExecutionError(f"manifest sample[{index}] has invalid split")
    return False


def _load_validation_refs(
    corpus_root: Path,
) -> tuple[tuple[TrainingSampleRef, str], ...]:
    manifest, raw_manifest = _load_canonical_json(
        corpus_root / "manifest.json", MAX_MANIFEST_BYTES, "manifest.json"
    )
    if sha256(raw_manifest).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise Stage7D3ExecutionError("manifest changed after D1 acceptance")

    build, _raw_build = _load_canonical_json(
        corpus_root / "build.json", MAX_BUILD_BYTES, "build.json"
    )
    expected_build = {
        "build_id": EXPECTED_BUILD_ID,
        "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sample_count": 1536,
        "target_count": 512,
        "image_count": 1536,
    }
    for key, value in expected_build.items():
        if build.get(key) != value:
            raise Stage7D3ExecutionError(f"build.json {key} changed after D1 acceptance")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 1536:
        raise Stage7D3ExecutionError("manifest sample list changed after D1 acceptance")

    refs: list[tuple[TrainingSampleRef, str]] = []
    target_cache: dict[str, tuple[int, ...]] = {}
    families: set[str] = set()

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise Stage7D3ExecutionError(f"manifest sample[{index}] is not an object")

        # Critical D3 boundary: TRAIN and TEST are skipped before any artifact
        # identity/path is derived or any artifact byte is read.
        if _skip_non_validation(sample, index):
            continue

        sample_id = _require_sha("sample_id", sample.get("sample_id"))
        image_sha = _require_sha("png_sha256", sample.get("png_sha256"))
        target_sha = _require_sha(
            "source_musicxml_sha256", sample.get("source_musicxml_sha256")
        )
        family_id = sample.get("family_id")
        width = sample.get("width")
        height = sample.get("height")
        if not isinstance(family_id, str) or not family_id:
            raise Stage7D3ExecutionError("validation family_id is invalid")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise Stage7D3ExecutionError("validation width is invalid")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise Stage7D3ExecutionError("validation height is invalid")

        target_path = corpus_root / "targets" / f"{target_sha}.musicxml"
        image_path = corpus_root / "images" / f"{image_sha}.png"
        if target_path.is_symlink() or not target_path.is_file():
            raise Stage7D3ExecutionError("validation target artifact missing or symlinked")
        if image_path.is_symlink() or not image_path.is_file():
            raise Stage7D3ExecutionError("validation image artifact missing or symlinked")

        token_ids = target_cache.get(target_sha)
        if token_ids is None:
            target_bytes = target_path.read_bytes()
            if sha256(target_bytes).hexdigest() != target_sha:
                raise Stage7D3ExecutionError("validation MusicXML changed after D1 acceptance")
            try:
                token_ids = tokenize_musicxml(target_bytes).token_ids
            except TokenizationError as exc:
                raise Stage7D3ExecutionError("validation MusicXML failed tokenization") from exc
            if len(token_ids) > STAGE7D2_FROZEN_RUN_CONFIG.max_decode_tokens:
                raise Stage7D3ExecutionError("validation target exceeds D2 decode ceiling")
            target_cache[target_sha] = token_ids

        image_bytes = image_path.read_bytes()
        if sha256(image_bytes).hexdigest() != image_sha:
            raise Stage7D3ExecutionError("validation PNG changed after D1 acceptance")
        preprocess_grayscale_png(
            image_bytes,
            STAGE7D2_FROZEN_PREPROCESS_CONFIG,
            expected_width=width,
            expected_height=height,
        )
        degradation = _classify_degradation(sample)

        refs.append(
            (
                TrainingSampleRef(
                    sample_id=sample_id,
                    family_id=family_id,
                    split=DatasetSplit.VALIDATION,
                    image_path=image_path,
                    image_sha256=image_sha,
                    target_path=target_path,
                    target_sha256=target_sha,
                    target_token_ids=token_ids,
                    source_width=width,
                    source_height=height,
                ),
                degradation,
            )
        )
        families.add(family_id)

    ordered = tuple(sorted(refs, key=lambda item: item[0].sample_id))
    if len(ordered) != STAGE7D2_FROZEN_RUN_CONFIG.validation_samples:
        raise Stage7D3ExecutionError("D3 validation sample count is not exact")
    if len(families) != STAGE7D2_FROZEN_RUN_CONFIG.validation_families:
        raise Stage7D3ExecutionError("D3 validation family count is not exact")
    if any(ref.split is not DatasetSplit.VALIDATION for ref, _profile in ordered):
        raise Stage7D3ExecutionError("non-validation sample crossed D3 loader boundary")
    return ordered


def _load_accepted_d2_model(
    checkpoint_path: Path,
    d2_verification_path: Path,
):
    checkpoint_bytes = _read_regular_bounded(
        checkpoint_path, maximum=MAX_CHECKPOINT_BYTES, label="D2 checkpoint"
    )
    if sha256(checkpoint_bytes).hexdigest() != EXPECTED_D2_CHECKPOINT_SHA256:
        raise Stage7D3ExecutionError("D2 checkpoint file hash mismatch")

    verification_bytes = _read_regular_bounded(
        d2_verification_path,
        maximum=MAX_VERIFICATION_BYTES,
        label="D2 verification",
    )
    if sha256(verification_bytes).hexdigest() != EXPECTED_D2_VERIFICATION_SHA256:
        raise Stage7D3ExecutionError("D2 verification file hash mismatch")
    try:
        verification = json.loads(verification_bytes.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D3ExecutionError("D2 verification is not valid ASCII JSON") from exc
    if (
        not isinstance(verification, dict)
        or _canonical_json_bytes(verification) != verification_bytes
    ):
        raise Stage7D3ExecutionError("D2 verification is not canonical JSON")

    expected = {
        "schema_version": STAGE7D2_VERIFICATION_SCHEMA,
        "repository_sha": EXPECTED_D2_SOURCE_REPOSITORY_SHA,
        "dataset_build_id": EXPECTED_BUILD_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_binding_sha256": EXPECTED_D1_ARTIFACT_BINDING_SHA256,
        "run_profile_fingerprint": STAGE7D2_FROZEN_RUN_FINGERPRINT,
        "run_id": EXPECTED_D2_RUN_ID,
        "metrics_sha256": EXPECTED_D2_METRICS_SHA256,
        "checkpoint_sha256": EXPECTED_D2_CHECKPOINT_SHA256,
        "checkpoint_state_sha256": EXPECTED_D2_CHECKPOINT_STATE_SHA256,
        "best_epoch": EXPECTED_D2_BEST_EPOCH,
        "validation_samples": STAGE7D2_FROZEN_RUN_CONFIG.validation_samples,
        "test_samples_exposed_to_model_development": 0,
        "d1_reverified_before_training": True,
        "source_tree_clean_before_and_after": True,
    }
    for key, value in expected.items():
        if verification.get(key) != value:
            raise Stage7D3ExecutionError(f"D2 verification {key} mismatch")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7D3ExecutionError("D2 checkpoint cannot be safely loaded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "epoch",
        "model_fingerprint",
        "model_state_dict",
    }:
        raise Stage7D3ExecutionError("D2 checkpoint structure mismatch")
    if checkpoint.get("epoch") != EXPECTED_D2_BEST_EPOCH:
        raise Stage7D3ExecutionError("D2 checkpoint epoch mismatch")
    if checkpoint.get("model_fingerprint") != model_config_fingerprint(
        STAGE7D2_FROZEN_MODEL_CONFIG
    ):
        raise Stage7D3ExecutionError("D2 checkpoint model fingerprint mismatch")

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise Stage7D3ExecutionError("D2 checkpoint state is invalid")
    model = build_baseline_model(STAGE7D2_FROZEN_MODEL_CONFIG, seed=0)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise Stage7D3ExecutionError("D2 checkpoint state cannot be loaded strictly") from exc
    assert_model_finite(model)
    if model_state_sha256(model) != EXPECTED_D2_CHECKPOINT_STATE_SHA256:
        raise Stage7D3ExecutionError("D2 checkpoint state SHA mismatch")
    model.eval()
    return model


@dataclass(frozen=True, slots=True)
class Stage7D3Result:
    run_id: str
    run_directory: Path
    diagnostics_path: Path
    diagnostics_sha256: str
    verification_path: Path
    verification_sha256: str

    def __post_init__(self) -> None:
        _require_sha("run_id", self.run_id)
        _require_sha("diagnostics_sha256", self.diagnostics_sha256)
        _require_sha("verification_sha256", self.verification_sha256)
        for label, path in (
            ("run_directory", self.run_directory),
            ("diagnostics_path", self.diagnostics_path),
            ("verification_path", self.verification_path),
        ):
            if not isinstance(path, Path):
                raise Stage7D3ExecutionError(f"{label} must be pathlib.Path")
        if not self.run_directory.is_dir():
            raise Stage7D3ExecutionError("run_directory must exist")
        if _sha256_file(self.diagnostics_path) != self.diagnostics_sha256:
            raise Stage7D3ExecutionError("diagnostics file hash mismatch")
        if _sha256_file(self.verification_path) != self.verification_sha256:
            raise Stage7D3ExecutionError("verification file hash mismatch")


def _progress(callback, event: str, **fields: object) -> None:
    if callback is not None:
        callback({"event": event, **fields})


def run_stage7d3_validation_analysis(
    corpus_root: str | Path,
    transport_archive: str | Path,
    checkpoint_path: str | Path,
    d2_verification_path: str | Path,
    run_root: str | Path,
    repository_root: str | Path,
    *,
    progress=None,
) -> Stage7D3Result:
    """Run deterministic validation-only diagnostics with TEST remaining sealed."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    for name, value in (
        ("corpus_root", corpus_root),
        ("transport_archive", transport_archive),
        ("checkpoint_path", checkpoint_path),
        ("d2_verification_path", d2_verification_path),
        ("run_root", run_root),
        ("repository_root", repository_root),
    ):
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{name} must be str or pathlib.Path")

    repository_path = Path(repository_root).resolve()
    output_root = Path(run_root).expanduser().resolve()
    if output_root == repository_path or repository_path in output_root.parents:
        raise Stage7D3ExecutionError("D3 evidence must stay outside the Git repository")

    repository_sha, repository_origin = verify_authoritative_repository(repository_path)
    runtime_versions = verify_stage7c_runtime()

    _progress(progress, "d1_reverification_started")
    receipt = verify_stage7d_corpus(corpus_root, transport_archive)
    _verify_d1_receipt(receipt)
    _progress(
        progress,
        "d1_reverification_completed",
        artifact_binding_sha256=receipt.artifact_binding_sha256,
    )

    model = _load_accepted_d2_model(
        Path(checkpoint_path),
        Path(d2_verification_path),
    )
    validation = _load_validation_refs(Path(corpus_root))
    _progress(
        progress,
        "validation_diagnostics_started",
        validation_samples_total=len(validation),
    )

    diagnostics: list[SampleDiagnostic] = []
    for index, (sample, degradation) in enumerate(validation, start=1):
        predicted = _greedy_decode_sample(
            model,
            sample,
            preprocess_config=STAGE7D2_FROZEN_PREPROCESS_CONFIG,
            max_decode_tokens=STAGE7D2_FROZEN_RUN_CONFIG.max_decode_tokens,
            measure_count=STAGE7D2_FROZEN_RUN_CONFIG.decode_measure_count,
        )
        diagnostics.append(
            analyze_validation_sample(
                sample_id=sample.sample_id,
                family_id=sample.family_id,
                target_token_ids=sample.target_token_ids,
                predicted_token_ids=predicted,
                extra_feature_tags=(f"degradation:{degradation}",),
            )
        )
        if index % _PROGRESS_INTERVAL == 0 or index == len(validation):
            _progress(
                progress,
                "validation_diagnostic_sample_completed",
                validation_samples_completed=index,
                validation_samples_total=len(validation),
            )

    identity = {
        "schema_version": STAGE7D3_EXECUTION_SCHEMA,
        "repository_sha": repository_sha,
        "source_run_id": EXPECTED_D2_RUN_ID,
        "checkpoint_sha256": EXPECTED_D2_CHECKPOINT_SHA256,
        "checkpoint_state_sha256": EXPECTED_D2_CHECKPOINT_STATE_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_binding_sha256": receipt.artifact_binding_sha256,
    }
    run_id = sha256(_canonical_json_bytes(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / run_id
    if run_directory.exists():
        raise Stage7D3ExecutionError("D3 run directory already exists; silent overwrite forbidden")
    run_directory.mkdir()

    report, diagnostic_bytes, diagnostic_sha = build_validation_diagnostic_report(
        tuple(diagnostics),
        repository_sha=repository_sha,
        checkpoint_sha256=EXPECTED_D2_CHECKPOINT_SHA256,
        checkpoint_state_sha256=EXPECTED_D2_CHECKPOINT_STATE_SHA256,
        source_run_id=EXPECTED_D2_RUN_ID,
    )
    diagnostics_path = run_directory / f"diagnostics-{diagnostic_sha}.json"
    diagnostics_path.write_bytes(diagnostic_bytes)

    ending_sha, ending_origin = verify_authoritative_repository(repository_path)
    ending_runtime = verify_stage7c_runtime()
    if ending_sha != repository_sha or ending_origin != repository_origin:
        raise Stage7D3ExecutionError("repository identity changed during D3 analysis")
    if ending_runtime != runtime_versions:
        raise Stage7D3ExecutionError("runtime identity changed during D3 analysis")

    verification = {
        "schema_version": STAGE7D3_VERIFICATION_SCHEMA,
        "repository_origin": repository_origin,
        "repository_sha": repository_sha,
        "run_id": run_id,
        "source_d2_repository_sha": EXPECTED_D2_SOURCE_REPOSITORY_SHA,
        "source_run_id": EXPECTED_D2_RUN_ID,
        "source_d2_verification_sha256": EXPECTED_D2_VERIFICATION_SHA256,
        "checkpoint_sha256": EXPECTED_D2_CHECKPOINT_SHA256,
        "checkpoint_state_sha256": EXPECTED_D2_CHECKPOINT_STATE_SHA256,
        "dataset_build_id": EXPECTED_BUILD_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_binding_sha256": receipt.artifact_binding_sha256,
        "diagnostics_sha256": diagnostic_sha,
        "validation_samples": len(validation),
        "validation_families": len({sample.family_id for sample, _profile in validation}),
        "train_samples_exposed_to_diagnostics": 0,
        "test_samples_exposed_to_diagnostics": 0,
        "optimizer_steps": 0,
        "runtime": runtime_versions,
        "aggregate": report["aggregate"],
        "feature_buckets": report["feature_buckets"],
        "source_tree_clean_before_and_after": True,
    }
    verification_bytes = _canonical_json_bytes(verification)
    verification_sha = sha256(verification_bytes).hexdigest()
    verification_path = run_directory / f"verification-{verification_sha}.json"
    verification_path.write_bytes(verification_bytes)

    complete = (
        f"{diagnostic_sha}  {diagnostics_path.name}\n"
        f"{verification_sha}  {verification_path.name}\n"
    ).encode("ascii")
    (run_directory / "COMPLETE").write_bytes(complete)

    _progress(
        progress,
        "stage7d3_completed",
        run_id=run_id,
        diagnostics_sha256=diagnostic_sha,
        verification_sha256=verification_sha,
    )
    return Stage7D3Result(
        run_id=run_id,
        run_directory=run_directory,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=diagnostic_sha,
        verification_path=verification_path,
        verification_sha256=verification_sha,
    )


def _stderr_progress(event: dict[str, object]) -> None:
    sys.stderr.write(_canonical_json_bytes(event).decode("ascii") + "\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 7-D3 validation-only error analysis"
    )
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--d2-verification", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)

    result = run_stage7d3_validation_analysis(
        args.corpus_root,
        args.archive,
        args.checkpoint,
        args.d2_verification,
        args.run_root,
        args.repository_root,
        progress=_stderr_progress,
    )
    payload = {
        "run_id": result.run_id,
        "diagnostics_sha256": result.diagnostics_sha256,
        "verification_sha256": result.verification_sha256,
    }
    sys.stdout.write(_canonical_json_bytes(payload).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
