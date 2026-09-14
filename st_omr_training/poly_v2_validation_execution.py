"""Hash-bound native Polyphonic V2 VALIDATION execution.

TR-POLY-09B7 connects one verified B5 quality checkpoint to the frozen B1
free-running decoder, B2 per-sample metrics and B3 aggregation over one exact
native V2 VALIDATION population.

Complexity and robustness metadata are explicit caller-supplied descriptors.
This module does not infer missing labels, default samples to ``clean``, open
TEST artifacts, tune a candidate, rank candidates, or grant production
authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Final, Iterable

from .dataset_manifest import DatasetSplit
from .poly_2d_inference import (
    Poly2DInferenceError,
    Poly2DInferenceResult,
    run_poly_2d_greedy_inference,
)
from .poly_2d_quality_checkpoint import (
    LoadedPoly2DQualityCheckpoint,
    load_and_verify_poly_2d_quality_checkpoint,
)
from .poly_2d_transformer import poly_2d_config_fingerprint
from .poly_evaluation_contract import BenchmarkIdentity, BenchmarkSampleDescriptor
from .polyphonic_representation import PolyScore
from .polyphonic_serialization import parse_canonical_polyphonic_json
from .poly_v2_dataset_materialization import (
    MAX_NATIVE_POLY_V2_SAMPLES,
    NativePolyV2DatasetBuild,
    native_poly_v2_materialization_fingerprint,
)
from .poly_v2_metrics import (
    PolyV2SampleMetricReport,
    evaluate_poly_v2_validation_sample,
)
from .poly_v2_quality_execution import materialize_native_poly_v2_quality_batches
from .poly_v2_validation_aggregation import (
    PolyV2ValidationBenchmarkReport,
    aggregate_poly_v2_validation_reports,
)


POLY_V2_VALIDATION_EXECUTION_VERSION: Final[str] = (
    "st-omr-poly-v2-validation-execution-v1"
)
POLY_V2_VALIDATION_DESCRIPTOR_POLICY: Final[str] = (
    "explicit-hash-bound-no-defaults-v1"
)
POLY_V2_VALIDATION_SELECTION_POLICY: Final[str] = (
    "complete-validation-population-sorted-sample-id-v1"
)


class PolyV2ValidationExecutionError(RuntimeError):
    """Raised when the B7 VALIDATION execution boundary fails closed."""


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PolyV2ValidationExecutionError(
            "B7 evidence is not canonical-JSON serializable"
        ) from exc


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolyV2ValidationExecutionError(
            f"{name} must be lowercase SHA-256 text"
        )
    return value


def _descriptor_payload(descriptor: BenchmarkSampleDescriptor) -> dict[str, object]:
    return {
        "sample_id": descriptor.sample_id,
        "family_id": descriptor.family_id,
        "split": descriptor.split,
        "complexity": asdict(descriptor.complexity),
        "robustness_bucket": descriptor.robustness_bucket.value,
    }


def _validation_samples(build: NativePolyV2DatasetBuild):
    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
    selected = tuple(
        sorted(
            (
                sample
                for sample in build.manifest.samples
                if sample.split is DatasetSplit.VALIDATION
            ),
            key=lambda item: item.sample_id,
        )
    )
    if not selected:
        raise PolyV2ValidationExecutionError(
            "native V2 build has no VALIDATION population"
        )
    if len(selected) > MAX_NATIVE_POLY_V2_SAMPLES:
        raise PolyV2ValidationExecutionError(
            "native V2 VALIDATION population exceeds the bounded contract"
        )
    return selected


def _validate_descriptors(
    build: NativePolyV2DatasetBuild,
    descriptors: Iterable[BenchmarkSampleDescriptor],
) -> tuple[BenchmarkSampleDescriptor, ...]:
    values = tuple(descriptors)
    if not values or len(values) > MAX_NATIVE_POLY_V2_SAMPLES:
        raise PolyV2ValidationExecutionError(
            "B7 requires a bounded non-empty VALIDATION descriptor set"
        )
    if any(not isinstance(item, BenchmarkSampleDescriptor) for item in values):
        raise PolyV2ValidationExecutionError(
            "B7 descriptors must be BenchmarkSampleDescriptor values"
        )
    if any(item.split != "validation" for item in values):
        raise PolyV2ValidationExecutionError(
            "B7 accepts VALIDATION descriptors only; TEST remains sealed"
        )
    sample_ids = [item.sample_id for item in values]
    if len(set(sample_ids)) != len(sample_ids):
        raise PolyV2ValidationExecutionError(
            "B7 descriptor set contains duplicate sample_id values"
        )

    samples = _validation_samples(build)
    sample_map = {sample.sample_id: sample for sample in samples}
    descriptor_map = {item.sample_id: item for item in values}
    if set(sample_map) != set(descriptor_map):
        missing = sorted(set(sample_map) - set(descriptor_map))
        unknown = sorted(set(descriptor_map) - set(sample_map))
        raise PolyV2ValidationExecutionError(
            "B7 descriptors must cover the exact complete VALIDATION population: "
            f"missing={missing}, unknown={unknown}"
        )
    for sample_id, descriptor in descriptor_map.items():
        if descriptor.family_id != sample_map[sample_id].family_id:
            raise PolyV2ValidationExecutionError(
                "B7 descriptor family_id differs from native V2 manifest"
            )
    return tuple(sorted(values, key=lambda item: item.sample_id))


def native_poly_v2_validation_split_manifest_sha256(
    *,
    build: NativePolyV2DatasetBuild,
    descriptors: Iterable[BenchmarkSampleDescriptor],
) -> str:
    """Bind explicit descriptor metadata to exact VALIDATION artifact identities."""

    ordered = _validate_descriptors(build, descriptors)
    sample_map = {sample.sample_id: sample for sample in _validation_samples(build)}
    rows: list[dict[str, object]] = []
    for descriptor in ordered:
        sample = sample_map[descriptor.sample_id]
        rows.append(
            {
                "descriptor": _descriptor_payload(descriptor),
                "target_sha256": sample.target_sha256,
                "representation_sha256": sample.representation_sha256,
                "image_sha256": sample.image_sha256,
                "width": sample.width,
                "height": sample.height,
                "target_token_count": sample.target_token_count,
            }
        )
    payload = {
        "version": POLY_V2_VALIDATION_EXECUTION_VERSION,
        "descriptor_policy": POLY_V2_VALIDATION_DESCRIPTOR_POLICY,
        "selection_policy": POLY_V2_VALIDATION_SELECTION_POLICY,
        "dataset_manifest_sha256": build.manifest_sha256,
        "dataset_build_id": build.build_id,
        "validation_samples": rows,
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def build_native_poly_v2_validation_benchmark_identity(
    *,
    build: NativePolyV2DatasetBuild,
    descriptors: Iterable[BenchmarkSampleDescriptor],
    benchmark_id: str,
    benchmark_version: str,
) -> BenchmarkIdentity:
    """Create the frozen TR-POLY-02 identity for one explicit native V2 split."""

    split_sha = native_poly_v2_validation_split_manifest_sha256(
        build=build,
        descriptors=descriptors,
    )
    return BenchmarkIdentity(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        dataset_manifest_sha256=build.manifest_sha256,
        split_manifest_sha256=split_sha,
    )


def _load_reference_score(
    *,
    dataset_root: Path,
    sample,
) -> PolyScore:
    target_path = dataset_root / "targets" / f"{sample.target_sha256}.json"
    if target_path.is_symlink() or not target_path.is_file():
        raise PolyV2ValidationExecutionError(
            "B7 reference target is missing or symlinked"
        )
    target_bytes = target_path.read_bytes()
    if sha256(target_bytes).hexdigest() != sample.target_sha256:
        raise PolyV2ValidationExecutionError("B7 reference target hash mismatch")
    try:
        reference = parse_canonical_polyphonic_json(target_bytes)
    except Exception as exc:
        raise PolyV2ValidationExecutionError(
            "B7 reference target is not canonical Polyphonic V2"
        ) from exc
    if reference.canonical_sha256() != sample.representation_sha256:
        raise PolyV2ValidationExecutionError(
            "B7 reference representation SHA-256 mismatch"
        )
    return reference


def _run_loaded_quality_inference(
    loaded: LoadedPoly2DQualityCheckpoint,
    image,
    *,
    max_decode_steps: int,
) -> Poly2DInferenceResult:
    """Run B1 once from an already-verified B5 artifact and bind its identity."""

    try:
        result = run_poly_2d_greedy_inference(
            loaded.model,
            image,
            max_decode_steps=max_decode_steps,
        )
    except Poly2DInferenceError as exc:
        raise PolyV2ValidationExecutionError(
            "B7 free-running inference violated the B1 contract"
        ) from exc
    metadata = loaded.metadata
    if result.identity.model_state_sha256 != metadata.selected_state_sha256:
        raise PolyV2ValidationExecutionError(
            "B7 checkpoint state differs from B1 inference state"
        )
    if result.identity.model_profile_sha256 != metadata.model_profile_sha256:
        raise PolyV2ValidationExecutionError(
            "B7 checkpoint model profile differs from B1 inference profile"
        )
    bound_identity = replace(
        result.identity,
        checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_metadata_file_sha256=loaded.metadata_sha256,
        checkpoint_receipt_sha256=loaded.receipt_sha256,
        checkpoint_metadata_fingerprint_sha256=metadata.fingerprint(),
        dataset_manifest_sha256=metadata.dataset_manifest_sha256,
        preprocess_fingerprint_sha256=metadata.preprocess_fingerprint_sha256,
        trainer_profile_sha256=metadata.quality_trainer_profile_sha256,
        provenance_sha256=metadata.provenance_sha256,
        registry_record_fingerprint_sha256=metadata.registry_record_fingerprint_sha256,
        repository_sha=metadata.repository_sha,
    )
    return replace(result, identity=bound_identity)


@dataclass(frozen=True, slots=True)
class PolyV2ValidationExecutionResult:
    benchmark: BenchmarkIdentity
    checkpoint_sha256: str
    checkpoint_metadata_sha256: str
    checkpoint_receipt_sha256: str
    checkpoint_metadata_fingerprint_sha256: str
    dataset_manifest_sha256: str
    dataset_build_id: str
    validation_split_manifest_sha256: str
    materialization_fingerprint_sha256: str
    candidate_identity_sha256: str
    inference_profile_sha256: str
    max_decode_steps: int
    sample_reports: tuple[PolyV2SampleMetricReport, ...]
    aggregate: PolyV2ValidationBenchmarkReport
    validation_benchmark_evidence: bool = True
    test_split_accessed: bool = False
    production_authority: bool = False
    execution_version: str = POLY_V2_VALIDATION_EXECUTION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, BenchmarkIdentity):
            raise PolyV2ValidationExecutionError(
                "B7 result benchmark must be BenchmarkIdentity"
            )
        for name in (
            "checkpoint_sha256",
            "checkpoint_metadata_sha256",
            "checkpoint_receipt_sha256",
            "checkpoint_metadata_fingerprint_sha256",
            "dataset_manifest_sha256",
            "dataset_build_id",
            "validation_split_manifest_sha256",
            "materialization_fingerprint_sha256",
            "candidate_identity_sha256",
            "inference_profile_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.dataset_manifest_sha256 != self.benchmark.dataset_manifest_sha256:
            raise PolyV2ValidationExecutionError(
                "B7 result dataset identity differs from benchmark"
            )
        if self.validation_split_manifest_sha256 != self.benchmark.split_manifest_sha256:
            raise PolyV2ValidationExecutionError(
                "B7 result split identity differs from benchmark"
            )
        if not _plain_int(self.max_decode_steps) or self.max_decode_steps < 1:
            raise PolyV2ValidationExecutionError(
                "max_decode_steps must be a positive plain integer"
            )
        if not self.sample_reports or any(
            not isinstance(item, PolyV2SampleMetricReport)
            for item in self.sample_reports
        ):
            raise PolyV2ValidationExecutionError(
                "B7 result requires immutable B2 sample reports"
            )
        report_ids = tuple(item.sample_id for item in self.sample_reports)
        if tuple(sorted(set(report_ids))) != report_ids:
            raise PolyV2ValidationExecutionError(
                "B7 sample reports must be sorted and unique"
            )
        if any(
            item.benchmark_identity_sha256 != self.benchmark.canonical_sha256()
            for item in self.sample_reports
        ):
            raise PolyV2ValidationExecutionError(
                "B7 sample report benchmark identity mismatch"
            )
        if any(
            item.candidate_identity_sha256 != self.candidate_identity_sha256
            for item in self.sample_reports
        ):
            raise PolyV2ValidationExecutionError(
                "B7 sample reports mix candidate identities"
            )
        if not isinstance(self.aggregate, PolyV2ValidationBenchmarkReport):
            raise PolyV2ValidationExecutionError(
                "B7 aggregate must be a B3 benchmark report"
            )
        if self.aggregate.benchmark_identity_sha256 != self.benchmark.canonical_sha256():
            raise PolyV2ValidationExecutionError(
                "B7 aggregate benchmark identity mismatch"
            )
        if self.aggregate.candidate_identity_sha256 != self.candidate_identity_sha256:
            raise PolyV2ValidationExecutionError(
                "B7 aggregate candidate identity mismatch"
            )
        if self.aggregate.sample_ids != report_ids:
            raise PolyV2ValidationExecutionError(
                "B7 aggregate sample population differs from sample reports"
            )
        if not self.validation_benchmark_evidence:
            raise PolyV2ValidationExecutionError(
                "a completed B7 result must identify itself as VALIDATION evidence"
            )
        if self.test_split_accessed or self.production_authority:
            raise PolyV2ValidationExecutionError(
                "B7 may not access TEST or claim production authority"
            )
        if self.execution_version != POLY_V2_VALIDATION_EXECUTION_VERSION:
            raise PolyV2ValidationExecutionError("B7 execution version mismatch")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.sample_reports)

    @property
    def sample_report_fingerprints(self) -> tuple[str, ...]:
        return tuple(item.fingerprint() for item in self.sample_reports)

    @property
    def full_metric_contract_ready(self) -> bool:
        return self.aggregate.full_metric_contract_ready

    @property
    def common_comparison_ready(self) -> bool:
        return self.aggregate.common_comparison_ready

    def fingerprint(self) -> str:
        payload = {
            "execution_version": self.execution_version,
            "benchmark_identity_sha256": self.benchmark.canonical_sha256(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_metadata_sha256": self.checkpoint_metadata_sha256,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "checkpoint_metadata_fingerprint_sha256": self.checkpoint_metadata_fingerprint_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_build_id": self.dataset_build_id,
            "validation_split_manifest_sha256": self.validation_split_manifest_sha256,
            "materialization_fingerprint_sha256": self.materialization_fingerprint_sha256,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "inference_profile_sha256": self.inference_profile_sha256,
            "max_decode_steps": self.max_decode_steps,
            "sample_ids": list(self.sample_ids),
            "sample_report_fingerprints": list(self.sample_report_fingerprints),
            "aggregate_fingerprint_sha256": self.aggregate.fingerprint(),
            "validation_benchmark_evidence": self.validation_benchmark_evidence,
            "test_split_accessed": self.test_split_accessed,
            "production_authority": self.production_authority,
        }
        return sha256(_canonical_json_bytes(_jsonable(payload))).hexdigest()


def execute_native_poly_v2_validation_benchmark(
    *,
    build: NativePolyV2DatasetBuild,
    dataset_root: str | Path,
    checkpoint_directory: Path,
    benchmark: BenchmarkIdentity,
    descriptors: Iterable[BenchmarkSampleDescriptor],
    max_decode_steps: int,
) -> PolyV2ValidationExecutionResult:
    """Execute one exact B5 candidate over the complete native V2 VALIDATION split."""

    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be str or pathlib.Path")
    if not isinstance(checkpoint_directory, Path):
        raise TypeError("checkpoint_directory must be pathlib.Path")
    if not isinstance(benchmark, BenchmarkIdentity):
        raise TypeError("benchmark must be BenchmarkIdentity")
    if not _plain_int(max_decode_steps) or max_decode_steps < 1:
        raise PolyV2ValidationExecutionError(
            "max_decode_steps must be a positive plain integer"
        )

    ordered_descriptors = _validate_descriptors(build, descriptors)
    expected_benchmark = build_native_poly_v2_validation_benchmark_identity(
        build=build,
        descriptors=ordered_descriptors,
        benchmark_id=benchmark.benchmark_id,
        benchmark_version=benchmark.benchmark_version,
    )
    if benchmark != expected_benchmark:
        raise PolyV2ValidationExecutionError(
            "B7 benchmark identity does not match exact descriptor/artifact binding"
        )

    loaded = load_and_verify_poly_2d_quality_checkpoint(checkpoint_directory)
    metadata = loaded.metadata
    if metadata.dataset_manifest_sha256 != build.manifest_sha256:
        raise PolyV2ValidationExecutionError(
            "B7 checkpoint was trained against a different dataset manifest"
        )
    expected_materialization = native_poly_v2_materialization_fingerprint(
        loaded.model.config
    )
    if metadata.preprocess_fingerprint_sha256 != expected_materialization:
        raise PolyV2ValidationExecutionError(
            "B7 checkpoint preprocessing identity differs from native V2 materialization"
        )
    if metadata.model_profile_sha256 != poly_2d_config_fingerprint(loaded.model.config):
        raise PolyV2ValidationExecutionError(
            "B7 checkpoint model profile differs from loaded model config"
        )
    if max_decode_steps > loaded.model.config.max_target_tokens - 1:
        raise PolyV2ValidationExecutionError(
            "max_decode_steps exceeds the verified checkpoint target boundary"
        )

    validation = materialize_native_poly_v2_quality_batches(
        build=build,
        dataset_root=dataset_root,
        split=DatasetSplit.VALIDATION,
        model_config=loaded.model.config,
        batch_size=1,
        max_samples=None,
    )
    descriptor_ids = tuple(item.sample_id for item in ordered_descriptors)
    if validation.sample_ids != descriptor_ids:
        raise PolyV2ValidationExecutionError(
            "B7 materialized VALIDATION population differs from descriptor manifest"
        )
    if validation.materialization_fingerprint_sha256 != expected_materialization:
        raise PolyV2ValidationExecutionError(
            "B7 VALIDATION materialization fingerprint drifted"
        )

    root = Path(dataset_root)
    sample_map = {
        sample.sample_id: sample
        for sample in _validation_samples(build)
    }
    descriptor_map = {item.sample_id: item for item in ordered_descriptors}
    reports: list[PolyV2SampleMetricReport] = []
    inference_profile_sha256: str | None = None
    candidate_identity_sha256: str | None = None

    for batch in validation.batches:
        if len(batch.sample_ids) != 1 or batch.images.shape[0] != 1:
            raise PolyV2ValidationExecutionError(
                "B7 requires one VALIDATION sample per inference call"
            )
        sample_id = batch.sample_ids[0]
        sample = sample_map[sample_id]
        descriptor = descriptor_map[sample_id]
        reference = _load_reference_score(dataset_root=root, sample=sample)
        prediction = _run_loaded_quality_inference(
            loaded,
            batch.images,
            max_decode_steps=max_decode_steps,
        )
        current_profile = prediction.identity.inference_profile_sha256
        current_candidate = prediction.identity.fingerprint()
        if inference_profile_sha256 is None:
            inference_profile_sha256 = current_profile
            candidate_identity_sha256 = current_candidate
        elif (
            current_profile != inference_profile_sha256
            or current_candidate != candidate_identity_sha256
        ):
            raise PolyV2ValidationExecutionError(
                "B7 inference identity changed within one benchmark execution"
            )
        reports.append(
            evaluate_poly_v2_validation_sample(
                reference=reference,
                prediction=prediction,
                benchmark=benchmark,
                descriptor=descriptor,
            )
        )

    if inference_profile_sha256 is None or candidate_identity_sha256 is None:
        raise PolyV2ValidationExecutionError(
            "B7 produced no VALIDATION inference evidence"
        )
    ordered_reports = tuple(sorted(reports, key=lambda item: item.sample_id))
    aggregate = aggregate_poly_v2_validation_reports(ordered_reports)
    split_sha = native_poly_v2_validation_split_manifest_sha256(
        build=build,
        descriptors=ordered_descriptors,
    )
    return PolyV2ValidationExecutionResult(
        benchmark=benchmark,
        checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_metadata_sha256=loaded.metadata_sha256,
        checkpoint_receipt_sha256=loaded.receipt_sha256,
        checkpoint_metadata_fingerprint_sha256=metadata.fingerprint(),
        dataset_manifest_sha256=build.manifest_sha256,
        dataset_build_id=build.build_id,
        validation_split_manifest_sha256=split_sha,
        materialization_fingerprint_sha256=expected_materialization,
        candidate_identity_sha256=candidate_identity_sha256,
        inference_profile_sha256=inference_profile_sha256,
        max_decode_steps=max_decode_steps,
        sample_reports=ordered_reports,
        aggregate=aggregate,
    )
