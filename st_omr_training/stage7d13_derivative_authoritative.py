"""Authoritative execution gate for Stage 7-D13 measure derivatives.

D13-3 binds the D13-1 builder and D13-2 independent verifier to an exact
repository/runtime identity.  Only after independent verification succeeds are
exact TRAIN measure-record counts converted into frozen-candidate optimizer-step
counts.  This module performs no model creation and no optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Final

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d13_measure_derivative_verifier import (
    Stage7D13VerificationReceipt,
    verify_stage7d13_measure_derivatives,
)
from .stage7d13_measure_derivatives import (
    Stage7D13DerivativeReceipt,
    build_stage7d13_measure_derivatives,
)
from .stage7d13_symbol_training_contract import FROZEN_D13_CONFIG, SPECIALIST_CLASSES


STAGE7D13_DERIVATIVE_AUTHORITATIVE_VERSION: Final[str] = (
    "stage7d13-derivative-authoritative-v1"
)
_HEX40: Final[frozenset[str]] = frozenset("0123456789abcdef")


class Stage7D13DerivativeAuthoritativeError(RuntimeError):
    """Raised when D13 authoritative derivative execution cannot be proven."""


def _fail(message: str) -> None:
    raise Stage7D13DerivativeAuthoritativeError(message)


def _git_sha40(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _HEX40 for character in value)
    ):
        _fail(f"{name} must be canonical lowercase 40-character Git SHA")
    return value


def _repository_identity(repository_root: Path) -> tuple[str, str]:
    identity = verify_authoritative_repository(repository_root)
    if (
        not isinstance(identity, tuple)
        or len(identity) != 2
        or not isinstance(identity[0], str)
        or not isinstance(identity[1], str)
    ):
        _fail("Stage 7-C repository verifier returned an unexpected identity")
    return _git_sha40("repository HEAD", identity[0]), identity[1]


def derive_optimizer_steps(train_record_count: int) -> dict[str, int]:
    """Derive exact per-specialist optimizer steps from verified TRAIN records."""
    if (
        not isinstance(train_record_count, int)
        or isinstance(train_record_count, bool)
        or train_record_count <= 0
    ):
        raise ValueError("train_record_count must be a positive integer")
    batches_per_epoch = math.ceil(train_record_count / FROZEN_D13_CONFIG.batch_size)
    per_specialist = batches_per_epoch * FROZEN_D13_CONFIG.epochs
    return {specialist: per_specialist for specialist in SPECIALIST_CLASSES}


def _assert_receipts_match(
    build: Stage7D13DerivativeReceipt,
    verified: Stage7D13VerificationReceipt,
) -> None:
    names = (
        "derivative_build_id",
        "manifest_sha256",
        "artifact_binding_sha256",
        "record_count",
        "image_count",
        "label_count",
        "source_sample_count",
        "family_count",
        "record_split_counts",
        "source_sample_split_counts",
        "family_split_counts",
        "observed_class_inventory",
        "target_instance_counts",
        "image_bytes_total",
        "label_bytes_total",
        "test_specialist_records",
        "optimizer_steps",
    )
    for name in names:
        if getattr(build, name) != getattr(verified, name):
            _fail(f"D13 builder/verifier {name} mismatch")
    if build.complete_marker_written:
        _fail("D13 builder unexpectedly reported COMPLETE")
    if verified.complete_marker_present:
        _fail("D13 verifier unexpectedly observed COMPLETE")
    if not verified.verification_passed:
        _fail("D13 independent derivative verification did not pass")


@dataclass(frozen=True, slots=True)
class Stage7D13DerivativeAuthoritativeReceipt:
    version: str
    repository_sha: str
    repository_origin: str
    derivative_build_id: str
    manifest_sha256: str
    artifact_binding_sha256: str
    record_count: int
    image_count: int
    label_count: int
    source_sample_count: int
    family_count: int
    record_split_counts: dict[str, int]
    source_sample_split_counts: dict[str, int]
    family_split_counts: dict[str, int]
    observed_class_inventory: dict[str, dict[str, dict[str, int]]]
    target_instance_counts: dict[str, dict[str, int]]
    image_bytes_total: int
    label_bytes_total: int
    expected_optimizer_steps: dict[str, int]
    expected_optimizer_steps_total: int
    test_specialist_records: int
    optimizer_steps_executed: int
    complete_marker_present: bool
    independent_verification_passed: bool


def run_verified_stage7d13_derivative_bundle(
    *,
    corpus_root: str | Path,
    d6_root: str | Path,
    d12_root: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_repository_sha: str,
) -> Stage7D13DerivativeAuthoritativeReceipt:
    """Build and independently verify D13 derivatives on an exact repository."""
    if not all(
        isinstance(value, (str, Path))
        for value in (corpus_root, d6_root, d12_root, output_root, repository_root)
    ):
        raise TypeError(
            "corpus_root, d6_root, d12_root, output_root and repository_root "
            "must be str or pathlib.Path"
        )
    expected = _git_sha40("expected_repository_sha", expected_repository_sha)
    repo = Path(repository_root)

    head_before, origin_before = _repository_identity(repo)
    if head_before != expected:
        _fail("D13 repository HEAD differs from the authorized exact head")
    verify_stage7c_runtime()

    build = build_stage7d13_measure_derivatives(
        corpus_root=corpus_root,
        d6_root=d6_root,
        d12_root=d12_root,
        output_root=output_root,
    )

    head_after_build, origin_after_build = _repository_identity(repo)
    if (head_after_build, origin_after_build) != (head_before, origin_before):
        _fail("repository identity changed during D13 derivative build")
    verify_stage7c_runtime()

    verified = verify_stage7d13_measure_derivatives(
        corpus_root=corpus_root,
        d6_root=d6_root,
        d12_root=d12_root,
        derivative_root=output_root,
    )
    _assert_receipts_match(build, verified)

    head_after_verify, origin_after_verify = _repository_identity(repo)
    if (head_after_verify, origin_after_verify) != (head_before, origin_before):
        _fail("repository identity changed during D13 derivative verification")
    verify_stage7c_runtime()

    train_records = verified.record_split_counts.get("train")
    if not isinstance(train_records, int) or train_records <= 0:
        _fail("verified D13 TRAIN record count is missing or invalid")
    expected_steps = derive_optimizer_steps(train_records)

    return Stage7D13DerivativeAuthoritativeReceipt(
        version=STAGE7D13_DERIVATIVE_AUTHORITATIVE_VERSION,
        repository_sha=head_before,
        repository_origin=origin_before,
        derivative_build_id=verified.derivative_build_id,
        manifest_sha256=verified.manifest_sha256,
        artifact_binding_sha256=verified.artifact_binding_sha256,
        record_count=verified.record_count,
        image_count=verified.image_count,
        label_count=verified.label_count,
        source_sample_count=verified.source_sample_count,
        family_count=verified.family_count,
        record_split_counts=verified.record_split_counts,
        source_sample_split_counts=verified.source_sample_split_counts,
        family_split_counts=verified.family_split_counts,
        observed_class_inventory=verified.observed_class_inventory,
        target_instance_counts=verified.target_instance_counts,
        image_bytes_total=verified.image_bytes_total,
        label_bytes_total=verified.label_bytes_total,
        expected_optimizer_steps=expected_steps,
        expected_optimizer_steps_total=sum(expected_steps.values()),
        test_specialist_records=verified.test_specialist_records,
        optimizer_steps_executed=verified.optimizer_steps,
        complete_marker_present=verified.complete_marker_present,
        independent_verification_passed=verified.verification_passed,
    )
