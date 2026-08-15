"""Authoritative execution gate for Stage 7-D12 symbol derivatives.

This module coordinates the already-frozen D12 builder and independent persisted
verifier on an exact repository/runtime identity.  It deliberately does not
write COMPLETE: D12-6 ends with a verified *uncompleted* bundle so closure
evidence can only be written in a later controlled step after the authoritative
inventory is reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d12_symbol_derivative_verifier import (
    Stage7D12VerificationReceipt,
    verify_stage7d12_symbol_derivatives,
)
from .stage7d12_symbol_derivatives import (
    Stage7D12DerivativeReceipt,
    build_stage7d12_symbol_derivatives,
)


STAGE7D12_AUTHORITATIVE_VERSION: Final[str] = "stage7d12-authoritative-bundle-v1"
_HEX40: Final[frozenset[str]] = frozenset("0123456789abcdef")


class Stage7D12AuthoritativeError(RuntimeError):
    """Raised when the authoritative D12 execution boundary cannot be proven."""


def _fail(message: str) -> None:
    raise Stage7D12AuthoritativeError(message)


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


def _assert_receipts_match(
    build: Stage7D12DerivativeReceipt,
    verified: Stage7D12VerificationReceipt,
) -> None:
    pairs = {
        "derivative_build_id": (build.derivative_build_id, verified.derivative_build_id),
        "manifest_sha256": (build.manifest_sha256, verified.manifest_sha256),
        "artifact_binding_sha256": (
            build.artifact_binding_sha256,
            verified.artifact_binding_sha256,
        ),
        "sample_count": (build.sample_count, verified.sample_count),
        "family_count": (build.family_count, verified.family_count),
        "label_count": (build.label_count, verified.label_count),
        "label_bytes_total": (build.label_bytes_total, verified.label_bytes_total),
        "sample_split_counts": (
            build.sample_split_counts,
            verified.sample_split_counts,
        ),
        "family_split_counts": (
            build.family_split_counts,
            verified.family_split_counts,
        ),
        "observed_class_inventory": (
            build.observed_class_inventory,
            verified.observed_class_inventory,
        ),
        "test_specialist_records": (
            build.test_specialist_records,
            verified.test_specialist_records,
        ),
        "optimizer_steps": (build.optimizer_steps, verified.optimizer_steps),
    }
    for name, (builder_value, verifier_value) in pairs.items():
        if builder_value != verifier_value:
            _fail(f"D12 builder/verifier {name} mismatch")
    if build.complete_marker_written:
        _fail("D12 builder unexpectedly reported COMPLETE")
    if verified.complete_marker_present:
        _fail("D12 verifier unexpectedly observed COMPLETE")
    if not verified.verification_passed:
        _fail("D12 independent verification did not pass")


@dataclass(frozen=True, slots=True)
class Stage7D12AuthoritativeReceipt:
    version: str
    repository_sha: str
    repository_origin: str
    derivative_build_id: str
    manifest_sha256: str
    artifact_binding_sha256: str
    sample_count: int
    family_count: int
    label_count: int
    label_bytes_total: int
    sample_split_counts: dict[str, int]
    family_split_counts: dict[str, int]
    observed_class_inventory: dict[str, dict[str, dict[str, int]]]
    test_specialist_records: int
    optimizer_steps: int
    complete_marker_present: bool
    independent_verification_passed: bool


def run_verified_stage7d12_authoritative_bundle(
    *,
    corpus_root: str | Path,
    d6_root: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_repository_sha: str,
) -> Stage7D12AuthoritativeReceipt:
    """Build and independently verify the exact D12 development-only bundle.

    The output root must be fresh and outside Git; the underlying builder enforces
    that rule.  TEST remains sealed and optimizer steps remain zero.  No COMPLETE
    marker is written by this function.
    """

    if not all(
        isinstance(value, (str, Path))
        for value in (corpus_root, d6_root, output_root, repository_root)
    ):
        raise TypeError(
            "corpus_root, d6_root, output_root and repository_root must be str or pathlib.Path"
        )
    expected = _git_sha40("expected_repository_sha", expected_repository_sha)
    repo = Path(repository_root)

    head_before, origin_before = _repository_identity(repo)
    if head_before != expected:
        _fail("D12 repository HEAD differs from the authorized exact head")
    verify_stage7c_runtime()

    build = build_stage7d12_symbol_derivatives(
        corpus_root=corpus_root,
        d6_root=d6_root,
        output_root=output_root,
    )

    head_after_build, origin_after_build = _repository_identity(repo)
    if (head_after_build, origin_after_build) != (head_before, origin_before):
        _fail("repository identity changed during D12 authoritative build")
    verify_stage7c_runtime()

    verified = verify_stage7d12_symbol_derivatives(
        corpus_root=corpus_root,
        d6_root=d6_root,
        derivative_root=output_root,
    )
    _assert_receipts_match(build, verified)

    head_after_verify, origin_after_verify = _repository_identity(repo)
    if (head_after_verify, origin_after_verify) != (head_before, origin_before):
        _fail("repository identity changed during D12 independent verification")
    verify_stage7c_runtime()

    return Stage7D12AuthoritativeReceipt(
        version=STAGE7D12_AUTHORITATIVE_VERSION,
        repository_sha=head_before,
        repository_origin=origin_before,
        derivative_build_id=verified.derivative_build_id,
        manifest_sha256=verified.manifest_sha256,
        artifact_binding_sha256=verified.artifact_binding_sha256,
        sample_count=verified.sample_count,
        family_count=verified.family_count,
        label_count=verified.label_count,
        label_bytes_total=verified.label_bytes_total,
        sample_split_counts=verified.sample_split_counts,
        family_split_counts=verified.family_split_counts,
        observed_class_inventory=verified.observed_class_inventory,
        test_specialist_records=verified.test_specialist_records,
        optimizer_steps=verified.optimizer_steps,
        complete_marker_present=verified.complete_marker_present,
        independent_verification_passed=verified.verification_passed,
    )
