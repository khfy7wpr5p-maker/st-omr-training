"""Authoritative Stage 7-D13 training + independent verification gate.

This module is executable infrastructure only; importing it performs no training.
A caller must explicitly provide the exact reviewed repository SHA and external
artifact roots. A full derivative/collision/parameter preflight runs before any
optimizer is created. Training is then followed by independent persisted-run
verification before verification.json and COMPLETE are written, then the
completed surface is reopened once more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d13_run_verifier import (
    STAGE7D13_RUN_VERIFIER_VERSION,
    Stage7D13RunVerificationReceipt,
    verify_stage7d13_run,
)
from .stage7d13_training import Stage7D13TrainingReceipt, run_stage7d13_training
from .stage7d13_training_preflight import verify_stage7d13_training_preflight


STAGE7D13_AUTHORITATIVE_VERSION: Final[str] = "stage7d13-authoritative-training-gate-v1"
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class Stage7D13AuthoritativeTrainingError(RuntimeError):
    """Raised when the D13 authoritative execution/verification boundary fails."""


def _fail(message: str) -> None:
    raise Stage7D13AuthoritativeTrainingError(message)


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
        raise Stage7D13AuthoritativeTrainingError("D13 authoritative payload is not canonical JSON") from exc


def _git_sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in _HEX for c in value):
        _fail("expected_repository_sha must be canonical lowercase Git SHA")
    return value


def _repo_identity(root: Path) -> tuple[str, str]:
    identity = verify_authoritative_repository(root)
    if not isinstance(identity, tuple) or len(identity) != 2:
        _fail("repository verifier returned unexpected identity")
    head, origin = identity
    if _git_sha(head) != head or not isinstance(origin, str) or not origin:
        _fail("repository identity is invalid")
    return head, origin


def _assert_training_and_verifier_match(
    training: Stage7D13TrainingReceipt,
    verified: Stage7D13RunVerificationReceipt,
) -> None:
    pairs = {
        "run_id": (training.run_id, verified.run_id),
        "repository_sha": (training.repository_sha, verified.repository_sha),
        "training_profile_fingerprint": (
            training.training_profile_fingerprint,
            verified.training_profile_fingerprint,
        ),
        "checkpoint_sha256": (training.checkpoint_sha256, verified.checkpoint_sha256),
        "metrics_sha256": (training.metrics_sha256, verified.metrics_sha256),
        "run_sha256": (training.run_sha256, verified.run_sha256),
        "optimizer_steps": (training.optimizer_steps, verified.optimizer_steps),
        "optimizer_steps_total": (
            training.optimizer_steps_total,
            verified.optimizer_steps_total,
        ),
        "acceptance": (training.acceptance, verified.acceptance),
        "test_opened": (training.test_opened, verified.test_opened),
    }
    for name, (left, right) in pairs.items():
        if left != right:
            _fail(f"D13 training/verifier {name} mismatch")
    if training.complete_marker_written:
        _fail("D13 trainer unexpectedly reported COMPLETE")
    if verified.complete_marker_present:
        _fail("uncompleted D13 verifier unexpectedly observed COMPLETE")
    if not verified.verification_passed:
        _fail("D13 independent persisted-run verification failed")


@dataclass(frozen=True, slots=True)
class Stage7D13AuthoritativeReceipt:
    version: str
    run_id: str
    repository_sha: str
    repository_origin: str
    preflight_record_count: int
    preflight_parameter_count_total: int
    checkpoint_sha256: str
    metrics_sha256: str
    run_sha256: str
    verification_sha256: str
    optimizer_steps: dict[str, int]
    optimizer_steps_total: int
    specialist_acceptance: dict[str, bool]
    acceptance: bool
    test_opened: bool
    complete_marker_present: bool
    independent_verification_passed: bool


def run_verified_stage7d13_training(
    *,
    derivative_root: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_repository_sha: str,
    heartbeat=print,
) -> Stage7D13AuthoritativeReceipt:
    """Run reviewed D13 training and independently close the persisted run."""
    expected = _git_sha(expected_repository_sha)
    repo = Path(repository_root)
    out = Path(output_root)
    head_before, origin_before = _repo_identity(repo)
    if head_before != expected:
        _fail("repository HEAD differs from explicitly authorized D13 head")
    verify_stage7c_runtime()

    preflight = verify_stage7d13_training_preflight(derivative_root)
    if not preflight.preflight_passed or not preflight.collision_free or preflight.test_opened:
        _fail("D13 preflight did not authorize optimizer creation")
    if heartbeat is not None:
        heartbeat(
            "D13 PREFLIGHT PASS | "
            f"records={preflight.record_count} | params={preflight.parameter_count_total} | TEST=False"
        )

    head_after_preflight, origin_after_preflight = _repo_identity(repo)
    verify_stage7c_runtime()
    if (head_after_preflight, origin_after_preflight) != (head_before, origin_before):
        _fail("repository identity changed during D13 preflight")

    training = run_stage7d13_training(
        derivative_root=derivative_root,
        output_root=out,
        repository_root=repo,
        expected_repository_sha=expected,
        heartbeat=heartbeat,
    )

    head_after_training, origin_after_training = _repo_identity(repo)
    verify_stage7c_runtime()
    if (head_after_training, origin_after_training) != (head_before, origin_before):
        _fail("repository identity changed during D13 training")

    verified = verify_stage7d13_run(out, require_complete=False)
    _assert_training_and_verifier_match(training, verified)

    verification_payload = asdict(verified)
    verification_payload["verifier_version"] = STAGE7D13_RUN_VERIFIER_VERSION
    verification_raw = _canonical_json(verification_payload)
    verification_sha = sha256(verification_raw).hexdigest()
    verification_path = out / "verification.json"
    if verification_path.exists() or verification_path.is_symlink():
        _fail("D13 verification.json unexpectedly already exists")
    verification_path.write_bytes(verification_raw)

    complete_payload = {
        "version": STAGE7D13_AUTHORITATIVE_VERSION,
        "run_id": verified.run_id,
        "verification_sha256": verification_sha,
        "acceptance": verified.acceptance,
        "test_opened": False,
    }
    complete_path = out / "COMPLETE"
    if complete_path.exists() or complete_path.is_symlink():
        _fail("D13 COMPLETE unexpectedly already exists")
    complete_path.write_bytes(_canonical_json(complete_payload))

    final_verified = verify_stage7d13_run(out, require_complete=True)
    if not final_verified.verification_passed or not final_verified.complete_marker_present:
        _fail("completed D13 run did not independently reopen")
    if final_verified.run_id != verified.run_id or final_verified.acceptance != verified.acceptance:
        _fail("completed D13 run changed verification identity")

    head_after, origin_after = _repo_identity(repo)
    verify_stage7c_runtime()
    if (head_after, origin_after) != (head_before, origin_before):
        _fail("repository identity changed during D13 completion verification")

    return Stage7D13AuthoritativeReceipt(
        version=STAGE7D13_AUTHORITATIVE_VERSION,
        run_id=final_verified.run_id,
        repository_sha=head_before,
        repository_origin=origin_before,
        preflight_record_count=preflight.record_count,
        preflight_parameter_count_total=preflight.parameter_count_total,
        checkpoint_sha256=final_verified.checkpoint_sha256,
        metrics_sha256=final_verified.metrics_sha256,
        run_sha256=final_verified.run_sha256,
        verification_sha256=verification_sha,
        optimizer_steps=final_verified.optimizer_steps,
        optimizer_steps_total=final_verified.optimizer_steps_total,
        specialist_acceptance=final_verified.specialist_acceptance,
        acceptance=final_verified.acceptance,
        test_opened=False,
        complete_marker_present=True,
        independent_verification_passed=True,
    )
