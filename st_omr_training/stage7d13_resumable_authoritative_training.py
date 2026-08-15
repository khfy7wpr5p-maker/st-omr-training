"""Crash-resilient authoritative Stage 7-D13 training gate.

This gate keeps the frozen preflight/training/verifier contract while allowing
Colab/runtime restarts. Epoch-level progress lives in the sibling resume root;
the final run directory appears only after all three specialists finish.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Final

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d13_resumable_training import run_stage7d13_resumable_training
from .stage7d13_run_verifier import (
    STAGE7D13_RUN_VERIFIER_VERSION,
    Stage7D13RunVerificationReceipt,
    verify_stage7d13_run,
)
from .stage7d13_training import Stage7D13TrainingReceipt
from .stage7d13_training_preflight import verify_stage7d13_training_preflight


STAGE7D13_RESUMABLE_AUTHORITATIVE_VERSION: Final[str] = (
    "stage7d13-resumable-authoritative-training-gate-v1"
)
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_BASE_TOP: Final[frozenset[str]] = frozenset({"checkpoint.pt", "metrics.json", "run.json"})
_COMPLETE_TOP: Final[frozenset[str]] = frozenset(
    {"checkpoint.pt", "metrics.json", "run.json", "verification.json", "COMPLETE"}
)
_VERIFICATION_ONLY_TOP: Final[frozenset[str]] = frozenset(
    {"checkpoint.pt", "metrics.json", "run.json", "verification.json"}
)


class Stage7D13ResumableAuthoritativeError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise Stage7D13ResumableAuthoritativeError(message)


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
        raise Stage7D13ResumableAuthoritativeError(
            "D13 authoritative payload is not canonical JSON"
        ) from exc


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


def _atomic_bytes(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verification_payload(
    verified: Stage7D13RunVerificationReceipt,
) -> tuple[bytes, str]:
    payload = asdict(verified)
    payload["verifier_version"] = STAGE7D13_RUN_VERIFIER_VERSION
    raw = _canonical_json(payload)
    return raw, sha256(raw).hexdigest()


def _assert_training_match(
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
    if verified.complete_marker_present or not verified.verification_passed:
        _fail("D13 uncompleted independent verification failed")


def _verify_base_via_shadow(root: Path) -> Stage7D13RunVerificationReceipt:
    """Verify the three-file base when verification.json was already persisted."""
    shadow = root.with_name(root.name + f".verify-shadow-{os.getpid()}")
    if shadow.exists() or shadow.is_symlink():
        _fail("D13 verification shadow unexpectedly exists")
    shadow.mkdir(parents=True)
    try:
        for name in _BASE_TOP:
            source = root / name
            if source.is_symlink() or not source.is_file():
                _fail("D13 recovery base file missing")
            destination = shadow / name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        return verify_stage7d13_run(shadow, require_complete=False)
    finally:
        if shadow.exists():
            shutil.rmtree(shadow)


@dataclass(frozen=True, slots=True)
class Stage7D13ResumableAuthoritativeReceipt:
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
    resumed_existing_run_surface: bool


def _receipt(
    *, verified: Stage7D13RunVerificationReceipt,
    repository_origin: str, preflight_record_count: int,
    preflight_parameter_count_total: int, verification_sha256: str,
    resumed_existing_run_surface: bool,
) -> Stage7D13ResumableAuthoritativeReceipt:
    return Stage7D13ResumableAuthoritativeReceipt(
        version=STAGE7D13_RESUMABLE_AUTHORITATIVE_VERSION,
        run_id=verified.run_id,
        repository_sha=verified.repository_sha,
        repository_origin=repository_origin,
        preflight_record_count=preflight_record_count,
        preflight_parameter_count_total=preflight_parameter_count_total,
        checkpoint_sha256=verified.checkpoint_sha256,
        metrics_sha256=verified.metrics_sha256,
        run_sha256=verified.run_sha256,
        verification_sha256=verification_sha256,
        optimizer_steps=verified.optimizer_steps,
        optimizer_steps_total=verified.optimizer_steps_total,
        specialist_acceptance=verified.specialist_acceptance,
        acceptance=verified.acceptance,
        test_opened=verified.test_opened,
        complete_marker_present=True,
        independent_verification_passed=verified.verification_passed,
        resumed_existing_run_surface=resumed_existing_run_surface,
    )


def run_verified_stage7d13_resumable_training(
    *, derivative_root: str | Path, output_root: str | Path,
    repository_root: str | Path, expected_repository_sha: str,
    resume_root: str | Path | None = None, heartbeat=print,
) -> Stage7D13ResumableAuthoritativeReceipt:
    expected = _git_sha(expected_repository_sha)
    repo = Path(repository_root)
    out = Path(output_root)
    head_before, origin_before = _repo_identity(repo)
    if head_before != expected:
        _fail("repository HEAD differs from explicitly authorized D13 head")
    verify_stage7c_runtime()

    preflight = verify_stage7d13_training_preflight(
        derivative_root, heartbeat=heartbeat
    )
    if (
        not preflight.preflight_passed
        or not preflight.collision_free
        or preflight.test_opened
    ):
        _fail("D13 preflight did not authorize optimizer creation")
    if heartbeat is not None:
        heartbeat(
            "D13 PREFLIGHT PASS | "
            f"records={preflight.record_count} | "
            f"params={preflight.parameter_count_total} | TEST=False"
        )
    if _repo_identity(repo) != (head_before, origin_before):
        _fail("repository identity changed during D13 preflight")
    verify_stage7c_runtime()

    names = (
        {p.name for p in out.iterdir()}
        if out.exists() and out.is_dir()
        else set()
    )
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        _fail("D13 final output root must be regular directory when present")

    if names == _COMPLETE_TOP:
        final_verified = verify_stage7d13_run(out, require_complete=True)
        if (
            final_verified.repository_sha != expected
            or not final_verified.verification_passed
        ):
            _fail("completed D13 recovery surface does not match authorized head")
        verification_sha = sha256(
            (out / "verification.json").read_bytes()
        ).hexdigest()
        if _repo_identity(repo) != (head_before, origin_before):
            _fail("repository identity changed during completed-run recovery")
        return _receipt(
            verified=final_verified,
            repository_origin=origin_before,
            preflight_record_count=preflight.record_count,
            preflight_parameter_count_total=preflight.parameter_count_total,
            verification_sha256=verification_sha,
            resumed_existing_run_surface=True,
        )

    training: Stage7D13TrainingReceipt | None = None
    resumed_surface = bool(names)
    if not names:
        training = run_stage7d13_resumable_training(
            derivative_root=derivative_root,
            output_root=out,
            repository_root=repo,
            expected_repository_sha=expected,
            resume_root=resume_root,
            heartbeat=heartbeat,
        )
        if _repo_identity(repo) != (head_before, origin_before):
            _fail("repository identity changed during D13 training")
        verify_stage7c_runtime()
        verified = verify_stage7d13_run(out, require_complete=False)
        _assert_training_match(training, verified)
    elif names == _BASE_TOP:
        verified = verify_stage7d13_run(out, require_complete=False)
        if heartbeat is not None:
            heartbeat(
                "D13 RECOVERY | persisted training surface reopened; "
                "skipping retraining"
            )
    elif names == _VERIFICATION_ONLY_TOP:
        verified = _verify_base_via_shadow(out)
        expected_verification_raw, expected_verification_sha = (
            _verification_payload(verified)
        )
        actual_verification_raw = (out / "verification.json").read_bytes()
        if actual_verification_raw != expected_verification_raw:
            _fail(
                "interrupted D13 verification.json does not match "
                "independent recomputation"
            )
        verification_sha = expected_verification_sha
        if verified.repository_sha != expected:
            _fail("recovered D13 run repository SHA mismatch")
        complete_payload = {
            "version": STAGE7D13_RESUMABLE_AUTHORITATIVE_VERSION,
            "run_id": verified.run_id,
            "verification_sha256": verification_sha,
            "acceptance": verified.acceptance,
            "test_opened": False,
        }
        _atomic_bytes(out / "COMPLETE", _canonical_json(complete_payload))
        final_verified = verify_stage7d13_run(out, require_complete=True)
        if _repo_identity(repo) != (head_before, origin_before):
            _fail("repository identity changed during D13 completion recovery")
        return _receipt(
            verified=final_verified,
            repository_origin=origin_before,
            preflight_record_count=preflight.record_count,
            preflight_parameter_count_total=preflight.parameter_count_total,
            verification_sha256=verification_sha,
            resumed_existing_run_surface=True,
        )
    else:
        _fail("D13 output root is in an unrecognized interrupted state")

    if verified.repository_sha != expected:
        _fail("D13 persisted run repository SHA mismatch")
    if verified.complete_marker_present or not verified.verification_passed:
        _fail("D13 independent persisted-run verification failed")

    verification_raw, verification_sha = _verification_payload(verified)
    verification_path = out / "verification.json"
    if verification_path.exists() or verification_path.is_symlink():
        _fail("D13 verification.json unexpectedly already exists")
    _atomic_bytes(verification_path, verification_raw)

    complete_payload = {
        "version": STAGE7D13_RESUMABLE_AUTHORITATIVE_VERSION,
        "run_id": verified.run_id,
        "verification_sha256": verification_sha,
        "acceptance": verified.acceptance,
        "test_opened": False,
    }
    complete_path = out / "COMPLETE"
    if complete_path.exists() or complete_path.is_symlink():
        _fail("D13 COMPLETE unexpectedly already exists")
    _atomic_bytes(complete_path, _canonical_json(complete_payload))

    final_verified = verify_stage7d13_run(out, require_complete=True)
    if (
        not final_verified.verification_passed
        or not final_verified.complete_marker_present
    ):
        _fail("completed D13 run did not independently reopen")
    if (
        final_verified.run_id != verified.run_id
        or final_verified.acceptance != verified.acceptance
    ):
        _fail("completed D13 run changed verification identity")
    if _repo_identity(repo) != (head_before, origin_before):
        _fail("repository identity changed during D13 completion verification")
    verify_stage7c_runtime()

    return _receipt(
        verified=final_verified,
        repository_origin=origin_before,
        preflight_record_count=preflight.record_count,
        preflight_parameter_count_total=preflight.parameter_count_total,
        verification_sha256=verification_sha,
        resumed_existing_run_surface=resumed_surface,
    )
