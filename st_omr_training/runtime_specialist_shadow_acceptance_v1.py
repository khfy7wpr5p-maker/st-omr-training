"""Shadow-only acceptance contract for real specialist artifacts.

No checkpoint is bundled or loaded here. This file freezes artifact/evidence
identities and fail-closed rules before any real specialist is wired to the
runtime Resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Final

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

D13_INPUT_SHAPE: Final[tuple[int, int, int]] = (1, 128, 512)
D13_OUTPUT_STRIDE: Final[int] = 4
D13_DECODER_SCORE_THRESHOLD: Final[float] = 0.25
D13_LOCAL_MAX_KERNEL: Final[int] = 3
D13_TOP_K: Final[int] = 256
D13_CENTER_TOLERANCE_PX: Final[float] = 4.0
D13_BBOX_IOU_THRESHOLD: Final[float] = 0.50


def _unit(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in 0..1")
    return result


def _sha(name: str, value: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class RealCheckpointSmoke:
    artifact_sha256: str
    output_digest_sha256: str
    identical_runs: int
    required_runs: int = 10
    fail_closed_passed: int = 4
    fail_closed_required: int = 4

    def __post_init__(self) -> None:
        _sha("artifact_sha256", self.artifact_sha256)
        _sha("output_digest_sha256", self.output_digest_sha256)
        for name, value in (
            ("identical_runs", self.identical_runs),
            ("required_runs", self.required_runs),
            ("fail_closed_passed", self.fail_closed_passed),
            ("fail_closed_required", self.fail_closed_required),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def passed(self) -> bool:
        return (
            self.identical_runs == self.required_runs
            and self.fail_closed_passed == self.fail_closed_required
        )


@dataclass(frozen=True, slots=True)
class D13Candidate:
    specialist: str
    classes: tuple[str, ...]
    smoke: RealCheckpointSmoke
    center_f1: float
    bbox_f1: float
    macro_f1: float
    center_gate: float
    bbox_gate: float
    macro_gate: float
    completed_epochs: int
    required_epochs: int

    def __post_init__(self) -> None:
        allowed = {
            "notehead": ("open", "filled"),
            "accidental": ("sharp", "flat", "natural"),
        }
        if self.specialist not in allowed or self.classes != allowed[self.specialist]:
            raise ValueError("D13 specialist/class vocabulary mismatch")
        for name in (
            "center_f1", "bbox_f1", "macro_f1",
            "center_gate", "bbox_gate", "macro_gate",
        ):
            _unit(name, getattr(self, name))
        for value in (self.completed_epochs, self.required_epochs):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("epoch counts must be positive integers")
        if self.completed_epochs > self.required_epochs:
            raise ValueError("completed epochs exceed frozen profile")

    @property
    def metrics_pass(self) -> bool:
        return (
            self.center_f1 >= self.center_gate
            and self.bbox_f1 >= self.bbox_gate
            and self.macro_f1 >= self.macro_gate
        )

    @property
    def shadow_decision(self) -> str:
        if not self.smoke.passed:
            return "REJECT"
        if not self.metrics_pass or self.completed_epochs != self.required_epochs:
            return "HOLD"
        return "PASS"


@dataclass(frozen=True, slots=True)
class RestVerifierEvidence:
    class_name: str
    artifact_kind: str
    artifact_sha256: str
    proposal_threshold: float
    verifier_threshold: float
    recall: float
    false_positive_reduction: float
    final_gate_pass: bool
    production_promotion: bool
    test_opened: bool

    def __post_init__(self) -> None:
        if self.class_name not in {"half", "quarter", "eighth"}:
            raise ValueError("unsupported Rest class")
        if self.artifact_kind not in {"checkpoint", "closure_fingerprint"}:
            raise ValueError("unsupported Rest artifact kind")
        _sha("artifact_sha256", self.artifact_sha256)
        for name in (
            "proposal_threshold", "verifier_threshold", "recall",
            "false_positive_reduction",
        ):
            _unit(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class RestR2Candidate:
    proposal_smoke: RealCheckpointSmoke
    verifiers: tuple[RestVerifierEvidence, ...]
    integrated_arbitration_ready: bool

    def __post_init__(self) -> None:
        names = tuple(item.class_name for item in self.verifiers)
        if names != ("half", "quarter", "eighth"):
            raise ValueError("Rest verifiers must be ordered half, quarter, eighth")
        if len(names) != len(set(names)):
            raise ValueError("Rest verifier classes must be unique")

    @property
    def class_gates_pass(self) -> bool:
        return all(item.final_gate_pass and not item.test_opened for item in self.verifiers)

    @property
    def shadow_decision(self) -> str:
        if not self.proposal_smoke.passed or not self.class_gates_pass:
            return "REJECT"
        if not self.integrated_arbitration_ready:
            return "HOLD"
        return "PASS"


NOTEHEAD_SHADOW: Final[D13Candidate] = D13Candidate(
    specialist="notehead",
    classes=("open", "filled"),
    smoke=RealCheckpointSmoke(
        artifact_sha256="f34de0a6a3627421ea2c7e0f23d007de94c67576e16366183b6d60b96c14a106",
        output_digest_sha256="1d78abb3923ef49a565b9b14b38919f6e604648b07915ff8251a4cf160176968",
        identical_runs=10,
    ),
    center_f1=0.9882845985331936,
    bbox_f1=0.9882845985331936,
    macro_f1=0.9855884316180947,
    center_gate=0.85,
    bbox_gate=0.75,
    macro_gate=0.90,
    completed_epochs=10,
    required_epochs=10,
)

ACCIDENTAL_SHADOW: Final[D13Candidate] = D13Candidate(
    specialist="accidental",
    classes=("sharp", "flat", "natural"),
    smoke=RealCheckpointSmoke(
        artifact_sha256="39d8c812ab793aaf318bb92881f94d4d286bbe6783a6319e4c326759994e3bf9",
        output_digest_sha256="d5f7c37bd0ca578fee0d5309b1c1918b734fc5e36d9ceb0dcf1402f4aefbc954",
        identical_runs=10,
    ),
    center_f1=0.9037118,
    bbox_f1=0.9034379,
    macro_f1=0.7938513,
    center_gate=0.80,
    bbox_gate=0.70,
    macro_gate=0.85,
    completed_epochs=5,
    required_epochs=10,
)

REST_R2_SHADOW: Final[RestR2Candidate] = RestR2Candidate(
    proposal_smoke=RealCheckpointSmoke(
        artifact_sha256="89dfe890961a42f13a8a2b29df4808649cefc2b20a9edf42b6107f6b75f3f35a",
        output_digest_sha256="6f27991728fcdf16f62cdaee3bb3db0eee5ca7fb3d2ccad92b299fdd83a55fb0",
        identical_runs=10,
    ),
    verifiers=(
        RestVerifierEvidence(
            "half", "closure_fingerprint",
            "69a4d16b5ab75594262544d5c6c025465738fa28907bec21b4d21daff89bf815",
            0.50, 0.070689357817173, 1.0, 0.8125, True, False, False,
        ),
        RestVerifierEvidence(
            "quarter", "checkpoint",
            "41d01db6ecbf8066b3b73966545611b0a73b39ddf7022631afcc57b9acebb5d5",
            0.20, 0.3782260715961457, 0.9890710382513661,
            0.7523689197725837, True, False, False,
        ),
        RestVerifierEvidence(
            "eighth", "checkpoint",
            "60a9ae6c8081cb8f8d47f155d8a31a6bcc6b0ea5d67231e6d7c76bf5de499668",
            0.50, 0.5620679259300232, 0.9856115107913669,
            0.792713567839196, True, False, False,
        ),
    ),
    integrated_arbitration_ready=True,
)


def resolver_connection_allowed() -> bool:
    """Shadow package never authorizes real Resolver wiring."""
    return False


def fail_closed_observation_status(
    *, score: float, threshold: float, bbox_finite: bool, class_conflict: bool = False
) -> str:
    """Return accepted only for finite, above-threshold, non-conflicting evidence."""
    try:
        score_value = _unit("score", score)
        threshold_value = _unit("threshold", threshold)
    except (TypeError, ValueError):
        return "rejected"
    if not bbox_finite:
        return "rejected"
    if class_conflict or score_value < threshold_value:
        return "ambiguous"
    return "accepted"
