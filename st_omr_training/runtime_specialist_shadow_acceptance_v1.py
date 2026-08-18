"""Shadow-only acceptance contract for real specialist artifacts.

No checkpoint is bundled or loaded here.  This module freezes the externally
verified artifact identities, thresholds, quality evidence and fail-closed
admission decision before any real specialist is wired to the runtime resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Final


_HEX64 = re.compile(r"^[0-9a-f]{64}$")

D13_INPUT_SHAPE: Final[tuple[int, int, int]] = (1, 128, 512)
D13_DECODER_SCORE_THRESHOLD: Final[float] = 0.25
D13_LOCAL_MAX_KERNEL: Final[int] = 3
D13_TOP_K: Final[int] = 256
D13_CENTER_TOLERANCE_PX: Final[float] = 4.0
D13_BBOX_IOU_THRESHOLD: Final[float] = 0.50


@dataclass(frozen=True, slots=True)
class QualityGates:
    center_f1_min: float
    bbox_f1_min: float
    macro_f1_min: float

    def __post_init__(self) -> None:
        for value in (self.center_f1_min, self.bbox_f1_min, self.macro_f1_min):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError("quality gates must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("quality gates must be finite values in 0..1")


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    center_f1: float
    bbox_f1: float
    macro_f1: float

    def __post_init__(self) -> None:
        for value in (self.center_f1, self.bbox_f1, self.macro_f1):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError("quality evidence must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("quality evidence must be finite values in 0..1")


@dataclass(frozen=True, slots=True)
class RealCheckpointSmoke:
    artifact_sha256: str
    output_digest_sha256: str
    identical_runs: int
    required_runs: int = 10
    fail_closed_passed: int = 4
    fail_closed_required: int = 4

    def __post_init__(self) -> None:
        if _HEX64.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be lowercase SHA-256")
        if _HEX64.fullmatch(self.output_digest_sha256) is None:
            raise ValueError("output_digest_sha256 must be lowercase SHA-256")
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
class D13ShadowCandidate:
    specialist: str
    classes: tuple[str, ...]
    smoke: RealCheckpointSmoke
    gates: QualityGates
    evidence: QualityEvidence
    completed_epochs: int
    required_epochs: int

    def __post_init__(self) -> None:
        allowed = {
            "notehead": ("open", "filled"),
            "accidental": ("sharp", "flat", "natural"),
        }
        if self.specialist not in allowed or self.classes != allowed[self.specialist]:
            raise ValueError("D13 shadow candidate class vocabulary mismatch")
        for value in (self.completed_epochs, self.required_epochs):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("epoch counts must be positive integers")
        if self.completed_epochs > self.required_epochs:
            raise ValueError("completed epochs cannot exceed required epochs")

    @property
    def metric_pass(self) -> bool:
        return (
            self.evidence.center_f1 >= self.gates.center_f1_min
            and self.evidence.bbox_f1 >= self.gates.bbox_f1_min
            and self.evidence.macro_f1 >= self.gates.macro_f1_min
        )

    @property
    def shadow_decision(self) -> str:
        if not self.smoke.passed:
            return "REJECT"
        if not self.metric_pass or self.completed_epochs != self.required_epochs:
            return "HOLD"
        return "PASS"


@dataclass(frozen=True, slots=True)
class RestClassVerifier:
    class_name: str
    checkpoint_sha256: str
    proposal_threshold: float
    verifier_threshold: float
    recall: float
    false_positive_reduction: float
    final_gate_pass: bool
    production_promotion: bool
    test_opened: bool

    def __post_init__(self) -> None:
        if self.class_name not in {"half", "quarter", "eighth"}:
            raise ValueError("unsupported Rest verifier class")
        if _HEX64.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("Rest checkpoint SHA must be lowercase SHA-256")
        for name, value in (
            ("proposal_threshold", self.proposal_threshold),
            ("verifier_threshold", self.verifier_threshold),
            ("recall", self.recall),
            ("false_positive_reduction", self.false_positive_reduction),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in 0..1")


@dataclass(frozen=True, slots=True)
class RestR2ShadowCandidate:
    proposal_smoke: RealCheckpointSmoke
    verifiers: tuple[RestClassVerifier, ...]
    integrated_arbitration_ready: bool

    def __post_init__(self) -> None:
        names = tuple(item.class_name for item in self.verifiers)
        if names != ("half", "quarter", "eighth"):
            raise ValueError("Rest verifiers must be fixed half, quarter, eighth order")
        if len(names) != len(set(names)):
            raise ValueError("Rest verifier classes must be unique")

    @property
    def class_gates_pass(self) -> bool:
        return all(
            item.final_gate_pass and not item.test_opened
            for item in self.verifiers
        )

    @property
    def shadow_decision(self) -> str:
        if not self.proposal_smoke.passed or not self.class_gates_pass:
            return "REJECT"
        if not self.integrated_arbitration_ready:
            return "HOLD"
        return "PASS"


NOTEHEAD_SHADOW: Final[D13ShadowCandidate] = D13ShadowCandidate(
    specialist="notehead",
    classes=("open", "filled"),
    smoke=RealCheckpointSmoke(
        artifact_sha256="1783f10cb136d024422bb5b07a0689c0f24256e486e164a3c047a68334fc929f",
        output_digest_sha256="f2cb505c9979fcf463b04e59e7524b910205815dabbbbff44c997b5f436b4e61",
        identical_runs=10,
    ),
    gates=QualityGates(0.85, 0.75, 0.90),
    evidence=QualityEvidence(0.9898990, 0.9900896, 0.9885898),
    completed_epochs=9,
    required_epochs=10,
)

ACCIDENTAL_SHADOW: Final[D13ShadowCandidate] = D13ShadowCandidate(
    specialist="accidental",
    classes=("sharp", "flat", "natural"),
    smoke=RealCheckpointSmoke(
        artifact_sha256="39d8c812ab793aaf318bb92881f94d4d286bbe6783a6319e4c326759994e3bf9",
        output_digest_sha256="d5f7c37bd0ca578fee0d5309b1c1918b734fc5e36d9ceb0dcf1402f4aefbc954",
        identical_runs=10,
    ),
    gates=QualityGates(0.80, 0.70, 0.85),
    evidence=QualityEvidence(0.9037118, 0.9034379, 0.7938513),
    completed_epochs=5,
    required_epochs=10,
)

REST_R2_SHADOW: Final[RestR2ShadowCandidate] = RestR2ShadowCandidate(
    proposal_smoke=RealCheckpointSmoke(
        artifact_sha256="89dfe890961a42f13a8a2b29df4808649cefc2b20a9edf42b6107f6b75f3f35a",
        output_digest_sha256="6f27991728fcdf16f62cdaee3bb3db0eee5ca7fb3d2ccad92b299fdd83a55fb0",
        identical_runs=10,
    ),
    verifiers=(
        RestClassVerifier(
            "half",
            "69a4d16b5ab75594262544d5c6c025465738fa28907bec21b4d21daff89bf815",
            0.50,
            0.070689357817173,
            1.0,
            0.8125,
            True,
            False,
            False,
        ),
        RestClassVerifier(
            "quarter",
            "41d01db6ecbf8066b3b73966545611b0a73b39ddf7022631afcc57b9acebb5d5",
            0.20,
            0.3782260715961457,
            0.9890710382513661,
            0.7523689197725837,
            True,
            False,
            False,
        ),
        RestClassVerifier(
            "eighth",
            "60a9ae6c8081cb8f8d47f155d8a31a6bcc6b0ea5d67231e6d7c76bf5de499668",
            0.50,
            0.5620679259300232,
            0.9856115107913669,
            0.792713567839196,
            True,
            False,
            False,
        ),
    ),
    integrated_arbitration_ready=False,
)


def resolver_connection_allowed() -> bool:
    """This package is shadow-only; real Resolver wiring is explicitly forbidden."""
    return False


def fail_closed_observation_status(
    *,
    score: float,
    threshold: float,
    bbox_finite: bool,
    class_conflict: bool = False,
) -> str:
    """Map an adapter candidate to accepted/ambiguous/rejected without guessing."""
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return "rejected"
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return "rejected"
    score = float(score)
    threshold = float(threshold)
    if not math.isfinite(score) or not math.isfinite(threshold):
        return "rejected"
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= score <= 1.0:
        return "rejected"
    if not bbox_finite:
        return "rejected"
    if class_conflict:
        return "ambiguous"
    if score < threshold:
        return "ambiguous"
    return "accepted"
