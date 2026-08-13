"""Canonical Stage 8-3A auxiliary target-preparation pipeline.

External Stage 8-3A orchestration must use this entrypoint rather than lower
parser/adapter helpers. It independently freezes the one-staff/five-line/G2/
key-0 MEI shape, then delegates accidental and conversion checks to the guarded
adapter boundary. Success is target preparation only, never Stage 8 admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import xml.etree.ElementTree as ET
from typing import Final

from .musicxml_validator import MAX_MUSICXML_BYTES
from .stage8_auxiliary_admission_gate import (
    GuardedAuxiliaryMusicXMLEvidence,
    Stage8AuxiliaryAdmissionGateError,
    adapt_guarded_primus_v1_to_musicxml,
    auxiliary_admission_gate_policy_fingerprint,
)


STAGE8_AUXILIARY_PIPELINE_VERSION: Final[str] = "st-stage8-auxiliary-pipeline-v1"
_XML_DECL_MARKERS: Final[tuple[bytes, ...]] = (
    b"<!" + b"DOC" + b"TYPE",
    b"<!" + b"ENT" + b"ITY",
)
_HEX = frozenset("0123456789abcdef")


class Stage8AuxiliaryPipelineError(ValueError):
    """Raised when the canonical auxiliary target-preparation pipeline vetoes."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def auxiliary_pipeline_policy_fingerprint() -> str:
    payload = {
        "pipeline_version": STAGE8_AUXILIARY_PIPELINE_VERSION,
        "guard_policy_fingerprint": auxiliary_admission_gate_policy_fingerprint(),
        "shape": "one-explicit-staffdef-n1-lines5-g2-explicit-key0",
        "xml_preflight": "reject-declarative-surface-before-shape-parse",
        "result": "validated-supported-v1-musicxml-plus-hash-only-evidence",
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _verify_exact_v1_mei_shape(mei: bytes) -> None:
    if not mei or len(mei) > MAX_MUSICXML_BYTES:
        raise Stage8AuxiliaryPipelineError("MEI bytes are outside the pipeline bounds")
    upper = mei.upper()
    if any(marker in upper for marker in _XML_DECL_MARKERS):
        raise Stage8AuxiliaryPipelineError(
            "declarative XML surface is forbidden before MEI shape verification"
        )
    try:
        root = ET.fromstring(mei)
    except ET.ParseError as exc:
        raise Stage8AuxiliaryPipelineError("MEI is not well-formed XML") from exc
    if _local_name(root.tag) != "mei":
        raise Stage8AuxiliaryPipelineError("auxiliary XML root must be MEI")

    score_defs = [node for node in root.iter() if _local_name(node.tag) == "scoreDef"]
    staff_defs = [node for node in root.iter() if _local_name(node.tag) == "staffDef"]
    if len(score_defs) != 1 or len(staff_defs) != 1:
        raise Stage8AuxiliaryPipelineError(
            "supported V1 requires exactly one scoreDef and one staffDef"
        )
    score_def = score_defs[0]
    staff_def = staff_defs[0]
    if staff_def.attrib.get("n") != "1":
        raise Stage8AuxiliaryPipelineError("supported V1 staffDef must explicitly be staff 1")
    if staff_def.attrib.get("lines") != "5":
        raise Stage8AuxiliaryPipelineError("supported V1 staff must explicitly have five lines")
    if staff_def.attrib.get("clef.shape") != "G" or staff_def.attrib.get("clef.line") != "2":
        raise Stage8AuxiliaryPipelineError("supported V1 clef must explicitly be G2")
    if score_def.attrib.get("key.sig") != "0":
        raise Stage8AuxiliaryPipelineError("supported V1 key signature must explicitly be zero")


@dataclass(frozen=True, slots=True)
class Stage8AuxiliaryTargetEvidence:
    mei_sha256: str
    semantic_sha256: str
    musicxml_sha256: str
    score_id: str
    measure_count: int
    note_count: int
    rest_count: int
    guarded_adapter_policy_fingerprint: str
    pipeline_policy_fingerprint: str
    pipeline_version: str = STAGE8_AUXILIARY_PIPELINE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "mei_sha256",
            "semantic_sha256",
            "musicxml_sha256",
            "guarded_adapter_policy_fingerprint",
            "pipeline_policy_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in _HEX for ch in value)
            ):
                raise Stage8AuxiliaryPipelineError(f"{name} must be lowercase SHA-256")
        if self.pipeline_policy_fingerprint != auxiliary_pipeline_policy_fingerprint():
            raise Stage8AuxiliaryPipelineError("auxiliary pipeline fingerprint mismatch")
        if self.pipeline_version != STAGE8_AUXILIARY_PIPELINE_VERSION:
            raise Stage8AuxiliaryPipelineError("auxiliary pipeline version mismatch")


def prepare_guarded_primus_v1_target(
    *,
    mei_bytes: object,
    semantic_bytes: object,
) -> tuple[bytes, Stage8AuxiliaryTargetEvidence]:
    """Produce a guarded supported-V1 MusicXML target and hash-only evidence."""

    if not isinstance(mei_bytes, bytes):
        raise Stage8AuxiliaryPipelineError("MEI must be bytes")
    _verify_exact_v1_mei_shape(mei_bytes)
    try:
        musicxml, guarded = adapt_guarded_primus_v1_to_musicxml(
            mei_bytes=mei_bytes,
            semantic_bytes=semantic_bytes,
        )
    except Stage8AuxiliaryAdmissionGateError as exc:
        raise Stage8AuxiliaryPipelineError(
            "guarded auxiliary adapter rejected the package"
        ) from exc

    assert isinstance(guarded, GuardedAuxiliaryMusicXMLEvidence)
    evidence = Stage8AuxiliaryTargetEvidence(
        mei_sha256=guarded.mei_sha256,
        semantic_sha256=guarded.semantic_sha256,
        musicxml_sha256=guarded.musicxml_sha256,
        score_id=guarded.score_id,
        measure_count=guarded.measure_count,
        note_count=guarded.note_count,
        rest_count=guarded.rest_count,
        guarded_adapter_policy_fingerprint=guarded.gate_policy_fingerprint,
        pipeline_policy_fingerprint=auxiliary_pipeline_policy_fingerprint(),
    )
    return musicxml, evidence
