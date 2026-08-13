"""Guarded Stage 8-3A auxiliary-to-MusicXML boundary.

This module is the only Stage 8-3A entrypoint allowed to promote a supported
PrIMuS-style MEI/semantic pair toward MusicXML preparation. It adds a separate
fail-closed preflight in front of the lower-level adapter: unsafe declarative
XML surfaces are rejected and visible accidental glyphs are corroborated
against gestural/sounding accidental state before conversion.

A successful return is preparation evidence only, never data admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import xml.etree.ElementTree as ET
from typing import Final

from .musicxml_validator import MAX_MUSICXML_BYTES
from .stage8_primus_adapter import (
    PrimusV1AdapterError,
    adapt_primus_v1_to_musicxml,
    primus_v1_adapter_policy_fingerprint,
)


STAGE8_AUXILIARY_ADMISSION_GATE_VERSION: Final[str] = (
    "st-stage8-auxiliary-admission-gate-v1"
)
_XML_DECL_MARKERS: Final[tuple[bytes, ...]] = (
    b"<!" + b"DOC" + b"TYPE",
    b"<!" + b"ENT" + b"ITY",
)
_ACCIDENTAL_TO_ALTER: Final[dict[str, int]] = {"s": 1, "f": -1, "n": 0}
_HEX = frozenset("0123456789abcdef")


class Stage8AuxiliaryAdmissionGateError(ValueError):
    """Raised when an auxiliary package fails the guarded Stage 8-3A boundary."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def auxiliary_admission_gate_policy_fingerprint() -> str:
    payload = {
        "gate_version": STAGE8_AUXILIARY_ADMISSION_GATE_VERSION,
        "lower_level_adapter_fingerprint": primus_v1_adapter_policy_fingerprint(),
        "xml_preflight": "reject-declarative-doctype-entity-surface-before-parse",
        "visible_accidental_source": "accid-attribute-or-child-accid",
        "gestural_accidental_source": "accid.ges-attribute-or-child-accid.ges",
        "sounding_precedence": "gestural-then-visible-then-key0-measure-state",
        "visible_policy": "exact-required-state-transition-no-redundant-courtesy-glyph",
        "measure_accidental_state": "reset-each-measure-key0",
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_mei_bytes(value: object) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise Stage8AuxiliaryAdmissionGateError("MEI must be non-empty bytes")
    if len(value) > MAX_MUSICXML_BYTES:
        raise Stage8AuxiliaryAdmissionGateError("MEI exceeds the Stage 8-3A byte limit")
    return value


def _preflight_xml(mei: bytes) -> None:
    upper = mei.upper()
    if any(marker in upper for marker in _XML_DECL_MARKERS):
        raise Stage8AuxiliaryAdmissionGateError(
            "declarative/external XML constructs are forbidden before auxiliary parsing"
        )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _single_accidental(values: list[str], *, label: str) -> int | None:
    if not values:
        return None
    converted: list[int] = []
    for value in values:
        try:
            converted.append(_ACCIDENTAL_TO_ALTER[value])
        except KeyError as exc:
            raise Stage8AuxiliaryAdmissionGateError(
                f"unsupported {label} accidental at guarded auxiliary boundary"
            ) from exc
    if len(set(converted)) != 1:
        raise Stage8AuxiliaryAdmissionGateError(
            f"conflicting {label} accidental values at guarded auxiliary boundary"
        )
    return converted[0]


def _note_accidentals(note: ET.Element) -> tuple[int | None, int | None]:
    visible_values: list[str] = []
    gestural_values: list[str] = []

    visible = note.attrib.get("accid")
    gestural = note.attrib.get("accid.ges")
    if visible is not None:
        visible_values.append(visible)
    if gestural is not None:
        gestural_values.append(gestural)

    for child in note:
        if _local_name(child.tag) != "accid":
            continue
        child_visible = child.attrib.get("accid")
        child_gestural = child.attrib.get("accid.ges")
        if child_visible is None and child_gestural is None:
            raise Stage8AuxiliaryAdmissionGateError(
                "accid element has no supported visible or gestural value"
            )
        if child_visible is not None:
            visible_values.append(child_visible)
        if child_gestural is not None:
            gestural_values.append(child_gestural)

    visible_alter = _single_accidental(visible_values, label="visible")
    gestural_alter = _single_accidental(gestural_values, label="gestural")
    if (
        visible_alter is not None
        and gestural_alter is not None
        and visible_alter != gestural_alter
    ):
        raise Stage8AuxiliaryAdmissionGateError(
            "visible and gestural accidental values disagree"
        )
    return visible_alter, gestural_alter


def _verify_visible_accidental_transitions(root: ET.Element) -> None:
    measures = [node for node in root.iter() if _local_name(node.tag) == "measure"]
    if not measures:
        raise Stage8AuxiliaryAdmissionGateError("MEI contains no measures")

    for measure in measures:
        state: dict[tuple[str, int], int] = {}
        notes = [node for node in measure.iter() if _local_name(node.tag) == "note"]
        for note in notes:
            pname = note.attrib.get("pname")
            octave_text = note.attrib.get("oct")
            if pname not in tuple("abcdefg") or octave_text is None or not octave_text.isdigit():
                raise Stage8AuxiliaryAdmissionGateError(
                    "MEI note pitch is not safe for accidental-glyph corroboration"
                )
            octave = int(octave_text)
            if not 0 <= octave <= 9:
                raise Stage8AuxiliaryAdmissionGateError(
                    "MEI note octave is outside guarded V1 bounds"
                )

            position = (pname.upper(), octave)
            previous = state.get(position, 0)
            visible, gestural = _note_accidentals(note)
            sounding = (
                gestural
                if gestural is not None
                else visible
                if visible is not None
                else previous
            )
            required_visible = None if sounding == previous else sounding
            if visible != required_visible:
                if required_visible is None:
                    raise Stage8AuxiliaryAdmissionGateError(
                        "redundant/courtesy visible accidental is outside frozen V1"
                    )
                if visible is None:
                    raise Stage8AuxiliaryAdmissionGateError(
                        "sounding accidental transition lacks a corroborating visible glyph"
                    )
                raise Stage8AuxiliaryAdmissionGateError(
                    "visible accidental does not match the required sounding transition"
                )
            state[position] = sounding


def _guard_mei(mei: bytes) -> None:
    _preflight_xml(mei)
    try:
        root = ET.fromstring(mei)
    except ET.ParseError as exc:
        raise Stage8AuxiliaryAdmissionGateError("MEI is not well-formed XML") from exc
    if _local_name(root.tag) != "mei":
        raise Stage8AuxiliaryAdmissionGateError("auxiliary XML root must be MEI")
    _verify_visible_accidental_transitions(root)


@dataclass(frozen=True, slots=True)
class GuardedAuxiliaryMusicXMLEvidence:
    mei_sha256: str
    semantic_sha256: str
    musicxml_sha256: str
    score_id: str
    measure_count: int
    note_count: int
    rest_count: int
    lower_level_adapter_policy_fingerprint: str
    gate_policy_fingerprint: str
    gate_version: str = STAGE8_AUXILIARY_ADMISSION_GATE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "mei_sha256",
            "semantic_sha256",
            "musicxml_sha256",
            "lower_level_adapter_policy_fingerprint",
            "gate_policy_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in _HEX for ch in value)
            ):
                raise Stage8AuxiliaryAdmissionGateError(
                    f"{name} must be lowercase SHA-256"
                )
        if self.gate_policy_fingerprint != auxiliary_admission_gate_policy_fingerprint():
            raise Stage8AuxiliaryAdmissionGateError("guard policy fingerprint mismatch")
        if self.gate_version != STAGE8_AUXILIARY_ADMISSION_GATE_VERSION:
            raise Stage8AuxiliaryAdmissionGateError("guard version mismatch")


def adapt_guarded_primus_v1_to_musicxml(
    *,
    mei_bytes: object,
    semantic_bytes: object,
) -> tuple[bytes, GuardedAuxiliaryMusicXMLEvidence]:
    """Guard one auxiliary pair, then invoke the lower-level V1 adapter.

    This is the canonical Stage 8-3A auxiliary conversion entrypoint. Success
    does not establish rights, image pairing, Stage 8-0/8-1 admission, duplicate
    clearance, split assignment, or training eligibility.
    """

    mei = _require_mei_bytes(mei_bytes)
    _guard_mei(mei)
    try:
        musicxml, lower = adapt_primus_v1_to_musicxml(
            mei_bytes=mei,
            semantic_bytes=semantic_bytes,
        )
    except PrimusV1AdapterError as exc:
        raise Stage8AuxiliaryAdmissionGateError(
            "lower-level PrIMuS V1 adapter rejected the guarded package"
        ) from exc

    evidence = GuardedAuxiliaryMusicXMLEvidence(
        mei_sha256=lower.mei_sha256,
        semantic_sha256=lower.semantic_sha256,
        musicxml_sha256=lower.musicxml_sha256,
        score_id=lower.score_id,
        measure_count=lower.measure_count,
        note_count=lower.note_count,
        rest_count=lower.rest_count,
        lower_level_adapter_policy_fingerprint=lower.policy_fingerprint,
        gate_policy_fingerprint=auxiliary_admission_gate_policy_fingerprint(),
    )
    return musicxml, evidence
