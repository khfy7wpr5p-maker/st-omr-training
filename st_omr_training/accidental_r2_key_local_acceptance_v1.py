"""Shadow-only deterministic evaluator for Accidental R2 key/local acceptance v1.

This module executes the frozen acceptance contract without opening sealed TEST,
loading checkpoints, changing training, or wiring the production Resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

SYMBOLS: Final[set[str]] = {
    "sharp", "flat", "natural", "double_sharp", "double_flat"
}
LOCAL_ALTER: Final[dict[str, int]] = {
    "sharp": 1,
    "flat": -1,
    "natural": 0,
    "double_sharp": 2,
    "double_flat": -2,
}
SHARP_ORDER: Final[tuple[str, ...]] = ("F#", "C#", "G#", "D#", "A#", "E#", "B#")
FLAT_ORDER: Final[tuple[str, ...]] = ("Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb")
POSSIBLE_KEYS: Final[dict[int, tuple[str, str]]] = {
    0: ("C major", "A minor"),
    1: ("G major", "E minor"),
    2: ("D major", "B minor"),
    3: ("A major", "F# minor"),
    4: ("E major", "C# minor"),
    5: ("B major", "G# minor"),
    6: ("F# major", "D# minor"),
    7: ("C# major", "A# minor"),
    -1: ("F major", "D minor"),
    -2: ("Bb major", "G minor"),
    -3: ("Eb major", "C minor"),
    -4: ("Ab major", "F minor"),
    -5: ("Db major", "Bb minor"),
    -6: ("Gb major", "Eb minor"),
    -7: ("Cb major", "Ab minor"),
}


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    symbols: tuple[str, ...]
    location: str  # staff_start | measure
    canonical_key_order: bool = True
    canonical_staff_positions: bool = True
    near_first_note: bool = False
    unique_staff: bool = True
    local_targets: tuple[str, ...] = ()
    confidence: float = 0.95
    bbox_valid: bool = True
    coordinates_finite: bool = True
    cancellation_context: bool = False

    def __post_init__(self) -> None:
        if not self.symbols or any(symbol not in SYMBOLS for symbol in self.symbols):
            raise ValueError("unsupported or empty accidental evidence")
        if self.location not in {"staff_start", "measure"}:
            raise ValueError("unsupported evidence location")


@dataclass(frozen=True, slots=True)
class KeySignatureResult:
    fifths: int
    accidentals: tuple[str, ...]
    possible_keys: tuple[str, str]
    mode: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class LocalAlteration:
    target: str
    alter: int


@dataclass(frozen=True, slots=True)
class AcceptanceOutcome:
    contexts: tuple[str, ...]
    key_signature: KeySignatureResult | None = None
    local_alterations: tuple[LocalAlteration, ...] = ()


def _is_finite_group(group: EvidenceGroup) -> bool:
    return (
        isinstance(group.confidence, (int, float))
        and not isinstance(group.confidence, bool)
        and math.isfinite(float(group.confidence))
        and 0.0 <= float(group.confidence) <= 1.0
        and group.coordinates_finite
        and group.bbox_valid
    )


def _route(group: EvidenceGroup) -> str:
    if not _is_finite_group(group):
        return "REJECTED"
    if not group.unique_staff:
        return "AMBIGUOUS"

    if group.location == "staff_start":
        if group.near_first_note:
            return "AMBIGUOUS"
        if group.symbols == ("natural",) and not group.cancellation_context:
            return "AMBIGUOUS"
        if not group.canonical_key_order or not group.canonical_staff_positions:
            return "AMBIGUOUS"
        if all(symbol == "sharp" for symbol in group.symbols):
            return "KEY_SIGNATURE"
        if all(symbol == "flat" for symbol in group.symbols):
            return "KEY_SIGNATURE"
        return "AMBIGUOUS"

    # measure-local evidence
    if len(group.symbols) != 1:
        return "AMBIGUOUS"
    if len(group.local_targets) == 1:
        return "LOCAL_ACCIDENTAL"
    if len(group.local_targets) > 1:
        return "AMBIGUOUS_TARGET"
    return "AMBIGUOUS_TARGET"


def _compose_key_signature(group: EvidenceGroup) -> KeySignatureResult:
    count = len(group.symbols)
    if not 1 <= count <= 7:
        raise ValueError("key signature accidental count must be 1..7")
    if all(symbol == "sharp" for symbol in group.symbols):
        fifths = count
        accidentals = SHARP_ORDER[:count]
    elif all(symbol == "flat" for symbol in group.symbols):
        fifths = -count
        accidentals = FLAT_ORDER[:count]
    else:
        raise ValueError("key signature must contain one accidental family")
    return KeySignatureResult(
        fifths=fifths,
        accidentals=accidentals,
        possible_keys=POSSIBLE_KEYS[fifths],
    )


def evaluate(groups: tuple[EvidenceGroup, ...]) -> AcceptanceOutcome:
    """Evaluate already-observed accidental evidence using only frozen rules."""
    if not groups:
        raise ValueError("at least one evidence group is required")

    routes = tuple(_route(group) for group in groups)
    terminal = tuple(route for route in routes if route in {"REJECTED", "AMBIGUOUS", "AMBIGUOUS_TARGET"})
    if terminal:
        # Fail closed: any unresolved group prevents musical mutation.
        priority = "REJECTED" if "REJECTED" in terminal else (
            "AMBIGUOUS_TARGET" if "AMBIGUOUS_TARGET" in terminal else "AMBIGUOUS"
        )
        return AcceptanceOutcome(contexts=(priority,))

    key_groups = tuple(group for group, route in zip(groups, routes) if route == "KEY_SIGNATURE")
    local_groups = tuple(group for group, route in zip(groups, routes) if route == "LOCAL_ACCIDENTAL")

    if len(key_groups) > 1:
        return AcceptanceOutcome(contexts=("AMBIGUOUS",))

    key_result = _compose_key_signature(key_groups[0]) if key_groups else None
    local_results = tuple(
        LocalAlteration(target=group.local_targets[0], alter=LOCAL_ALTER[group.symbols[0]])
        for group in local_groups
    )

    contexts: list[str] = []
    if key_result is not None:
        contexts.append("KEY_SIGNATURE")
    if local_results:
        contexts.append("LOCAL_ACCIDENTAL")
    return AcceptanceOutcome(
        contexts=tuple(contexts),
        key_signature=key_result,
        local_alterations=local_results,
    )


def acceptance_cases() -> dict[str, tuple[tuple[EvidenceGroup, ...], AcceptanceOutcome]]:
    """Return all 22 frozen cases from ACCIDENTAL_R2_KEY_LOCAL_ACCEPTANCE_V1.md."""
    KS = KeySignatureResult
    LA = LocalAlteration
    AO = AcceptanceOutcome
    EG = EvidenceGroup

    return {
        "K01": ((EG(("sharp",), "staff_start"),), AO(("KEY_SIGNATURE",), KS(1, ("F#",), ("G major", "E minor")))),
        "K02": ((EG(("sharp", "sharp"), "staff_start"),), AO(("KEY_SIGNATURE",), KS(2, ("F#", "C#"), ("D major", "B minor")))),
        "K03": ((EG(("sharp", "sharp", "sharp"), "staff_start"),), AO(("KEY_SIGNATURE",), KS(3, ("F#", "C#", "G#"), ("A major", "F# minor")))),
        "K04": ((EG(("flat",), "staff_start"),), AO(("KEY_SIGNATURE",), KS(-1, ("Bb",), ("F major", "D minor")))),
        "K05": ((EG(("flat", "flat"), "staff_start"),), AO(("KEY_SIGNATURE",), KS(-2, ("Bb", "Eb"), ("Bb major", "G minor")))),
        "K06": ((EG(("flat", "flat", "flat"), "staff_start"),), AO(("KEY_SIGNATURE",), KS(-3, ("Bb", "Eb", "Ab"), ("Eb major", "C minor")))),
        "L01": ((EG(("sharp",), "measure", local_targets=("note_1",)),), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", 1),))),
        "L02": ((EG(("flat",), "measure", local_targets=("note_1",)),), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", -1),))),
        "L03": ((EG(("natural",), "measure", local_targets=("note_1",)),), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", 0),))),
        "L04": ((EG(("double_sharp",), "measure", local_targets=("note_1",)),), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", 2),))),
        "L05": ((EG(("double_flat",), "measure", local_targets=("note_1",)),), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", -2),))),
        "L06": ((EG(("sharp",), "staff_start"), EG(("natural",), "measure", local_targets=("F",))), AO(("KEY_SIGNATURE", "LOCAL_ACCIDENTAL"), KS(1, ("F#",), ("G major", "E minor")), (LA("F", 0),))),
        "L07": ((EG(("flat", "flat"), "staff_start"), EG(("sharp",), "measure", local_targets=("note_1",))), AO(("KEY_SIGNATURE", "LOCAL_ACCIDENTAL"), KS(-2, ("Bb", "Eb"), ("Bb major", "G minor")), (LA("note_1", 1),))),
        "L08": ((EG(("sharp", "sharp"), "staff_start"), EG(("flat",), "measure", local_targets=("note_1",))), AO(("KEY_SIGNATURE", "LOCAL_ACCIDENTAL"), KS(2, ("F#", "C#"), ("D major", "B minor")), (LA("note_1", -1),))),
        "L09": ((EG(("sharp",), "measure", local_targets=("note_1",)), EG(("sharp",), "measure", local_targets=("note_2",))), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", 1), LA("note_2", 1)))),
        "L10": ((EG(("double_sharp",), "measure", local_targets=("note_1",)),), AO(("LOCAL_ACCIDENTAL",), local_alterations=(LA("note_1", 2),))),
        "A01": ((EG(("sharp",), "staff_start", near_first_note=True),), AO(("AMBIGUOUS",))),
        "A02": ((EG(("sharp", "sharp"), "staff_start", canonical_staff_positions=False),), AO(("AMBIGUOUS",))),
        "A03": ((EG(("natural",), "staff_start", cancellation_context=False),), AO(("AMBIGUOUS",))),
        "A04": ((EG(("sharp",), "staff_start", unique_staff=False),), AO(("AMBIGUOUS",))),
        "A05": ((EG(("flat",), "measure", local_targets=("note_1", "note_2")),), AO(("AMBIGUOUS_TARGET",))),
        "A06": ((EG(("sharp",), "measure", local_targets=("note_1",), confidence=math.nan),), AO(("REJECTED",))),
    }


def resolver_connection_allowed() -> bool:
    """This shadow acceptance package never authorizes runtime wiring."""
    return False
