"""Additive ScoreGraph V2 contract over the frozen Polyphonic Representation V2 core.

The graph envelope does not replace or mutate PolyScore. It adds deterministic
node views, source-order navigation evidence, semantic relations that are not
part of the frozen PolyScore surface, and capability outcomes for adapters.

MusicXML parsing, normalization, dataset admission, training and TEST access are
explicitly outside this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Final

from .polyphonic_representation import ExactRational, PolyScore


SCOREGRAPH_V2_VERSION: Final[str] = "st-omr-scoregraph-v2-envelope-v1"


class ScoreGraphV2Error(ValueError):
    """Raised when the additive ScoreGraph V2 contract is violated."""


class CapabilityOutcome(str, Enum):
    SUPPORTED = "supported"
    REVIEW_REQUIRED = "review-required"
    BLOCKED = "blocked"


class BlockReason(str, Enum):
    SAFETY = "safety"
    UNPARSEABLE = "unparseable"
    BOUNDEDNESS = "boundedness"


class SourceNavigationKind(str, Enum):
    BACKUP = "backup"
    FORWARD = "forward"


class GraphAnchorKind(str, Enum):
    PART = "part"
    STAFF = "staff"
    VOICE = "voice"
    MEASURE = "measure"
    EVENT = "event"
    NOTEHEAD = "notehead"


class RelationKind(str, Enum):
    TIE = "tie"
    SLUR = "slur"
    BEAM = "beam"
    TUPLET = "tuplet"
    CHORD_MEMBERSHIP = "chord-membership"
    CROSS_STAFF = "cross-staff"
    REPEAT_ENDING = "repeat-ending"


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoreGraphV2Error(f"{name} must be non-empty text")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScoreGraphV2Error(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScoreGraphV2Error(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True, order=True)
class StaffNode:
    part_id: str
    staff: int

    def __post_init__(self) -> None:
        _text("part_id", self.part_id)
        _positive_int("staff", self.staff)


@dataclass(frozen=True, slots=True, order=True)
class VoiceNode:
    part_id: str
    voice: int

    def __post_init__(self) -> None:
        _text("part_id", self.part_id)
        _positive_int("voice", self.voice)


@dataclass(frozen=True, slots=True)
class SourceNavigationEvent:
    kind: SourceNavigationKind
    part_id: str
    measure_index: int
    sequence_index: int
    duration: ExactRational

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SourceNavigationKind):
            raise ScoreGraphV2Error("source navigation kind must be backup or forward")
        _text("part_id", self.part_id)
        _positive_int("measure_index", self.measure_index)
        _nonnegative_int("sequence_index", self.sequence_index)
        if not isinstance(self.duration, ExactRational) or self.duration.is_zero:
            raise ScoreGraphV2Error("backup/forward duration must be positive ExactRational")


@dataclass(frozen=True, slots=True)
class GraphAnchor:
    kind: GraphAnchorKind
    ref_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GraphAnchorKind):
            raise ScoreGraphV2Error("anchor kind must be GraphAnchorKind")
        _text("anchor ref_id", self.ref_id)


@dataclass(frozen=True, slots=True)
class ScoreGraphRelation:
    relation_id: str
    kind: RelationKind
    source: GraphAnchor
    target: GraphAnchor
    number: int | None = None

    def __post_init__(self) -> None:
        _text("relation_id", self.relation_id)
        if not isinstance(self.kind, RelationKind):
            raise ScoreGraphV2Error("relation kind must be RelationKind")
        if not isinstance(self.source, GraphAnchor) or not isinstance(self.target, GraphAnchor):
            raise ScoreGraphV2Error("relation endpoints must be GraphAnchor values")
        if self.number is not None:
            _positive_int("relation number", self.number)


@dataclass(frozen=True, slots=True)
class ScoreGraphCapability:
    code: str
    outcome: CapabilityOutcome
    path: str
    block_reason: BlockReason | None = None

    def __post_init__(self) -> None:
        _text("capability code", self.code)
        _text("capability path", self.path)
        if not isinstance(self.outcome, CapabilityOutcome):
            raise ScoreGraphV2Error("capability outcome must be CapabilityOutcome")
        if self.outcome is CapabilityOutcome.BLOCKED:
            if not isinstance(self.block_reason, BlockReason):
                raise ScoreGraphV2Error(
                    "BLOCKED requires safety, unparseable or boundedness reason"
                )
        elif self.block_reason is not None:
            raise ScoreGraphV2Error("non-BLOCKED capability must not carry block_reason")


@dataclass(frozen=True, slots=True)
class ScoreGraphV2:
    core: PolyScore
    source_navigation: tuple[SourceNavigationEvent, ...] = ()
    relations: tuple[ScoreGraphRelation, ...] = ()
    capabilities: tuple[ScoreGraphCapability, ...] = ()
    version: str = SCOREGRAPH_V2_VERSION

    def __post_init__(self) -> None:
        if self.version != SCOREGRAPH_V2_VERSION:
            raise ScoreGraphV2Error("unsupported ScoreGraph V2 envelope version")
        if not isinstance(self.core, PolyScore):
            raise ScoreGraphV2Error("core must be frozen PolyScore")
        if not isinstance(self.source_navigation, tuple) or any(
            not isinstance(item, SourceNavigationEvent) for item in self.source_navigation
        ):
            raise ScoreGraphV2Error("source_navigation must be immutable SourceNavigationEvent tuple")
        if not isinstance(self.relations, tuple) or any(
            not isinstance(item, ScoreGraphRelation) for item in self.relations
        ):
            raise ScoreGraphV2Error("relations must be immutable ScoreGraphRelation tuple")
        if not isinstance(self.capabilities, tuple) or any(
            not isinstance(item, ScoreGraphCapability) for item in self.capabilities
        ):
            raise ScoreGraphV2Error("capabilities must be immutable ScoreGraphCapability tuple")

        part_by_id = {part.part_id: part for part in self.core.parts}

        navigation_keys = tuple(
            (item.part_id, item.measure_index, item.sequence_index, item.kind.value)
            for item in self.source_navigation
        )
        if navigation_keys != tuple(sorted(navigation_keys)):
            raise ScoreGraphV2Error("source navigation evidence must be in canonical order")
        if len(set(navigation_keys)) != len(navigation_keys):
            raise ScoreGraphV2Error("duplicate source navigation evidence")
        for item in self.source_navigation:
            part = part_by_id.get(item.part_id)
            if part is None or item.measure_index > len(part.measures):
                raise ScoreGraphV2Error("source navigation references missing part/measure")

        valid_anchors = self._valid_anchor_refs()
        relation_ids: set[str] = set()
        relation_keys = tuple(item.relation_id for item in self.relations)
        if relation_keys != tuple(sorted(relation_keys)):
            raise ScoreGraphV2Error("relations must be in canonical relation_id order")
        for relation in self.relations:
            if relation.relation_id in relation_ids:
                raise ScoreGraphV2Error("duplicate relation_id")
            relation_ids.add(relation.relation_id)
            if relation.source.ref_id not in valid_anchors[relation.source.kind]:
                raise ScoreGraphV2Error("relation source anchor does not exist in core")
            if relation.target.ref_id not in valid_anchors[relation.target.kind]:
                raise ScoreGraphV2Error("relation target anchor does not exist in core")

        capability_keys = tuple(
            (item.code, item.path, item.outcome.value) for item in self.capabilities
        )
        if capability_keys != tuple(sorted(capability_keys)):
            raise ScoreGraphV2Error("capabilities must be in canonical code/path/outcome order")
        if len(set(capability_keys)) != len(capability_keys):
            raise ScoreGraphV2Error("duplicate capability finding")

    @property
    def core_sha256(self) -> str:
        return self.core.canonical_sha256()

    @property
    def staff_nodes(self) -> tuple[StaffNode, ...]:
        return tuple(
            StaffNode(part.part_id, staff)
            for part in self.core.parts
            for staff in range(1, part.staff_count + 1)
        )

    @property
    def voice_nodes(self) -> tuple[VoiceNode, ...]:
        nodes = {
            VoiceNode(part.part_id, event.voice)
            for part in self.core.parts
            for measure in part.measures
            for event in measure.events
        }
        return tuple(sorted(nodes))

    def _valid_anchor_refs(self) -> dict[GraphAnchorKind, set[str]]:
        refs: dict[GraphAnchorKind, set[str]] = {kind: set() for kind in GraphAnchorKind}
        for part in self.core.parts:
            refs[GraphAnchorKind.PART].add(part.part_id)
            for staff in range(1, part.staff_count + 1):
                refs[GraphAnchorKind.STAFF].add(f"{part.part_id}:staff:{staff}")
            voices: set[int] = set()
            for measure in part.measures:
                refs[GraphAnchorKind.MEASURE].add(
                    f"{part.part_id}:measure:{measure.measure_index}"
                )
                for event in measure.events:
                    voices.add(event.voice)
                    refs[GraphAnchorKind.EVENT].add(event.event_id)
                    for atom in event.noteheads:
                        refs[GraphAnchorKind.NOTEHEAD].add(atom.atom_id)
            for voice in voices:
                refs[GraphAnchorKind.VOICE].add(f"{part.part_id}:voice:{voice}")
        return refs

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "core_sha256": self.core_sha256,
            "core": self.core.canonical_payload(),
            "source_navigation": [_jsonable(asdict(item)) for item in self.source_navigation],
            "relations": [_jsonable(asdict(item)) for item in self.relations],
            "capabilities": [_jsonable(asdict(item)) for item in self.capabilities],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def canonical_sha256(self) -> str:
        return sha256(self.canonical_json().encode("ascii")).hexdigest()


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
