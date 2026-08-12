"""Fail-closed MusicXML 4.0 validation for ST-OMR V1.

Stage 2-C keeps two independent gates:

1. offline validation against the pinned official W3C MusicXML 4.0 XSD;
2. ST-OMR V1 semantic validation that recomputes project invariants.

The module never downloads schemas, resolves document entities, loads DTDs, or
silently normalizes unsupported MusicXML into the V1 subset.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Final

from .validator import ValidationIssue, ValidationResult


MAX_MUSICXML_BYTES: Final[int] = 8 * 1024 * 1024
SCHEMA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "schemas" / "musicxml-4.0"

MUSICXML_SCHEMA_SOURCE_COMMIT: Final[str] = "799e2defb2ece0ae7bafe08dcbcac25b2c631d53"
MUSICXML_SCHEMA_SOURCE_BLOBS: Final[dict[str, str]] = {
    "musicxml.xsd": "2f2d116e94095cf069a1b3daf3691297b83106d1",
    "xlink.xsd": "7b5e5e831189fcc5952c69aad52bc353125eb7bd",
    "xml.xsd": "eeb9db56093d2382951cbcd1b61c2ccd9d674c92",
    "catalog.xml": "fc241d1f4f6dc15eb96f6ed4a07b9c8db84ca00b",
}
# Filled only with independently computed SHA-256 values for the exact vendored
# bytes. A pending value deliberately makes the XSD gate fail closed.
MUSICXML_SCHEMA_SHA256: Final[dict[str, str]] = {
    "musicxml.xsd": "__PENDING__",
    "xlink.xsd": "__PENDING__",
    "xml.xsd": "__PENDING__",
    "catalog.xml": "__PENDING__",
}

_IMPORT_URLS: Final[dict[str, str]] = {
    "http://www.musicxml.org/xsd/xml.xsd": "xml.xsd",
    "http://www.musicxml.org/xsd/xlink.xsd": "xlink.xsd",
}
_SUPPORTED_SIGNATURES: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2, 4), (3, 4), (4, 4)}
)
_DURATION_BY_TYPE: Final[dict[str, Fraction]] = {
    "whole": Fraction(1, 1),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
}
_ALLOWED_ACCIDENTALS: Final[frozenset[str]] = frozenset({"sharp", "flat", "natural"})


class _SchemaAssetError(RuntimeError):
    pass


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, path=path, message=message)


def _preflight_bytes(data: object) -> tuple[bytes | None, list[ValidationIssue]]:
    if not isinstance(data, bytes):
        return None, [_issue("musicxml.input_type", "$", "MusicXML input must be bytes")]
    if not data:
        return None, [_issue("musicxml.empty", "$", "MusicXML input must not be empty")]
    if len(data) > MAX_MUSICXML_BYTES:
        return None, [
            _issue(
                "musicxml.too_large",
                "$",
                f"MusicXML input exceeds {MAX_MUSICXML_BYTES} bytes",
            )
        ]
    if b"<!DOCTYPE" in data.upper():
        return None, [
            _issue(
                "musicxml.doctype_forbidden",
                "$",
                "DOCTYPE declarations are forbidden for ST-OMR MusicXML validation",
            )
        ]
    return data, []


def _import_lxml():
    try:
        from lxml import etree
    except Exception as exc:  # pragma: no cover - exercised when dependency absent
        raise _SchemaAssetError("lxml 6.1.1 is required for MusicXML XSD validation") from exc
    return etree


def _secure_parser(etree, *, resolver=None):
    parser = etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        dtd_validation=False,
        no_network=True,
        recover=False,
        huge_tree=False,
        remove_comments=False,
        remove_pis=False,
        strip_cdata=False,
    )
    if resolver is not None:
        parser.resolvers.add(resolver)
    return parser


def _schema_asset_hashes(schema_dir: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name in MUSICXML_SCHEMA_SHA256:
        path = schema_dir / name
        if not path.is_file():
            raise _SchemaAssetError(f"missing pinned schema asset: {name}")
        actual[name] = sha256(path.read_bytes()).hexdigest()
    return actual


def verify_musicxml_schema_assets(schema_dir: Path | None = None) -> ValidationResult:
    directory = schema_dir or SCHEMA_DIR
    issues: list[ValidationIssue] = []
    try:
        actual = _schema_asset_hashes(directory)
    except OSError as exc:
        return ValidationResult(
            (_issue("musicxml.schema_io", "$schema", f"schema asset read failed: {exc}"),)
        )
    except _SchemaAssetError as exc:
        return ValidationResult((_issue("musicxml.schema_missing", "$schema", str(exc)),))

    for name, expected in MUSICXML_SCHEMA_SHA256.items():
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            issues.append(
                _issue(
                    "musicxml.schema_hash_unpinned",
                    f"$schema.{name}",
                    "schema SHA-256 is not pinned",
                )
            )
        elif actual[name] != expected:
            issues.append(
                _issue(
                    "musicxml.schema_hash_mismatch",
                    f"$schema.{name}",
                    "schema asset SHA-256 does not match the pinned value",
                )
            )
    return ValidationResult(tuple(issues))


def _compile_schema(schema_dir: Path):
    integrity = verify_musicxml_schema_assets(schema_dir)
    if not integrity.is_valid:
        codes = ", ".join(issue.code for issue in integrity.issues)
        raise _SchemaAssetError(f"schema integrity gate failed: {codes}")

    etree = _import_lxml()

    class _Resolver(etree.Resolver):
        def resolve(self, url, pubid, context):
            name = _IMPORT_URLS.get(url)
            if name is None:
                raise OSError(f"external schema resolution is forbidden: {url}")
            target = (schema_dir / name).resolve()
            if target.parent != schema_dir.resolve():
                raise OSError("schema resolver escaped pinned schema directory")
            return self.resolve_filename(str(target), context)

    parser = _secure_parser(etree, resolver=_Resolver())
    schema_document = etree.parse(str(schema_dir / "musicxml.xsd"), parser)
    return etree, etree.XMLSchema(schema_document)


def validate_musicxml_xsd(data: object, *, schema_dir: Path | None = None) -> ValidationResult:
    payload, issues = _preflight_bytes(data)
    if issues:
        return ValidationResult(tuple(issues))
    assert payload is not None

    directory = schema_dir or SCHEMA_DIR
    try:
        etree, schema = _compile_schema(directory)
        document = etree.fromstring(payload, parser=_secure_parser(etree))
    except _SchemaAssetError as exc:
        return ValidationResult((_issue("musicxml.schema_unavailable", "$schema", str(exc)),))
    except Exception as exc:
        # XMLSyntaxError and schema parser failures both fail closed. The exact
        # parser text is intentionally not exposed as a stable API contract.
        return ValidationResult(
            (_issue("musicxml.xsd_parse_error", "$", f"XML/XSD parse failed: {type(exc).__name__}"),)
        )

    if not schema.validate(document):
        return ValidationResult(
            (_issue("musicxml.xsd_invalid", "$", "document is not valid MusicXML 4.0 XSD"),)
        )
    return ValidationResult()


def _parse_semantic_tree(data: bytes):
    try:
        etree = _import_lxml()
        return etree.fromstring(data, parser=_secure_parser(etree)), None
    except _SchemaAssetError as exc:
        return None, _issue("musicxml.lxml_unavailable", "$", str(exc))
    except Exception:
        return None, _issue("musicxml.malformed", "$", "MusicXML is not well-formed XML")


def _plain_int(text: object) -> int | None:
    if not isinstance(text, str) or not text:
        return None
    if text == "0":
        return 0
    if text.startswith("-"):
        digits = text[1:]
        if not digits or not digits.isdigit() or digits.startswith("0"):
            return None
    elif not text.isdigit() or text.startswith("0"):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _child_tags(element) -> list[str]:
    return [child.tag for child in element]


def _text(element, child_name: str) -> str | None:
    child = element.find(child_name)
    return None if child is None else child.text


def _validate_first_attributes(attributes, path: str) -> tuple[list[ValidationIssue], int | None, tuple[int, int] | None]:
    issues: list[ValidationIssue] = []
    if attributes.attrib:
        issues.append(_issue("musicxml.attributes_attributes", path, "attributes element must not carry XML attributes"))
    expected_tags = ["divisions", "key", "time", "clef"]
    if _child_tags(attributes) != expected_tags:
        issues.append(
            _issue(
                "musicxml.first_attributes_shape",
                path,
                "first measure attributes must be divisions, key, time, clef in V1 order",
            )
        )

    divisions = _plain_int(_text(attributes, "divisions"))
    if divisions is None or divisions <= 0:
        issues.append(_issue("musicxml.divisions", f"{path}.divisions", "divisions must be a positive canonical integer"))
        divisions = None

    key = attributes.find("key")
    if key is None or key.attrib or _child_tags(key) != ["fifths"] or _text(key, "fifths") != "0":
        issues.append(_issue("musicxml.key_signature", f"{path}.key", "V1 key signature must be exactly fifths = 0"))

    signature, time_issues = _validate_time(attributes.find("time"), f"{path}.time")
    issues.extend(time_issues)

    clef = attributes.find("clef")
    if clef is None or clef.attrib or _child_tags(clef) != ["sign", "line"] or _text(clef, "sign") != "G" or _text(clef, "line") != "2":
        issues.append(_issue("musicxml.clef", f"{path}.clef", "V1 clef must be exactly G on line 2"))

    return issues, divisions, signature


def _validate_time(time, path: str) -> tuple[tuple[int, int] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if time is None or time.attrib or _child_tags(time) != ["beats", "beat-type"]:
        return None, [_issue("musicxml.time_shape", path, "time must contain only beats and beat-type")]
    beats = _plain_int(_text(time, "beats"))
    beat_type = _plain_int(_text(time, "beat-type"))
    if beats is None or beat_type is None or (beats, beat_type) not in _SUPPORTED_SIGNATURES:
        issues.append(_issue("musicxml.time_signature", path, "V1 time signature must be 2/4, 3/4, or 4/4"))
        return None, issues
    return (beats, beat_type), issues


def _parse_note(note, path: str, divisions: int) -> tuple[dict[str, object] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if note.attrib:
        issues.append(_issue("musicxml.note_attributes", path, "V1 note elements must not carry XML attributes"))

    children = list(note)
    tags = [child.tag for child in children]
    chord_continuation = bool(tags and tags[0] == "chord")
    index = 1 if chord_continuation else 0

    if index >= len(tags) or tags[index] not in {"pitch", "rest"}:
        issues.append(_issue("musicxml.note_kind", path, "V1 note must contain exactly one pitch or rest"))
        return None, issues
    kind = tags[index]
    index += 1

    required_tail = ["duration", "voice", "type"]
    if tags[index:index + 3] != required_tail:
        issues.append(_issue("musicxml.note_shape", path, "V1 note child order is invalid"))
        return None, issues
    index += 3

    accidental = None
    if index < len(tags) and tags[index] == "accidental":
        accidental = children[index].text
        index += 1
    if index >= len(tags) or tags[index] != "staff" or index + 1 != len(tags):
        issues.append(_issue("musicxml.note_shape", path, "V1 note must terminate with staff after supported fields"))
        return None, issues

    duration = _plain_int(_text(note, "duration"))
    if duration is None or duration <= 0:
        issues.append(_issue("musicxml.duration", f"{path}.duration", "duration must be a positive canonical integer"))
    voice = _plain_int(_text(note, "voice"))
    if voice != 1:
        issues.append(_issue("musicxml.voice", f"{path}.voice", "V1 requires voice 1"))
    staff = _plain_int(_text(note, "staff"))
    if staff != 1:
        issues.append(_issue("musicxml.staff", f"{path}.staff", "V1 requires staff 1"))
    note_type = _text(note, "type")
    if note_type not in _DURATION_BY_TYPE:
        issues.append(_issue("musicxml.note_type", f"{path}.type", "unsupported V1 note/rest type"))
    elif kind == "rest" and note_type == "whole":
        issues.append(_issue("musicxml.whole_rest_deferred", f"{path}.type", "whole rests are not V1 rhythmic rests"))

    if duration is not None and duration > 0 and note_type in _DURATION_BY_TYPE:
        expected = _DURATION_BY_TYPE[note_type] * 4 * divisions
        if expected.denominator != 1 or duration != expected.numerator:
            issues.append(_issue("musicxml.duration_type_mismatch", path, "duration does not exactly match note type and divisions"))

    pitch_identity: tuple[str, int, int] | None = None
    if kind == "rest":
        rest = children[1 if chord_continuation else 0]
        if chord_continuation:
            issues.append(_issue("musicxml.chord_rest", path, "a chord continuation cannot be a rest"))
        if rest.attrib or len(rest) or (rest.text not in (None, "")):
            issues.append(_issue("musicxml.rest_shape", path, "V1 rest must be an empty rest element"))
        if accidental is not None:
            issues.append(_issue("musicxml.rest_accidental", path, "rests cannot carry accidental intent"))
    else:
        pitch = children[1 if chord_continuation else 0]
        if pitch.attrib:
            issues.append(_issue("musicxml.pitch_attributes", f"{path}.pitch", "V1 pitch must not carry XML attributes"))
        pitch_tags = _child_tags(pitch)
        if pitch_tags not in (["step", "octave"], ["step", "alter", "octave"]):
            issues.append(_issue("musicxml.pitch_shape", f"{path}.pitch", "pitch must be step, optional alter, octave"))
        step = _text(pitch, "step")
        if step not in "ABCDEFG" if isinstance(step, str) else True:
            issues.append(_issue("musicxml.pitch_step", f"{path}.pitch.step", "pitch step must be A through G"))
        alter_text = _text(pitch, "alter")
        if alter_text is None:
            alter = 0
        else:
            alter = _plain_int(alter_text)
            if alter not in {-1, 1}:
                issues.append(_issue("musicxml.pitch_alter", f"{path}.pitch.alter", "V1 alter must be omitted or -1/+1"))
        octave = _plain_int(_text(pitch, "octave"))
        if octave is None or not 0 <= octave <= 9:
            issues.append(_issue("musicxml.pitch_octave", f"{path}.pitch.octave", "V1 octave must be 0 through 9"))
        if isinstance(step, str) and step in "ABCDEFG" and isinstance(alter, int) and alter in {-1, 0, 1} and isinstance(octave, int) and 0 <= octave <= 9:
            pitch_identity = (step, alter, octave)

        if accidental is not None:
            accidental_element = note.find("accidental")
            if accidental_element is not None and (accidental_element.attrib or len(accidental_element)):
                issues.append(_issue("musicxml.accidental_shape", f"{path}.accidental", "V1 accidental must be text-only"))
            if accidental not in _ALLOWED_ACCIDENTALS:
                issues.append(_issue("musicxml.accidental", f"{path}.accidental", "V1 accidental must be sharp, flat, or natural"))
            else:
                expected_alter = {"sharp": 1, "flat": -1, "natural": 0}[accidental]
                if alter != expected_alter:
                    issues.append(_issue("musicxml.accidental_mismatch", path, "visible accidental is incoherent with pitch alter"))

    chord = note.find("chord")
    if chord is not None and (chord.attrib or len(chord) or chord.text not in (None, "")):
        issues.append(_issue("musicxml.chord_shape", f"{path}.chord", "chord marker must be empty"))

    return {
        "kind": kind,
        "chord": chord_continuation,
        "duration": duration,
        "voice": voice,
        "staff": staff,
        "type": note_type,
        "pitch": pitch_identity,
    }, issues


def validate_musicxml_semantics(data: object) -> ValidationResult:
    payload, issues = _preflight_bytes(data)
    if issues:
        return ValidationResult(tuple(issues))
    assert payload is not None

    root, parse_issue = _parse_semantic_tree(payload)
    if parse_issue is not None:
        return ValidationResult((parse_issue,))

    issues = []
    if root.tag != "score-partwise":
        return ValidationResult((_issue("musicxml.root", "$", "V1 root must be unnamespaced score-partwise"),))
    if root.attrib != {"version": "4.0"}:
        issues.append(_issue("musicxml.version", "$.@version", "V1 root must have exactly version=4.0"))
    if _child_tags(root) != ["part-list", "part"]:
        issues.append(_issue("musicxml.root_shape", "$", "V1 score contains unsupported top-level elements"))

    part_list = root.find("part-list")
    part = root.find("part")
    if part_list is None or part is None:
        return ValidationResult(tuple(issues + [_issue("musicxml.parts_missing", "$", "V1 requires part-list and part")]))

    score_parts = part_list.findall("score-part")
    if part_list.attrib or len(score_parts) != 1 or len(part_list) != 1:
        issues.append(_issue("musicxml.part_list", "$.part-list", "V1 requires exactly one score-part"))
    if score_parts:
        score_part = score_parts[0]
        if score_part.attrib != {"id": "P1"}:
            issues.append(_issue("musicxml.score_part_id", "$.part-list.score-part", "V1 score-part id must be P1"))
        if _child_tags(score_part) != ["part-name"] or _text(score_part, "part-name") != "ST-OMR Synthetic":
            issues.append(_issue("musicxml.part_name", "$.part-list.score-part.part-name", "V1 part name is not canonical"))

    if part.attrib != {"id": "P1"}:
        issues.append(_issue("musicxml.part_id", "$.part", "V1 part id must be exactly P1"))
    measures = list(part)
    if not measures:
        issues.append(_issue("musicxml.measure_empty", "$.part", "V1 part must contain measures"))
        return ValidationResult(tuple(issues))
    if any(measure.tag != "measure" for measure in measures):
        issues.append(_issue("musicxml.unsupported_element", "$.part", "V1 part may contain only measure elements"))

    divisions: int | None = None
    active_signature: tuple[int, int] | None = None

    for measure_index, measure in enumerate(measures):
        path = f"$.part.measure[{measure_index}]"
        if measure.tag != "measure":
            continue
        if measure.attrib != {"number": str(measure_index + 1)}:
            issues.append(_issue("musicxml.measure_number", path, f"expected canonical measure number {measure_index + 1}"))

        children = list(measure)
        attributes_items = [child for child in children if child.tag == "attributes"]
        if len(attributes_items) > 1:
            issues.append(_issue("musicxml.attributes_count", path, "a measure may contain at most one attributes element"))

        if measure_index == 0:
            if not children or children[0].tag != "attributes":
                issues.append(_issue("musicxml.first_attributes_missing", path, "first measure must begin with attributes"))
            elif len(attributes_items) == 1:
                attr_issues, parsed_divisions, parsed_signature = _validate_first_attributes(attributes_items[0], f"{path}.attributes")
                issues.extend(attr_issues)
                divisions = parsed_divisions
                active_signature = parsed_signature
        elif attributes_items:
            attributes = attributes_items[0]
            if children[0] is not attributes:
                issues.append(_issue("musicxml.attributes_position", path, "measure attributes must appear before notes"))
            if attributes.attrib or _child_tags(attributes) != ["time"]:
                issues.append(_issue("musicxml.later_attributes_shape", f"{path}.attributes", "later V1 attributes may contain only a time change"))
            new_signature, time_issues = _validate_time(attributes.find("time"), f"{path}.attributes.time")
            issues.extend(time_issues)
            if new_signature is not None:
                if new_signature == active_signature:
                    issues.append(_issue("musicxml.time_redundant", f"{path}.attributes.time", "unchanged time signature must not be redundantly emitted"))
                active_signature = new_signature

        for child in children:
            if child.tag not in {"attributes", "note"}:
                issues.append(_issue("musicxml.unsupported_element", path, f"unsupported V1 measure element: {child.tag}"))

        if divisions is None or active_signature is None:
            continue

        capacity = Fraction(active_signature[0], active_signature[1]) * 4 * divisions
        if capacity.denominator != 1:
            issues.append(_issue("musicxml.measure_capacity", path, "time signature does not map to integral divisions"))
            continue
        cursor = 0
        chord_group: list[dict[str, object]] = []

        def close_chord_group() -> None:
            nonlocal chord_group
            if len(chord_group) > 1:
                if len(chord_group) > 4:
                    issues.append(_issue("musicxml.chord_size", path, "V1 chord must contain 2 through 4 notes"))
                base = chord_group[0]
                seen: set[object] = set()
                for member in chord_group:
                    if member["kind"] != "pitch":
                        issues.append(_issue("musicxml.chord_kind", path, "V1 chord members must be pitched notes"))
                    for key in ("duration", "voice", "staff", "type"):
                        if member[key] != base[key]:
                            issues.append(_issue("musicxml.chord_member_mismatch", path, f"chord member {key} must match the base note"))
                    pitch = member["pitch"]
                    if pitch is not None:
                        if pitch in seen:
                            issues.append(_issue("musicxml.chord_duplicate_pitch", path, "duplicate chord pitches are forbidden"))
                        seen.add(pitch)
            chord_group = []

        note_index = 0
        for child in children:
            if child.tag != "note":
                continue
            note_path = f"{path}.note[{note_index}]"
            note_index += 1
            parsed, note_issues = _parse_note(child, note_path, divisions)
            issues.extend(note_issues)
            if parsed is None:
                close_chord_group()
                continue

            if parsed["chord"]:
                if not chord_group:
                    issues.append(_issue("musicxml.chord_without_base", note_path, "chord continuation requires an immediately preceding pitched base note"))
                    chord_group = [parsed]
                else:
                    chord_group.append(parsed)
            else:
                close_chord_group()
                chord_group = [parsed]
                duration = parsed["duration"]
                if isinstance(duration, int) and duration > 0:
                    cursor += duration
                    if cursor > capacity.numerator:
                        issues.append(_issue("musicxml.measure_overflow", note_path, "measure duration exceeds its time-signature capacity"))
        close_chord_group()
        if cursor < capacity.numerator:
            issues.append(_issue("musicxml.measure_underflow", path, "measure duration does not exactly fill its time signature"))
        elif cursor > capacity.numerator:
            issues.append(_issue("musicxml.measure_duration_overflow", path, "measure duration exceeds its time signature"))

    return ValidationResult(tuple(issues))


def validate_musicxml(data: object, *, schema_dir: Path | None = None) -> ValidationResult:
    payload, preflight = _preflight_bytes(data)
    if preflight:
        return ValidationResult(tuple(preflight))
    assert payload is not None

    combined = list(validate_musicxml_xsd(payload, schema_dir=schema_dir).issues)
    combined.extend(validate_musicxml_semantics(payload).issues)

    unique: list[ValidationIssue] = []
    seen: set[ValidationIssue] = set()
    for issue in combined:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)
    return ValidationResult(tuple(unique))
