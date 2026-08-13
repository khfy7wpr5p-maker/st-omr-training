"""Stage 8-3A pilot source preparation and auxiliary-package triage.

This module is deliberately in-memory only. It does not persist real data,
create dataset manifests, admit samples, load checkpoints, run training, or
access a sealed test split.

The first Stage 8-3A pilot source policy is intentionally narrow: source images
must already be PNG. The source bytes are never modified. A deterministic
8-bit grayscale training PNG derivative may be produced for later Stage 8-1
byte validation. PrIMuS-style MEI/semantic/agnostic files are auxiliary
triage evidence only; they are not silently promoted to trusted MusicXML.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import re
import struct
import xml.etree.ElementTree as ET
from typing import Final

from .real_data_intake import (
    EXPECTED_PILLOW_VERSION,
    MAX_SOURCE_DOCUMENT_BYTES,
    MAX_TRAINING_IMAGE_BYTES,
    MAX_TRAINING_IMAGE_PIXELS,
)


STAGE8_PILOT_PREPARATION_VERSION: Final[str] = "st-stage8-pilot-preparation-v1"
SOURCE_PNG_POLICY_VERSION: Final[str] = "st-stage8-source-png-to-training-png-v1"
PRIMUS_AUXILIARY_POLICY_VERSION: Final[str] = "st-stage8-primus-auxiliary-triage-v1"
SUPPORTED_SOURCE_PNG_MODES: Final[tuple[str, ...]] = ("1", "L", "P", "RGB")
_SUPPORTED_V1_METERS: Final[frozenset[tuple[int, int]]] = frozenset({(2, 4), (3, 4), (4, 4)})
_SUPPORTED_V1_NOTE_DURS: Final[frozenset[int]] = frozenset({1, 2, 4, 8})
_SUPPORTED_V1_REST_DURS: Final[frozenset[int]] = frozenset({2, 4, 8})
_FORBIDDEN_MEI_ELEMENTS: Final[frozenset[str]] = frozenset(
    {"tie", "slur", "tuplet", "tupletSpan", "multiRest"}
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_HEX = frozenset("0123456789abcdef")

_MAJOR_FIFTHS: Final[dict[str, int]] = {
    "CbM": -7,
    "GbM": -6,
    "DbM": -5,
    "AbM": -4,
    "EbM": -3,
    "BbM": -2,
    "FM": -1,
    "CM": 0,
    "GM": 1,
    "DM": 2,
    "AM": 3,
    "EM": 4,
    "BM": 5,
    "F#M": 6,
    "C#M": 7,
}
_MINOR_FIFTHS: Final[dict[str, int]] = {
    "Abm": -7,
    "Ebm": -6,
    "Bbm": -5,
    "Fm": -4,
    "Cm": -3,
    "Gm": -2,
    "Dm": -1,
    "Am": 0,
    "Em": 1,
    "Bm": 2,
    "F#m": 3,
    "C#m": 4,
    "G#m": 5,
    "D#m": 6,
    "A#m": 7,
}
_KEY_NAME_TO_FIFTHS: Final[dict[str, int]] = {**_MAJOR_FIFTHS, **_MINOR_FIFTHS}


class Stage8PilotPreparationError(ValueError):
    """Raised when Stage 8-3A preparation or triage must fail closed."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pilot_preparation_policy_fingerprint() -> str:
    payload = {
        "preparation_version": STAGE8_PILOT_PREPARATION_VERSION,
        "source_png_policy_version": SOURCE_PNG_POLICY_VERSION,
        "primus_auxiliary_policy_version": PRIMUS_AUXILIARY_POLICY_VERSION,
        "pillow_version": EXPECTED_PILLOW_VERSION,
        "source_contract": "png-only-single-frame-no-transparency",
        "supported_source_modes": list(SUPPORTED_SOURCE_PNG_MODES),
        "geometry_policy": "preserve-width-height-no-crop-resize-rotate",
        "grayscale_policy": "Pillow-convert-L",
        "metadata_policy": "strip-by-new-L-image-from-pixel-bytes",
        "png_save": {"optimize": False, "compress_level": 9},
        "training_image_contract": "png-grayscale-8bit-noninterlaced-single-frame",
        "max_source_bytes": MAX_SOURCE_DOCUMENT_BYTES,
        "max_training_image_bytes": MAX_TRAINING_IMAGE_BYTES,
        "max_pixels": MAX_TRAINING_IMAGE_PIXELS,
        "auxiliary_formats": ["mei", "semantic", "agnostic"],
        "auxiliary_trust": "triage-only-not-MusicXML-admission",
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_bytes(name: str, value: object, *, maximum: int) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise Stage8PilotPreparationError(f"{name} must be non-empty bytes")
    if len(value) > maximum:
        raise Stage8PilotPreparationError(f"{name} exceeds the Stage 8-3A byte limit")
    return value


def _verify_pillow_runtime() -> None:
    try:
        actual = metadata.version("Pillow")
    except metadata.PackageNotFoundError as exc:
        raise Stage8PilotPreparationError(
            f"Pillow=={EXPECTED_PILLOW_VERSION} is required"
        ) from exc
    if actual != EXPECTED_PILLOW_VERSION:
        raise Stage8PilotPreparationError(
            f"Stage 8-3A requires Pillow=={EXPECTED_PILLOW_VERSION}; got {actual}"
        )


def _inspect_training_png_header(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE):
        raise Stage8PilotPreparationError("derived training image is not PNG")
    length = struct.unpack(">I", data[8:12])[0]
    chunk_type = data[12:16]
    if length != 13 or chunk_type != b"IHDR":
        raise Stage8PilotPreparationError("derived PNG does not begin with canonical IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    if width < 1 or height < 1 or width * height > MAX_TRAINING_IMAGE_PIXELS:
        raise Stage8PilotPreparationError("derived training image dimensions are outside bounds")
    if (bit_depth, color_type, compression, filtering, interlace) != (8, 0, 0, 0, 0):
        raise Stage8PilotPreparationError(
            "derived image must be 8-bit non-interlaced grayscale PNG"
        )
    return width, height


@dataclass(frozen=True, slots=True)
class PreparedTrainingImageEvidence:
    source_sha256: str
    training_image_sha256: str
    width: int
    height: int
    source_mode: str
    policy_fingerprint: str
    preparation_version: str = STAGE8_PILOT_PREPARATION_VERSION

    def __post_init__(self) -> None:
        for name in ("source_sha256", "training_image_sha256", "policy_fingerprint"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in _HEX for ch in value)
            ):
                raise Stage8PilotPreparationError(f"{name} must be lowercase SHA-256")
        if self.width < 1 or self.height < 1 or self.width * self.height > MAX_TRAINING_IMAGE_PIXELS:
            raise Stage8PilotPreparationError("prepared image dimensions are outside bounds")
        if self.source_mode not in SUPPORTED_SOURCE_PNG_MODES:
            raise Stage8PilotPreparationError("source_mode is outside the frozen PNG pilot policy")
        if self.policy_fingerprint != pilot_preparation_policy_fingerprint():
            raise Stage8PilotPreparationError("preparation policy fingerprint mismatch")
        if self.preparation_version != STAGE8_PILOT_PREPARATION_VERSION:
            raise Stage8PilotPreparationError("preparation version mismatch")


def prepare_training_png(source_png_bytes: object) -> tuple[bytes, PreparedTrainingImageEvidence]:
    """Derive one deterministic Stage 8-1-compatible training PNG in memory.

    The source bytes are never rewritten. No crop, resize, rotation, or geometry
    normalization is allowed by this first-pilot policy.
    """

    source = _require_bytes(
        "source PNG", source_png_bytes, maximum=MAX_SOURCE_DOCUMENT_BYTES
    )
    if not source.startswith(_PNG_SIGNATURE):
        raise Stage8PilotPreparationError("first-pilot source must already be PNG")
    _verify_pillow_runtime()

    try:
        from PIL import Image
    except Exception as exc:
        raise Stage8PilotPreparationError("Pillow is required for PNG preparation") from exc

    try:
        with Image.open(BytesIO(source)) as image:
            if image.format != "PNG":
                raise Stage8PilotPreparationError("source must decode as PNG")
            if getattr(image, "n_frames", 1) != 1:
                raise Stage8PilotPreparationError("multi-frame source PNG is not allowed")
            if image.mode not in SUPPORTED_SOURCE_PNG_MODES:
                raise Stage8PilotPreparationError(
                    f"source PNG mode {image.mode!r} is outside the frozen pilot policy"
                )
            if "transparency" in image.info:
                raise Stage8PilotPreparationError("transparent source PNG is not allowed")
            width, height = image.size
            source_mode = image.mode
            if width < 1 or height < 1 or width * height > MAX_TRAINING_IMAGE_PIXELS:
                raise Stage8PilotPreparationError("source PNG dimensions are outside bounds")
            image.verify()

        with Image.open(BytesIO(source)) as image:
            if image.format != "PNG" or image.size != (width, height) or image.mode != source_mode:
                raise Stage8PilotPreparationError("source PNG identity changed across verification")
            if getattr(image, "n_frames", 1) != 1 or "transparency" in image.info:
                raise Stage8PilotPreparationError("source PNG changed frame/transparency identity")
            image.load()
            grayscale = image.convert("L")
            clean = Image.frombytes("L", grayscale.size, grayscale.tobytes())

        output = BytesIO()
        clean.save(output, format="PNG", optimize=False, compress_level=9)
        derived = output.getvalue()
    except Stage8PilotPreparationError:
        raise
    except Exception as exc:
        raise Stage8PilotPreparationError("source PNG failed deterministic preparation") from exc

    _require_bytes("derived training PNG", derived, maximum=MAX_TRAINING_IMAGE_BYTES)
    derived_width, derived_height = _inspect_training_png_header(derived)
    if (derived_width, derived_height) != (width, height):
        raise Stage8PilotPreparationError("derived PNG geometry differs from source")

    try:
        with Image.open(BytesIO(derived)) as check:
            if check.format != "PNG" or check.mode != "L" or check.size != (width, height):
                raise Stage8PilotPreparationError("derived PNG failed grayscale decode identity")
            if getattr(check, "n_frames", 1) != 1:
                raise Stage8PilotPreparationError("derived training PNG is multi-frame")
            check.verify()
    except Stage8PilotPreparationError:
        raise
    except Exception as exc:
        raise Stage8PilotPreparationError("derived training PNG failed verification") from exc

    evidence = PreparedTrainingImageEvidence(
        source_sha256=sha256(source).hexdigest(),
        training_image_sha256=sha256(derived).hexdigest(),
        width=width,
        height=height,
        source_mode=source_mode,
        policy_fingerprint=pilot_preparation_policy_fingerprint(),
    )
    return derived, evidence


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_mei_key_signature(value: str | None) -> int | None:
    if value is None or value in {"", "0"}:
        return 0
    match = re.fullmatch(r"([1-7])([fs])", value)
    if match is None:
        return None
    count = int(match.group(1))
    return count if match.group(2) == "s" else -count


def _semantic_key_fifths(tokens: tuple[str, ...]) -> int | None:
    matches = [token for token in tokens if token.startswith("keySignature-")]
    if len(matches) != 1:
        return None
    return _KEY_NAME_TO_FIFTHS.get(matches[0].removeprefix("keySignature-"))


def _semantic_meter(tokens: tuple[str, ...]) -> tuple[int, int] | str | None:
    matches = [token for token in tokens if token.startswith("timeSignature-")]
    if len(matches) != 1:
        return None
    value = matches[0].removeprefix("timeSignature-")
    if value in {"C", "C/"}:
        return value
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _semantic_clef(tokens: tuple[str, ...]) -> str | None:
    matches = [token for token in tokens if token.startswith("clef-")]
    if len(matches) != 1:
        return None
    return matches[0].removeprefix("clef-")


@dataclass(frozen=True, slots=True)
class PrimusAuxiliaryInspection:
    mei_sha256: str
    semantic_sha256: str
    agnostic_sha256: str
    mei_measure_count: int
    mei_note_count: int
    mei_rest_count: int
    semantic_token_count: int
    agnostic_token_count: int
    clef: str | None
    key_signature_fifths: int | None
    meter: tuple[int, int] | str | None
    metadata_coherent: bool
    v1_eligible: bool
    v1_rejection_reasons: tuple[str, ...]
    policy_fingerprint: str


def inspect_primus_auxiliary_package(
    *,
    mei_bytes: object,
    semantic_bytes: object,
    agnostic_bytes: object,
) -> PrimusAuxiliaryInspection:
    """Triage one PrIMuS-style auxiliary package without admitting it.

    This checks parseability, key header agreement, and obvious V1 exclusions.
    It does not prove image pairing, legal rights, MusicXML equivalence, or
    training eligibility. A later admission path must still cross Stage 8-0
    and Stage 8-1 with a supported-V1 MusicXML target.
    """

    mei = _require_bytes("MEI", mei_bytes, maximum=8 * 1024 * 1024)
    semantic = _require_bytes("semantic", semantic_bytes, maximum=8 * 1024 * 1024)
    agnostic = _require_bytes("agnostic", agnostic_bytes, maximum=8 * 1024 * 1024)

    try:
        root = ET.fromstring(mei)
    except ET.ParseError as exc:
        raise Stage8PilotPreparationError("MEI is not well-formed XML") from exc
    if _local_name(root.tag) != "mei":
        raise Stage8PilotPreparationError("auxiliary XML root must be MEI")

    try:
        semantic_text = semantic.decode("utf-8")
        agnostic_text = agnostic.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage8PilotPreparationError("semantic/agnostic files must be UTF-8 text") from exc
    semantic_tokens = tuple(token for token in re.split(r"\s+", semantic_text.strip()) if token)
    agnostic_tokens = tuple(token for token in re.split(r"\s+", agnostic_text.strip()) if token)
    if not semantic_tokens or not agnostic_tokens:
        raise Stage8PilotPreparationError("semantic/agnostic token streams must be non-empty")

    score_defs = [node for node in root.iter() if _local_name(node.tag) == "scoreDef"]
    staff_defs = [node for node in root.iter() if _local_name(node.tag) == "staffDef"]
    measures = [node for node in root.iter() if _local_name(node.tag) == "measure"]
    notes = [node for node in root.iter() if _local_name(node.tag) == "note"]
    rests = [node for node in root.iter() if _local_name(node.tag) == "rest"]
    staves = [node for node in root.iter() if _local_name(node.tag) == "staff"]
    layers = [node for node in root.iter() if _local_name(node.tag) == "layer"]

    if len(score_defs) != 1 or len(staff_defs) != 1:
        raise Stage8PilotPreparationError("pilot MEI requires exactly one scoreDef and one staffDef")
    score_def = score_defs[0]
    staff_def = staff_defs[0]

    clef_shape = staff_def.attrib.get("clef.shape")
    clef_line = staff_def.attrib.get("clef.line")
    clef = f"{clef_shape}{clef_line}" if clef_shape and clef_line else None
    key_fifths = _parse_mei_key_signature(score_def.attrib.get("key.sig"))

    meter: tuple[int, int] | str | None
    meter_sym = score_def.attrib.get("meter.sym")
    if meter_sym in {"common", "cut"}:
        meter = "C" if meter_sym == "common" else "C/"
    else:
        try:
            meter = (int(score_def.attrib["meter.count"]), int(score_def.attrib["meter.unit"]))
        except (KeyError, ValueError):
            meter = None

    semantic_clef = _semantic_clef(semantic_tokens)
    semantic_key = _semantic_key_fifths(semantic_tokens)
    semantic_meter = _semantic_meter(semantic_tokens)
    metadata_coherent = (
        clef is not None
        and semantic_clef == clef
        and key_fifths is not None
        and semantic_key == key_fifths
        and meter is not None
        and semantic_meter == meter
    )

    reasons: list[str] = []
    if not metadata_coherent:
        reasons.append("mei_semantic_header_mismatch")
    if clef != "G2":
        reasons.append("unsupported_clef")
    if key_fifths != 0:
        reasons.append("unsupported_key_signature")
    if not isinstance(meter, tuple) or meter not in _SUPPORTED_V1_METERS:
        reasons.append("unsupported_meter_or_meter_symbol")
    if len(staves) != len(measures) or len(layers) != len(measures):
        reasons.append("unsupported_staff_or_layer_structure")

    local_names = tuple(_local_name(node.tag) for node in root.iter())
    if "beam" in local_names:
        reasons.append("explicit_beam_notation_deferred")
    if any(name in _FORBIDDEN_MEI_ELEMENTS for name in local_names):
        reasons.append("unsupported_mei_structure")
    if "dot" in local_names or any(node.attrib.get("dots") not in {None, "", "0"} for node in root.iter()):
        reasons.append("dotted_duration_unsupported")

    for node in notes:
        try:
            dur = int(node.attrib["dur"])
        except (KeyError, ValueError):
            reasons.append("invalid_note_duration")
            break
        if dur not in _SUPPORTED_V1_NOTE_DURS:
            reasons.append("unsupported_note_duration")
            break
    for node in rests:
        try:
            dur = int(node.attrib["dur"])
        except (KeyError, ValueError):
            reasons.append("invalid_rest_duration")
            break
        if dur not in _SUPPORTED_V1_REST_DURS:
            reasons.append("unsupported_rest_duration")
            break

    if any("_sixteenth" in token or "_32nd" in token or "_64th" in token for token in semantic_tokens):
        reasons.append("semantic_duration_outside_v1")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return PrimusAuxiliaryInspection(
        mei_sha256=sha256(mei).hexdigest(),
        semantic_sha256=sha256(semantic).hexdigest(),
        agnostic_sha256=sha256(agnostic).hexdigest(),
        mei_measure_count=len(measures),
        mei_note_count=len(notes),
        mei_rest_count=len(rests),
        semantic_token_count=len(semantic_tokens),
        agnostic_token_count=len(agnostic_tokens),
        clef=clef,
        key_signature_fifths=key_fifths,
        meter=meter,
        metadata_coherent=metadata_coherent,
        v1_eligible=not unique_reasons,
        v1_rejection_reasons=unique_reasons,
        policy_fingerprint=pilot_preparation_policy_fingerprint(),
    )
