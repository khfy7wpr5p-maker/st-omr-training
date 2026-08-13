"""Stage 8-1 quarantine/intake byte validation for real-data candidates.

This module is intentionally bytes-in / hash-only-evidence-out. It performs no
filesystem storage, no network access, no model loading, no training, and no
sealed-test enumeration. Repository tests use synthetic fixtures only.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import struct
import zlib
from typing import Final

from .real_data_contract import (
    RealDataContractError,
    RealDataManifest,
    RealDataSample,
    RealDataSplit,
    SealedTestAccessError,
    validate_quarantine_record,
    validate_real_data_manifest,
)
from .training_tokens import TokenizationError, TokenizedTarget, tokenize_musicxml, tokenizer_fingerprint


STAGE8_INTAKE_VERSION: Final[str] = "st-real-data-intake-v1"
STAGE8_SEMANTIC_FINGERPRINT_VERSION: Final[str] = "st-real-data-semantic-token-ids-v1"
STAGE8_PERCEPTUAL_HASH_VERSION: Final[str] = "st-real-data-dhash64-v1"
EXPECTED_PILLOW_VERSION: Final[str] = "12.3.0"

MAX_SOURCE_DOCUMENT_BYTES: Final[int] = 64 * 1024 * 1024
MAX_TRAINING_IMAGE_BYTES: Final[int] = 64 * 1024 * 1024
MAX_TRAINING_IMAGE_PIXELS: Final[int] = 16_000_000
NEAR_DUPLICATE_MAX_HAMMING_DISTANCE: Final[int] = 4
MAX_NEAR_DUPLICATE_COMPARISONS: Final[int] = 1_000_000
MAX_NEAR_DUPLICATE_RESULTS: Final[int] = 100_000

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DHASH_SEGMENT_WIDTHS: Final[tuple[int, ...]] = (13, 13, 13, 13, 12)


class RealDataIntakeError(RealDataContractError):
    """Raised when Stage 8-1 byte validation or handoff fails closed."""


@dataclass(frozen=True, slots=True)
class RealDataByteReceipt:
    """Hash-only evidence emitted after a quarantined candidate passes byte gates."""

    receipt_sha256: str
    sample_id: str
    family_id: str
    split: RealDataSplit
    source_document_sha256: str
    image_sha256: str
    musicxml_sha256: str
    semantic_fingerprint: str
    tokenizer_fingerprint: str
    image_width: int
    image_height: int
    perceptual_hash64: str
    token_count: int
    policy_fingerprint: str
    intake_version: str = STAGE8_INTAKE_VERSION

    def __post_init__(self) -> None:
        _validate_receipt_integrity(self)


@dataclass(frozen=True, slots=True)
class NearDuplicateCandidate:
    left_sample_id: str
    right_sample_id: str
    hamming_distance: int

    def __post_init__(self) -> None:
        _require_hex64("left_sample_id", self.left_sample_id)
        _require_hex64("right_sample_id", self.right_sample_id)
        if self.left_sample_id >= self.right_sample_id:
            raise RealDataIntakeError("near-duplicate sample ids must be canonically ordered")
        if (
            not isinstance(self.hamming_distance, int)
            or isinstance(self.hamming_distance, bool)
            or not 0 <= self.hamming_distance <= 64
        ):
            raise RealDataIntakeError("hamming_distance is outside the dHash64 range")


def _require_hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise RealDataIntakeError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def intake_policy_fingerprint() -> str:
    # Sample-level receipt identity binds byte/semantic/perceptual semantics.
    # Corpus-level search budgets are fail-closed operational guardrails and do
    # not change the identity of an already validated sample receipt.
    payload = {
        "intake_version": STAGE8_INTAKE_VERSION,
        "semantic_fingerprint_version": STAGE8_SEMANTIC_FINGERPRINT_VERSION,
        "perceptual_hash_version": STAGE8_PERCEPTUAL_HASH_VERSION,
        "pillow_version": EXPECTED_PILLOW_VERSION,
        "max_source_document_bytes": MAX_SOURCE_DOCUMENT_BYTES,
        "max_training_image_bytes": MAX_TRAINING_IMAGE_BYTES,
        "max_training_image_pixels": MAX_TRAINING_IMAGE_PIXELS,
        "training_image_contract": "png-grayscale-8bit-noninterlaced-single-frame",
        "perceptual_resize": [9, 8],
        "perceptual_resampling": "LANCZOS",
        "perceptual_comparison": "left-greater-than-right",
        "near_duplicate_max_hamming_distance": NEAR_DUPLICATE_MAX_HAMMING_DISTANCE,
        "near_duplicate_segment_widths": list(_DHASH_SEGMENT_WIDTHS),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _semantic_fingerprint(tokenized: TokenizedTarget) -> str:
    payload = {
        "semantic_fingerprint_version": STAGE8_SEMANTIC_FINGERPRINT_VERSION,
        "tokenizer_fingerprint": tokenized.tokenizer_fingerprint,
        "token_ids": list(tokenized.token_ids),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def semantic_fingerprint_from_musicxml(musicxml_bytes: object) -> str:
    """Compute the frozen Stage 8-1 supported-V1 semantic identity."""

    try:
        return _semantic_fingerprint(tokenize_musicxml(musicxml_bytes))
    except (TokenizationError, ValueError, TypeError) as exc:
        raise RealDataIntakeError(
            "MusicXML failed XSD/semantic/supported-V1/token round-trip validation"
        ) from exc


def _verify_pillow_runtime() -> None:
    try:
        actual = metadata.version("Pillow")
    except metadata.PackageNotFoundError as exc:
        raise RealDataIntakeError("Pillow 12.3.0 is required for Stage 8-1 image verification") from exc
    if actual != EXPECTED_PILLOW_VERSION:
        raise RealDataIntakeError(
            f"Stage 8-1 requires Pillow=={EXPECTED_PILLOW_VERSION}; got {actual}"
        )


def _validate_source_document_bytes(data: object) -> bytes:
    if not isinstance(data, bytes) or not data:
        raise RealDataIntakeError("source document must be non-empty bytes")
    if len(data) > MAX_SOURCE_DOCUMENT_BYTES:
        raise RealDataIntakeError("source document exceeds the Stage 8-1 byte limit")
    return data


def _inspect_png_header(data: bytes) -> tuple[int, int]:
    if len(data) > MAX_TRAINING_IMAGE_BYTES:
        raise RealDataIntakeError("training image exceeds the Stage 8-1 byte limit")
    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE):
        raise RealDataIntakeError("training image must be PNG bytes")
    length = struct.unpack(">I", data[8:12])[0]
    chunk_type = data[12:16]
    if length != 13 or chunk_type != b"IHDR":
        raise RealDataIntakeError("PNG must begin with the canonical IHDR chunk")
    ihdr = data[16:29]
    expected_crc = struct.unpack(">I", data[29:33])[0]
    actual_crc = zlib.crc32(chunk_type + ihdr) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise RealDataIntakeError("PNG IHDR CRC mismatch")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if width < 1 or height < 1 or width * height > MAX_TRAINING_IMAGE_PIXELS:
        raise RealDataIntakeError("training image dimensions are outside Stage 8-1 bounds")
    if (bit_depth, color_type, compression, filtering, interlace) != (8, 0, 0, 0, 0):
        raise RealDataIntakeError(
            "Stage 8-1 requires 8-bit non-interlaced grayscale PNG training images"
        )
    return width, height


def _load_verified_grayscale_png(data: object):
    if not isinstance(data, bytes) or not data:
        raise RealDataIntakeError("training image must be non-empty bytes")
    _verify_pillow_runtime()
    width, height = _inspect_png_header(data)
    try:
        from PIL import Image
    except Exception as exc:
        raise RealDataIntakeError("Pillow 12.3.0 is required for Stage 8-1 image verification") from exc

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "L":
                raise RealDataIntakeError("training image must decode as grayscale PNG mode L")
            if image.size != (width, height):
                raise RealDataIntakeError("decoded PNG dimensions do not match IHDR")
            if getattr(image, "n_frames", 1) != 1:
                raise RealDataIntakeError("animated/multi-frame PNG is not allowed")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (width, height):
                raise RealDataIntakeError("PNG changed identity across verification/decode")
            image.load()
            return image.copy()
    except RealDataIntakeError:
        raise
    except Exception as exc:
        raise RealDataIntakeError("training PNG failed full decode verification") from exc


def _dhash64(image) -> str:
    try:
        from PIL import Image

        resized = image.resize((9, 8), resample=Image.Resampling.LANCZOS)
        pixels = tuple(resized.get_flattened_data())
    except Exception as exc:
        raise RealDataIntakeError("could not compute frozen Stage 8-1 perceptual hash") from exc

    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def _receipt_payload(receipt: RealDataByteReceipt) -> dict[str, object]:
    return {
        "intake_version": receipt.intake_version,
        "sample_id": receipt.sample_id,
        "family_id": receipt.family_id,
        "split": receipt.split.value,
        "source_document_sha256": receipt.source_document_sha256,
        "image_sha256": receipt.image_sha256,
        "musicxml_sha256": receipt.musicxml_sha256,
        "semantic_fingerprint": receipt.semantic_fingerprint,
        "tokenizer_fingerprint": receipt.tokenizer_fingerprint,
        "image_width": receipt.image_width,
        "image_height": receipt.image_height,
        "perceptual_hash64": receipt.perceptual_hash64,
        "token_count": receipt.token_count,
        "policy_fingerprint": receipt.policy_fingerprint,
    }


def _receipt_sha256(receipt: RealDataByteReceipt) -> str:
    return sha256(_canonical_json_bytes(_receipt_payload(receipt))).hexdigest()


def _validate_receipt_integrity(receipt: object) -> None:
    if not isinstance(receipt, RealDataByteReceipt):
        raise RealDataIntakeError("receipt must be RealDataByteReceipt")
    if receipt.intake_version != STAGE8_INTAKE_VERSION:
        raise RealDataIntakeError("unsupported Stage 8-1 intake version")
    if not isinstance(receipt.split, RealDataSplit):
        raise RealDataIntakeError("receipt split must be RealDataSplit")
    for name in (
        "receipt_sha256",
        "sample_id",
        "source_document_sha256",
        "image_sha256",
        "musicxml_sha256",
        "semantic_fingerprint",
        "tokenizer_fingerprint",
        "policy_fingerprint",
    ):
        _require_hex64(name, getattr(receipt, name))
    if not isinstance(receipt.family_id, str) or not receipt.family_id:
        raise RealDataIntakeError("receipt family_id must be non-empty")
    if (
        not isinstance(receipt.perceptual_hash64, str)
        or len(receipt.perceptual_hash64) != 16
        or any(ch not in "0123456789abcdef" for ch in receipt.perceptual_hash64)
    ):
        raise RealDataIntakeError("perceptual_hash64 must be 16 lowercase hex characters")
    for name in ("image_width", "image_height", "token_count"):
        value = getattr(receipt, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RealDataIntakeError(f"{name} must be a positive integer")
    if receipt.image_width * receipt.image_height > MAX_TRAINING_IMAGE_PIXELS:
        raise RealDataIntakeError("receipt image dimensions exceed Stage 8-1 bounds")
    if receipt.tokenizer_fingerprint != tokenizer_fingerprint():
        raise RealDataIntakeError("receipt tokenizer fingerprint is not the frozen tokenizer")
    if receipt.policy_fingerprint != intake_policy_fingerprint():
        raise RealDataIntakeError("receipt policy fingerprint is not the frozen Stage 8-1 policy")
    if receipt.receipt_sha256 != _receipt_sha256(receipt):
        raise RealDataIntakeError("receipt_sha256 does not match canonical receipt payload")


def _new_receipt(
    *,
    record: RealDataSample,
    image_width: int,
    image_height: int,
    perceptual_hash64: str,
    token_count: int,
) -> RealDataByteReceipt:
    values = {
        "intake_version": STAGE8_INTAKE_VERSION,
        "sample_id": record.sample_id,
        "family_id": record.family_id,
        "split": record.split,
        "source_document_sha256": record.source_document_sha256,
        "image_sha256": record.image_sha256,
        "musicxml_sha256": record.musicxml_sha256,
        "semantic_fingerprint": record.semantic_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "image_width": image_width,
        "image_height": image_height,
        "perceptual_hash64": perceptual_hash64,
        "token_count": token_count,
        "policy_fingerprint": intake_policy_fingerprint(),
    }
    payload = {**values, "split": record.split.value}
    digest = sha256(_canonical_json_bytes(payload)).hexdigest()
    return RealDataByteReceipt(receipt_sha256=digest, **values)


def validate_quarantined_sample_bytes(
    record: object,
    *,
    source_document_bytes: object,
    training_image_png_bytes: object,
    musicxml_bytes: object,
) -> RealDataByteReceipt:
    """Validate one quarantined train/validation candidate and emit a receipt."""

    if not isinstance(record, RealDataSample):
        raise RealDataIntakeError("record must be RealDataSample")
    if record.split is RealDataSplit.TEST:
        raise SealedTestAccessError("Stage 8-1 test bytes are sealed until Stage 9")

    quarantine = validate_quarantine_record(record)
    if not quarantine.is_valid:
        first = quarantine.issues[0]
        raise RealDataIntakeError(
            f"quarantine metadata rejected: {first.code} at {first.path}: {first.message}"
        )

    source = _validate_source_document_bytes(source_document_bytes)
    if sha256(source).hexdigest() != record.source_document_sha256:
        raise RealDataIntakeError("source_document_sha256 does not match source bytes")

    if not isinstance(training_image_png_bytes, bytes) or not training_image_png_bytes:
        raise RealDataIntakeError("training image must be non-empty bytes")
    if sha256(training_image_png_bytes).hexdigest() != record.image_sha256:
        raise RealDataIntakeError("image_sha256 does not match training image bytes")
    image = _load_verified_grayscale_png(training_image_png_bytes)

    if not isinstance(musicxml_bytes, bytes) or not musicxml_bytes:
        raise RealDataIntakeError("MusicXML must be non-empty bytes")
    if sha256(musicxml_bytes).hexdigest() != record.musicxml_sha256:
        raise RealDataIntakeError("musicxml_sha256 does not match MusicXML bytes")
    try:
        tokenized = tokenize_musicxml(musicxml_bytes)
    except (TokenizationError, ValueError, TypeError) as exc:
        raise RealDataIntakeError(
            "MusicXML failed XSD/semantic/supported-V1/token round-trip validation"
        ) from exc
    semantic = _semantic_fingerprint(tokenized)
    if semantic != record.semantic_fingerprint:
        raise RealDataIntakeError("semantic_fingerprint does not match MusicXML semantics")

    return _new_receipt(
        record=record,
        image_width=image.width,
        image_height=image.height,
        perceptual_hash64=_dhash64(image),
        token_count=len(tokenized.token_ids),
    )


def validate_byte_receipt(sample: object, receipt: object) -> None:
    """Independently bind a receipt to the exact metadata record it claims to verify."""

    if not isinstance(sample, RealDataSample):
        raise RealDataIntakeError("sample must be RealDataSample")
    if sample.split is RealDataSplit.TEST:
        raise SealedTestAccessError("Stage 8-1 test records are sealed until Stage 9")
    _validate_receipt_integrity(receipt)
    assert isinstance(receipt, RealDataByteReceipt)
    for name in (
        "sample_id",
        "family_id",
        "split",
        "source_document_sha256",
        "image_sha256",
        "musicxml_sha256",
        "semantic_fingerprint",
    ):
        if getattr(receipt, name) != getattr(sample, name):
            raise RealDataIntakeError(f"receipt {name} does not match sample metadata")


def _hamming_distance64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _dhash_segment_keys(value_hex: str) -> tuple[tuple[int, int], ...]:
    """Return five disjoint dHash chunks used for radius-4 candidate indexing."""

    value = int(value_hex, 16)
    remaining = 64
    keys: list[tuple[int, int]] = []
    for index, width in enumerate(_DHASH_SEGMENT_WIDTHS):
        remaining -= width
        keys.append((index, (value >> remaining) & ((1 << width) - 1)))
    return tuple(keys)


def find_near_duplicate_candidates(receipts: object) -> tuple[NearDuplicateCandidate, ...]:
    """Flag dHash64 near-duplicates with deterministic bounded candidate search."""

    if not isinstance(receipts, tuple):
        raise RealDataIntakeError("receipts must be an immutable tuple")
    ordered = sorted(receipts, key=lambda item: item.sample_id if isinstance(item, RealDataByteReceipt) else "")
    for receipt in ordered:
        _validate_receipt_integrity(receipt)
        assert isinstance(receipt, RealDataByteReceipt)
        if receipt.split is RealDataSplit.TEST:
            raise SealedTestAccessError("Stage 8-1 test receipts are sealed until Stage 9")

    buckets: dict[tuple[int, int], list[RealDataByteReceipt]] = {}
    candidates: list[NearDuplicateCandidate] = []
    comparisons = 0

    for right in ordered:
        assert isinstance(right, RealDataByteReceipt)
        keys = _dhash_segment_keys(right.perceptual_hash64)
        possible: dict[str, RealDataByteReceipt] = {}
        for key in keys:
            for left in buckets.get(key, ()):
                possible[left.sample_id] = left

        for left_id in sorted(possible):
            comparisons += 1
            if comparisons > MAX_NEAR_DUPLICATE_COMPARISONS:
                raise RealDataIntakeError(
                    "near-duplicate search exceeded the frozen comparison safety budget"
                )
            left = possible[left_id]
            distance = _hamming_distance64(left.perceptual_hash64, right.perceptual_hash64)
            if distance <= NEAR_DUPLICATE_MAX_HAMMING_DISTANCE:
                candidates.append(
                    NearDuplicateCandidate(
                        left_sample_id=left.sample_id,
                        right_sample_id=right.sample_id,
                        hamming_distance=distance,
                    )
                )
                if len(candidates) > MAX_NEAR_DUPLICATE_RESULTS:
                    raise RealDataIntakeError(
                        "near-duplicate search exceeded the frozen result safety budget"
                    )

        for key in keys:
            buckets.setdefault(key, []).append(right)

    return tuple(candidates)


def validate_stage8_development_handoff(
    manifest: object,
    receipts: object,
) -> tuple[NearDuplicateCandidate, ...]:
    """Validate the byte-evidence handoff required by any later Stage 8 loader."""

    manifest_result = validate_real_data_manifest(manifest)
    if not manifest_result.is_valid:
        first = manifest_result.issues[0]
        raise RealDataIntakeError(
            f"real-data manifest rejected: {first.code} at {first.path}: {first.message}"
        )
    assert isinstance(manifest, RealDataManifest)
    if not isinstance(receipts, tuple) or len(receipts) != len(manifest.samples):
        raise RealDataIntakeError("every admitted manifest sample requires exactly one byte receipt")

    receipt_by_id: dict[str, RealDataByteReceipt] = {}
    sample_by_id = {sample.sample_id: sample for sample in manifest.samples}
    if len(sample_by_id) != len(manifest.samples):
        raise RealDataIntakeError("manifest contains duplicate sample ids")

    for receipt in receipts:
        _validate_receipt_integrity(receipt)
        assert isinstance(receipt, RealDataByteReceipt)
        if receipt.sample_id in receipt_by_id:
            raise RealDataIntakeError("duplicate byte receipt for one sample")
        receipt_by_id[receipt.sample_id] = receipt

    if set(receipt_by_id) != set(sample_by_id):
        raise RealDataIntakeError("manifest samples and byte receipts do not bind the same ids")
    for sample_id, sample in sample_by_id.items():
        validate_byte_receipt(sample, receipt_by_id[sample_id])

    near = find_near_duplicate_candidates(tuple(receipt_by_id.values()))
    for candidate in near:
        left = sample_by_id[candidate.left_sample_id]
        right = sample_by_id[candidate.right_sample_id]
        if left.family_id != right.family_id:
            raise RealDataIntakeError(
                "perceptual near-duplicate appears under different families; keep it quarantined"
            )
        if left.split is not right.split:
            raise RealDataIntakeError(
                "perceptual near-duplicate family appears across development splits"
            )
    return near
