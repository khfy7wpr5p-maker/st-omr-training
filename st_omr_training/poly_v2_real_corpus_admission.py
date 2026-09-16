"""Fail-closed lineage admission for the first real Native Polyphonic V2 corpus.

TR-POLY-09B8I bridges two already-frozen safety boundaries without weakening
either one:

* B8A proves that persisted Native V2 TRAIN/VALIDATION artifacts match their
  exact manifest/build identity while TEST artifact bytes remain absent; and
* Stage 8-0/8-1 proves that the real source/image/MusicXML material has admitted
  rights/provenance/pairing metadata plus byte-level receipts.

This module requires an explicit one-to-one lineage binding between those two
surfaces before the Native V2 corpus may be considered eligible for the B8
quality-training lane.  It does not read source, image, MusicXML, Native V2
artifact, or TEST bytes; it consumes already-verified immutable objects only.
It does not run training and grants no production or commercial-use authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final, Iterable

from .dataset_manifest import DatasetSplit
from .poly_v2_dataset_reload import LoadedNativePolyV2Dataset
from .real_data_contract import RealDataManifest, RealDataSample, RealDataSplit
from .real_data_intake import (
    RealDataByteReceipt,
    RealDataIntakeError,
    validate_stage8_development_handoff,
)


POLY_V2_REAL_CORPUS_ADMISSION_VERSION: Final[str] = (
    "st-omr-native-poly-v2-real-corpus-admission-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NativePolyV2RealCorpusAdmissionError(ValueError):
    """Raised when the real-data → Native V2 lineage gate fails closed."""


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NativePolyV2RealCorpusAdmissionError(
            f"{name} must be lowercase SHA-256 text"
        )
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise NativePolyV2RealCorpusAdmissionError(
            "B8I evidence is not canonical-JSON serializable"
        ) from exc


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


def _real_data_manifest_sha256(manifest: RealDataManifest) -> str:
    payload = {
        "schema_version": manifest.schema_version,
        "source_class": manifest.source_class,
        "split_policy": manifest.split_policy,
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "sealed_test_manifest_sha256": manifest.sealed_test_manifest_sha256,
        "samples": [
            _jsonable(asdict(sample))
            for sample in sorted(manifest.samples, key=lambda item: item.sample_id)
        ],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _receipt_manifest_sha256(receipts: tuple[RealDataByteReceipt, ...]) -> str:
    payload = [
        {
            "sample_id": receipt.sample_id,
            "receipt_sha256": receipt.receipt_sha256,
        }
        for receipt in sorted(receipts, key=lambda item: item.sample_id)
    ]
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _sealed_native_test_population_sha256(sample_ids: tuple[str, ...]) -> str:
    payload = {
        "policy": "metadata-only-no-test-bytes-v1",
        "sample_ids": list(sample_ids),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePolyV2RealDataBinding:
    """One reviewed lineage edge from admitted real data to one Native V2 target."""

    native_sample_id: str
    real_sample_id: str
    v2_conversion_profile_sha256: str
    v2_target_review_evidence_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "native_sample_id",
            "real_sample_id",
            "v2_conversion_profile_sha256",
            "v2_target_review_evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(asdict(self))).hexdigest()


def _binding_manifest_sha256(
    bindings: tuple[NativePolyV2RealDataBinding, ...],
) -> str:
    payload = [
        asdict(binding)
        for binding in sorted(bindings, key=lambda item: item.native_sample_id)
    ]
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePolyV2RealCorpusAdmissionReceipt:
    dataset_manifest_sha256: str
    dataset_build_id: str
    b8a_preflight_receipt_sha256: str
    real_data_manifest_sha256: str
    real_data_byte_receipt_manifest_sha256: str
    binding_manifest_sha256: str
    sealed_real_test_manifest_sha256: str
    sealed_native_test_population_sha256: str
    train_native_sample_ids: tuple[str, ...]
    validation_native_sample_ids: tuple[str, ...]
    admitted_real_sample_ids: tuple[str, ...]
    near_duplicate_candidate_count: int
    admission_version: str = POLY_V2_REAL_CORPUS_ADMISSION_VERSION
    quality_training_eligible: bool = True
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False
    commercial_use_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "dataset_build_id",
            "b8a_preflight_receipt_sha256",
            "real_data_manifest_sha256",
            "real_data_byte_receipt_manifest_sha256",
            "binding_manifest_sha256",
            "sealed_real_test_manifest_sha256",
            "sealed_native_test_population_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "train_native_sample_ids",
            "validation_native_sample_ids",
            "admitted_real_sample_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise NativePolyV2RealCorpusAdmissionError(
                    f"{name} must be a non-empty immutable tuple"
                )
            if tuple(sorted(set(values))) != values:
                raise NativePolyV2RealCorpusAdmissionError(
                    f"{name} must be sorted and unique"
                )
            for value in values:
                _require_sha256(name, value)
        if set(self.train_native_sample_ids) & set(self.validation_native_sample_ids):
            raise NativePolyV2RealCorpusAdmissionError(
                "admission receipt contains TRAIN/VALIDATION sample leakage"
            )
        if len(self.admitted_real_sample_ids) != (
            len(self.train_native_sample_ids) + len(self.validation_native_sample_ids)
        ):
            raise NativePolyV2RealCorpusAdmissionError(
                "admitted real sample count must equal the Native V2 development population"
            )
        if (
            not isinstance(self.near_duplicate_candidate_count, int)
            or isinstance(self.near_duplicate_candidate_count, bool)
            or self.near_duplicate_candidate_count < 0
        ):
            raise NativePolyV2RealCorpusAdmissionError(
                "near_duplicate_candidate_count must be a non-negative integer"
            )
        if self.admission_version != POLY_V2_REAL_CORPUS_ADMISSION_VERSION:
            raise NativePolyV2RealCorpusAdmissionError("unsupported B8I admission version")
        if not self.quality_training_eligible:
            raise NativePolyV2RealCorpusAdmissionError(
                "a completed B8I receipt must identify an admitted quality-training corpus"
            )
        if (
            self.test_artifact_bytes_accessed
            or self.production_authority
            or self.commercial_use_authority
        ):
            raise NativePolyV2RealCorpusAdmissionError(
                "B8I may not access TEST bytes or grant production/commercial authority"
            )

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["train_native_sample_ids"] = list(self.train_native_sample_ids)
        payload["validation_native_sample_ids"] = list(
            self.validation_native_sample_ids
        )
        payload["admitted_real_sample_ids"] = list(self.admitted_real_sample_ids)
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def admit_native_poly_v2_real_corpus(
    *,
    loaded: LoadedNativePolyV2Dataset,
    real_manifest: RealDataManifest,
    byte_receipts: Iterable[RealDataByteReceipt],
    bindings: Iterable[NativePolyV2RealDataBinding],
) -> NativePolyV2RealCorpusAdmissionReceipt:
    """Bind a complete admitted real development corpus to the B8A Native V2 root.

    The target-review evidence is intentionally external to this machine gate:
    the validator can bind that evidence to exact source/target identities, but
    cannot independently decide whether a human/organizational musical review
    was substantively correct.
    """

    if not isinstance(loaded, LoadedNativePolyV2Dataset):
        raise TypeError("loaded must be LoadedNativePolyV2Dataset")
    if not isinstance(real_manifest, RealDataManifest):
        raise TypeError("real_manifest must be RealDataManifest")

    receipts = tuple(byte_receipts)
    binding_values = tuple(bindings)
    if not receipts:
        raise NativePolyV2RealCorpusAdmissionError(
            "B8I requires Stage 8-1 byte receipts"
        )
    if not binding_values:
        raise NativePolyV2RealCorpusAdmissionError(
            "B8I requires explicit real-data → Native V2 lineage bindings"
        )

    try:
        near_duplicates = validate_stage8_development_handoff(
            real_manifest,
            receipts,
        )
    except RealDataIntakeError as exc:
        raise NativePolyV2RealCorpusAdmissionError(
            "Stage 8 real-data development handoff is not admitted"
        ) from exc

    build = loaded.build
    receipt = loaded.receipt
    native_development = tuple(
        sorted(
            (
                sample
                for sample in build.manifest.samples
                if sample.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
            ),
            key=lambda item: item.sample_id,
        )
    )
    native_by_id = {sample.sample_id: sample for sample in native_development}
    real_by_id = {sample.sample_id: sample for sample in real_manifest.samples}
    receipt_by_id = {item.sample_id: item for item in receipts}

    expected_train = tuple(
        sample.sample_id
        for sample in native_development
        if sample.split is DatasetSplit.TRAIN
    )
    expected_validation = tuple(
        sample.sample_id
        for sample in native_development
        if sample.split is DatasetSplit.VALIDATION
    )
    if expected_train != receipt.train_sample_ids:
        raise NativePolyV2RealCorpusAdmissionError(
            "B8I TRAIN population differs from the exact B8A preflight receipt"
        )
    if expected_validation != receipt.validation_sample_ids:
        raise NativePolyV2RealCorpusAdmissionError(
            "B8I VALIDATION population differs from the exact B8A preflight receipt"
        )

    if len(real_by_id) != len(real_manifest.samples):
        raise NativePolyV2RealCorpusAdmissionError(
            "real-data manifest contains duplicate sample ids"
        )
    if len(receipt_by_id) != len(receipts):
        raise NativePolyV2RealCorpusAdmissionError(
            "real-data byte receipts contain duplicate sample ids"
        )
    if set(real_by_id) != set(receipt_by_id):
        raise NativePolyV2RealCorpusAdmissionError(
            "real-data manifest and byte receipts cover different samples"
        )
    if len(real_by_id) != len(native_by_id):
        raise NativePolyV2RealCorpusAdmissionError(
            "B8I requires one admitted real source record per Native V2 development sample"
        )

    binding_by_native: dict[str, NativePolyV2RealDataBinding] = {}
    bound_real_ids: set[str] = set()
    for binding in binding_values:
        if not isinstance(binding, NativePolyV2RealDataBinding):
            raise TypeError("bindings must contain NativePolyV2RealDataBinding values")
        if binding.native_sample_id in binding_by_native:
            raise NativePolyV2RealCorpusAdmissionError(
                "duplicate lineage binding for one Native V2 sample"
            )
        if binding.real_sample_id in bound_real_ids:
            raise NativePolyV2RealCorpusAdmissionError(
                "one admitted real sample may not feed multiple Native V2 samples"
            )
        binding_by_native[binding.native_sample_id] = binding
        bound_real_ids.add(binding.real_sample_id)

    if set(binding_by_native) != set(native_by_id):
        raise NativePolyV2RealCorpusAdmissionError(
            "lineage bindings do not cover the exact Native V2 development population"
        )
    if bound_real_ids != set(real_by_id):
        raise NativePolyV2RealCorpusAdmissionError(
            "lineage bindings do not cover the exact admitted real-data population"
        )

    for native_id in sorted(native_by_id):
        native_sample = native_by_id[native_id]
        binding = binding_by_native[native_id]
        real_sample: RealDataSample = real_by_id[binding.real_sample_id]
        real_receipt = receipt_by_id[real_sample.sample_id]

        expected_real_split = RealDataSplit(native_sample.split.value)
        if real_sample.split is not expected_real_split:
            raise NativePolyV2RealCorpusAdmissionError(
                "Native V2 / real-data split mismatch"
            )
        if native_sample.family_id != real_sample.family_id:
            raise NativePolyV2RealCorpusAdmissionError(
                "Native V2 / real-data family mismatch"
            )
        if native_sample.image_sha256 != real_sample.image_sha256:
            raise NativePolyV2RealCorpusAdmissionError(
                "Native V2 image identity differs from the admitted real-data image"
            )
        if native_sample.image_sha256 != real_receipt.image_sha256:
            raise NativePolyV2RealCorpusAdmissionError(
                "Native V2 image identity differs from the Stage 8-1 byte receipt"
            )
        if (native_sample.width, native_sample.height) != (
            real_receipt.image_width,
            real_receipt.image_height,
        ):
            raise NativePolyV2RealCorpusAdmissionError(
                "Native V2 image dimensions differ from the Stage 8-1 byte receipt"
            )

    train_ids = tuple(sorted(expected_train))
    validation_ids = tuple(sorted(expected_validation))
    admitted_real_ids = tuple(sorted(real_by_id))
    sealed_native_test_sha = _sealed_native_test_population_sha256(
        receipt.sealed_test_sample_ids
    )

    return NativePolyV2RealCorpusAdmissionReceipt(
        dataset_manifest_sha256=build.manifest_sha256,
        dataset_build_id=build.build_id,
        b8a_preflight_receipt_sha256=receipt.fingerprint(),
        real_data_manifest_sha256=_real_data_manifest_sha256(real_manifest),
        real_data_byte_receipt_manifest_sha256=_receipt_manifest_sha256(receipts),
        binding_manifest_sha256=_binding_manifest_sha256(binding_values),
        sealed_real_test_manifest_sha256=real_manifest.sealed_test_manifest_sha256,
        sealed_native_test_population_sha256=sealed_native_test_sha,
        train_native_sample_ids=train_ids,
        validation_native_sample_ids=validation_ids,
        admitted_real_sample_ids=admitted_real_ids,
        near_duplicate_candidate_count=len(near_duplicates),
    )
