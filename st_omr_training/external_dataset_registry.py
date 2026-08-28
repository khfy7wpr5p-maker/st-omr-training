"""TR-POLY-03 external OMR dataset license and admission registry.

The registry deliberately separates legal-use metadata from local installation
admission. Merely being downloadable or publicly visible never makes a dataset
training-ready. No function in this module downloads, opens, or stores dataset
bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Final, Iterable


EXTERNAL_DATASET_REGISTRY_VERSION: Final[str] = "st-omr-external-dataset-registry-v1"


class ExternalDatasetRegistryError(ValueError):
    """Raised when external dataset metadata violates the registry contract."""


class DataUseClass(str, Enum):
    COMMERCIAL_CLEAN = "COMMERCIAL_CLEAN"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    EVALUATION_ONLY = "EVALUATION_ONLY"
    LICENSE_REVIEW_REQUIRED = "LICENSE_REVIEW_REQUIRED"


class RegistryState(str, Enum):
    CANDIDATE = "CANDIDATE"
    LICENSE_VERIFIED = "LICENSE_VERIFIED"
    INSTALL_PINNED = "INSTALL_PINNED"


def _require_nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalDatasetRegistryError(f"{name} must be non-empty text")
    return value


def _require_https_url(name: str, value: object) -> str:
    text = _require_nonempty_text(name, value)
    if not text.startswith("https://"):
        raise ExternalDatasetRegistryError(f"{name} must use https")
    return text


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExternalDatasetRegistryError(f"{name} must be lowercase SHA-256 text")
    return value


@dataclass(frozen=True, slots=True)
class ExternalDatasetRecord:
    dataset_name: str
    dataset_component: str
    source: str
    version: str
    license_id: str
    license_evidence: str
    redistribution_allowed: bool | None
    commercial_use_allowed: bool | None
    training_allowed: bool | None
    evaluation_allowed: bool | None
    derivative_restrictions: str
    data_use_class: DataUseClass
    registry_state: RegistryState
    artifact_sha256: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_text("dataset_name", self.dataset_name)
        _require_nonempty_text("dataset_component", self.dataset_component)
        _require_https_url("source", self.source)
        _require_nonempty_text("version", self.version)
        _require_nonempty_text("license_id", self.license_id)
        _require_https_url("license_evidence", self.license_evidence)
        _require_nonempty_text("derivative_restrictions", self.derivative_restrictions)
        if not isinstance(self.data_use_class, DataUseClass):
            raise ExternalDatasetRegistryError("data_use_class must be DataUseClass")
        if not isinstance(self.registry_state, RegistryState):
            raise ExternalDatasetRegistryError("registry_state must be RegistryState")
        if not isinstance(self.notes, str):
            raise ExternalDatasetRegistryError("notes must be text")

        permissions = (
            self.redistribution_allowed,
            self.commercial_use_allowed,
            self.training_allowed,
            self.evaluation_allowed,
        )
        if any(value is not None and not isinstance(value, bool) for value in permissions):
            raise ExternalDatasetRegistryError("permission fields must be bool or None")

        if self.data_use_class is DataUseClass.LICENSE_REVIEW_REQUIRED:
            if self.registry_state is not RegistryState.CANDIDATE:
                raise ExternalDatasetRegistryError(
                    "license-review-required data must remain CANDIDATE"
                )
            if any(value is not None for value in permissions):
                raise ExternalDatasetRegistryError(
                    "unverified licenses must not assert usage permissions"
                )
            if self.artifact_sha256 is not None:
                raise ExternalDatasetRegistryError(
                    "license-review-required data must not be installation-pinned"
                )
            return

        if self.registry_state is RegistryState.CANDIDATE:
            raise ExternalDatasetRegistryError(
                "verified use classes require LICENSE_VERIFIED or INSTALL_PINNED state"
            )
        if any(value is None for value in permissions):
            raise ExternalDatasetRegistryError(
                "verified licenses require explicit usage permissions"
            )

        if self.data_use_class is DataUseClass.COMMERCIAL_CLEAN:
            if not (
                self.commercial_use_allowed
                and self.training_allowed
                and self.evaluation_allowed
            ):
                raise ExternalDatasetRegistryError(
                    "COMMERCIAL_CLEAN requires commercial, training and evaluation permission"
                )
        elif self.data_use_class is DataUseClass.RESEARCH_ONLY:
            if self.commercial_use_allowed:
                raise ExternalDatasetRegistryError(
                    "RESEARCH_ONLY must prohibit commercial use"
                )
            if not self.training_allowed or not self.evaluation_allowed:
                raise ExternalDatasetRegistryError(
                    "RESEARCH_ONLY must permit research training and evaluation"
                )
        elif self.data_use_class is DataUseClass.EVALUATION_ONLY:
            if self.training_allowed or not self.evaluation_allowed:
                raise ExternalDatasetRegistryError(
                    "EVALUATION_ONLY must prohibit training and permit evaluation"
                )

        if self.registry_state is RegistryState.INSTALL_PINNED:
            _require_sha256("artifact_sha256", self.artifact_sha256)
        elif self.artifact_sha256 is not None:
            raise ExternalDatasetRegistryError(
                "artifact_sha256 is allowed only for INSTALL_PINNED records"
            )

    @property
    def research_training_ready(self) -> bool:
        return (
            self.registry_state is RegistryState.INSTALL_PINNED
            and self.training_allowed is True
        )

    @property
    def evaluation_ready(self) -> bool:
        return (
            self.registry_state is RegistryState.INSTALL_PINNED
            and self.evaluation_allowed is True
        )

    @property
    def commercial_candidate_training_ready(self) -> bool:
        return (
            self.research_training_ready
            and self.data_use_class is DataUseClass.COMMERCIAL_CLEAN
            and self.commercial_use_allowed is True
        )

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return sha256(payload).hexdigest()


def validate_registry(records: Iterable[ExternalDatasetRecord]) -> tuple[ExternalDatasetRecord, ...]:
    values = tuple(records)
    if not values:
        raise ExternalDatasetRegistryError("registry must contain at least one dataset record")
    seen: set[tuple[str, str, str]] = set()
    for record in values:
        if not isinstance(record, ExternalDatasetRecord):
            raise ExternalDatasetRegistryError(
                "registry entries must be ExternalDatasetRecord values"
            )
        identity = (record.dataset_name, record.dataset_component, record.version)
        if identity in seen:
            raise ExternalDatasetRegistryError("duplicate external dataset registry identity")
        seen.add(identity)
    return tuple(
        sorted(
            values,
            key=lambda record: (
                record.dataset_name,
                record.dataset_component,
                record.version,
            ),
        )
    )


def validate_training_admission(
    records: Iterable[ExternalDatasetRecord],
    *,
    commercial_candidate: bool,
) -> tuple[ExternalDatasetRecord, ...]:
    values = validate_registry(records)
    for record in values:
        if not record.research_training_ready:
            raise ExternalDatasetRegistryError(
                f"dataset is not research-training ready: {record.dataset_name} / "
                f"{record.dataset_component}"
            )
        if commercial_candidate and not record.commercial_candidate_training_ready:
            raise ExternalDatasetRegistryError(
                f"dataset is not commercial-candidate clean: {record.dataset_name} / "
                f"{record.dataset_component}"
            )
    return values


def validate_evaluation_admission(
    records: Iterable[ExternalDatasetRecord],
) -> tuple[ExternalDatasetRecord, ...]:
    values = validate_registry(records)
    for record in values:
        if not record.evaluation_ready:
            raise ExternalDatasetRegistryError(
                f"dataset is not evaluation ready: {record.dataset_name} / "
                f"{record.dataset_component}"
            )
    return values


# Legal/source facts below are intentionally conservative. A record can be
# license-verified without being installation-pinned; no listed candidate is
# made training/evaluation-ready by this module.
EXTERNAL_DATASET_CANDIDATES: Final[tuple[ExternalDatasetRecord, ...]] = validate_registry(
    (
        ExternalDatasetRecord(
            dataset_name="Muse OMR Benchmark",
            dataset_component="1077 symbolic-score + augmented-PDF pairs",
            source="https://huggingface.co/datasets/musegroup/omr_benchmark",
            version="dataset card current at TR-POLY-03 review",
            license_id="CC0-1.0",
            license_evidence="https://huggingface.co/datasets/musegroup/omr_benchmark",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            training_allowed=True,
            evaluation_allowed=True,
            derivative_restrictions="CC0-1.0; no license restrictions, attribution appreciated by publisher",
            data_use_class=DataUseClass.COMMERCIAL_CLEAN,
            registry_state=RegistryState.LICENSE_VERIFIED,
            notes="No local artifact checksum is pinned yet; not admitted for training/evaluation.",
        ),
        ExternalDatasetRecord(
            dataset_name="DeepScoresV2",
            dataset_component="complete/dense music-object detection dataset",
            source="https://zenodo.org/records/4012193",
            version="2.0",
            license_id="CC-BY-4.0",
            license_evidence="https://digitalcollection.zhaw.ch/items/1376f518-211d-49c8-a658-dbc7f085d2b1",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            training_allowed=True,
            evaluation_allowed=True,
            derivative_restrictions="Attribution required under CC BY 4.0",
            data_use_class=DataUseClass.COMMERCIAL_CLEAN,
            registry_state=RegistryState.LICENSE_VERIFIED,
            notes="Zenodo publishes MD5 file identities, but ST-OMR requires a separately verified SHA-256 installation pin.",
        ),
        ExternalDatasetRecord(
            dataset_name="MUSCIMA++",
            dataset_component="MUSCIMA++ annotations",
            source="https://github.com/OMR-Research/muscima-pp",
            version="2.1",
            license_id="CC-BY-NC-SA-4.0",
            license_evidence="https://github.com/OMR-Research/muscima-pp/blob/master/LICENSE.txt",
            redistribution_allowed=True,
            commercial_use_allowed=False,
            training_allowed=True,
            evaluation_allowed=True,
            derivative_restrictions="Attribution, NonCommercial and ShareAlike requirements apply",
            data_use_class=DataUseClass.RESEARCH_ONLY,
            registry_state=RegistryState.LICENSE_VERIFIED,
            notes="Underlying CVC-MUSCIMA image acquisition/licensing remains a separate component boundary.",
        ),
        ExternalDatasetRecord(
            dataset_name="OLiMPiC",
            dataset_component="synthetic 1.0",
            source="https://github.com/ufal/olimpic-icdar24/releases",
            version="1.0 / 2024-02-12 release artifact",
            license_id="CC-BY-SA (version unspecified in cited dataset README)",
            license_evidence="https://github.com/ufal/olimpic-icdar24/blob/master/README.md",
            redistribution_allowed=None,
            commercial_use_allowed=None,
            training_allowed=None,
            evaluation_allowed=None,
            derivative_restrictions="Exact CC BY-SA version and upstream OpenScore Lieder obligations require review",
            data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
            registry_state=RegistryState.CANDIDATE,
            notes="Do not download or use until exact dataset-license version/upstream obligations are pinned.",
        ),
        ExternalDatasetRecord(
            dataset_name="OLiMPiC",
            dataset_component="scanned 1.0",
            source="https://github.com/ufal/olimpic-icdar24/releases",
            version="1.0 / 2024-02-12 release artifact",
            license_id="CC-BY-SA (version unspecified in cited dataset README)",
            license_evidence="https://github.com/ufal/olimpic-icdar24/blob/master/README.md",
            redistribution_allowed=None,
            commercial_use_allowed=None,
            training_allowed=None,
            evaluation_allowed=None,
            derivative_restrictions="Exact CC BY-SA version, scanned-source rights and upstream obligations require review",
            data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
            registry_state=RegistryState.CANDIDATE,
            notes="Evaluation harness may be designed later, but bytes remain blocked pending license review.",
        ),
        ExternalDatasetRecord(
            dataset_name="GrandStaff-LMX",
            dataset_component="added .lmx and .musicxml annotations only",
            source="https://github.com/ufal/olimpic-icdar24/releases",
            version="2024-02-12 extension artifact",
            license_id="CC-BY-SA (version unspecified in cited dataset README)",
            license_evidence="https://github.com/ufal/olimpic-icdar24/blob/master/README.md",
            redistribution_allowed=None,
            commercial_use_allowed=None,
            training_allowed=None,
            evaluation_allowed=None,
            derivative_restrictions="License statement covers only added LMX/MusicXML; original GrandStaff remains separate",
            data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
            registry_state=RegistryState.CANDIDATE,
            notes="Combined GrandStaff use is blocked until original dataset terms are independently verified.",
        ),
        ExternalDatasetRecord(
            dataset_name="GrandStaff",
            dataset_component="original pianoform dataset",
            source="https://github.com/multiscore/e2e-pianoform",
            version="original dataset used by IJDAR 2023 work",
            license_id="UNVERIFIED_DATASET_LICENSE",
            license_evidence="https://github.com/multiscore/e2e-pianoform",
            redistribution_allowed=None,
            commercial_use_allowed=None,
            training_allowed=None,
            evaluation_allowed=None,
            derivative_restrictions="Public replication availability is not treated as a dataset license",
            data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
            registry_state=RegistryState.CANDIDATE,
            notes="MIT repository code license must not be confused with GrandStaff dataset rights.",
        ),
        ExternalDatasetRecord(
            dataset_name="DoReMi",
            dataset_component="published openly distributable score subset",
            source="https://github.com/steinbergmedia/DoReMi",
            version="v1.0 publication family",
            license_id="UNVERIFIED_DATASET_LICENSE",
            license_evidence="https://github.com/steinbergmedia/DoReMi/blob/main/README.md",
            redistribution_allowed=None,
            commercial_use_allowed=None,
            training_allowed=None,
            evaluation_allowed=None,
            derivative_restrictions="Copyright-free/openly distributable source statement is not an explicit dataset license grant",
            data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
            registry_state=RegistryState.CANDIDATE,
            notes="Published repo root contains no explicit dataset LICENSE file at TR-POLY-03 review time.",
        ),
    )
)
