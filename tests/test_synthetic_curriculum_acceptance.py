from __future__ import annotations

import copy
import json
import unittest

from st_omr_training.synthetic_curriculum_acceptance import (
    EXPECTED_BUILD_ID,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_TRANSPORT_SHA256,
    SyntheticCurriculumAcceptanceError,
    verify_synthetic_curriculum_export_evidence,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "st-omr-colab-synthetic-export-v1",
        "source_repository": "https://github.com/khfy7wpr5p-maker/st-omr-training.git",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "runtime": {
            "python": "3.12.13",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "platform_release": "test",
            "cairo_runtime": "1.18.4",
            "packages": {
                "CairoSVG": "2.8.2",
                "Pillow": "12.3.0",
                "lxml": "6.1.1",
                "verovio": "6.2.1",
            },
        },
        "plan": {
            "schema_version": "st-synthetic-curriculum-plan-v1",
            "profile_version": "st-synthetic-curriculum-v1",
            "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
            "family_count": 512,
            "measure_count": 8,
            "raster_width": 1000,
            "degradation_profiles": ["clean", "light", "medium"],
            "family_split_counts": {"test": 51, "train": 410, "validation": 51},
            "family_profile_counts": {
                "chord-only": 64,
                "mixed": 64,
                "no-accidentals": 64,
                "note-only": 64,
                "rest-only": 64,
                "time-2-4": 64,
                "time-3-4": 64,
                "time-4-4": 64,
            },
        },
        "build": {
            "profile_version": "st-synthetic-curriculum-v1",
            "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
            "build_id": EXPECTED_BUILD_ID,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "sample_count": 1536,
            "target_count": 512,
            "image_count": 1536,
        },
        "transport_archive": "st-omr-synthetic-curriculum-v1-d9320e362f162cd2a.tar.gz",
        "transport_sha256": EXPECTED_TRANSPORT_SHA256,
    }


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"


class SyntheticCurriculumAcceptanceTests(unittest.TestCase):
    def test_exact_export_identity_is_accepted(self):
        receipt = verify_synthetic_curriculum_export_evidence(_canonical(_payload()))
        self.assertEqual(receipt.build_id, EXPECTED_BUILD_ID)
        self.assertEqual(receipt.manifest_sha256, EXPECTED_MANIFEST_SHA256)
        self.assertEqual(receipt.config_fingerprint, EXPECTED_CONFIG_FINGERPRINT)
        self.assertEqual(receipt.transport_sha256, EXPECTED_TRANSPORT_SHA256)
        self.assertEqual(len(receipt.evidence_sha256), 64)

    def test_manifest_tamper_is_rejected(self):
        payload = copy.deepcopy(_payload())
        payload["build"]["manifest_sha256"] = "0" * 64
        with self.assertRaises(SyntheticCurriculumAcceptanceError):
            verify_synthetic_curriculum_export_evidence(_canonical(payload))

    def test_split_drift_is_rejected(self):
        payload = copy.deepcopy(_payload())
        payload["plan"]["family_split_counts"] = {"test": 50, "train": 411, "validation": 51}
        with self.assertRaises(SyntheticCurriculumAcceptanceError):
            verify_synthetic_curriculum_export_evidence(_canonical(payload))

    def test_runtime_package_drift_is_rejected(self):
        payload = copy.deepcopy(_payload())
        payload["runtime"]["packages"]["Pillow"] = "12.2.0"
        with self.assertRaises(SyntheticCurriculumAcceptanceError):
            verify_synthetic_curriculum_export_evidence(_canonical(payload))

    def test_noncanonical_json_is_rejected(self):
        raw = json.dumps(_payload(), indent=2, sort_keys=True).encode("ascii") + b"\n"
        with self.assertRaises(SyntheticCurriculumAcceptanceError):
            verify_synthetic_curriculum_export_evidence(raw)


if __name__ == "__main__":
    unittest.main()
