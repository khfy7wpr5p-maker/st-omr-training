from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import unittest

from st_omr_training.stage7d12_symbol_derivative_verifier import (
    EXPECTED_D6_ARTIFACT_BINDING_SHA256,
    EXPECTED_D6_DERIVATIVE_BUILD_ID,
    EXPECTED_D6_MANIFEST_SHA256,
    Stage7D12VerificationError,
    _empty_inventory,
    _validate_symbol_label,
    development_rows,
)
from st_omr_training.stage7d12_symbol_derivatives import (
    STAGE7D12_DERIVATIVE_VERSION,
    STAGE7D12_LABEL_SCHEMA,
)
from st_omr_training.stage7d12_symbol_geometry import STAGE7D12_SYMBOL_GEOMETRY_VERSION
from st_omr_training.stage7d12_symbol_gt_contract import stage7d12_contract_fingerprint
from st_omr_training.stage7d5_geometry import STAGE7D5_TRANSFORM_VERSION


H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64
H5 = "5" * 64
H6 = "6" * 64
H7 = "7" * 64
H8 = "8" * 64
H9 = "9" * 64


class _HostileTestRow(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"D12 verifier touched sealed TEST field: {key}")

    def __iter__(self):
        yield "split"

    def __len__(self) -> int:
        return 1

    def get(self, key: str, default=None):
        if key == "split":
            return "test"
        raise AssertionError(f"D12 verifier touched sealed TEST field: {key}")


def _source() -> dict[str, object]:
    return {
        "sample_id": H1,
        "family_id": "family-1",
        "split": "train",
        "page_number": 1,
        "png_sha256": H2,
        "width": 100,
        "height": 80,
        "source_musicxml_sha256": H3,
        "source_svg_sha256": H4,
        "renderer_config_fingerprint": H5,
        "degradation_config_fingerprint": H6,
    }


def _label() -> dict[str, object]:
    return {
        "schema_version": STAGE7D12_LABEL_SCHEMA,
        "stage7d12_derivative_version": STAGE7D12_DERIVATIVE_VERSION,
        "contract_fingerprint": stage7d12_contract_fingerprint(),
        "sample_id": H1,
        "family_id": "family-1",
        "split": "train",
        "page_number": 1,
        "image": {
            "png_sha256": H2,
            "width": 100,
            "height": 80,
            "mode": "L",
            "image_format": "png",
        },
        "accepted_d6": {
            "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "label_sha256": H7,
        },
        "lineage": {
            "source_musicxml_sha256": H3,
            "source_svg_sha256": H4,
            "renderer_config_fingerprint": H5,
            "degradation_config_fingerprint": H6,
            "symbol_geometry_version": STAGE7D12_SYMBOL_GEOMETRY_VERSION,
            "geometry_instrumentation_fingerprint": H8,
            "geometry_svg_sha256": H9,
            "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
            "geometry_transform_fingerprint": "a" * 64,
        },
        "symbol_geometry": {
            "coordinate_space": "final_png_pixels",
            "view_box": [0.0, 0.0, 1000.0, 800.0],
            "measures": [
                {
                    "measure_number": 1,
                    "renderer_measure_id": "measure-r1",
                    "measure_bbox": {
                        "x_min": 5.0,
                        "y_min": 5.0,
                        "x_max": 95.0,
                        "y_max": 75.0,
                    },
                    "noteheads": [
                        {
                            "canonical_event_id": "m1-e0",
                            "renderer_id": "note-r1",
                            "notehead_bbox": {
                                "x_min": 20.0,
                                "y_min": 30.0,
                                "x_max": 28.0,
                                "y_max": 36.0,
                            },
                            "notehead_center": {"x": 24.0, "y": 33.0},
                            "fill_class": "filled",
                        }
                    ],
                    "rests": [
                        {
                            "canonical_event_id": "m1-e1",
                            "renderer_id": "rest-r1",
                            "rest_bbox": {
                                "x_min": 40.0,
                                "y_min": 25.0,
                                "x_max": 47.0,
                                "y_max": 39.0,
                            },
                            "rest_class": "quarter",
                            "duration_class": "quarter",
                        }
                    ],
                    "accidentals": [
                        {
                            "canonical_event_id": "m1-e0",
                            "renderer_id": "accid-r1",
                            "accidental_bbox": {
                                "x_min": 15.0,
                                "y_min": 28.0,
                                "x_max": 19.0,
                                "y_max": 38.0,
                            },
                            "accidental_class": "sharp",
                        }
                    ],
                }
            ],
        },
    }


class Stage7D12PersistedVerifierTests(unittest.TestCase):
    def test_verifier_test_seal_touches_only_split(self) -> None:
        rows = development_rows(
            [
                _HostileTestRow(),
                {"split": "train", "sample_id": "not-read-here"},
                {"split": "validation", "sample_id": "not-read-here"},
            ]
        )
        self.assertEqual([row.get("split") for row in rows], ["train", "validation"])

    def test_verifier_freezes_independent_accepted_d6_identity(self) -> None:
        self.assertEqual(
            EXPECTED_D6_DERIVATIVE_BUILD_ID,
            "0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519",
        )
        self.assertEqual(
            EXPECTED_D6_MANIFEST_SHA256,
            "e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9",
        )
        self.assertEqual(
            EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1",
        )

    def test_persisted_label_is_reopened_and_inventory_recomputed(self) -> None:
        inventory = _empty_inventory()
        _validate_symbol_label(
            _label(),
            source=_source(),
            d6_record={"sample_id": H1, "png_sha256": H2},
            d6_label_sha=H7,
            d6_geometry_lineage=(H8, H9, "a" * 64),
            inventory=inventory,
        )
        self.assertEqual(inventory["train"]["notehead"]["filled"], 1)
        self.assertEqual(inventory["train"]["rest"]["quarter"], 1)
        self.assertEqual(inventory["train"]["accidental"]["sharp"], 1)
        self.assertEqual(sum(inventory["validation"]["notehead"].values()), 0)

    def test_tampered_notehead_center_fails_closed(self) -> None:
        label = deepcopy(_label())
        label["symbol_geometry"]["measures"][0]["noteheads"][0][
            "notehead_center"
        ] = {"x": 90.0, "y": 70.0}
        with self.assertRaises(Stage7D12VerificationError):
            _validate_symbol_label(
                label,
                source=_source(),
                d6_record={"sample_id": H1, "png_sha256": H2},
                d6_label_sha=H7,
                d6_geometry_lineage=(H8, H9, "a" * 64),
                inventory=_empty_inventory(),
            )

    def test_tampered_d6_transform_lineage_fails_closed(self) -> None:
        label = deepcopy(_label())
        label["lineage"]["geometry_transform_fingerprint"] = "b" * 64
        with self.assertRaises(Stage7D12VerificationError):
            _validate_symbol_label(
                label,
                source=_source(),
                d6_record={"sample_id": H1, "png_sha256": H2},
                d6_label_sha=H7,
                d6_geometry_lineage=(H8, H9, "a" * 64),
                inventory=_empty_inventory(),
            )


if __name__ == "__main__":
    unittest.main()
