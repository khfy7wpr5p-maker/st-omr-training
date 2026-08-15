from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from st_omr_training.stage7d13_symbol_models import build_symbol_model
from st_omr_training.stage7d13_symbol_training_contract import (
    MAX_PARAMETERS_COMBINED,
    SPECIALIST_CLASSES,
)
from st_omr_training.stage7d13_training_preflight import (
    Stage7D13PreflightError,
    _assert_build_identity,
    verify_stage7d13_training_preflight,
)
from st_omr_training.stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_IMAGE_COUNT,
    D13_LABEL_COUNT,
    D13_RECORD_COUNT,
    D13_RECORD_SPLIT_COUNTS,
)
from st_omr_training.training_model import count_trainable_parameters


class Stage7D13TrainingPreflightTests(unittest.TestCase):
    def test_combined_specialist_parameter_budget_is_satisfied(self) -> None:
        counts = {
            specialist: count_trainable_parameters(build_symbol_model(specialist))
            for specialist in SPECIALIST_CLASSES
        }
        self.assertTrue(all(value > 0 for value in counts.values()))
        self.assertLessEqual(sum(counts.values()), MAX_PARAMETERS_COMBINED)

    def test_incomplete_derivative_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(Stage7D13PreflightError):
                verify_stage7d13_training_preflight(Path(temporary))

    def test_build_identity_rejects_artifact_binding_drift_before_record_scan(self) -> None:
        build = {
            "derivative_build_id": D13_DERIVATIVE_BUILD_ID,
            "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
            "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
            "record_count": D13_RECORD_COUNT,
            "image_count": D13_IMAGE_COUNT,
            "label_count": D13_LABEL_COUNT,
            "record_split_counts": D13_RECORD_SPLIT_COUNTS,
            "test_specialist_records": 0,
            "optimizer_steps": 0,
            "complete_marker_written": False,
        }
        _assert_build_identity(build)

        tampered = dict(build)
        tampered["artifact_binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            Stage7D13PreflightError,
            "artifact_binding_sha256 mismatch",
        ):
            _assert_build_identity(tampered)


if __name__ == "__main__":
    unittest.main()
