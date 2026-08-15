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
    verify_stage7d13_training_preflight,
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


if __name__ == "__main__":
    unittest.main()
